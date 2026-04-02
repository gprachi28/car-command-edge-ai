"""Fine-tuning pipeline using HF TRL + LoRA for car command intent classification.

Loads each base model, applies LoRA adapters, trains on the processed JSONL
dataset, and saves merged checkpoints ready for MLX quantization.

Public API:
    - load_base_model(model_key, hf_token) -> tuple[AutoModelForCausalLM, AutoTokenizer]
    - build_lora_model(model) -> PeftModel
    - load_train_dataset(processed_dir) -> Dataset
    - run_finetune(model_key, hf_token, processed_dir, output_dir) -> Path
    - finetune_all(hf_token, processed_dir, output_dir) -> dict[str, Path]
"""

from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.utils import (
    LORA_CONFIG,
    MODEL_IDS,
    TRAINING_CONFIG,
    get_logger,
    get_models_dir,
    get_processed_dir,
)

logger = get_logger(__name__)


def load_base_model(
    model_key: str,
    hf_token: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Download and load a base model and tokenizer from HuggingFace.

    Args:
        model_key: One of "llama", "qwen", "smollm2".
        hf_token: HuggingFace API token.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        KeyError: If model_key is not in MODEL_IDS.
    """
    if model_key not in MODEL_IDS:
        raise KeyError(f"Unknown model key '{model_key}'. Choose from: {list(MODEL_IDS)}")

    model_id = MODEL_IDS[model_key]
    logger.info("Loading base model: %s", model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map="auto" does not route to MPS on Apple Silicon.
    # Explicit MPS mapping + bfloat16 (requires PyTorch >= 2.1) halves memory
    # vs float32: 3B model uses ~3 GB instead of ~6 GB.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map={"": "mps"} if torch.backends.mps.is_available() else "cpu",
    )

    logger.info("Loaded %s (%s params)", model_id, _count_params(model))
    return model, tokenizer


def build_lora_model(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Wrap a base model with LoRA adapters.

    Uses the LORA_CONFIG constants from utils: rank=8, alpha=16,
    targeting q_proj and v_proj attention layers.

    Args:
        model: Base causal LM loaded via load_base_model.

    Returns:
        Model with LoRA adapters attached.
    """
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        target_modules=LORA_CONFIG["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_train_dataset(processed_dir: Path | None = None) -> Dataset:
    """Load the training JSONL file as a HuggingFace Dataset.

    Args:
        processed_dir: Directory containing train.jsonl.
            Defaults to data/processed/.

    Returns:
        HuggingFace Dataset with a single "text" column.

    Raises:
        FileNotFoundError: If train.jsonl does not exist.
    """
    if processed_dir is None:
        processed_dir = get_processed_dir()
    train_path = Path(processed_dir) / "train.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"train.jsonl not found at {train_path}. Run dataset.py first."
        )

    dataset = load_dataset("json", data_files=str(train_path), split="train")
    logger.info("Loaded training dataset: %d examples", len(dataset))
    return dataset


def run_finetune(
    model_key: str,
    hf_token: str,
    processed_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Fine-tune a single model with LoRA and save the merged checkpoint.

    Steps:
        1. Load base model + tokenizer
        2. Attach LoRA adapters
        3. Load training data
        4. Train with SFTTrainer (HF TRL)
        5. Merge LoRA weights into base model
        6. Save merged model + tokenizer to output_dir/<model_key>-finetuned/

    Args:
        model_key: One of "llama", "qwen", "smollm2".
        hf_token: HuggingFace API token.
        processed_dir: Directory with train.jsonl. Defaults to data/processed/.
        output_dir: Parent dir for saving checkpoints.
            Defaults to models/finetuned/.

    Returns:
        Path to the saved merged model directory.
    """
    if output_dir is None:
        output_dir = get_models_dir() / "finetuned"
    save_path = Path(output_dir) / f"{model_key}-finetuned"
    save_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_base_model(model_key, hf_token)
    model = build_lora_model(model)
    dataset = load_train_dataset(processed_dir)

    # bf16 requires PyTorch >= 2.1 and MPS / CUDA with bfloat16 support.
    use_bf16 = torch.backends.mps.is_available() or (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    sft_config = SFTConfig(
        output_dir=str(save_path / "checkpoints"),
        num_train_epochs=TRAINING_CONFIG["num_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        max_seq_length=TRAINING_CONFIG["max_seq_length"],
        dataset_text_field="text",
        logging_steps=50,
        save_strategy="epoch",
        fp16=False,
        bf16=use_bf16,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting fine-tuning for %s", model_key)
    trainer.train()
    logger.info("Training complete for %s", model_key)

    # Merge LoRA weights into base model before saving
    logger.info("Merging LoRA weights into base model")
    merged = model.merge_and_unload()
    merged.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    logger.info("Saved merged model to %s", save_path)

    # Release MPS memory before the next model load
    del model, merged
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return save_path


def finetune_all(
    hf_token: str,
    processed_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Fine-tune all three models sequentially.

    Args:
        hf_token: HuggingFace API token.
        processed_dir: Directory with train.jsonl. Defaults to data/processed/.
        output_dir: Parent dir for saving checkpoints.
            Defaults to models/finetuned/.

    Returns:
        Dict mapping model_key -> saved checkpoint path.
    """
    results: dict[str, Path] = {}
    for model_key in MODEL_IDS:
        logger.info("=== Fine-tuning: %s ===", model_key)
        results[model_key] = run_finetune(
            model_key,
            hf_token=hf_token,
            processed_dir=processed_dir,
            output_dir=output_dir,
        )
    logger.info("All models fine-tuned: %s", list(results.keys()))
    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _count_params(model: AutoModelForCausalLM) -> str:
    """Return human-readable parameter count."""
    total = sum(p.numel() for p in model.parameters())
    if total >= 1e9:
        return f"{total / 1e9:.1f}B"
    return f"{total / 1e6:.0f}M"


if __name__ == "__main__":
    from src.utils import load_config

    config = load_config()
    finetune_all(hf_token=config["hf_token"])
