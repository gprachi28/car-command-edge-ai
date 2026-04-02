"""Fine-tune Llama 3.2 3B, Qwen 2.5 3B, SmolLM2 1.7B using MLX-LM LoRA.

Uses Apple Silicon's native MLX framework instead of PyTorch/MPS.
Mirrors finetune.py in structure for easy comparison, but replaces
HF TRL + PEFT with mlx_lm.lora — which runs entirely on Metal with no
CPU fallbacks and produces adapters ready for mlx_lm quantization.

Why MLX over TRL on Mac:
    - MPS backend in PyTorch has incomplete op coverage → silent CPU fallbacks
    - MLX is Metal-native: all ops run on GPU, unified memory handled correctly
    - Output is already in MLX format, so quantize.py needs no conversion step

Data format expected by mlx_lm.lora:
    data/processed/
        train.jsonl   {"text": "Command: ...\nAction: ..."}
        valid.jsonl   symlinked to test.jsonl (created automatically here)

Pipeline per model:
    1. mlx_lm.lora  → trains LoRA adapter, saves to models/finetuned/<key>-mlx-adapter/
    2. mlx_lm.fuse  → merges adapter into base model, saves to models/finetuned/<key>-mlx/

Public API:
    - run_finetune(model_key, hf_token, processed_dir, output_dir) -> Path
    - finetune_all(hf_token, processed_dir, output_dir) -> dict[str, Path]
"""

import subprocess
import sys
from pathlib import Path

from src.utils import MODEL_IDS, TRAINING_CONFIG, get_logger, get_models_dir, get_processed_dir

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MLX-LM training hyperparameters
# ---------------------------------------------------------------------------

# mlx_lm.lora uses --num-layers (transformer layers to apply LoRA to, from end),
# not LoRA rank directly. Default rank in mlx_lm is 8, matching our LORA_CONFIG.
MLX_LORA_LAYERS = 8        # how many transformer layers get LoRA adapters
MLX_BATCH_SIZE = 4         # 4 is safer than 8 for 3B models on 18 GB unified memory
MLX_GRAD_ACCUM = 2         # effective batch = 4 * 2 = 8, matching TRAINING_CONFIG
MLX_LEARNING_RATE = TRAINING_CONFIG["learning_rate"]   # 2e-4
MLX_MAX_SEQ_LEN = TRAINING_CONFIG["max_seq_length"]    # 256
MLX_STEPS_PER_REPORT = 50
MLX_STEPS_PER_EVAL = 200
MLX_SAVE_EVERY = 200


def _compute_iters(processed_dir: Path, epochs: int, batch_size: int) -> int:
    """Compute training iterations from dataset size and epoch count.

    Args:
        processed_dir: Directory containing train.jsonl.
        epochs: Number of training epochs.
        batch_size: Per-step batch size.

    Returns:
        Total number of training steps.
    """
    train_path = processed_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"train.jsonl not found at {train_path}")
    n_examples = sum(1 for line in train_path.open() if line.strip())
    steps_per_epoch = max(1, n_examples // batch_size)
    iters = steps_per_epoch * epochs
    logger.info(
        "Dataset: %d examples → %d steps/epoch × %d epochs = %d iters",
        n_examples, steps_per_epoch, epochs, iters,
    )
    return iters


def _ensure_valid_split(processed_dir: Path) -> None:
    """Create valid.jsonl as a symlink to test.jsonl if it doesn't exist.

    mlx_lm.lora expects {train, valid}.jsonl in the data directory.
    Our pipeline produces train.jsonl + test.jsonl; this bridges the gap.
    """
    valid_path = processed_dir / "valid.jsonl"
    test_path = processed_dir / "test.jsonl"
    if valid_path.exists() or valid_path.is_symlink():
        return
    if not test_path.exists():
        raise FileNotFoundError(f"test.jsonl not found at {test_path}")
    valid_path.symlink_to(test_path.name)  # relative symlink within same dir
    logger.info("Created valid.jsonl → test.jsonl symlink in %s", processed_dir)


def run_finetune(
    model_key: str,
    hf_token: str,
    processed_dir: Path | None = None,
    output_dir: Path | None = None,
    epochs: int = TRAINING_CONFIG["num_epochs"],
) -> Path:
    """Fine-tune a single model with MLX-LM LoRA and fuse the adapter.

    Steps:
        1. Ensure valid.jsonl exists (symlinked to test.jsonl)
        2. Train LoRA adapter via mlx_lm.lora
        3. Fuse adapter into base model via mlx_lm.fuse
        4. Save fused model to output_dir/<model_key>-mlx/

    Args:
        model_key: One of "llama", "qwen", "smollm2".
        hf_token: HuggingFace API token (required for gated models like Llama).
        processed_dir: Directory with train.jsonl + test.jsonl. Defaults to data/processed/.
        output_dir: Parent dir for saving outputs. Defaults to models/finetuned/.
        epochs: Number of training epochs (default: 3).

    Returns:
        Path to the saved fused model directory (<model_key>-mlx/).
    """
    if model_key not in MODEL_IDS:
        raise KeyError(f"Unknown model key '{model_key}'. Choose from: {list(MODEL_IDS)}")
    if processed_dir is None:
        processed_dir = get_processed_dir()
    if output_dir is None:
        output_dir = get_models_dir() / "finetuned"

    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    model_id = MODEL_IDS[model_key]

    adapter_path = output_dir / f"{model_key}-mlx-adapter"
    fused_path = output_dir / f"{model_key}-mlx"
    adapter_path.mkdir(parents=True, exist_ok=True)
    fused_path.mkdir(parents=True, exist_ok=True)

    _ensure_valid_split(processed_dir)

    iters = _compute_iters(processed_dir, epochs=epochs, batch_size=MLX_BATCH_SIZE)

    # --- Step 1: Train LoRA adapter ---
    logger.info("=== Training LoRA adapter: %s ===", model_id)
    train_cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_id,
        "--train",
        "--data", str(processed_dir),
        "--adapter-path", str(adapter_path),
        "--num-layers", str(MLX_LORA_LAYERS),
        "--batch-size", str(MLX_BATCH_SIZE),
        "--grad-accumulation-steps", str(MLX_GRAD_ACCUM),
        "--iters", str(iters),
        "--learning-rate", str(MLX_LEARNING_RATE),
        "--max-seq-length", str(MLX_MAX_SEQ_LEN),
        "--steps-per-report", str(MLX_STEPS_PER_REPORT),
        "--steps-per-eval", str(MLX_STEPS_PER_EVAL),
        "--save-every", str(MLX_SAVE_EVERY),
        "--val-batches", "-1",   # use full validation set
        "--mask-prompt",         # compute loss on Action output only, not Command input
        "--grad-checkpoint",     # reduce memory for 3B models
        "--seed", "42",
    ]

    _run_subprocess(train_cmd, step="lora-train", model_key=model_key)

    # --- Step 2: Fuse adapter into base model ---
    logger.info("=== Fusing adapter into base model: %s ===", model_id)
    fuse_cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model_id,
        "--adapter-path", str(adapter_path),
        "--save-path", str(fused_path),
    ]

    _run_subprocess(fuse_cmd, step="fuse", model_key=model_key)

    logger.info("Fused model saved to %s", fused_path)
    return fused_path


def finetune_all(
    hf_token: str,
    processed_dir: Path | None = None,
    output_dir: Path | None = None,
    epochs: int = TRAINING_CONFIG["num_epochs"],
) -> dict[str, Path]:
    """Fine-tune all three models sequentially with MLX-LM LoRA.

    Args:
        hf_token: HuggingFace API token.
        processed_dir: Directory with train.jsonl + test.jsonl. Defaults to data/processed/.
        output_dir: Parent dir for outputs. Defaults to models/finetuned/.
        epochs: Number of training epochs (default: 3).

    Returns:
        Dict mapping model_key -> fused model path.
    """
    results: dict[str, Path] = {}
    for model_key in MODEL_IDS:
        logger.info("=== Fine-tuning: %s ===", model_key)
        results[model_key] = run_finetune(
            model_key=model_key,
            hf_token=hf_token,
            processed_dir=processed_dir,
            output_dir=output_dir,
            epochs=epochs,
        )
    logger.info("All models fine-tuned: %s", list(results.keys()))
    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _run_subprocess(cmd: list[str], step: str, model_key: str) -> None:
    """Run a subprocess command, streaming output and raising on failure.

    Args:
        cmd: Command and arguments list.
        step: Label for logging (e.g. "lora-train").
        model_key: Model being processed, for error context.

    Raises:
        subprocess.CalledProcessError: If the command exits non-zero.
    """
    logger.info("[%s | %s] Running: %s", model_key, step, " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            f"[{model_key} | {step}] exited with code {result.returncode}",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from src.utils import load_config

    parser = argparse.ArgumentParser(description="Fine-tune all models with MLX-LM LoRA")
    parser.add_argument(
        "--model",
        choices=[*list(MODEL_IDS.keys()), "all"],
        default="all",
        help="Which model to fine-tune (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_CONFIG["num_epochs"],
        help=f"Training epochs (default: {TRAINING_CONFIG['num_epochs']})",
    )
    args = parser.parse_args()

    config = load_config()

    if args.model == "all":
        finetune_all(hf_token=config["hf_token"], epochs=args.epochs)
    else:
        run_finetune(model_key=args.model, hf_token=config["hf_token"], epochs=args.epochs)
