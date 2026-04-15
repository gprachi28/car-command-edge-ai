"""Fine-tune Llama 3.2 3B, Qwen 2.5 3B, SmolLM2 1.7B using MLX-LM LoRA.

Uses Apple Silicon's native MLX framework instead of PyTorch/MPS.
Runs entirely on Metal via mlx_lm.lora with no CPU fallbacks.
Produces adapters ready for mlx_lm quantization.

Why MLX over TRL on Mac:
    - MPS backend in PyTorch has incomplete op coverage → silent CPU fallbacks
    - MLX is Metal-native: all ops run on GPU, unified memory handled correctly
    - Output is already in MLX format, so quantize.py needs no conversion step

Data format expected by mlx_lm.lora:
    data/processed/
        train.jsonl   {"text": "Command: ...\nAction: ..."}
        valid.jsonl   symlinked to test.jsonl (created automatically here)

Pipeline per model:
    1. mlx_lm.lora  → trains LoRA adapter, saves to
                       models/finetuned/<key>-mlx-adapter/
    2. mlx_lm.fuse  → merges adapter into base model, saves to
                       models/finetuned/<key>-mlx/

Public API:
    - run_finetune(model_key, hf_token, processed_dir, output_dir) -> Path
    - finetune_all(hf_token, processed_dir, output_dir) -> dict[str, Path]
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from src.utils import (
    MODEL_IDS,
    TRAINING_CONFIG,
    get_data_dir,
    get_logger,
    get_models_dir,
    get_processed_dir,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MLX-LM training hyperparameters
# ---------------------------------------------------------------------------

# mlx_lm.lora uses --num-layers (transformer layers to apply LoRA to, from end),
# not LoRA rank directly. Default rank in mlx_lm is 8, matching our LORA_CONFIG.
MLX_LORA_LAYERS = 8  # how many transformer layers get LoRA adapters
MLX_BATCH_SIZE = 4  # 4 is safer than 8 for 3B models on 18 GB unified memory
MLX_GRAD_ACCUM = 2  # effective batch = 4 * 2 = 8, matching TRAINING_CONFIG
MLX_LEARNING_RATE = TRAINING_CONFIG["learning_rate"]  # 2e-4
MLX_MAX_SEQ_LEN = TRAINING_CONFIG["max_seq_length"]  # 256
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
        n_examples,
        steps_per_epoch,
        epochs,
        iters,
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
    learning_rate: float = MLX_LEARNING_RATE,
    lora_rank: int = 8,
    run_suffix: str = "",
) -> Path:
    """Fine-tune a single model with MLX-LM LoRA and fuse the adapter.

    Steps:
        1. Ensure valid.jsonl exists (symlinked to test.jsonl)
        2. Train LoRA adapter via mlx_lm.lora
        3. Fuse adapter into base model via mlx_lm.fuse
        4. Save fused model to output_dir/<model_key>-mlx<suffix>/

    Args:
        model_key: One of "llama", "qwen", "smollm2".
        hf_token: HuggingFace API token (required for gated models like Llama).
        processed_dir: Directory with train.jsonl + test.jsonl.
            Defaults to data/processed/.
        output_dir: Parent dir for saving outputs. Defaults to models/finetuned/.
        epochs: Number of training epochs (default: 3).
        learning_rate: Adam learning rate (default: 2e-4).
        lora_rank: LoRA rank (default: 8). Passed via YAML config to mlx_lm.
        run_suffix: Optional suffix for output directory names (e.g. "-v2").

    Returns:
        Path to the saved fused model directory (<model_key>-mlx<suffix>/).
    """
    if model_key not in MODEL_IDS:
        raise KeyError(
            f"Unknown model key '{model_key}'. Choose from: {list(MODEL_IDS)}"
        )
    if processed_dir is None:
        processed_dir = get_processed_dir()
    if output_dir is None:
        output_dir = get_models_dir() / "finetuned"

    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    model_id = MODEL_IDS[model_key]

    adapter_path = output_dir / f"{model_key}-mlx-adapter{run_suffix}"
    fused_path = output_dir / f"{model_key}-mlx{run_suffix}"
    adapter_path.mkdir(parents=True, exist_ok=True)
    fused_path.mkdir(parents=True, exist_ok=True)

    _ensure_valid_split(processed_dir)

    iters = _compute_iters(processed_dir, epochs=epochs, batch_size=MLX_BATCH_SIZE)

    # --- Step 1: Train LoRA adapter ---
    logger.info(
        "=== Training LoRA adapter: %s | lr=%.0e rank=%d suffix='%s' ===",
        model_id,
        learning_rate,
        lora_rank,
        run_suffix,
    )
    train_cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        model_id,
        "--train",
        "--data",
        str(processed_dir),
        "--adapter-path",
        str(adapter_path),
        "--num-layers",
        str(MLX_LORA_LAYERS),
        "--batch-size",
        str(MLX_BATCH_SIZE),
        "--grad-accumulation-steps",
        str(MLX_GRAD_ACCUM),
        "--iters",
        str(iters),
        "--learning-rate",
        str(learning_rate),
        "--max-seq-length",
        str(MLX_MAX_SEQ_LEN),
        "--steps-per-report",
        str(MLX_STEPS_PER_REPORT),
        "--steps-per-eval",
        str(MLX_STEPS_PER_EVAL),
        "--save-every",
        str(MLX_SAVE_EVERY),
        "--val-batches",
        "-1",
        "--grad-checkpoint",
        "--seed",
        "42",
    ]

    # Write YAML config for non-default LoRA rank (rank can only be set via config file)
    yaml_config_path = None
    if lora_rank != 8:
        yaml_config = {
            "lora_parameters": {"rank": lora_rank, "dropout": 0.0, "scale": 20.0}
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(yaml_config, tmp)
        tmp.flush()
        yaml_config_path = Path(tmp.name)
        train_cmd += ["-c", str(yaml_config_path)]
        logger.info("LoRA rank=%d set via config: %s", lora_rank, yaml_config_path)

    loss_log = _run_subprocess_capture(
        train_cmd, step="lora-train", model_key=model_key
    )
    _save_loss_log(loss_log, model_key=model_key, run_suffix=run_suffix)

    if yaml_config_path and yaml_config_path.exists():
        yaml_config_path.unlink()

    # --- Step 2: Fuse adapter into base model ---
    logger.info("=== Fusing adapter into base model: %s ===", model_id)
    fuse_cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "fuse",
        "--model",
        model_id,
        "--adapter-path",
        str(adapter_path),
        "--save-path",
        str(fused_path),
    ]

    _run_subprocess_capture(fuse_cmd, step="fuse", model_key=model_key)

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
        processed_dir: Directory with train.jsonl + test.jsonl.
            Defaults to data/processed/.
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


def _run_subprocess_capture(cmd: list[str], step: str, model_key: str) -> list[dict]:
    """Run a subprocess, stream output to terminal, and return parsed loss entries.

    Parses lines of the form:
        "Iter N: Train loss X, ..."
        "Iter N: Val loss X, ..."

    Args:
        cmd: Command and arguments list.
        step: Label for logging (e.g. "lora-train").
        model_key: Model being processed, for error context.

    Returns:
        List of dicts with keys: iter, train_loss, val_loss (either may be None).

    Raises:
        subprocess.CalledProcessError: If the command exits non-zero.
    """
    logger.info("[%s | %s] Running: %s", model_key, step, " ".join(cmd))

    train_re = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+)")
    val_re = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")

    loss_by_iter: dict[int, dict] = {}

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)  # stream to terminal
        tm = train_re.search(line)
        if tm:
            it = int(tm.group(1))
            loss_by_iter.setdefault(it, {"iter": it})["train_loss"] = float(tm.group(2))
        vm = val_re.search(line)
        if vm:
            it = int(vm.group(1))
            loss_by_iter.setdefault(it, {"iter": it})["val_loss"] = float(vm.group(2))

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            cmd,
            f"[{model_key} | {step}] exited with code {proc.returncode}",
        )

    return sorted(loss_by_iter.values(), key=lambda x: x["iter"])


def _save_loss_log(
    loss_log: list[dict],
    model_key: str,
    run_suffix: str = "",
) -> None:
    """Save parsed loss entries to data/results/loss_curves/<model_key><suffix>.json.

    Args:
        loss_log: List of loss dicts from _run_subprocess_capture.
        model_key: Model key for filename.
        run_suffix: Optional suffix (e.g. "-v2").
    """
    if not loss_log:
        return
    out_dir = get_data_dir() / "results" / "loss_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}{run_suffix}.json"
    with out_path.open("w") as f:
        json.dump(loss_log, f, indent=2)
    logger.info("Loss log saved to %s (%d entries)", out_path, len(loss_log))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from src.utils import load_config

    parser = argparse.ArgumentParser(
        description="Fine-tune all models with MLX-LM LoRA"
    )
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
    parser.add_argument(
        "--lr",
        type=float,
        default=MLX_LEARNING_RATE,
        help=f"Learning rate (default: {MLX_LEARNING_RATE})",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix for output directory names, e.g. '-v2' (default: '')",
    )
    args = parser.parse_args()

    config = load_config()

    if args.model == "all":
        finetune_all(hf_token=config["hf_token"], epochs=args.epochs)
    else:
        run_finetune(
            model_key=args.model,
            hf_token=config["hf_token"],
            epochs=args.epochs,
            learning_rate=args.lr,
            lora_rank=args.lora_rank,
            run_suffix=args.suffix,
        )
