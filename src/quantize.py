"""Quantize fine-tuned MLX models to 4-bit and 8-bit variants.

Reads fused models from models/finetuned/<key>-mlx/ (already in MLX format)
and applies mlx_lm convert quantization, saving to models/quantized/.

Pipeline per model:
    models/finetuned/<key>-mlx/  →  mlx_lm convert -q --q-bits 4
                                 →  models/quantized/<key>-4bit/
                                 →  mlx_lm convert -q --q-bits 8
                                 →  models/quantized/<key>-8bit/

Public API:
    - quantize_model(model_key, bits, finetuned_dir, output_dir) -> Path
    - quantize_all(finetuned_dir, output_dir) -> dict[str, dict[int, Path]]
"""

import subprocess
import sys
from pathlib import Path

from src.utils import MODEL_IDS, dir_size_mb, get_logger, get_models_dir

logger = get_logger(__name__)

QUANT_BITS = [4, 8]


def quantize_model(
    model_key: str,
    bits: int,
    finetuned_dir: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
    run_suffix: str = "",
) -> Path:
    """Quantize a single fused MLX model to the specified bit depth.

    Args:
        model_key: One of "llama", "qwen", "smollm2".
        bits: Quantization bit depth (4 or 8).
        finetuned_dir: Directory containing <key>-mlx/ fused models.
            Defaults to models/finetuned/.
        output_dir: Directory for quantized outputs.
            Defaults to models/quantized/.
        force: If True, overwrite existing quantized output. Default False.
        run_suffix: Optional suffix matching the one used in finetune_mlx.py
            (e.g. "-v2"), applied to both source and destination paths.

    Returns:
        Path to the saved quantized model directory.

    Raises:
        KeyError: If model_key is not recognised.
        FileNotFoundError: If the fused model directory does not exist.
        ValueError: If bits is not 4 or 8.
        subprocess.CalledProcessError: If mlx_lm convert fails.
    """
    if model_key not in MODEL_IDS:
        raise KeyError(
            f"Unknown model key '{model_key}'. Choose from: {list(MODEL_IDS)}"
        )
    if bits not in QUANT_BITS:
        raise ValueError(f"bits must be one of {QUANT_BITS}, got {bits}")

    if finetuned_dir is None:
        finetuned_dir = get_models_dir() / "finetuned"
    if output_dir is None:
        output_dir = get_models_dir() / "quantized"

    finetuned_dir = Path(finetuned_dir)
    output_dir = Path(output_dir)

    src_path = finetuned_dir / f"{model_key}-mlx{run_suffix}"
    dst_path = output_dir / f"{model_key}-{bits}bit{run_suffix}"

    if not src_path.exists():
        raise FileNotFoundError(
            f"Fused model not found at {src_path}. Run finetune_mlx.py first."
        )

    if dst_path.exists() and not force:
        logger.info(
            "Skipping %s %dbit — already exists at %s (use --force to overwrite)",
            model_key,
            bits,
            dst_path,
        )
        return dst_path

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    src_size_mb = dir_size_mb(src_path)
    logger.info(
        "Quantizing %s to %d-bit | src: %.0f MB → %s",
        model_key,
        bits,
        src_size_mb,
        dst_path,
    )

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        str(src_path),
        "--mlx-path",
        str(dst_path),
        "--quantize",
        "--q-bits",
        str(bits),
    ]

    logger.info("[%s | %dbit] Running: %s", model_key, bits, " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            f"[{model_key} | {bits}bit] mlx_lm convert exited with code"
            f" {result.returncode}",
        )

    dst_size_mb = dir_size_mb(dst_path)
    compression = (1 - dst_size_mb / src_size_mb) * 100 if src_size_mb > 0 else 0
    logger.info(
        "Done: %s %dbit | %.0f MB → %.0f MB (%.1f%% reduction)",
        model_key,
        bits,
        src_size_mb,
        dst_size_mb,
        compression,
    )
    return dst_path


def quantize_all(
    bits_to_run: list[int] = QUANT_BITS,
    finetuned_dir: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict[str, dict[int, Path]]:
    """Quantize all three models to the specified bit depths.

    Args:
        bits_to_run: List of bit depths to quantize to. Defaults to [4, 8].
        finetuned_dir: Directory containing fused models. Defaults to models/finetuned/.
        output_dir: Directory for outputs. Defaults to models/quantized/.
        force: If True, overwrite existing outputs. Default False.

    Returns:
        Nested dict: {model_key: {bits: quantized_path}}
    """
    results: dict[str, dict[int, Path]] = {}
    for model_key in MODEL_IDS:
        results[model_key] = {}
        for bits in bits_to_run:
            logger.info("=== Quantizing %s to %d-bit ===", model_key, bits)
            results[model_key][bits] = quantize_model(
                model_key=model_key,
                bits=bits,
                finetuned_dir=finetuned_dir,
                output_dir=output_dir,
                force=force,
            )
    logger.info("All quantization complete: %s", list(results.keys()))
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize fine-tuned MLX models")
    parser.add_argument(
        "--model",
        choices=[*MODEL_IDS, "all"],
        default="all",
        help="Which model to quantize (default: all)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=QUANT_BITS,
        default=None,
        help="Bit depth to quantize to (default: both 4 and 8)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing quantized outputs",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Run suffix used during fine-tuning (e.g. use --suffix=-v2)",
    )
    args = parser.parse_args()

    bits_to_run = [args.bits] if args.bits else QUANT_BITS

    if args.model == "all":
        quantize_all(bits_to_run=bits_to_run, force=args.force)
    else:
        for b in bits_to_run:
            quantize_model(
                model_key=args.model, bits=b, force=args.force, run_suffix=args.suffix
            )
