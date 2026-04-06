"""Plot training and validation loss curves from loss log files.

Reads JSON loss logs produced by finetune_mlx.py and writes PNG plots to
data/results/loss_curves/.

Each log file: data/results/loss_curves/<model_key><suffix>.json
    [{"iter": N, "train_loss": X, "val_loss": Y}, ...]
    (val_loss may be absent on train-only entries)

Public API:
    - plot_loss_curve(model_key, run_suffix, output_dir) -> Path
    - plot_all(output_dir) -> list[Path]
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils import get_data_dir, get_logger

logger = get_logger(__name__)


def plot_loss_curve(
    model_key: str,
    run_suffix: str = "",
    output_dir: Path | None = None,
) -> Path:
    """Plot train + val loss for one model run.

    Args:
        model_key: e.g. "llama", "qwen", "smollm2"
        run_suffix: Optional suffix used when training (e.g. "-v2").
        output_dir: Directory to write PNG to. Defaults to data/results/loss_curves/.

    Returns:
        Path to saved PNG.
    """
    loss_dir = get_data_dir() / "results" / "loss_curves"
    log_path = loss_dir / f"{model_key}{run_suffix}.json"
    if not log_path.exists():
        raise FileNotFoundError(f"Loss log not found: {log_path}")

    with log_path.open() as f:
        entries = json.load(f)

    train_iters = [e["iter"] for e in entries if "train_loss" in e]
    train_losses = [e["train_loss"] for e in entries if "train_loss" in e]
    val_iters = [e["iter"] for e in entries if "val_loss" in e]
    val_losses = [e["val_loss"] for e in entries if "val_loss" in e]

    fig, ax = plt.subplots(figsize=(8, 4))
    if train_iters:
        ax.plot(train_iters, train_losses, label="Train loss", linewidth=1.5)
    if val_iters:
        ax.plot(val_iters, val_losses, label="Val loss", marker="o", linewidth=1.5)

    title = f"{model_key}{run_suffix} — loss curve"
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_dir is None:
        output_dir = loss_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_key}{run_suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved loss curve to %s", out_path)
    return out_path


def plot_all(output_dir: Path | None = None) -> list[Path]:
    """Plot loss curves for every JSON log in data/results/loss_curves/.

    Args:
        output_dir: Where to save PNGs. Defaults to same directory as logs.

    Returns:
        List of paths to saved PNGs.
    """
    loss_dir = get_data_dir() / "results" / "loss_curves"
    log_files = sorted(loss_dir.glob("*.json"))
    if not log_files:
        logger.warning("No loss log files found in %s", loss_dir)
        return []

    paths = []
    for log_file in log_files:
        stem = log_file.stem  # e.g. "llama-v2"
        # Use the full stem directly — no model_key/suffix split needed
        try:
            p = _plot_from_stem(stem, output_dir)
            paths.append(p)
        except Exception as exc:
            logger.error("Failed to plot %s: %s", log_file, exc)
    return paths


def _plot_from_stem(stem: str, output_dir: Path | None) -> Path:
    """Plot directly from a loss-log file stem (no model_key/suffix split needed)."""
    loss_dir = get_data_dir() / "results" / "loss_curves"
    log_path = loss_dir / f"{stem}.json"
    with log_path.open() as f:
        entries = json.load(f)

    train_iters = [e["iter"] for e in entries if "train_loss" in e]
    train_losses = [e["train_loss"] for e in entries if "train_loss" in e]
    val_iters = [e["iter"] for e in entries if "val_loss" in e]
    val_losses = [e["val_loss"] for e in entries if "val_loss" in e]

    fig, ax = plt.subplots(figsize=(8, 4))
    if train_iters:
        ax.plot(train_iters, train_losses, label="Train loss", linewidth=1.5)
    if val_iters:
        ax.plot(val_iters, val_losses, label="Val loss", marker="o", linewidth=1.5)

    ax.set_title(f"{stem} — loss curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_dir is None:
        output_dir = loss_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved loss curve to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot training loss curves")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model stem to plot (e.g. 'llama-v2') or 'all' for every log",
    )
    args = parser.parse_args()

    if args.model == "all":
        saved = plot_all()
        print(f"Plotted {len(saved)} curve(s):")
        for p in saved:
            print(f"  {p}")
    else:
        p = _plot_from_stem(args.model, output_dir=None)
        print(f"Saved: {p}")
