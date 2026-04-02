"""Shared dataset utilities: split, save, and log training pairs.

Backend-agnostic — works with any utterance source (Ollama, Gemini, etc.).
Used by generate_dataset.py; not an entry point.

Public API:
    - split_dataset(examples, train_ratio, seed) -> tuple[list, list]
    - save_dataset(train, test, output_dir) -> None
    - log_metadata(train, test) -> None

See src/generate_dataset.py for the generation entry point.
"""

import json
import random
from pathlib import Path

from src.utils import get_logger, get_processed_dir

logger = get_logger(__name__)


def split_dataset(
    examples: list[dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Stratified shuffle-split by intent.

    Args:
        examples: List of {"utterance": ..., "intent": ..., "slots": {...}} dicts.
        train_ratio: Fraction for training (default 0.8).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_examples, test_examples).
    """
    rng = random.Random(seed)
    by_intent: dict[str, list] = {}
    for ex in examples:
        by_intent.setdefault(ex["intent"], []).append(ex)

    train, test = [], []
    for intent_examples in by_intent.values():
        shuffled = list(intent_examples)
        rng.shuffle(shuffled)
        # Ensure at least 1 example goes to train even for single-example intents
        split = max(1, int(len(shuffled) * train_ratio))
        train.extend(shuffled[:split])
        test.extend(shuffled[split:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def save_dataset(
    train: list[dict],
    test: list[dict],
    output_dir: Path | None = None,
) -> None:
    """Save train and test splits as JSONL in fine-tuning format.

    Each line: {"text": "Command: <utterance>\\nAction: <intent> <slots_json>"}

    Args:
        train: Training examples.
        test: Test examples.
        output_dir: Directory to write train.jsonl and test.jsonl.
            Defaults to data/processed/.
    """
    if output_dir is None:
        output_dir = get_processed_dir()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(train, output_dir / "train.jsonl")
    _write_jsonl(test, output_dir / "test.jsonl")
    logger.info(
        "Saved %d train / %d test examples to %s", len(train), len(test), output_dir
    )


def log_metadata(
    train: list[dict],
    test: list[dict],
) -> None:
    """Log dataset statistics: counts, intent distribution, avg utterance length.

    Args:
        train: Training examples.
        test: Test examples.
    """
    all_examples = train + test
    intents: dict[str, int] = {}
    lengths: list[int] = []

    for ex in all_examples:
        intents[ex["intent"]] = intents.get(ex["intent"], 0) + 1
        lengths.append(len(ex["utterance"].split()))

    avg_len = sum(lengths) / len(lengths) if lengths else 0

    logger.info("--- Dataset Metadata ---")
    logger.info(
        "Total examples : %d (train=%d, test=%d)",
        len(all_examples),
        len(train),
        len(test),
    )
    logger.info("Unique intents : %d", len(intents))
    logger.info("Avg utterance  : %.1f words", avg_len)
    logger.info("Intent distribution:")
    for intent, count in sorted(intents.items(), key=lambda x: -x[1]):
        logger.info("  %-30s %d", intent, count)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _write_jsonl(examples: list[dict], path: Path) -> None:
    """Write examples as JSONL in fine-tuning format."""
    with path.open("w") as f:
        for ex in examples:
            action = json.dumps({"intent": ex["intent"], "slots": ex.get("slots", {})})
            record = {"text": f"Command: {ex['utterance']}\nAction: {action}"}
            f.write(json.dumps(record) + "\n")
