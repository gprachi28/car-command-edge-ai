"""Tests for src/dataset.py — synthetic dataset split/save/metadata utilities."""

import json
from pathlib import Path

from src.dataset import log_metadata, save_dataset, split_dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INTENTS = [
    "set_climate",
    "navigate",
    "play_media",
    "adjust_volume",
    "call_contact",
]


def _make_examples(n: int, intents: list[str] | None = None) -> list[dict]:
    """Generate synthetic example dicts for testing."""
    intents = intents or INTENTS
    return [
        {
            "utterance": f"utterance {i}",
            "intent": intents[i % len(intents)],
            "slots": {"zone": "front"} if i % 2 == 0 else {},
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------


def test_split_dataset_ratio() -> None:
    """80/20 split produces correct total length."""
    examples = _make_examples(100)
    train, test = split_dataset(examples, train_ratio=0.8, seed=42)

    assert len(train) + len(test) == 100
    # Allow ±1 per intent due to stratification rounding
    assert 78 <= len(train) <= 82


def test_split_dataset_deterministic() -> None:
    """Same seed produces identical splits."""
    examples = _make_examples(50)
    train1, test1 = split_dataset(examples, seed=42)
    train2, test2 = split_dataset(examples, seed=42)

    assert train1 == train2
    assert test1 == test2


def test_split_dataset_no_overlap() -> None:
    """Train and test sets share no utterances."""
    examples = _make_examples(50)
    train, test = split_dataset(examples, seed=42)

    train_utts = {ex["utterance"] for ex in train}
    test_utts = {ex["utterance"] for ex in test}
    assert train_utts.isdisjoint(test_utts)


def test_split_dataset_stratified() -> None:
    """All intents appear in both train and test when data allows."""
    examples = _make_examples(50, intents=["set_climate", "navigate"])
    train, test = split_dataset(examples, train_ratio=0.8, seed=42)

    train_intents = {ex["intent"] for ex in train}
    test_intents = {ex["intent"] for ex in test}
    assert "set_climate" in train_intents
    assert "navigate" in train_intents
    assert "set_climate" in test_intents
    assert "navigate" in test_intents


def test_split_dataset_single_example_per_intent() -> None:
    """Does not crash when an intent has only 1 example; goes to train."""
    examples = [
        {"utterance": "cool it down", "intent": "set_climate", "slots": {}},
        {"utterance": "play something", "intent": "play_media", "slots": {}},
    ]
    train, test = split_dataset(examples, train_ratio=0.8, seed=42)
    assert len(train) + len(test) == 2
    assert len(test) == 0


# ---------------------------------------------------------------------------
# save_dataset
# ---------------------------------------------------------------------------


def test_save_dataset_creates_jsonl(tmp_path: Path) -> None:
    """train.jsonl and test.jsonl are created."""
    train = _make_examples(4)
    test = _make_examples(1)

    save_dataset(train, test, output_dir=tmp_path)

    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()


def test_save_dataset_correct_line_count(tmp_path: Path) -> None:
    """Each JSONL file has exactly as many lines as examples."""
    train = _make_examples(8)
    test = _make_examples(2)

    save_dataset(train, test, output_dir=tmp_path)

    train_lines = (tmp_path / "train.jsonl").read_text().strip().splitlines()
    test_lines = (tmp_path / "test.jsonl").read_text().strip().splitlines()
    assert len(train_lines) == 8
    assert len(test_lines) == 2


def test_save_dataset_format(tmp_path: Path) -> None:
    """Each line is valid JSON with a 'text' key containing Command/Action."""
    examples = [
        {
            "utterance": "cool the front to 20",
            "intent": "set_climate",
            "slots": {"zone": "front", "temperature": 20},
        }
    ]
    save_dataset(examples, [], output_dir=tmp_path)

    line = (tmp_path / "train.jsonl").read_text().strip()
    record = json.loads(line)
    assert "text" in record
    assert "Command: cool the front to 20" in record["text"]
    assert '"intent": "set_climate"' in record["text"]
    assert "Action:" in record["text"]


def test_save_dataset_empty_slots(tmp_path: Path) -> None:
    """Examples with empty slots are saved without error."""
    examples = [{"utterance": "answer the call", "intent": "call_contact", "slots": {}}]
    save_dataset(examples, [], output_dir=tmp_path)

    line = (tmp_path / "train.jsonl").read_text().strip()
    record = json.loads(line)
    assert '"slots": {}' in record["text"]


def test_save_dataset_creates_output_dir(tmp_path: Path) -> None:
    """Output directory is created if it does not exist."""
    output_dir = tmp_path / "new" / "nested" / "dir"
    assert not output_dir.exists()

    save_dataset(_make_examples(2), [], output_dir=output_dir)
    assert output_dir.exists()


# ---------------------------------------------------------------------------
# log_metadata
# ---------------------------------------------------------------------------


def test_log_metadata_runs() -> None:
    """Smoke test: log_metadata does not raise on valid input."""
    train = _make_examples(20)
    test = _make_examples(5)
    log_metadata(train, test)  # should not raise


def test_log_metadata_empty_inputs() -> None:
    """log_metadata handles empty inputs without raising (no division-by-zero)."""
    log_metadata([], [])
