"""Tests for src/dataset.py — audio-based Car-Command dataset pipeline.

Tests use a fake folder structure and monkeypatched mlx-whisper; no real
audio files or Kaggle API calls are made.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.dataset import (
    parse_dataset,
    split_dataset,
    save_dataset,
    log_metadata,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INTENTS = [
    "Turn AC ON",
    "Turn AC OFF",
    "Open cars manual",
    "Pause the music",
    "Navigate home",
    "Answer the call",
    "Enable cruise control",
    "Check battery level",
    "Skip the song",
    "Turn on Bluetooth",
]

TRANSCRIPTS = {intent: f"Spoken command for {intent}" for intent in INTENTS}


def _make_audio_dir(tmp_path: Path, intents: list[str], files_per_intent: int = 3) -> Path:
    """Create a fake intent/timestamp/uuid.mp3 directory structure."""
    data_dir = tmp_path / "car_command"
    for intent in intents:
        ts_dir = data_dir / intent / "2025-01-01T00-00-00.000Z"
        ts_dir.mkdir(parents=True)
        for i in range(files_per_intent):
            (ts_dir / f"fake_{i:02d}.mp3").write_bytes(b"fake audio")
    return data_dir


def _mock_transcribe(path, **kwargs):
    """Return a deterministic transcript based on the intent folder name."""
    audio_path = Path(path)
    intent = audio_path.parts[-3]  # <data_dir>/<intent>/<timestamp>/<file>
    return {"text": TRANSCRIPTS.get(intent, "unknown command")}


# ---------------------------------------------------------------------------
# parse_dataset
# ---------------------------------------------------------------------------


def test_parse_dataset_returns_command_action_pairs(tmp_path: Path) -> None:
    """parse_dataset returns dicts with 'command' and 'action' keys."""
    data_dir = _make_audio_dir(tmp_path, INTENTS[:3], files_per_intent=2)

    with patch("src.dataset.mlx_whisper.transcribe", side_effect=_mock_transcribe), \
         patch("src.dataset.get_processed_dir", return_value=tmp_path / "processed"):
        examples = parse_dataset(data_dir)

    assert len(examples) == 6  # 3 intents × 2 files
    for ex in examples:
        assert "command" in ex
        assert "action" in ex
        assert isinstance(ex["command"], str)
        assert ex["action"] in INTENTS[:3]


def test_parse_dataset_action_matches_folder_name(tmp_path: Path) -> None:
    """Action label equals the intent folder name exactly."""
    data_dir = _make_audio_dir(tmp_path, ["Turn AC ON"], files_per_intent=1)

    with patch("src.dataset.mlx_whisper.transcribe", return_value={"text": "turn on the AC"}), \
         patch("src.dataset.get_processed_dir", return_value=tmp_path / "processed"):
        examples = parse_dataset(data_dir)

    assert examples[0]["action"] == "Turn AC ON"
    assert examples[0]["command"] == "turn on the AC"


def test_parse_dataset_missing_dir(tmp_path: Path) -> None:
    """Raises FileNotFoundError when no intent folders exist."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        parse_dataset(empty)


def test_parse_dataset_respects_max_per_intent(tmp_path: Path) -> None:
    """At most max_per_intent files are transcribed per intent."""
    data_dir = _make_audio_dir(tmp_path, ["Turn AC ON"], files_per_intent=10)

    with patch("src.dataset.mlx_whisper.transcribe", return_value={"text": "turn on AC"}), \
         patch("src.dataset.get_processed_dir", return_value=tmp_path / "processed"):
        examples = parse_dataset(data_dir, max_per_intent=5)

    assert len(examples) == 5


def test_parse_dataset_uses_cache(tmp_path: Path) -> None:
    """Second call uses cached transcriptions; mlx_whisper not called again."""
    data_dir = _make_audio_dir(tmp_path, ["Pause the music"], files_per_intent=2)
    processed_dir = tmp_path / "processed"

    with patch("src.dataset.mlx_whisper.transcribe", side_effect=_mock_transcribe) as mock_t, \
         patch("src.dataset.get_processed_dir", return_value=processed_dir):
        parse_dataset(data_dir)
        first_call_count = mock_t.call_count

    with patch("src.dataset.mlx_whisper.transcribe", side_effect=_mock_transcribe) as mock_t, \
         patch("src.dataset.get_processed_dir", return_value=processed_dir):
        parse_dataset(data_dir)
        second_call_count = mock_t.call_count

    assert first_call_count == 2
    assert second_call_count == 0  # all served from cache


def test_parse_dataset_skips_empty_transcriptions(tmp_path: Path) -> None:
    """Examples with empty transcriptions are excluded."""
    data_dir = _make_audio_dir(tmp_path, ["Turn AC ON"], files_per_intent=3)

    responses = [{"text": "turn on AC"}, {"text": ""}, {"text": "  "}]

    with patch("src.dataset.mlx_whisper.transcribe", side_effect=responses), \
         patch("src.dataset.get_processed_dir", return_value=tmp_path / "processed"):
        examples = parse_dataset(data_dir)

    assert len(examples) == 1
    assert examples[0]["command"] == "turn on AC"


def test_parse_dataset_skips_failed_transcriptions(tmp_path: Path) -> None:
    """Files where mlx_whisper raises are skipped; run continues."""
    data_dir = _make_audio_dir(tmp_path, ["Turn AC ON"], files_per_intent=3)

    def _raise_on_second(path, **kwargs):
        if "fake_01" in path:
            raise RuntimeError("corrupt audio")
        return {"text": "turn on AC"}

    with patch("src.dataset.mlx_whisper.transcribe", side_effect=_raise_on_second), \
         patch("src.dataset.get_processed_dir", return_value=tmp_path / "processed"):
        examples = parse_dataset(data_dir)

    # 2 of 3 files succeed; the corrupt one is skipped
    assert len(examples) == 2


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------


def test_split_dataset_ratio() -> None:
    """Verify 80/20 split with correct total length."""
    examples = [{"command": f"cmd {i}", "action": f"intent_{i % 5}"} for i in range(100)]
    train, test = split_dataset(examples, train_ratio=0.8, seed=42)

    assert len(train) == 80
    assert len(test) == 20
    assert len(train) + len(test) == len(examples)


def test_split_dataset_deterministic() -> None:
    """Same seed produces identical splits."""
    examples = [{"command": f"cmd {i}", "action": f"intent_{i}"} for i in range(50)]
    train1, test1 = split_dataset(examples, seed=42)
    train2, test2 = split_dataset(examples, seed=42)

    assert train1 == train2
    assert test1 == test2


def test_split_dataset_no_overlap() -> None:
    """Train and test sets share no commands."""
    examples = [{"command": f"cmd {i}", "action": f"intent_{i}"} for i in range(50)]
    train, test = split_dataset(examples, seed=42)

    train_cmds = {ex["command"] for ex in train}
    test_cmds = {ex["command"] for ex in test}
    assert train_cmds.isdisjoint(test_cmds)


# ---------------------------------------------------------------------------
# save_dataset
# ---------------------------------------------------------------------------


def test_save_dataset_creates_jsonl(tmp_path: Path) -> None:
    """train.jsonl and test.jsonl are created with correct format."""
    train = [
        {"command": "turn on AC", "action": "Turn AC ON"},
        {"command": "play music", "action": "Resume the music"},
    ]
    test = [
        {"command": "enable cruise control", "action": "Enable cruise control"},
    ]

    output_dir = tmp_path / "processed"
    save_dataset(train, test, output_dir=output_dir)

    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()

    train_lines = (output_dir / "train.jsonl").read_text().strip().splitlines()
    assert len(train_lines) == 2

    first = json.loads(train_lines[0])
    assert "text" in first
    assert "Command: turn on AC" in first["text"]
    assert "Action: Turn AC ON" in first["text"]


def test_save_dataset_test_split(tmp_path: Path) -> None:
    """test.jsonl has correct number of entries."""
    train = [{"command": f"cmd {i}", "action": "X"} for i in range(5)]
    test = [{"command": "one command", "action": "Y"}]

    save_dataset(train, test, output_dir=tmp_path / "out")
    lines = (tmp_path / "out" / "test.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


# ---------------------------------------------------------------------------
# log_metadata
# ---------------------------------------------------------------------------


def test_log_metadata_runs() -> None:
    """Smoke test: log_metadata does not raise on valid input."""
    train = [
        {"command": "turn on AC", "action": "Turn AC ON"},
        {"command": "play music", "action": "Resume the music"},
        {"command": "set temp to 20", "action": "Turn AC ON"},
    ]
    test = [
        {"command": "skip this track", "action": "Skip the song"},
    ]
    log_metadata(train, test)  # should not raise


def test_log_metadata_empty_inputs() -> None:
    """log_metadata handles empty train and test lists without raising."""
    log_metadata([], [])  # division-by-zero guard must hold
