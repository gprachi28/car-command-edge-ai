"""Dataset pipeline for Kaggle Car-Command audio dataset.

Walks the folder structure (intent/timestamp/audio.mp3), transcribes audio
files with mlx-whisper (cached), splits 80/20, and saves JSONL pairs.

Public API:
    - parse_dataset(data_dir, max_per_intent, whisper_model) -> list[dict]
    - split_dataset(examples, train_ratio, seed) -> tuple[list, list]
    - save_dataset(train, test, output_dir) -> None
    - log_metadata(train, test) -> None
"""

import json
import random
import re
from pathlib import Path

import mlx_whisper

from src.utils import get_logger, get_processed_dir

logger = get_logger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a"}
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"
# Cache lives in data/processed/ (writable, separate from raw inputs)
CACHE_FILENAME = "transcription_cache.json"
# Quality filter thresholds
_MAX_WORDS = 20          # car commands are short; longer = likely hallucination loop
_MAX_WORD_REPEAT = 3     # any single word appearing > this many times = repetition loop


def parse_dataset(
    data_dir: Path,
    max_per_intent: int = 75,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
) -> list[dict[str, str]]:
    """Walk intent folders, transcribe audio, return command/action pairs.

    Expects structure: data_dir/<intent>/<timestamp>/<uuid>.mp3
    Uses a JSON cache (data_dir/transcription_cache.json) to avoid
    re-transcribing files across runs.

    Args:
        data_dir: Root folder containing one sub-folder per intent.
        max_per_intent: Maximum audio files to transcribe per intent.
            Sampled randomly with seed=42 when the folder has more.
        whisper_model: HuggingFace repo or local path for mlx-whisper.

    Returns:
        List of dicts with keys "command" (transcribed text) and
        "action" (intent label, i.e. the folder name).

    Raises:
        FileNotFoundError: If data_dir has no intent sub-folders.
    """
    data_dir = Path(data_dir)
    intent_dirs = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not intent_dirs:
        raise FileNotFoundError(f"No intent folders found in {data_dir}")

    # Cache goes to data/processed/ — writable and separate from raw inputs
    cache_path = get_processed_dir() / CACHE_FILENAME
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_path)

    examples: list[dict[str, str]] = []
    rng = random.Random(42)

    for intent_dir in intent_dirs:
        intent = intent_dir.name
        audio_files = sorted(
            f
            for f in intent_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        )

        if not audio_files:
            logger.warning("No audio files in intent folder: %s", intent)
            continue

        if len(audio_files) > max_per_intent:
            audio_files = rng.sample(audio_files, max_per_intent)

        logger.info("Transcribing %d files for intent: %s", len(audio_files), intent)

        for audio_path in audio_files:
            cache_key = str(audio_path.resolve())
            if cache_key in cache:
                text = cache[cache_key]
            else:
                text = _transcribe(audio_path, whisper_model)
                cache[cache_key] = text
                _save_cache(cache, cache_path)

            text = text.strip()
            if text and _is_valid_transcription(text):
                examples.append({"command": text, "action": intent})
            elif text:
                logger.debug("Filtered low-quality transcription: %s", text[:80])

    if not examples:
        raise FileNotFoundError(f"No transcribable audio found under {data_dir}")

    logger.info("Parsed %d examples across %d intents", len(examples), len(intent_dirs))
    return examples


def split_dataset(
    examples: list[dict[str, str]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Shuffle and split examples into train and test sets.

    Args:
        examples: List of {"command": ..., "action": ...} dicts.
        train_ratio: Fraction of data for training (default 0.8).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_examples, test_examples).
    """
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]


def save_dataset(
    train: list[dict[str, str]],
    test: list[dict[str, str]],
    output_dir: Path | None = None,
) -> None:
    """Save train and test splits as JSONL files in fine-tuning format.

    Each line is {"text": "Command: <utterance>\\nAction: <intent>"}.

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
    train: list[dict[str, str]],
    test: list[dict[str, str]],
) -> None:
    """Log dataset statistics: counts, intent distribution, utterance length.

    Args:
        train: Training examples.
        test: Test examples.
    """
    all_examples = train + test
    intents: dict[str, int] = {}
    lengths: list[int] = []

    for ex in all_examples:
        intents[ex["action"]] = intents.get(ex["action"], 0) + 1
        lengths.append(len(ex["command"].split()))

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
        logger.info("  %-45s %d", intent, count)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_valid_transcription(text: str) -> bool:
    """Return True if the transcription looks like a real car command.

    Filters out two common mlx-whisper failure modes:
    - Repetition loops: "edge edge edge edge..." (Whisper hallucinating on bad audio)
    - Wrong language: non-ASCII output when language detection picks the wrong language

    Args:
        text: Stripped transcription string.

    Returns:
        True if the text passes all quality checks.
    """
    # Wrong language: reject if any non-ASCII character is present
    if not text.isascii():
        return False

    words = text.split()

    # Too long: car commands are short phrases; >20 words is a hallucination loop
    if len(words) > _MAX_WORDS:
        return False

    # Repetition loop: any single word appearing more than _MAX_WORD_REPEAT times
    word_counts: dict[str, int] = {}
    for word in words:
        normalised = re.sub(r"[^a-z]", "", word.lower())
        if normalised:
            word_counts[normalised] = word_counts.get(normalised, 0) + 1
    if any(count > _MAX_WORD_REPEAT for count in word_counts.values()):
        return False

    return True


def _transcribe(audio_path: Path, model: str) -> str:
    """Transcribe a single audio file using mlx-whisper.

    Args:
        audio_path: Path to the audio file.
        model: mlx-whisper model repo or local path.

    Returns:
        Transcribed text string (empty string if Whisper fails or returns
        nothing — caller skips empty results via the `if text:` guard).
    """
    try:
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=model,
            language="en",  # force English — prevents wrong-language hallucinations
        )
        return result.get("text", "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Transcription failed for %s: %s", audio_path.name, exc)
        return ""


def _load_cache(cache_path: Path) -> dict[str, str]:
    """Load transcription cache from JSON, returning empty dict if absent."""
    if cache_path.exists():
        with cache_path.open() as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict[str, str], cache_path: Path) -> None:
    """Persist transcription cache to JSON."""
    with cache_path.open("w") as f:
        json.dump(cache, f, indent=2)


def _write_jsonl(examples: list[dict[str, str]], path: Path) -> None:
    """Write examples as JSONL in fine-tuning format."""
    with path.open("w") as f:
        for ex in examples:
            record = {"text": f"Command: {ex['command']}\nAction: {ex['action']}"}
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    from src.utils import get_data_dir

    data_dir = get_data_dir() / "raw" / "car_command"
    examples = parse_dataset(data_dir)
    train, test = split_dataset(examples)
    save_dataset(train, test)
    log_metadata(train, test)
