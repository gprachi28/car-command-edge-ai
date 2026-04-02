"""Generate synthetic car command dataset using a local or cloud LLM.

Entry point. Run from project root:
    python -m src.generate_dataset [--backend {ollama,gemini}] [--model MODEL]
                                   [--delay SECONDS] [--dry-run]

Options:
    --backend      LLM backend to use: 'ollama' (default) or 'gemini'
    --model        Model name (default: llama3.1:8b for ollama, gemini-2.0-flash for gemini)
    --delay        Seconds to wait between intent API calls (default: 1.0)
    --dry-run      Print prompts without calling any LLM
"""

import argparse
import json
import time
from pathlib import Path

from src.dataset import log_metadata, save_dataset, split_dataset
from src.utils import get_data_dir, get_logger, load_config

logger = get_logger(__name__)

OLLAMA_DEFAULT_MODEL = "llama3.1:8b"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"
OLLAMA_BATCH_SIZE = 20  # examples per API call; keeps response within context limits

# ---------------------------------------------------------------------------
# Intent schemas: intent → {count, slots}
# Slot values describe accepted options; used in the prompt and for validation.
# ---------------------------------------------------------------------------

INTENT_SCHEMAS: dict[str, dict] = {
    # Deep intents (130 utterances each)
    "set_climate": {
        "count": 130,
        "slots": {
            "zone": "string: front|rear|driver|passenger|left|right",
            "temperature": "number (e.g. 20, 22.5)",
            "mode": "string: cool|heat|auto|fan",
            "fan_speed": "integer 1–5",
            "unit": "string: celsius|fahrenheit",
        },
    },
    "navigate": {
        "count": 130,
        "slots": {
            "destination": "string (place name, address, or type)",
            "destination_type": "string: address|poi|home|work|hospital|gas_station",
            "route_preference": "string: fastest|scenic|avoid_tolls|avoid_highways",
            "arrival_time": "string (e.g. '8am', 'in 30 minutes')",
            "waypoint": "string (intermediate stop)",
        },
    },
    "seat_control": {
        "count": 130,
        "slots": {
            "seat": "string: driver|passenger|rear_left|rear_right|all",
            "adjustment": "string: forward|backward|up|down|recline|upright",
            "lumbar": "integer 1–5",
            "heat": "string: off|low|medium|high",
            "memory": "integer 1–3 (recall preset)",
        },
    },
    "set_lighting": {
        "count": 130,
        "slots": {
            "zone": "string: interior|ambient|dashboard|cabin|footwell|reading",
            "color": "string (color name or hex)",
            "brightness": "integer 0–100",
            "mode": "string: off|dim|full|reading|night",
        },
    },
    # Medium intents (120 utterances each)
    "play_media": {
        "count": 120,
        "slots": {
            "source": "string: radio|spotify|bluetooth|podcast|usb|apple_music",
            "query": "string (song, podcast, or genre name)",
            "artist": "string",
            "genre": "string",
            "station": "string (radio station name or frequency)",
        },
    },
    "adjust_volume": {
        "count": 120,
        "slots": {
            "direction": "string: up|down|mute|unmute",
            "level": "integer 0–100",
            "step": "integer (relative change, e.g. 5)",
        },
    },
    "window_control": {
        "count": 120,
        "slots": {
            "window": "string: driver|passenger|rear_left|rear_right|all|sunroof",
            "action": "string: open|close|vent|crack|half",
            "percentage": "integer 0–100",
        },
    },
    "cruise_control": {
        "count": 120,
        "slots": {
            "action": "string: enable|disable|set|increase|decrease|resume",
            "speed": "integer",
            "unit": "string: mph|kph",
            "gap": "string: close|medium|far (following distance)",
        },
    },
    "connectivity": {
        "count": 120,
        "slots": {
            "feature": "string: bluetooth|wifi|hotspot|carplay|android_auto|nfc",
            "action": "string: enable|disable|connect|disconnect|pair",
            "device_name": "string",
        },
    },
    # Shallow intents (100 utterances each)
    "call_contact": {
        "count": 100,
        "slots": {
            "contact_name": "string",
            "contact_type": "string: mobile|home|work",
        },
    },
    "read_message": {
        "count": 100,
        "slots": {
            "contact_name": "string",
            "message_type": "string: text|email|voicemail|whatsapp",
        },
    },
    "safety_assist": {
        "count": 100,
        "slots": {
            "feature": "string: lane_assist|parking_sensors|blind_spot|collision_warning|driver_alert|speed_limiter",
            "action": "string: enable|disable|status",
        },
    },
    "vehicle_info": {
        "count": 100,
        "slots": {
            "query_type": "string: fuel|battery|tire_pressure|oil_level|range|mileage|speed|rpm|temperature",
        },
    },
    "drive_mode": {
        "count": 100,
        "slots": {
            "mode": "string: eco|sport|normal|comfort|offroad|snow|rain|track",
        },
    },
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_prompt(intent: str, count: int, slots: dict[str, str]) -> str:
    """Build the generation prompt for a given intent and slot schema."""
    slot_lines = "\n".join(f"  - {key}: {desc}" for key, desc in slots.items())
    valid_keys = ", ".join(f'"{k}"' for k in slots)
    return f"""You are generating training data for a car voice command AI.

Generate exactly {count} diverse English voice commands for the intent "{intent}".

Slot schema (ONLY use these keys):
{slot_lines}

Requirements:
- Use natural, varied language: formal requests, casual commands, questions, abbreviated phrases
- Cover different slot combinations — not every slot needs to be present in each example
- Avoid repetitive phrasing patterns; maximise lexical variety
- Slots must use ONLY valid keys: {valid_keys}
- Slot values must match the schema descriptions

Respond with ONLY a valid JSON object in this exact format:
{{"examples": [{{"utterance": "<voice command text>", "slots": {{<slot key/value pairs or {{}}}}}}]}}

Do not include any explanation, markdown, or text outside the JSON."""


# ---------------------------------------------------------------------------
# Response parser (handles both array and {"examples": [...]} formats)
# ---------------------------------------------------------------------------


def _parse_llm_response(text: str) -> list[dict]:
    """Parse LLM response text into a list of example dicts.

    Handles:
    - {"examples": [...]}  (preferred)
    - [...]                (direct array)
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("JSON parse error: %s", e)
        logger.debug("Raw response (first 500): %s", text[:500])
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Look for the first list value
        for v in data.values():
            if isinstance(v, list):
                return v
    logger.warning("Unexpected response structure: %s", type(data))
    return []


# ---------------------------------------------------------------------------
# Backend: Ollama
# ---------------------------------------------------------------------------


def _call_ollama(model: str, prompt: str) -> list[dict]:
    """Call a local Ollama model and return parsed examples."""
    import ollama as _ollama  # imported lazily — only needed for this backend

    try:
        response = _ollama.generate(
            model=model,
            prompt=prompt,
            format="json",
            options={"temperature": 1.0, "num_predict": 8192},
        )
        return _parse_llm_response(response.get("response", ""))
    except Exception as e:
        logger.error("Ollama call failed: %s", e)
        return []


def _generate_ollama(model: str, intent: str, schema: dict) -> list[dict]:
    """Generate examples for one intent via Ollama, batched to stay within context."""
    target = schema["count"]
    valid_slot_keys = set(schema["slots"].keys())
    collected: list[dict] = []
    attempts = 0
    max_attempts = (target // OLLAMA_BATCH_SIZE + 1) * 3  # generous retry budget

    while len(collected) < target and attempts < max_attempts:
        remaining = target - len(collected)
        batch_size = min(OLLAMA_BATCH_SIZE, remaining)
        prompt = _build_prompt(intent, batch_size, schema["slots"])
        raw = _call_ollama(model, prompt)

        for ex in raw:
            if _validate_example(ex, intent, valid_slot_keys):
                collected.append(
                    {
                        "utterance": ex["utterance"].strip(),
                        "intent": intent,
                        "slots": ex.get("slots", {}),
                    }
                )
            else:
                logger.debug("  Skipped invalid: %s", str(ex)[:100])

        attempts += 1
        logger.info(
            "  [%s] batch %d: got %d valid → total %d/%d",
            intent,
            attempts,
            len(raw),
            len(collected),
            target,
        )

    if len(collected) < target:
        logger.warning(
            "  Only collected %d/%d examples for '%s' after %d attempts",
            len(collected),
            target,
            intent,
            attempts,
        )
    return collected[:target]


# ---------------------------------------------------------------------------
# Backend: Gemini
# ---------------------------------------------------------------------------


def _call_gemini(client: object, model: str, prompt: str) -> list[dict]:
    """Call Gemini and return parsed examples."""
    from google.genai import types  # lazy import

    try:
        response = client.models.generate_content(  # type: ignore[attr-defined]
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=1.0,
            ),
        )
        return _parse_llm_response(response.text)
    except Exception as e:
        logger.error("Gemini call failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_example(ex: object, intent: str, valid_slot_keys: set[str]) -> bool:
    """Return True if ex is a valid example dict for the given intent."""
    if not isinstance(ex, dict):
        return False
    utterance = ex.get("utterance", "")
    if not isinstance(utterance, str) or not utterance.strip():
        return False
    slots = ex.get("slots", {})
    if not isinstance(slots, dict):
        return False
    invalid_keys = set(slots.keys()) - valid_slot_keys
    if invalid_keys:
        logger.debug("Unknown slot keys %s for '%s'", invalid_keys, intent)
        return False
    return True


# ---------------------------------------------------------------------------
# Raw per-intent persistence (resume-safe)
# ---------------------------------------------------------------------------


def _load_raw_intent(raw_dir: Path, intent: str) -> list[dict] | None:
    """Return existing examples if the intent file exists, else None."""
    path = raw_dir / f"{intent}.jsonl"
    if not path.exists():
        return None
    examples = []
    dropped = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    dropped += 1
    if dropped:
        logger.warning("  %s: dropped %d corrupt lines while loading", path.name, dropped)
    return examples


def _save_raw_intent(raw_dir: Path, intent: str, examples: list[dict]) -> None:
    """Write examples as JSONL to data/raw/synthetic/<intent>.jsonl.

    Uses a write-to-temp-then-rename pattern so the file is never left in a
    partially written state if the process is interrupted.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{intent}.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    tmp.replace(path)
    logger.info("  Saved %d examples → %s", len(examples), path)


# ---------------------------------------------------------------------------
# Per-intent generation (dispatch to selected backend)
# ---------------------------------------------------------------------------


def generate_intent(
    intent: str,
    schema: dict,
    raw_dir: Path,
    backend: str,
    model: str,
    dry_run: bool,
    gemini_client: object = None,
) -> list[dict]:
    """Generate or load examples for a single intent.

    Returns examples in canonical form:
        {"utterance": str, "intent": str, "slots": dict}
    """
    existing = _load_raw_intent(raw_dir, intent)
    if existing is not None:
        logger.info("  Resume: %d existing examples for '%s'", len(existing), intent)
        return existing

    if dry_run:
        logger.info("  [DRY RUN] Would generate %d examples for '%s'", schema["count"], intent)
        return []

    if backend == "ollama":
        examples = _generate_ollama(model, intent, schema)
    else:
        valid_slot_keys = set(schema["slots"].keys())
        prompt = _build_prompt(intent, schema["count"], schema["slots"])
        raw = _call_gemini(gemini_client, model, prompt)
        examples = []
        for ex in raw:
            if _validate_example(ex, intent, valid_slot_keys):
                examples.append(
                    {
                        "utterance": ex["utterance"].strip(),
                        "intent": intent,
                        "slots": ex.get("slots", {}),
                    }
                )

    _save_raw_intent(raw_dir, intent, examples)
    return examples


# ---------------------------------------------------------------------------
# Metadata JSON save
# ---------------------------------------------------------------------------


def _save_metadata(train: list[dict], test: list[dict], output_dir: Path) -> None:
    """Save dataset statistics to data/processed/metadata.json."""
    all_ex = train + test
    intent_counts: dict[str, dict] = {}
    for ex in all_ex:
        intent = ex["intent"]
        if intent not in intent_counts:
            intent_counts[intent] = {"train": 0, "test": 0, "total": 0}
        intent_counts[intent]["total"] += 1
    for ex in train:
        intent_counts[ex["intent"]]["train"] += 1
    for ex in test:
        intent_counts[ex["intent"]]["test"] += 1

    lengths = [len(ex["utterance"].split()) for ex in all_ex]
    metadata = {
        "total": len(all_ex),
        "train": len(train),
        "test": len(test),
        "intents": len(intent_counts),
        "avg_utterance_words": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "intent_distribution": intent_counts,
    }

    out = output_dir / "metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata → %s", out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic car command dataset")
    parser.add_argument(
        "--backend",
        choices=["ollama", "gemini"],
        default="ollama",
        help="LLM backend (default: ollama)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: llama3.1:8b for ollama, gemini-2.0-flash for gemini)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between intent calls for gemini backend (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print without calling any LLM",
    )
    args = parser.parse_args()

    model = args.model or (OLLAMA_DEFAULT_MODEL if args.backend == "ollama" else GEMINI_DEFAULT_MODEL)

    gemini_client = None
    if args.backend == "gemini" and not args.dry_run:
        from google import genai as _genai  # lazy import

        config = load_config()
        gemini_client = _genai.Client(api_key=config["gemini_api_key"])

    logger.info("Backend: %s | Model: %s", args.backend, model)

    raw_dir = get_data_dir() / "raw" / "synthetic"
    processed_dir = get_data_dir() / "processed"
    total_intents = len(INTENT_SCHEMAS)
    all_examples: list[dict] = []

    gemini_calls = 0
    for i, (intent, schema) in enumerate(INTENT_SCHEMAS.items(), start=1):
        logger.info("[%d/%d] Intent: %s (%d examples)", i, total_intents, intent, schema["count"])
        is_cached = (get_data_dir() / "raw" / "synthetic" / f"{intent}.jsonl").exists()
        if args.backend == "gemini" and not args.dry_run and not is_cached:
            if gemini_calls > 0:
                time.sleep(args.delay)
            gemini_calls += 1

        examples = generate_intent(
            intent=intent,
            schema=schema,
            raw_dir=raw_dir,
            backend=args.backend,
            model=model,
            dry_run=args.dry_run,
            gemini_client=gemini_client,
        )
        all_examples.extend(examples)

    if args.dry_run:
        logger.info("[DRY RUN] Complete. No data written to processed/.")
        return

    # Deduplicate on normalised utterance text
    seen: set[str] = set()
    deduped: list[dict] = []
    for ex in all_examples:
        key = ex["utterance"].lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(ex)

    removed = len(all_examples) - len(deduped)
    if removed:
        logger.info("Deduplication: removed %d duplicates (%d → %d)", removed, len(all_examples), len(deduped))

    train, test = split_dataset(deduped, train_ratio=0.8, seed=42)
    save_dataset(train, test, output_dir=processed_dir)
    _save_metadata(train, test, processed_dir)
    log_metadata(train, test)

    logger.info("Dataset generation complete: %d train / %d test", len(train), len(test))


if __name__ == "__main__":
    main()
