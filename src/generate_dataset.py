"""Generate synthetic car command dataset — tighter slot quality, density tiers.

Features:
  - Density-tier generation: each intent is generated in three tiers (full / partial /
    minimal) with explicit minimum slot counts. Realistic sparsity is preserved in the
    minimal tier rather than emerging as a generation accident.
  - Gold examples per tier embedded in the prompt — the model sees the exact JSON shape
    it must produce at each density level.
  - Inline validation: None values, empty strings, out-of-schema keys, and
    question/status utterances are rejected at generation time, not in a cleanup pass.
  - Explicit prohibition in prompt: status queries, "?"-terminated utterances,
    None/null values, multi-intent commands.
  - Target ~1,200 examples across 14 intents.

Usage:
    python -m src.generate_dataset [--dry-run] [--model MODEL]

Output:  data/processed/train.jsonl  (80%)
         data/processed/test.jsonl   (20%)
Raw per-intent files: data/raw/synthetic/<intent>.jsonl  (resume-safe)
"""

import argparse
import json
import re
from pathlib import Path

from src.dataset import log_metadata, save_dataset, split_dataset
from src.utils import get_data_dir, get_logger

logger = get_logger(__name__)

OLLAMA_DEFAULT_MODEL = "llama3.1:8b"
BATCH_SIZE = 15  # smaller batches → tighter outputs, fewer run-on responses

# ---------------------------------------------------------------------------
# Density tier configuration
# ---------------------------------------------------------------------------
# Each intent depth class splits its total count across three tiers.
# min_slots = minimum required slots for an example to be accepted in that tier.
# Fractions are approximate; final counts are rounded.

_TIER_CONFIG: dict[str, list[dict]] = {
    "deep": [
        {
            "name": "full",
            "fraction": 0.50,
            "min_slots": 3,
            "description": "3 or more distinct slots all expressed in the utterance",
        },
        {
            "name": "partial",
            "fraction": 0.30,
            "min_slots": 2,
            "description": "exactly 2 slots present",
        },
        {
            "name": "minimal",
            "fraction": 0.20,
            "min_slots": 1,
            "description": "exactly 1 slot — realistic single-concept commands",
        },
    ],
    "medium": [
        {
            "name": "full",
            "fraction": 0.50,
            "min_slots": 2,
            "description": "2 or more slots expressed",
        },
        {"name": "partial", "fraction": 0.30, "min_slots": 2, "description": "2 slots"},
        {
            "name": "minimal",
            "fraction": 0.20,
            "min_slots": 1,
            "description": "1 slot — the most specific one in the utterance",
        },
    ],
    "shallow": [
        {
            "name": "full",
            "fraction": 0.60,
            "min_slots": 1,
            "description": "all slots expressed (shallow intents have 1–2 slots max)",
        },
        {"name": "partial", "fraction": 0.20, "min_slots": 1, "description": "1 slot"},
        {
            "name": "minimal",
            "fraction": 0.20,
            "min_slots": 1,
            "description": "1 slot, minimal phrasing",
        },
    ],
}

# ---------------------------------------------------------------------------
# Intent schemas + gold examples
# ---------------------------------------------------------------------------
# Each intent has:
#   depth     : "deep" | "medium" | "shallow"
#   count     : total target examples
#   slots     : allowed slot keys and their descriptions (same as v1)
#   examples  : 2 gold examples per tier for few-shot prompting

INTENT_SCHEMAS_V2: dict[str, dict] = {
    # ------------------------------------------------------------------
    # Deep intents
    # ------------------------------------------------------------------
    "set_climate": {
        "depth": "deep",
        "count": 90,
        "slots": {
            "zone": "string: front|rear|driver|passenger|left|right",
            "temperature": "number (e.g. 20, 22.5)",
            "mode": "string: cool|heat|auto|fan",
            "fan_speed": "integer 1–5",
            "unit": "string: celsius|fahrenheit",
        },
        "examples": {
            "full": [
                {
                    "utterance": "Set front zone to 22 celsius in cool mode, fan on 3",
                    "slots": {
                        "zone": "front",
                        "temperature": 22,
                        "mode": "cool",
                        "unit": "celsius",
                        "fan_speed": 3,
                    },
                },
                {
                    "utterance": "Heat the rear to 75 fahrenheit, max fan",
                    "slots": {
                        "zone": "rear",
                        "temperature": 75,
                        "mode": "heat",
                        "unit": "fahrenheit",
                        "fan_speed": 5,
                    },
                },
            ],
            "partial": [
                {
                    "utterance": "Cool down the front area",
                    "slots": {"zone": "front", "mode": "cool"},
                },
                {
                    "utterance": "Set passenger side to 21 degrees",
                    "slots": {"zone": "passenger", "temperature": 21},
                },
            ],
            "minimal": [
                {"utterance": "Make it cooler", "slots": {"mode": "cool"}},
                {"utterance": "Turn off the AC", "slots": {"mode": "fan"}},
            ],
        },
    },
    "navigate": {
        "depth": "deep",
        "count": 90,
        "slots": {
            "destination": "string (place name or address)",
            "destination_type": "string: address|poi|home|work|hospital|gas_station",
            "route_preference": "string: fastest|scenic|avoid_tolls|avoid_highways",
            "arrival_time": "string (e.g. '8am', 'in 30 minutes')",
            "waypoint": "string (intermediate stop)",
        },
        "examples": {
            "full": [
                {
                    "utterance": "Take the fastest route to 123 Main Street, I need to be there by 9am",  # noqa: E501
                    "slots": {
                        "destination": "123 Main Street",
                        "route_preference": "fastest",
                        "arrival_time": "9am",
                    },
                },
                {
                    "utterance": "Navigate to work avoiding highways, stop by Starbucks on the way",  # noqa: E501
                    "slots": {
                        "destination_type": "work",
                        "route_preference": "avoid_highways",
                        "waypoint": "Starbucks",
                    },
                },
            ],
            "partial": [
                {
                    "utterance": "Get me to the nearest gas station on the scenic route",  # noqa: E501
                    "slots": {
                        "destination_type": "gas_station",
                        "route_preference": "scenic",
                    },
                },
                {
                    "utterance": "Head home, avoid tolls",
                    "slots": {
                        "destination_type": "home",
                        "route_preference": "avoid_tolls",
                    },
                },
            ],
            "minimal": [
                {"utterance": "Navigate home", "slots": {"destination_type": "home"}},
                {
                    "utterance": "Take me to Central Park",
                    "slots": {"destination": "Central Park"},
                },
            ],
        },
    },
    "seat_control": {
        "depth": "deep",
        "count": 90,
        "slots": {
            "seat": "string: driver|passenger|rear_left|rear_right|all",
            "adjustment": "string: forward|backward|up|down|recline|upright",
            "lumbar": "integer 1–5",
            "heat": "string: off|low|medium|high",
            "memory": "integer 1–3 (recall preset)",
        },
        "examples": {
            "full": [
                {
                    "utterance": "Recline driver seat and set heat to medium, lumbar to 3",  # noqa: E501
                    "slots": {
                        "seat": "driver",
                        "adjustment": "recline",
                        "heat": "medium",
                        "lumbar": 3,
                    },
                },
                {
                    "utterance": "Move passenger seat backward and turn heat to high",
                    "slots": {
                        "seat": "passenger",
                        "adjustment": "backward",
                        "heat": "high",
                    },
                },
            ],
            "partial": [
                {
                    "utterance": "Recline the rear left seat a bit",
                    "slots": {"seat": "rear_left", "adjustment": "recline"},
                },
                {
                    "utterance": "Turn up seat heat for driver to high",
                    "slots": {"seat": "driver", "heat": "high"},
                },
            ],
            "minimal": [
                {"utterance": "Recline my seat", "slots": {"adjustment": "recline"}},
                {
                    "utterance": "Heat the passenger seat",
                    "slots": {"seat": "passenger"},
                },
            ],
        },
    },
    "set_lighting": {
        "depth": "deep",
        "count": 90,
        "slots": {
            "zone": "string: interior|ambient|dashboard|cabin|footwell|reading",
            "color": "string (color name or hex)",
            "brightness": "integer 0–100",
            "mode": "string: off|dim|full|reading|night",
        },
        "examples": {
            "full": [
                {
                    "utterance": "Set cabin ambient lights to blue, brightness 60, night mode",  # noqa: E501
                    "slots": {
                        "zone": "cabin",
                        "color": "blue",
                        "brightness": 60,
                        "mode": "night",
                    },
                },
                {
                    "utterance": "Footwell lights to red at 80 percent brightness",
                    "slots": {"zone": "footwell", "color": "red", "brightness": 80},
                },
            ],
            "partial": [
                {
                    "utterance": "Dim the reading light",
                    "slots": {"zone": "reading", "mode": "dim"},
                },
                {
                    "utterance": "Set ambient lights to 50 percent",
                    "slots": {"zone": "ambient", "brightness": 50},
                },
            ],
            "minimal": [
                {"utterance": "Turn off interior lights", "slots": {"mode": "off"}},
                {"utterance": "Night mode", "slots": {"mode": "night"}},
            ],
        },
    },
    # ------------------------------------------------------------------
    # Medium intents
    # ------------------------------------------------------------------
    "play_media": {
        "depth": "medium",
        "count": 88,
        "slots": {
            "source": "string: radio|spotify|bluetooth|podcast|usb|apple_music",
            "query": "string (song, podcast, or playlist name)",
            "artist": "string",
            "genre": "string",
            "station": "string (radio station name or frequency)",
        },
        "examples": {
            "full": [
                {
                    "utterance": "Play jazz on Spotify by Miles Davis",
                    "slots": {
                        "source": "spotify",
                        "genre": "jazz",
                        "artist": "Miles Davis",
                    },
                },
                {
                    "utterance": "Put on KISS 108 radio station",
                    "slots": {"source": "radio", "station": "KISS 108"},
                },
            ],
            "partial": [
                {
                    "utterance": "Play some hip hop on Bluetooth",
                    "slots": {"source": "bluetooth", "genre": "hip hop"},
                },
                {
                    "utterance": "Start the morning commute podcast",
                    "slots": {"source": "podcast", "query": "morning commute"},
                },
            ],
            "minimal": [
                {"utterance": "Play Spotify", "slots": {"source": "spotify"}},
                {
                    "utterance": "Play some classical music",
                    "slots": {"genre": "classical"},
                },
            ],
        },
    },
    "adjust_volume": {
        "depth": "medium",
        "count": 88,
        "slots": {
            "direction": "string: up|down|mute|unmute",
            "level": "integer 0–100",
            "step": "integer (relative change, e.g. 5)",
        },
        "examples": {
            "full": [
                {
                    "utterance": "Turn volume up by 10",
                    "slots": {"direction": "up", "step": 10},
                },
                {"utterance": "Set volume to 40", "slots": {"level": 40}},
            ],
            "partial": [
                {
                    "utterance": "Lower it by 5",
                    "slots": {"direction": "down", "step": 5},
                },
                {"utterance": "Volume to 70 percent", "slots": {"level": 70}},
            ],
            "minimal": [
                {"utterance": "Mute", "slots": {"direction": "mute"}},
                {"utterance": "Turn it up", "slots": {"direction": "up"}},
            ],
        },
    },
    "window_control": {
        "depth": "medium",
        "count": 88,
        "slots": {
            "window": "string: driver|passenger|rear_left|rear_right|all|sunroof",
            "action": "string: open|close|vent|crack|half",
            "percentage": "integer 0–100",
        },
        "examples": {
            "full": [
                {
                    "utterance": "Open the driver window halfway",
                    "slots": {"window": "driver", "action": "open", "percentage": 50},
                },
                {
                    "utterance": "Crack the sunroof open to 20 percent",
                    "slots": {"window": "sunroof", "action": "crack", "percentage": 20},
                },
            ],
            "partial": [
                {
                    "utterance": "Close all windows",
                    "slots": {"window": "all", "action": "close"},
                },
                {
                    "utterance": "Vent the passenger window",
                    "slots": {"window": "passenger", "action": "vent"},
                },
            ],
            "minimal": [
                {"utterance": "Close the sunroof", "slots": {"window": "sunroof"}},
                {"utterance": "Open my window", "slots": {"window": "driver"}},
            ],
        },
    },
    "cruise_control": {
        "depth": "medium",
        "count": 88,
        "slots": {
            "action": "string: enable|disable|set|increase|decrease|resume",
            "speed": "integer",
            "unit": "string: mph|kph",
            "gap": "string: close|medium|far (following distance)",
        },
        "examples": {
            "full": [
                {
                    "utterance": "Set cruise control to 70 mph, medium gap",
                    "slots": {
                        "action": "set",
                        "speed": 70,
                        "unit": "mph",
                        "gap": "medium",
                    },
                },
                {
                    "utterance": "Increase speed to 110 kph",
                    "slots": {"action": "increase", "speed": 110, "unit": "kph"},
                },
            ],
            "partial": [
                {
                    "utterance": "Enable cruise, keep a far gap",
                    "slots": {"action": "enable", "gap": "far"},
                },
                {
                    "utterance": "Set to 65 miles per hour",
                    "slots": {"action": "set", "speed": 65, "unit": "mph"},
                },
            ],
            "minimal": [
                {"utterance": "Enable cruise control", "slots": {"action": "enable"}},
                {"utterance": "Resume cruise", "slots": {"action": "resume"}},
            ],
        },
    },
    "connectivity": {
        "depth": "medium",
        "count": 88,
        "slots": {
            "feature": "string: bluetooth|wifi|hotspot|carplay|android_auto|nfc",
            "action": "string: enable|disable|connect|disconnect|pair",
            "device_name": "string",
        },
        "examples": {
            "full": [
                {
                    "utterance": "Pair my iPhone via Bluetooth",
                    "slots": {
                        "feature": "bluetooth",
                        "action": "pair",
                        "device_name": "iPhone",
                    },
                },
                {
                    "utterance": "Connect to Samsung Galaxy hotspot",
                    "slots": {
                        "feature": "hotspot",
                        "action": "connect",
                        "device_name": "Samsung Galaxy",
                    },
                },
            ],
            "partial": [
                {
                    "utterance": "Enable Android Auto",
                    "slots": {"feature": "android_auto", "action": "enable"},
                },
                {
                    "utterance": "Disconnect Bluetooth",
                    "slots": {"feature": "bluetooth", "action": "disconnect"},
                },
            ],
            "minimal": [
                {"utterance": "Turn on WiFi", "slots": {"feature": "wifi"}},
                {"utterance": "Enable CarPlay", "slots": {"feature": "carplay"}},
            ],
        },
    },
    # ------------------------------------------------------------------
    # Shallow intents
    # ------------------------------------------------------------------
    "call_contact": {
        "depth": "shallow",
        "count": 80,
        "slots": {
            "contact_name": "string",
            "contact_type": "string: mobile|home|work",
        },
        "examples": {
            "full": [
                {
                    "utterance": "Call Rachel on her mobile",
                    "slots": {"contact_name": "Rachel", "contact_type": "mobile"},
                },
                {
                    "utterance": "Ring John at work",
                    "slots": {"contact_name": "John", "contact_type": "work"},
                },
            ],
            "partial": [
                {"utterance": "Call Mom", "slots": {"contact_name": "Mom"}},
                {
                    "utterance": "Dial Sarah's mobile",
                    "slots": {"contact_name": "Sarah", "contact_type": "mobile"},
                },
            ],
            "minimal": [
                {"utterance": "Call David", "slots": {"contact_name": "David"}},
                {"utterance": "Ring the office", "slots": {"contact_type": "work"}},
            ],
        },
    },
    "read_message": {
        "depth": "shallow",
        "count": 80,
        "slots": {
            "contact_name": "string",
            "message_type": "string: text|email|voicemail|whatsapp",
        },
        "examples": {
            "full": [
                {
                    "utterance": "Read the WhatsApp message from Alex",
                    "slots": {"contact_name": "Alex", "message_type": "whatsapp"},
                },
                {
                    "utterance": "Play back the voicemail from work",
                    "slots": {"contact_name": "work", "message_type": "voicemail"},
                },
            ],
            "partial": [
                {"utterance": "Read my latest text", "slots": {"message_type": "text"}},
                {
                    "utterance": "Play voicemail from Sarah",
                    "slots": {"contact_name": "Sarah", "message_type": "voicemail"},
                },
            ],
            "minimal": [
                {"utterance": "Read my messages", "slots": {"message_type": "text"}},
                {
                    "utterance": "Play back voicemail",
                    "slots": {"message_type": "voicemail"},
                },
            ],
        },
    },
    "safety_assist": {
        "depth": "shallow",
        "count": 80,
        "slots": {
            "feature": "string: lane_assist|parking_sensors|blind_spot|collision_warning|driver_alert|speed_limiter",  # noqa: E501
            "action": "string: enable|disable|status",
        },
        "examples": {
            "full": [
                {
                    "utterance": "Enable lane assist",
                    "slots": {"feature": "lane_assist", "action": "enable"},
                },
                {
                    "utterance": "Check blind spot warning status",
                    "slots": {"feature": "blind_spot", "action": "status"},
                },
            ],
            "partial": [
                {
                    "utterance": "Turn off parking sensors",
                    "slots": {"feature": "parking_sensors", "action": "disable"},
                },
                {
                    "utterance": "Enable collision warning",
                    "slots": {"feature": "collision_warning", "action": "enable"},
                },
            ],
            "minimal": [
                {"utterance": "Disable speed limiter", "slots": {"action": "disable"}},
                {
                    "utterance": "Turn on driver alert",
                    "slots": {"feature": "driver_alert"},
                },
            ],
        },
    },
    "vehicle_info": {
        "depth": "shallow",
        "count": 80,
        "slots": {
            "query_type": "string: fuel|battery|tire_pressure|oil_level|range|mileage|speed|rpm|temperature",  # noqa: E501
        },
        "examples": {
            "full": [
                {
                    "utterance": "Tell me the fuel level",
                    "slots": {"query_type": "fuel"},
                },
                {
                    "utterance": "Show tire pressure",
                    "slots": {"query_type": "tire_pressure"},
                },
            ],
            "partial": [
                {"utterance": "Check the battery", "slots": {"query_type": "battery"}},
                {"utterance": "What's my range", "slots": {"query_type": "range"}},
            ],
            "minimal": [
                {"utterance": "Oil level", "slots": {"query_type": "oil_level"}},
                {"utterance": "Show mileage", "slots": {"query_type": "mileage"}},
            ],
        },
    },
    "drive_mode": {
        "depth": "shallow",
        "count": 80,
        "slots": {
            "mode": "string: eco|sport|normal|comfort|offroad|snow|rain|track",
        },
        "examples": {
            "full": [
                {"utterance": "Switch to sport mode", "slots": {"mode": "sport"}},
                {"utterance": "Activate eco mode", "slots": {"mode": "eco"}},
            ],
            "partial": [
                {"utterance": "Enable offroad", "slots": {"mode": "offroad"}},
                {"utterance": "Snow mode on", "slots": {"mode": "snow"}},
            ],
            "minimal": [
                {"utterance": "Normal mode", "slots": {"mode": "normal"}},
                {"utterance": "Comfort", "slots": {"mode": "comfort"}},
            ],
        },
    },
}

# Question/status-query detection
_QUERY_STARTERS = (
    "what ",
    "what'",
    "how ",
    "why ",
    "is ",
    "are ",
    "do ",
    "does ",
    "which ",
    "who ",
    "who'",
    "whose ",
    "when ",
    "where ",
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_status_query(utterance: str) -> bool:
    """True for question/status utterances — never valid training commands."""
    u = utterance.strip().lower()
    if not u.endswith("?"):
        return False
    return any(u.startswith(s) for s in _QUERY_STARTERS)


def _validate_example(
    ex: object,
    intent: str,
    valid_keys: set[str],
    min_slots: int,
) -> bool:
    """Return True only if example passes all quality gates."""
    if not isinstance(ex, dict):
        return False
    utterance = ex.get("utterance", "")
    if not isinstance(utterance, str) or not utterance.strip():
        return False
    if _is_status_query(utterance):
        return False
    slots = ex.get("slots")
    if not isinstance(slots, dict):
        return False
    # Reject any out-of-schema keys
    if set(slots.keys()) - valid_keys:
        return False
    # Reject None / empty values
    for v in slots.values():
        if v is None or v == "" or v == []:
            return False
    # Enforce minimum slot count for this tier
    if len(slots) < min_slots:
        return False
    return True


# ---------------------------------------------------------------------------
# Prompt builder — density-tier aware with gold examples
# ---------------------------------------------------------------------------


def _build_tier_prompt(
    intent: str,
    count: int,
    slots: dict[str, str],
    tier: dict,
    gold: list[dict],
) -> str:
    slot_lines = "\n".join(f"  - {k}: {v}" for k, v in slots.items())
    valid_keys = ", ".join(f'"{k}"' for k in slots)
    gold_str = "\n".join(
        f'  {{"utterance": "{g["utterance"]}", "slots": {json.dumps(g["slots"])}}}'
        for g in gold
    )
    return f"""You are generating training data for a car voice assistant NLU model.

Task: Generate exactly {count} natural English voice commands for intent "{intent}".

DENSITY TIER: {tier["name"].upper()} — {tier["description"]}
Every example MUST have at least {tier["min_slots"]} slot(s).

Allowed slot keys (use ONLY these):
{slot_lines}

Gold examples for this tier:
{gold_str}

Rules — read carefully:
1. These are COMMANDS or REQUESTS, never questions or status queries.
   NEVER end with "?".
   NEVER start with: what, how, why, is, are, do, does, who, when, where.
2. If a slot value is NOT present in the utterance, OMIT the key entirely.
   NEVER use null, None, or empty string as a slot value.
3. Use only the allowed keys listed above: {valid_keys}
4. Use natural, varied phrasing — short commands, polite requests, casual phrases.
   Avoid repeating the same sentence structure.
5. Generate EXACTLY {count} examples.

Respond with ONLY valid JSON in this exact format, no markdown, no explanation:
{{"examples": [{{"utterance": "...", "slots": {{...}}}}, ...]}}"""


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------


def _call_ollama(model: str, prompt: str) -> list[dict]:
    import ollama as _ollama

    try:
        response = _ollama.generate(
            model=model,
            prompt=prompt,
            format="json",
            options={"temperature": 0.9, "num_predict": 8192},
        )
        raw = response.get("response", "")
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            examples = data.get("examples")
            if isinstance(examples, list):
                return examples
        return []
    except Exception as e:
        logger.error("Ollama call failed: %s", e)
        return []


def _generate_tier(
    model: str,
    intent: str,
    slots: dict,
    tier: dict,
    gold: list[dict],
    target: int,
) -> list[dict]:
    """Generate examples for one intent × one density tier."""
    valid_keys = set(slots.keys())
    min_slots = tier["min_slots"]
    collected: list[dict] = []
    attempts = 0
    max_attempts = (target // BATCH_SIZE + 2) * 4

    while len(collected) < target and attempts < max_attempts:
        remaining = target - len(collected)
        batch = min(BATCH_SIZE, remaining)
        prompt = _build_tier_prompt(intent, batch, slots, tier, gold)
        raw = _call_ollama(model, prompt)

        valid_count = 0
        for ex in raw:
            if _validate_example(ex, intent, valid_keys, min_slots):
                collected.append(
                    {
                        "utterance": ex["utterance"].strip(),
                        "intent": intent,
                        "slots": ex["slots"],
                    }
                )
                valid_count += 1
            else:
                logger.debug(
                    "  [%s/%s] rejected: %s", intent, tier["name"], str(ex)[:80]
                )

        attempts += 1
        logger.info(
            "  [%s | %s] attempt %d: %d/%d valid → %d/%d collected",
            intent,
            tier["name"],
            attempts,
            valid_count,
            len(raw),
            len(collected),
            target,
        )

    if len(collected) < target:
        logger.warning(
            "  [%s | %s] only %d/%d collected after %d attempts",
            intent,
            tier["name"],
            len(collected),
            target,
            attempts,
        )
    return collected[:target]


# ---------------------------------------------------------------------------
# Per-intent generation with resume support
# ---------------------------------------------------------------------------


def _raw_path(raw_dir: Path, intent: str) -> Path:
    return raw_dir / f"{intent}.jsonl"


def _load_raw(raw_dir: Path, intent: str) -> list[dict] | None:
    path = _raw_path(raw_dir, intent)
    if not path.exists():
        return None
    examples, dropped = [], 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                dropped += 1
    if dropped:
        logger.warning("  %s: dropped %d corrupt lines", path.name, dropped)
    return examples


def _save_raw(raw_dir: Path, intent: str, examples: list[dict]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = _raw_path(raw_dir, intent)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    tmp.replace(path)
    logger.info("  Saved %d examples → %s", len(examples), path.name)


def _tier_counts(total: int, tiers: list[dict]) -> list[int]:
    """Distribute total examples across tiers respecting fractions.

    Assigns leftover to the first (full) tier to avoid undercounting.
    """
    counts = [max(1, round(total * t["fraction"])) for t in tiers]
    diff = total - sum(counts)
    counts[0] += diff
    return counts


def generate_intent_v2(
    intent: str,
    schema: dict,
    raw_dir: Path,
    model: str,
    dry_run: bool,
) -> list[dict]:
    """Generate or load examples for one intent across all density tiers."""
    existing = _load_raw(raw_dir, intent)
    if existing is not None:
        logger.info("  Resume: %d cached examples for '%s'", len(existing), intent)
        return existing

    if dry_run:
        logger.info(
            "  [dry-run] Would generate %d examples for '%s'", schema["count"], intent
        )
        return []

    depth = schema["depth"]
    tiers = _TIER_CONFIG[depth]
    counts = _tier_counts(schema["count"], tiers)
    gold_by_tier = schema["examples"]

    all_examples: list[dict] = []
    for tier, count in zip(tiers, counts):
        gold = gold_by_tier.get(tier["name"], [])
        examples = _generate_tier(model, intent, schema["slots"], tier, gold, count)
        all_examples.extend(examples)

    _save_raw(raw_dir, intent, all_examples)
    return all_examples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v2 car command dataset")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be generated without calling Ollama",
    )
    parser.add_argument(
        "--model",
        default=OLLAMA_DEFAULT_MODEL,
        help=f"Ollama model (default: {OLLAMA_DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    raw_dir = get_data_dir() / "raw" / "synthetic"
    processed_dir = get_data_dir() / "processed"

    logger.info("Model: %s | dry-run: %s", args.model, args.dry_run)
    logger.info(
        "Target: %d total examples across %d intents",
        sum(s["count"] for s in INTENT_SCHEMAS_V2.values()),
        len(INTENT_SCHEMAS_V2),
    )

    all_examples: list[dict] = []
    total = len(INTENT_SCHEMAS_V2)

    for i, (intent, schema) in enumerate(INTENT_SCHEMAS_V2.items(), 1):
        logger.info(
            "[%d/%d] %s (%s, %d examples)",
            i,
            total,
            intent,
            schema["depth"],
            schema["count"],
        )
        examples = generate_intent_v2(intent, schema, raw_dir, args.model, args.dry_run)
        all_examples.extend(examples)

    if args.dry_run:
        logger.info("[dry-run] Done. No files written.")
        return

    # Deduplicate
    seen: set[str] = set()
    deduped: list[dict] = []
    for ex in all_examples:
        key = re.sub(r"\s+", " ", ex["utterance"].lower().strip())
        if key not in seen:
            seen.add(key)
            deduped.append(ex)

    dupes = len(all_examples) - len(deduped)
    if dupes:
        logger.info(
            "Deduplication: removed %d duplicates (%d → %d)",
            dupes,
            len(all_examples),
            len(deduped),
        )

    train, test = split_dataset(deduped, train_ratio=0.8, seed=42)
    save_dataset(train, test, output_dir=processed_dir)
    log_metadata(train, test)
    logger.info("Done: %d train / %d test", len(train), len(test))


if __name__ == "__main__":
    main()
