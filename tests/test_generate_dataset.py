"""Tests for pure utility functions in src/generate_dataset.py."""

import json

from src.generate_dataset import _parse_llm_response, _validate_example

VALID_SLOT_KEYS = {"zone", "temperature", "mode"}


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------


def test_parse_llm_response_direct_array() -> None:
    """Parses a top-level JSON array."""
    data = [{"utterance": "cool it", "slots": {}}]
    assert _parse_llm_response(json.dumps(data)) == data


def test_parse_llm_response_examples_key() -> None:
    """Parses {"examples": [...]} wrapper format (preferred LLM output)."""
    data = [{"utterance": "cool it", "slots": {}}]
    wrapped = {"examples": data}
    assert _parse_llm_response(json.dumps(wrapped)) == data


def test_parse_llm_response_other_key() -> None:
    """Finds the first list value regardless of key name."""
    data = [{"utterance": "cool it", "slots": {}}]
    wrapped = {"items": data}
    assert _parse_llm_response(json.dumps(wrapped)) == data


def test_parse_llm_response_invalid_json() -> None:
    """Returns empty list on malformed JSON."""
    assert _parse_llm_response("not json at all") == []


def test_parse_llm_response_non_list_scalar() -> None:
    """Returns empty list when JSON is a scalar."""
    assert _parse_llm_response(json.dumps("just a string")) == []


def test_parse_llm_response_empty_string() -> None:
    """Returns empty list on empty input."""
    assert _parse_llm_response("") == []


# ---------------------------------------------------------------------------
# _validate_example
# ---------------------------------------------------------------------------


def test_validate_example_valid() -> None:
    """Returns True for a well-formed example."""
    ex = {"utterance": "cool the front", "slots": {"zone": "front"}}
    assert _validate_example(ex, "set_climate", VALID_SLOT_KEYS) is True


def test_validate_example_empty_slots() -> None:
    """Empty slots dict is valid."""
    ex = {"utterance": "turn on AC", "slots": {}}
    assert _validate_example(ex, "set_climate", VALID_SLOT_KEYS) is True


def test_validate_example_missing_utterance() -> None:
    """Returns False when utterance key is absent."""
    ex = {"slots": {"zone": "front"}}
    assert _validate_example(ex, "set_climate", VALID_SLOT_KEYS) is False


def test_validate_example_empty_utterance() -> None:
    """Returns False for blank utterance string."""
    ex = {"utterance": "   ", "slots": {}}
    assert _validate_example(ex, "set_climate", VALID_SLOT_KEYS) is False


def test_validate_example_unknown_slot_key() -> None:
    """Returns False when slots contain an unrecognised key."""
    ex = {"utterance": "cool it", "slots": {"unknown_key": "value"}}
    assert _validate_example(ex, "set_climate", VALID_SLOT_KEYS) is False


def test_validate_example_slots_not_dict() -> None:
    """Returns False when slots is not a dict."""
    ex = {"utterance": "cool it", "slots": ["zone", "front"]}
    assert _validate_example(ex, "set_climate", VALID_SLOT_KEYS) is False


def test_validate_example_not_dict() -> None:
    """Returns False for non-dict input."""
    assert _validate_example("just a string", "set_climate", VALID_SLOT_KEYS) is False
    assert _validate_example(None, "set_climate", VALID_SLOT_KEYS) is False
