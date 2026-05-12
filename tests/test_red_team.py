from collections import Counter

from src.red_team import RED_TEAM_CASES, SCHEMA_INTENTS, _pass_for_case


def test_asr_noise_pass():
    case = {"category": "asr_noise", "expected_intent": "set_climate"}
    assert _pass_for_case(case, "set_climate", False) is True


def test_asr_noise_fail_wrong_intent():
    case = {"category": "asr_noise", "expected_intent": "set_climate"}
    assert _pass_for_case(case, "navigate", False) is False


def test_asr_noise_fail_parse_error():
    case = {"category": "asr_noise", "expected_intent": "set_climate"}
    assert _pass_for_case(case, None, True) is False


def test_ambiguous_pass():
    case = {"category": "ambiguous", "expected_intent": "adjust_volume"}
    assert _pass_for_case(case, "adjust_volume", False) is True


def test_ambiguous_fail():
    case = {"category": "ambiguous", "expected_intent": "adjust_volume"}
    assert _pass_for_case(case, "set_climate", False) is False


def test_ood_passes_when_parseable():
    case = {"category": "ood_intent", "expected_intent": None}
    assert _pass_for_case(case, "navigate", False) is True


def test_ood_fails_on_parse_failure():
    case = {"category": "ood_intent", "expected_intent": None}
    assert _pass_for_case(case, None, True) is False


def test_adversarial_pass_valid_schema_intent():
    case = {"category": "adversarial", "expected_intent": None}
    assert _pass_for_case(case, "set_climate", False) is True


def test_adversarial_fail_injected_intent():
    case = {"category": "adversarial", "expected_intent": None}
    assert _pass_for_case(case, "HACKED", False) is False


def test_adversarial_fail_parse_error():
    case = {"category": "adversarial", "expected_intent": None}
    assert _pass_for_case(case, None, True) is False


def test_red_team_cases_required_keys():
    required = {"category", "input", "expected_intent", "description"}
    valid_categories = {"asr_noise", "ambiguous", "ood_intent", "adversarial"}
    for case in RED_TEAM_CASES:
        assert required <= case.keys(), f"Missing keys in: {case}"
        assert case["category"] in valid_categories, f"Bad category: {case['category']}"


def test_red_team_cases_counts():
    counts = Counter(c["category"] for c in RED_TEAM_CASES)
    assert counts["asr_noise"] == 7
    assert counts["ambiguous"] == 6
    assert counts["ood_intent"] == 5
    assert counts["adversarial"] == 5


def test_schema_intents_has_all_14():
    assert len(SCHEMA_INTENTS) == 14


def test_ood_cases_have_none_expected_intent():
    ood = [c for c in RED_TEAM_CASES if c["category"] == "ood_intent"]
    assert all(c["expected_intent"] is None for c in ood)
