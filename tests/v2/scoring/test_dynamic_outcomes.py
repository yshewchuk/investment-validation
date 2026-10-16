"""Independent seven-member accounting oracles; no scoring or data fixtures."""
from __future__ import annotations

import builtins
import socket
from dataclasses import FrozenInstanceError, fields, replace
from itertools import permutations, product
from pathlib import Path

import pytest

from engine.v2.registry import DYNAMIC_MENU
from engine.v2.scoring import application
from engine.v2.scoring.dynamic_outcomes import (
    DYNAMIC_MENU_REFUSAL_CODES,
    DynamicCandidateOutcome,
    DynamicMenuIdentity,
    DynamicMenuRefusal,
    validate_dynamic_menu,
)

MENU = ("TWIN-P", "TWIN-P5", "CND-PS", "BFLY-P", "BFLY-P5", "RAMP7", "CTR5")
IDENTITY = {
    "ticker": "SYNTHETIC", "event_date": "2026-10-16T16:01:00",
    "session": "AMC", "event_id": "synthetic-event", "event_revision": "revision-1",
    "calendar_revision": "calendar-1", "snapshot_id": "snapshot-1",
    "release_id": "release-1", "deployment_id": "deployment-1",
    "decision_clock_id": "entry-close", "requested_decision_at": "2026-10-15T20:00:00Z",
    "mode": "shadow", "fill_model_hash": "fill-hash-1", "atm_policy_hash": "atm-hash-1",
}


def identity(**updates):
    return DynamicMenuIdentity(**(IDENTITY | updates))


def record(strategy, /, *, context=None, **updates):
    return DynamicCandidateOutcome(**({
        "strategy": strategy, "identity": context or identity(),
        "request_hash": f"request:{strategy}", "score_id": f"score:{strategy}",
        "validation_status": "scored", "readiness": "ready",
        "reason_codes": ("WIDE_MARKET",), "warnings": ("synthetic warning",),
    } | updates))


def refusal(strategy, *, context=None, **updates):
    return DynamicCandidateOutcome(**({
        "strategy": strategy, "identity": context or identity(),
        "refusal_code": "MISSING_STAGED_INPUT", "reason_codes": ("SOURCE_NOT_FINAL",),
    } | updates))


def menu():
    return tuple(record(strategy) for strategy in MENU)


def assert_invalid(call):
    with pytest.raises(DynamicMenuRefusal) as caught:
        call()
    error = caught.value
    assert type(error) is DynamicMenuRefusal
    assert error.code == "INVALID_CHOOSER_MENU"
    assert error.detail == "candidate outcomes do not form one valid event menu"
    assert error.missing_members == ()
    assert str(error) == "INVALID_CHOOSER_MENU: candidate outcomes do not form one valid event menu"
    assert error.__cause__ is None


def assert_missing(outcomes, expected):
    with pytest.raises(DynamicMenuRefusal) as caught:
        validate_dynamic_menu(identity(), outcomes)
    error = caught.value
    assert type(error) is DynamicMenuRefusal
    assert error.code == "CHOOSER_MENU_INCOMPLETE"
    assert error.missing_members == expected
    assert error.detail == "missing candidate outcomes: " + ", ".join(expected)
    assert str(error) == "CHOOSER_MENU_INCOMPLETE: " + error.detail
    assert error.__cause__ is None


def test_registry_and_closed_error_vocabulary_are_exact():
    assert DYNAMIC_MENU == MENU
    assert DYNAMIC_MENU_REFUSAL_CODES == frozenset({
        "INVALID_CHOOSER_MENU", "CHOOSER_MENU_INCOMPLETE",
    })
    with pytest.raises(ValueError, match="^unknown dynamic menu refusal code$"):
        DynamicMenuRefusal("NO_CHOOSER_CANDIDATE")


def test_all_5040_input_orders_return_the_identical_registry_order():
    expected = menu()
    for ordering in permutations(expected):
        actual = validate_dynamic_menu(identity(), ordering)
        assert actual == expected
        assert tuple(item.strategy for item in actual) == MENU
    assert validate_dynamic_menu(identity(), list(reversed(expected))) == expected


@pytest.mark.parametrize("missing", [(), ("TWIN-P",), ("CTR5",), ("TWIN-P5", "BFLY-P", "CTR5"), MENU])
def test_missing_members_are_named_in_registry_order(missing):
    outcomes = tuple(item for item in menu() if item.strategy not in missing)
    if not missing:
        assert validate_dynamic_menu(identity(), outcomes) == menu()
    else:
        assert_missing(outcomes, missing)
        assert_missing(tuple(reversed(outcomes)), missing)


@pytest.mark.parametrize("member", MENU)
@pytest.mark.parametrize("factory", [record, refusal])
def test_duplicates_and_record_refusal_overlap_refuse_in_both_orders(member, factory):
    outcomes = menu() + (factory(member),)
    for ordering in (outcomes, tuple(reversed(outcomes))):
        assert_invalid(lambda: validate_dynamic_menu(identity(), ordering))


def test_duplicate_refusals_do_not_fill_a_missing_member():
    outcomes = (refusal("TWIN-P"), refusal("TWIN-P"))
    assert_invalid(lambda: validate_dynamic_menu(identity(), outcomes))


@pytest.mark.parametrize("strategy", ["DYN-SV", "STR-THRU", "STR-RUNUP", "unknown", "twin-p"])
def test_unexpected_member_is_invalid_even_when_another_is_missing(strategy):
    outcomes = menu()[:-1] + (record(strategy),)
    assert_invalid(lambda: validate_dynamic_menu(identity(), outcomes))


@pytest.mark.parametrize("field", tuple(IDENTITY))
@pytest.mark.parametrize("factory", [record, refusal])
def test_every_identity_component_is_compared_exactly(field, factory):
    changed = identity(**{field: IDENTITY[field] + "-other"})
    outcomes = menu()[:-1] + (factory("CTR5", context=changed),)
    for ordering in (outcomes, tuple(reversed(outcomes))):
        assert_invalid(lambda: validate_dynamic_menu(identity(), ordering))
    assert validate_dynamic_menu(changed, tuple(factory(s, context=changed) for s in MENU))


@pytest.mark.parametrize("changes", [
    {"event_date": "2026-10-16"}, {"event_date": "2026-10-16T16:02:00"},
    {"session": "BMO"}, {"event_revision": None},
    {"requested_decision_at": "2026-10-15T20:00:01Z"},
])
def test_planted_cross_event_and_clock_joins_are_not_normalized(changes):
    wrong = replace(menu()[3], identity=identity(**changes))
    assert_invalid(lambda: validate_dynamic_menu(identity(), (*menu()[:3], wrong, *menu()[4:])))


def test_interleaved_events_are_rejected_then_validate_separately():
    first = menu()
    other_context = identity(session="BMO", event_id="second-event")
    second = tuple(record(s, context=other_context) for s in MENU)
    combined = tuple(value for pair in zip(first, second, strict=True) for value in pair)
    assert_invalid(lambda: validate_dynamic_menu(identity(), combined))
    assert validate_dynamic_menu(identity(), combined[::2]) == first
    assert validate_dynamic_menu(other_context, combined[1::2]) == second


@pytest.mark.parametrize("field", ["request_hash", "score_id"])
def test_planted_reused_candidate_id_under_another_strategy_refuses(field):
    outcomes = menu()
    wrong = replace(outcomes[-1], **{field: getattr(outcomes[0], field)})
    assert_invalid(lambda: validate_dynamic_menu(identity(), (*outcomes[:-1], wrong)))


def test_refusal_request_identity_may_be_absent_but_cannot_overlap_a_record():
    outcomes = (refusal("TWIN-P"), *menu()[1:])
    assert validate_dynamic_menu(identity(), outcomes) == outcomes
    wrong = replace(outcomes[0], request_hash=outcomes[1].request_hash)
    assert_invalid(lambda: validate_dynamic_menu(identity(), (wrong, *outcomes[1:])))
    prebuilt = (replace(outcomes[0], request_hash="prebuilt"), *outcomes[1:])
    assert validate_dynamic_menu(identity(), prebuilt) == prebuilt


def test_all_refused_and_mixed_menus_are_complete_without_selecting():
    refused = tuple(refusal(s) for s in MENU)
    assert validate_dynamic_menu(identity(), tuple(reversed(refused))) == refused
    mixed = (
        record("TWIN-P", validation_status="refused", readiness="refused",
               reason_codes=("MODEL_NOT_READY", "CHOOSER_MISSING_FEATURES")),
        record("TWIN-P5", reason_codes=("ENTRY_RULE_NO", "WIDE_MARKET")),
        *refused[2:],
    )
    assert validate_dynamic_menu(identity(), mixed) == mixed
    assert mixed[0].reason_codes == ("MODEL_NOT_READY", "CHOOSER_MISSING_FEATURES")
    assert mixed[1].reason_codes == ("ENTRY_RULE_NO", "WIDE_MARKET")
    assert tuple(item.refusal_code for item in mixed[2:]) == ("MISSING_STAGED_INPUT",) * 5


def test_all_128_record_refusal_combinations_preserve_exact_summaries():
    for factories in product((record, refusal), repeat=7):
        outcomes = tuple(factory(member) for factory, member in zip(factories, MENU, strict=True))
        assert validate_dynamic_menu(identity(), tuple(reversed(outcomes))) == outcomes


def test_reason_and_warning_sequences_are_detached_and_immutable():
    reasons = ["NO_SCORE", "NO_SCORE"]
    warnings = ["first", "second"]
    outcome = record("TWIN-P", reason_codes=reasons, warnings=warnings)
    reasons.append("unexpected")
    warnings.clear()
    assert outcome.reason_codes == ("NO_SCORE", "NO_SCORE")
    assert outcome.warnings == ("first", "second")
    with pytest.raises(FrozenInstanceError):
        outcome.reason_codes = ()
    with pytest.raises(FrozenInstanceError):
        outcome.identity.session = "BMO"
    result = validate_dynamic_menu(identity(), (outcome, *menu()[1:]))
    with pytest.raises(TypeError):
        result[0] = result[1]
    assert hash(outcome) == hash(record("TWIN-P", reason_codes=reasons[:2], warnings=["first", "second"]))


@pytest.mark.parametrize("field", tuple(IDENTITY))
@pytest.mark.parametrize("value", ["", " ", 0, False, [], {}])
def test_malformed_identity_cannot_masquerade_as_complete(field, value):
    assert_invalid(lambda: identity(**{field: value}))


@pytest.mark.parametrize("field", [name for name in IDENTITY if name != "event_revision"])
def test_required_identity_fields_cannot_be_none(field):
    assert_invalid(lambda: identity(**{field: None}))


def test_optional_event_revision_can_be_absent_without_losing_identity():
    context = identity(event_revision=None)
    outcomes = tuple(record(s, context=context) for s in MENU)
    assert validate_dynamic_menu(context, outcomes) == outcomes
    assert_invalid(lambda: validate_dynamic_menu(identity(), outcomes))


@pytest.mark.parametrize("updates", [
    {"identity": None}, {"identity": IDENTITY}, {"strategy": None}, {"strategy": ""},
    {"request_hash": None}, {"score_id": None}, {"readiness": None}, {"validation_status": None},
    {"request_hash": ""}, {"score_id": False}, {"readiness": []}, {"validation_status": {}},
    {"refusal_code": "QUOTE_MISSING"}, {"refusal_code": ""},
    {"reason_codes": "NO_SCORE"}, {"reason_codes": [None]}, {"reason_codes": [" "]},
    {"reason_codes": ["NO_SCORE", None]}, {"reason_codes": [None, "NO_SCORE"]},
    {"warnings": {"warning": "text"}}, {"warnings": [42]}, {"warnings": None},
    {"warnings": ["synthetic warning", 42]}, {"warnings": [42, "synthetic warning"]},
])
def test_malformed_record_summaries_and_single_item_overlap_refuse(updates):
    assert_invalid(lambda: record("TWIN-P", **updates))


@pytest.mark.parametrize("updates", [
    {"refusal_code": None}, {"refusal_code": ""}, {"refusal_code": 42},
    {"score_id": "score"}, {"readiness": "refused"}, {"validation_status": "refused"},
])
def test_malformed_refusal_summaries_refuse(updates):
    assert_invalid(lambda: refusal("TWIN-P", **updates))


@pytest.mark.parametrize("outcomes", [None, {}, {"TWIN-P": "record"}, "menu", 7, (None,)])
def test_malformed_envelopes_are_typed_and_never_missing_menu(outcomes):
    assert_invalid(lambda: validate_dynamic_menu(identity(), outcomes))


@pytest.mark.parametrize("outcomes", [menu(), (), None])
def test_malformed_expected_identity_refuses(outcomes):
    assert_invalid(lambda: validate_dynamic_menu(IDENTITY, outcomes))


def test_invalid_outcome_takes_precedence_over_incomplete_accounting_in_every_order():
    outcomes = (record("TWIN-P"), refusal("CTR5", context=identity(release_id="wrong")))
    for ordering in permutations(outcomes):
        assert_invalid(lambda: validate_dynamic_menu(identity(), ordering))


def test_summary_has_no_economics_selection_or_source_admission_fields():
    assert {field.name for field in fields(DynamicCandidateOutcome)} == {
        "strategy", "identity", "request_hash", "score_id", "validation_status",
        "readiness", "refusal_code", "reason_codes", "warnings",
    }
    assert {field.name for field in fields(DynamicMenuIdentity)} == set(IDENTITY)


def test_validation_does_no_scoring_fitting_file_or_network_io(monkeypatch):
    from engine.v2.models.loader import FrozenInference
    from engine.v2.models.no_fit import no_fit_guard

    expected = menu()

    def forbidden(*args, **kwargs):
        raise AssertionError("outcome validation attempted execution or I/O")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(socket, "socket", forbidden)
        patch.setattr(application, "score_one", forbidden)
        patch.setattr(application, "score_event", forbidden)
        patch.setattr(application, "score_batch", forbidden)
        patch.setattr(FrozenInference, "infer", forbidden)
        with no_fit_guard():
            assert validate_dynamic_menu(identity(), tuple(reversed(expected))) == expected
            assert_missing(expected[:-1], ("CTR5",))
            assert_invalid(lambda: validate_dynamic_menu(identity(), expected + expected))
