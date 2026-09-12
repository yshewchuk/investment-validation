"""Acceptance for the ComparisonReceipt and the staged comparator (§5).

The four stated properties:

* two records differing in three unrelated fields produce **three** findings,
  not one;
* a record compared against itself produces ``agree``;
* an empty input produces ``incomparable``;
* removing a field from the digest removes it from the comparison, with no
  edit to the comparator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.diagnosis import (  # noqa: E402
    AGREE,
    DIFFER,
    INCOMPARABLE,
    SCORER_V1,
    Tolerance,
    TolerancePolicy,
    canonical_json,
    compare_records,
    content_hash,
    flatten,
    load_stage_plan,
    merge_receipts,
    problem,
)


def record(**overrides) -> dict:
    """A ScoreRecord-shaped mapping, one field per stage."""
    base = {
        "ticker": "AAPL",
        "strategy": "TWIN-P5",
        "event_date": "2026-10-29",
        "model_inputs": {"or_implied": 6.5, "mcap_log": 28.1},
        "forecast_abs_move": 5.21874,
        "legs": [{"name": "a", "strike": 240.0}, {"name": "b", "strike": 250.0}],
        "entry_cost": 3.4512,
        "ci_low": -0.011,
        "ci_high": 0.042,
        "exp_pnl_sim": 0.0187,
        "gate_pass": True,
        "flags": ["THIN_ANALOGS"],
        "chooser_score": 0.0042,
        "structure_params": {"width_moneyness": 0.05123456789},
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# the four acceptance properties
# --------------------------------------------------------------------------


def test_three_unrelated_differences_produce_three_findings():
    left = record()
    right = record(
        forecast_abs_move=None,          # forecast
        ci_low=-0.019,                   # analogs
        structure_params={"width_moneyness": 0.051235},  # serialization
    )
    receipt = compare_records(left, right)
    assert receipt.verdict == DIFFER
    assert len(receipt.findings) == 3, receipt.summary()
    assert receipt.stages_named() == ("forecast", "analogs", "serialization")


def test_a_record_compared_against_itself_agrees():
    receipt = compare_records(record(), record())
    assert receipt.verdict == AGREE
    assert receipt.findings == ()
    assert receipt.first_differing_stage is None


def test_an_empty_input_is_incomparable_not_agreement():
    for left, right in (({}, {}), (record(), {}), ({}, record())):
        receipt = compare_records(left, right)
        assert receipt.verdict != AGREE, (left, right)
    assert compare_records({}, {}).verdict == INCOMPARABLE


def test_a_missing_input_is_incomparable():
    receipt = compare_records(None, record())
    assert receipt.verdict == INCOMPARABLE
    assert receipt.problems[0]["code"] == "MISSING_INPUT"
    assert receipt.problems[0]["category"] == "dependency"


def test_removing_a_field_removes_it_from_the_comparison():
    """No hand-maintained field list — `28cf8b1`, restated as a property."""
    left, right = record(), record(ci_low=-0.9)
    assert len(compare_records(left, right).findings) == 1
    left.pop("ci_low")
    right.pop("ci_low")
    receipt = compare_records(left, right)
    assert receipt.verdict == AGREE
    assert "ci_low" not in canonical_json(receipt.payload())


def test_adding_a_field_adds_it_to_the_comparison():
    left = record(new_field_nobody_declared=1)
    right = record(new_field_nobody_declared=2)
    receipt = compare_records(left, right)
    assert [f.field_path for f in receipt.findings] == ["new_field_nobody_declared"]
    assert receipt.findings[0].first_differing_stage == "unassigned"


# --------------------------------------------------------------------------
# stage localization
# --------------------------------------------------------------------------


def test_a_finding_names_a_stage_and_a_field_path_never_only_a_row():
    receipt = compare_records(
        record(), record(structure_params={"width_moneyness": 0.051235})
    )
    finding = receipt.findings[0]
    assert finding.first_differing_stage == "serialization"
    assert finding.field_path == "structure_params.width_moneyness"
    assert "row" not in finding.describe()


def test_stage_hashes_cover_every_stage_in_order():
    receipt = compare_records(record(), record())
    assert tuple(r.stage_id for r in receipt.stage_hashes) == SCORER_V1.stage_ids()


def test_first_differing_stage_is_the_earliest_with_agreeing_inputs():
    receipt = compare_records(record(), record(model_inputs={"or_implied": 9.9}))
    assert receipt.first_differing_stage == "features"


def test_a_downstream_finding_is_not_claimed_independent():
    """features differs, so forecast's inputs differ: no independence claim."""
    left = record()
    right = record(model_inputs={"or_implied": 9.9}, forecast_abs_move=1.0)
    receipt = compare_records(left, right)
    by_stage = {f.first_differing_stage: f for f in receipt.findings}
    assert by_stage["forecast"].not_downstream_of == ()
    assert by_stage["features"].first_differing_stage == "features"


def test_unrelated_findings_name_each_other_as_independent():
    left = record()
    right = record(ci_low=-0.9, structure_params={"width_moneyness": 0.9})
    receipt = compare_records(left, right)
    ids = {f.finding_id for f in receipt.findings}
    for finding in receipt.findings:
        assert set(finding.not_downstream_of) == ids - {finding.finding_id}


def test_analogs_do_not_depend_on_the_forecast():
    """The dependency that makes 2026-09-11's two bugs separable."""
    assert "forecast" not in SCORER_V1.depends_on("analogs")


def test_unknown_stage_plan_raises_rather_than_defaulting():
    with pytest.raises(KeyError):
        load_stage_plan("scorer.v99")


# --------------------------------------------------------------------------
# null masks, missing fields, types
# --------------------------------------------------------------------------


def test_a_null_mask_difference_is_its_own_kind():
    receipt = compare_records(record(), record(forecast_abs_move=None))
    finding = receipt.findings[0]
    assert finding.kind == "null_mask"
    assert finding.null_mask_right and not finding.null_mask_left


def test_both_null_agrees():
    left = record(forecast_abs_move=None)
    right = record(forecast_abs_move=None)
    assert compare_records(left, right).verdict == AGREE


def test_a_missing_field_is_not_the_same_as_a_null_one():
    left, right = record(), record()
    right.pop("chooser_score")
    finding = compare_records(left, right).findings[0]
    assert finding.kind == "missing_field"

    right["chooser_score"] = None
    finding = compare_records(left, right).findings[0]
    assert finding.kind == "null_mask"


def test_a_shorter_leg_list_names_the_missing_positions():
    left = record()
    right = record(legs=[{"name": "a", "strike": 240.0}])
    paths = {f.field_path for f in compare_records(left, right).findings}
    assert paths == {"legs[1].name", "legs[1].strike"}


def test_a_type_change_is_reported():
    finding = compare_records(record(), record(gate_pass="True")).findings[0]
    assert finding.kind == "type"


def test_bool_and_int_are_not_compared_numerically():
    assert compare_records(record(), record(gate_pass=1)).verdict == DIFFER


# --------------------------------------------------------------------------
# tolerances
# --------------------------------------------------------------------------


def test_the_default_policy_is_exact():
    receipt = compare_records(record(), record(entry_cost=3.4512000000001))
    assert receipt.verdict == DIFFER
    assert receipt.findings[0].tolerance_applied == "exact"


def test_a_declared_per_field_tolerance_absorbs_a_declared_field_only():
    policy = TolerancePolicy(
        policy_id="test.v1",
        rules=(("entry_cost", Tolerance(absolute=1e-9, reason="mid-fill rounding")),),
    )
    left = record()
    near = record(entry_cost=3.4512 + 5e-10)
    assert compare_records(left, near, tolerance_policy=policy).verdict == AGREE
    far = record(ci_low=-0.011 + 5e-10)
    assert compare_records(left, far, tolerance_policy=policy).verdict == DIFFER


def test_exceeded_by_reports_how_far_outside_tolerance():
    policy = TolerancePolicy(
        policy_id="t", rules=(("entry_cost", Tolerance(absolute=0.01, reason="r")),)
    )
    receipt = compare_records(record(), record(entry_cost=3.4712),
                              tolerance_policy=policy)
    assert receipt.findings[0].exceeded_by == pytest.approx(0.01, rel=1e-6)


# --------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------


def test_merge_reports_incomparable_when_the_corpus_collapsed():
    """Forty fixtures that silently became zero is not agreement."""
    merged = merge_receipts([], comparison_kind="tier0_replay", expected=40)
    assert merged.verdict == INCOMPARABLE
    assert merged.population.expected == 40
    assert merged.population.compared == 0


def test_merge_reports_incomparable_when_some_pairs_went_missing():
    receipts = [compare_records(record(), record()) for _ in range(3)]
    merged = merge_receipts(receipts, comparison_kind="tier0_replay", expected=5)
    assert merged.verdict == INCOMPARABLE


def test_merge_of_agreeing_receipts_agrees():
    receipts = [compare_records(record(), record()) for _ in range(3)]
    merged = merge_receipts(receipts, comparison_kind="tier0_replay", expected=3)
    assert merged.verdict == AGREE


def test_merge_keeps_every_finding_from_every_pair():
    receipts = [
        compare_records(record(), record(ci_low=-0.9)),
        compare_records(record(), record(entry_cost=9.9)),
    ]
    merged = merge_receipts(receipts, comparison_kind="tier0_replay", expected=2)
    assert len(merged.findings) == 2
    assert merged.stages_named() == ("analogs", "pricing") or set(
        merged.stages_named()) == {"analogs", "pricing"}


# --------------------------------------------------------------------------
# canonical form and the envelope
# --------------------------------------------------------------------------


def test_canonical_json_sorts_keys_and_preserves_array_order():
    assert canonical_json({"b": 1, "a": [3, 1, 2]}) == '{"a":[3,1,2],"b":1}'


def test_canonical_json_never_rounds():
    """`b33036c` and `6b9d5cf` were both identity taken over a rounded value."""
    value = 0.05123456789012345
    assert repr(value)[2:10] in canonical_json({"w": value})


def test_content_hash_shape():
    digest = content_hash({"a": 1})
    assert digest.startswith("sha256:") and len(digest) == 7 + 64


def test_the_envelope_is_excluded_from_the_payload():
    receipt = compare_records(record(), record())
    assert "envelope" not in receipt.payload()
    assert receipt.envelope.duration_seconds is not None


def test_two_receipts_over_the_same_records_have_the_same_payload():
    """A replay reproduces a payload without reproducing elapsed time."""
    one = compare_records(record(), record(ci_low=-0.9))
    two = compare_records(record(), record(ci_low=-0.9))
    assert canonical_json(one.payload()) == canonical_json(two.payload())
    assert one.envelope != two.envelope or one.envelope.started_at is not None


def test_problem_rejects_an_undeclared_category():
    with pytest.raises(ValueError):
        problem("X", "y", category="whatever")


def test_flatten_distinguishes_an_empty_container_from_an_absent_one():
    assert flatten({"a": {}}) == {"a": {}}
    assert flatten({"a": []}) == {"a": []}
    assert flatten({"a": {"b": 1}}) == {"a.b": 1}


# --------------------------------------------------------------------------
# the 2026-09-12 review defects: a comparator that can call different records
# equal is worse than no comparator
# --------------------------------------------------------------------------


def test_adjacent_integers_above_2pow53_are_a_finding():
    """float(2**53) == float(2**53 + 1); ints must never route through float."""
    left = record(n_analogs=2**53)
    right = record(n_analogs=2**53 + 1)
    receipt = compare_records(left, right)
    assert receipt.verdict == DIFFER
    finding = receipt.findings[0]
    assert finding.field_path == "n_analogs"
    assert finding.delta == 1 and isinstance(finding.delta, int)


def test_equal_large_integers_still_agree_exactly():
    assert compare_records(record(n_analogs=2**53 + 1),
                           record(n_analogs=2**53 + 1)).verdict == AGREE


def test_a_leg_list_and_a_numeric_keyed_dict_do_not_compare_equal():
    left = record(legs=[{"strike": 240.0}])
    right = record()
    right["legs"] = {0: {"strike": 240.0}}
    receipt = compare_records(left, right)
    assert receipt.verdict == DIFFER
    paths = {f.field_path for f in receipt.findings}
    assert "legs[0].strike" in paths and "legs.0.strike" in paths


def test_a_dotted_key_cannot_collide_with_a_nested_path():
    left = {"model_inputs": {"a.b": 1}}
    right = {"model_inputs": {"a": {"b": 1}}}
    receipt = compare_records(left, right)
    assert receipt.verdict == DIFFER
    assert {f.field_path for f in receipt.findings} == {
        r"model_inputs.a\.b", "model_inputs.a.b"}


def test_a_dotted_root_key_is_assigned_by_its_full_key():
    from engine.v2.diagnosis.record_comparator import root_of
    assert root_of(r"a\.b[0].c") == "a.b"
    assert root_of("legs[12].strike") == "legs"
    assert SCORER_V1.stage_of(r"model_inputs.a\.b") == "features"


def test_canonical_json_serializes_numbers_per_ecmascript():
    assert canonical_json(1.0) == "1"
    assert canonical_json(-0.0) == "0"
    assert canonical_json(1e-5) == "0.00001"
    assert canonical_json(1e-7) == "1e-7"
    assert canonical_json(1e21) == "1e+21"
    assert canonical_json(1e20) == "100000000000000000000"
    assert canonical_json(2**53 + 1) == "9007199254740993"
    assert canonical_json(-1.5e22) == "-1.5e+22"


def test_canonical_json_sorts_keys_by_utf16_code_units():
    """RFC 8785 §3.2.3: astral chars (surrogate-first) sort below U+FB33."""
    import json as _json
    doc = {"\u20ac": "a", "\r": "b", "\ufb33": "c", "1": "d",
           "\U0001f602": "e", "\u00f6": "f", "</script>": "g"}
    order = _json.loads(canonical_json(doc),
                        object_pairs_hook=lambda p: [k for k, _ in p])
    assert order == ["\r", "1", "</script>", "\u00f6", "\u20ac",
                     "\U0001f602", "\ufb33"]


def test_localization_links_do_not_claim_causal_independence():
    """One bootstrap defect moves BOTH endpoints; the link must not deny that."""
    receipt = compare_records(record(ci_low=-0.9, ci_high=-0.5),
                              record(ci_low=0.1, ci_high=0.5))
    assert len(receipt.findings) == 2
    for finding in receipt.findings:
        assert finding.first_differing_stage == "analogs"
    # the field states what is known — graph separation, not cause separation
    assert not hasattr(receipt.findings[0], "independent_of")
