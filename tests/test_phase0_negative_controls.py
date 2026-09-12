"""The seeded negative controls, at the comparator level — tier 0, synthetic.

A check that has never failed is not known to work. On 2026-09-11 a suite of
1,567 tests passed over five live defects. So the evidence that phase 0 works is
not a pass count: it is that every seeded cause produces the findings it is
SPECIFIED to produce (``checks/replay_identity.SEEDED_CONTROLS``), in the stage
it is specified to land in, and nothing else.

The same specification is judged in two more places, against real data:
``checks/tier0_corpus.py`` plants the causes into the real frozen corpus, and
``tools/replay_tier1.py --seed-defects`` plants them into the real engine
stages. This file proves the comparator and the judgement themselves, on a
clean checkout with no corpus.

An earlier version of this file asserted ``len(causes) == 4`` over a literal
dict, asserted a docstring existed, and seeded the `6b9d5cf` control as a
``digest`` field that ANY record edit would also have moved. Those assertions
could not fail; the ones below can.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.replay_identity import FORECAST_BLOCK, SEEDED_CONTROLS, check_control  # noqa: E402
from checks.tier0_corpus import finding_dicts, round_params  # noqa: E402
from engine.v2.diagnosis import (  # noqa: E402
    AGREE,
    DIFFER,
    INCOMPARABLE,
    compare_records,
    content_hash,
    flatten,
    merge_receipts,
)

CAUSES = tuple(SEEDED_CONTROLS)
TIER0_KINDS = ("record", "integrity")


# --------------------------------------------------------------------------
# a baseline record, shaped like the board's
# --------------------------------------------------------------------------


def baseline() -> dict:
    """One frozen tier-0 record, in the shape `ScoreResult.as_dict()` produces."""
    return {
        "ticker": "MTN",
        "strategy": "TWIN-P5",
        "as_of": "2026-09-11",
        "event_date": "2026-09-28",
        "session": "AMC",
        "entry_date": "2026-09-28",
        "exit_date": "2026-09-29",
        "evidence_cutoff": "2026-09-28",
        "snapshot_hash": "sha256:1111",
        "model_inputs": {"or_implied": 6.5312, "mcap_log": 28.1147},
        "model_input_as_of": "2026-09-28",
        "model_versions": {"size": "size_v1_4", "gate": "gate_midfill_str_thru"},
        "implied_move": 6.5312,
        "forecast_abs_move": 5.218743916,
        "forecast_p10": 2.9114,
        "forecast_p90": 8.0041,
        "forecast_sd": 1.9933,
        "forecast_model": "size_v1_4",
        "forecast_fold": "2025-01-01",
        "legs": [
            {"name": "dn1", "strike": 240.0, "qty": 1},
            {"name": "atm", "strike": 250.0, "qty": -4},
            {"name": "up1", "strike": 260.0, "qty": 1},
        ],
        "strike": 250.0,
        "expiry": "2026-10-02",
        "structure_width": 10.0,
        "structure_peak": 40.0,
        "entry_cost": 3.4512778,
        "spot": 249.8812,
        "fill": 0.5,
        "rel_spread": 0.0871,
        "exp_pnl_analog": -0.0026114,
        "win_analog": 0.4412,
        "ci_low": -0.01193117,
        "ci_high": 0.04217742,
        "n_analogs": 184,
        "analog_widened": 0,
        "analog_buckets": {"im_t1": "6-8", "mcap": "10-50B"},
        "exp_pnl_model": 0.01871,
        "win_model": 0.5133,
        "exp_pnl_sim": 0.018712244,
        "win_sim": 0.5142,
        "payoff": {"peak": 40.0, "breakeven_down": 243.1},
        "gate_score": 0.0611,
        "gate_threshold": 0.0508851760203979,
        "gate_pass": True,
        "flags": [],
        "chooser_score": 0.004211,
        "structure_params": {
            "width_moneyness": 0.05123456789012,
            "wing_multiple": 3,
            "steps": 1,
        },
        "structure_spec": {"factory": "twin_peak_5"},
        "detail": "",
    }


# --------------------------------------------------------------------------
# the seeds
# --------------------------------------------------------------------------


def seed(cause: str | None, record: dict) -> dict:
    """The record as a seeded defect leaves it at the defect's own stage."""
    out = copy.deepcopy(record)
    if cause == "forecast_suppressed":          # e845f3e
        out.update({key: None for key in FORECAST_BLOCK})
    elif cause == "analog_bootstrap_reseeded":  # b9aa1fd: only the interval moves
        out["ci_low"], out["ci_high"] = -0.01488201, 0.03911905
    elif cause == "replay_input_rounded":       # b33036c
        out = round_params(out)
    return out


def seeded_pair(cause: str | None):
    """``(record receipt, integrity receipt)`` for one pair carrying ``cause``.

    ``rounded_after_digest`` (`6b9d5cf`) lives on the WRITE path: the digest is
    taken over the record, then the bytes written are re-rounded. It is caught
    by comparing the stored digest with the digest of what was written — not by
    a record field, which every other cause would also move.
    """
    frozen = baseline()
    replayed = seed(cause, frozen)
    stored = content_hash(replayed)
    written = round_params(replayed) if cause == "rounded_after_digest" else replayed
    return (
        compare_records(frozen, replayed, comparison_kind="seeded_record"),
        compare_records({"payload_hash": stored}, {"payload_hash": content_hash(written)},
                        comparison_kind="seeded_integrity"),
    )


def by_kind(pair) -> dict[str, list[dict]]:
    record, integrity = pair
    return {"record": finding_dicts(record), "integrity": finding_dicts(integrity)}


# --------------------------------------------------------------------------
# each cause, and all of them in one pass
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cause", CAUSES)
def test_each_seeded_cause_alone_produces_exactly_its_specified_findings(cause):
    assert check_control(cause, by_kind(seeded_pair(cause)), kinds=TIER0_KINDS) == []


def test_an_unseeded_pair_is_clean_in_both_receipts():
    record, integrity = seeded_pair(None)
    assert record.verdict == AGREE and integrity.verdict == AGREE


def test_all_seeded_causes_in_one_pass_each_meet_their_spec():
    """§8's headline: every cause at once, one pass, each localized, nothing else.

    Each cause rides on its own pair — as it does over the real corpus — so the
    one pass's findings are attributable, and the clean pair proves the pass
    does not invent findings of its own.
    """
    pairs = {cause: seeded_pair(cause) for cause in CAUSES} | {"clean": seeded_pair(None)}
    receipts = [receipt for pair in pairs.values() for receipt in pair]
    one_pass = merge_receipts(receipts, comparison_kind="seeded_one_pass",
                              expected=len(receipts))
    assert one_pass.verdict == DIFFER
    for cause in CAUSES:
        assert check_control(cause, by_kind(pairs[cause]), kinds=TIER0_KINDS) == [], cause
    assert by_kind(pairs["clean"]) == {"record": [], "integrity": []}
    assert set(one_pass.stages_named()) == {"forecast", "analogs", "serialization"}
    for finding in one_pass.findings:
        assert finding.field_path and finding.first_differing_stage


def test_the_digest_control_is_moved_by_no_other_cause():
    """`6b9d5cf` must not be a consequence of the others.

    The first version seeded it as a `digest` field on the record, which moved
    with any record edit — and then asserted it was not downstream of them.
    """
    for cause in CAUSES:
        _, integrity = seeded_pair(cause)
        assert (integrity.verdict == DIFFER) == (cause == "rounded_after_digest"), cause


def test_rounding_twice_on_one_record_hides_the_second_rounding():
    """Why every control gets its own pair.

    Rounding is idempotent. A record already rounded before its digest
    (`b33036c`) re-rounded after it (`6b9d5cf`) writes the same bytes the digest
    names, so seeding both on one record would show one cause, not two.
    """
    rounded = round_params(baseline())
    assert content_hash(round_params(rounded)) == content_hash(rounded)


def test_every_seeded_pair_compares_exactly_its_own_leaves():
    """`28cf8b1` under seeding: the compared population is the records' leaves."""
    for cause in CAUSES:
        frozen = baseline()
        record, _ = seeded_pair(cause)
        leaves = set(flatten(frozen)) | set(flatten(seed(cause, frozen)))
        assert record.population.compared == len(leaves), cause


def test_a_suppressed_forecast_that_empties_the_simulation_is_one_localized_cause():
    """What `e845f3e` actually did: the forecast blanked, and the layers behind it.

    Four stages' fields move; one stage is named. That is the difference
    between "the forecast is broken" and four separate-looking red rows.
    """
    replayed = seed("forecast_suppressed", baseline()) | {
        "exp_pnl_model": None, "win_model": None, "gate_score": None,
        "chooser_score": None,
        # The narrative moves with it — the real engine's seeded run showed
        # this, and `detail` filed under serialization reported it as a
        # second, independent root.
        "detail": "TWIN-P5: entry rule fails: no forecast",
    }
    receipt = compare_records(baseline(), replayed)
    assert receipt.stages_named() == ("forecast",)
    assert {f.owning_stage for f in receipt.findings} >= {
        "forecast", "simulation", "gate", "chooser"}


# --------------------------------------------------------------------------
# the judgement is itself a check, so it has negative controls
# --------------------------------------------------------------------------


def test_a_control_landing_in_the_wrong_stage_is_reported():
    found = {"record": finding_dicts(compare_records(baseline(), {**baseline(), "entry_cost": 9.9})),
             "integrity": []}
    problems = check_control("forecast_suppressed", found, kinds=TIER0_KINDS)
    assert any("localized to" in p for p in problems)
    assert any("no finding at" in p for p in problems)


def test_an_undetected_control_is_reported():
    assert check_control("analog_bootstrap_reseeded", {"record": [], "integrity": []},
                         kinds=TIER0_KINDS) == [
        "record: no finding — the seeded defect went undetected"]


def test_a_control_that_leaks_into_a_receipt_it_should_leave_clean_is_reported():
    found = by_kind(seeded_pair("replay_input_rounded"))
    found["integrity"] = found["record"]
    assert any(p.startswith("integrity:")
               for p in check_control("replay_input_rounded", found, kinds=TIER0_KINDS))


def test_a_control_that_moves_a_field_its_defect_leaves_alone_is_reported():
    replayed = seed("analog_bootstrap_reseeded", baseline()) | {"n_analogs": 171}
    found = {"record": finding_dicts(compare_records(baseline(), replayed)), "integrity": []}
    assert any("moved fields" in p for p in check_control(
        "analog_bootstrap_reseeded", found, kinds=TIER0_KINDS))


def test_a_tier_one_receipt_kind_missing_from_a_tier_one_run_is_reported():
    found = by_kind(seeded_pair("rounded_after_digest"))
    assert any("no round_trip receipt" in p
               for p in check_control("rounded_after_digest", found))


# --------------------------------------------------------------------------
# the §11 controls
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,mutate,stage,paths",
    [
        ("timestamp", lambda r: {**r, "evidence_cutoff": "2026-09-29"},
         "resolve_context", {"evidence_cutoff"}),
        ("feature builder",
         lambda r: {**r, "model_inputs": {**r["model_inputs"], "or_implied": 6.9}},
         "features", {"model_inputs.or_implied"}),
        ("geometry",
         lambda r: {**r, "legs": [{**r["legs"][0], "strike": 237.5}, *r["legs"][1:]]},
         "geometry", {"legs[0].strike"}),
        ("model hash",
         lambda r: {**r, "model_versions": {**r["model_versions"], "size": "size_v1_3"}},
         "features", {"model_versions.size"}),
        ("dataset membership",
         lambda r: {**r, "n_analogs": 171,
                    "analog_buckets": {**r["analog_buckets"], "im_t1": "4-6"}},
         "analogs", {"n_analogs", "analog_buckets.im_t1"}),
    ],
)
def test_section_11_controls(label, mutate, stage, paths):
    """§11: corrupt a timestamp, a feature builder, a geometry, a model hash
    and a dataset membership, and prove the corresponding check fails."""
    receipt = compare_records(baseline(), mutate(baseline()))
    assert receipt.verdict == DIFFER, label
    assert {f.field_path for f in receipt.findings} == paths, label
    assert {f.first_differing_stage for f in receipt.findings} == {stage}, label


def test_a_cross_entity_break_is_caught_rather_than_silently_replayed():
    """data_model §4: a score must resolve every dependency to a retained object."""
    broken = {**baseline(), "forecast_model": "size_v1_3",
              "model_versions": {"gate": "gate_midfill_str_thru"}}
    receipt = compare_records(baseline(), broken)
    assert receipt.verdict == DIFFER
    assert {f.owning_stage for f in receipt.findings} == {"features", "forecast"}
    # The forecast field moved, but its stage's inputs already differed: the
    # finding is localized upstream, to the model-version block that changed.
    assert receipt.stages_named() == ("features",)
    assert any(f.kind == "missing_field" for f in receipt.findings)


# --------------------------------------------------------------------------
# the vacuous-pass controls
# --------------------------------------------------------------------------


def test_an_empty_population_never_reports_agreement():
    assert compare_records({}, {}).verdict == INCOMPARABLE
    assert merge_receipts([], comparison_kind="tier0", expected=12).verdict == INCOMPARABLE


def test_a_shrunken_corpus_is_incomparable_not_agreement():
    healthy = [compare_records(baseline(), baseline()) for _ in range(12)]
    assert merge_receipts(healthy, comparison_kind="tier0", expected=12).verdict == AGREE
    assert merge_receipts(healthy[:3], comparison_kind="tier0",
                          expected=12).verdict == INCOMPARABLE


def test_a_receipt_survives_a_json_round_trip():
    receipt = compare_records(baseline(), seed("forecast_suppressed", baseline()))
    text = json.dumps(receipt.payload(), sort_keys=True, default=str)
    assert json.loads(text)["verdict"] == DIFFER
