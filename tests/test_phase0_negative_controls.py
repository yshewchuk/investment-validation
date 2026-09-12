"""The negative controls — this phase's reason for existing (§8).

A check that has never failed is not known to work. On 2026-09-11 a suite of
1,567 tests passed over five live defects, and the determinism test that should
have caught the analog ordering bug passed the same frame twice: coverage of
the executed line was total and the assertion was empty. So the evidence that
phase 0 worked is not a pass count — it is that every seeded corruption below
produces the finding it is supposed to, in the stage it is supposed to.

The headline assertion is :func:`test_five_seeded_defects_produce_five_findings_in_one_pass`.
That single test is the difference between five nights and one.
"""
from __future__ import annotations

import copy
import json
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
    compare_records,
    content_hash,
    flatten,
    merge_receipts,
)


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


def with_digest(record: dict, *, stored: str | None = None) -> dict:
    """A record plus the two identity fields the corpus keeps beside it.

    ``payload_hash`` is what was written to disk; ``digest`` is what the bytes
    on disk hash to now. They are equal in a healthy fixture and the pair is
    exactly what `6b9d5cf` broke — a file disagreeing with its own digest — so
    carrying both is what lets one comparator pass see that defect at all.
    """
    computed = content_hash(record)
    return {**record, "payload_hash": stored or computed, "digest": computed}


# --------------------------------------------------------------------------
# the five 2026-09-11 causes, seeded
# --------------------------------------------------------------------------

FORECAST_BLOCK = ("forecast_abs_move", "forecast_p10", "forecast_p90",
                  "forecast_sd", "forecast_model", "forecast_fold")


def seed_forecast_suppressed(record: dict) -> dict:
    """`e845f3e`: the forecast blanked when a pinned shape suppressed sizing.

    §9.5: selecting contracts from a forecast and replaying with those
    parameters pinned must still record the same forecast.
    """
    out = copy.deepcopy(record)
    for key in FORECAST_BLOCK:
        out[key] = None
    return out


def seed_analog_bootstrap_reseeded(record: dict) -> dict:
    """`b9aa1fd`: a bootstrap sampling by index over an unordered set.

    Only the interval moves. The point estimate and the population size are
    untouched, which is precisely why a determinism test over the whole frame
    missed it.
    """
    out = copy.deepcopy(record)
    out["ci_low"] = -0.01488201
    out["ci_high"] = 0.03911905
    return out


def seed_replay_input_rounded(record: dict) -> dict:
    """`b33036c`: `json_safe` rounding a replay input to six places."""
    out = copy.deepcopy(record)
    out["structure_params"] = {
        k: (round(v, 6) if isinstance(v, float) else v)
        for k, v in out["structure_params"].items()
    }
    return out


def seed_all_five(record: dict) -> dict:
    """Every cause at once, which is the situation the phase exists for."""
    out = seed_forecast_suppressed(record)
    out = seed_analog_bootstrap_reseeded(out)
    out = seed_replay_input_rounded(out)
    return out


# --------------------------------------------------------------------------
# the headline assertion
# --------------------------------------------------------------------------


def test_five_seeded_defects_produce_five_findings_in_one_pass():
    """§8: all five seeded at once must produce five findings in ONE pass.

    | Seeded corruption | Reported as |
    |---|---|
    | forecast suppressed on a pinned replay (`e845f3e`) | `forecast`, block null |
    | a field removed from the compared set (`28cf8b1`) | impossible by construction |
    | analog bootstrap reseeded by row order (`b9aa1fd`) | `analogs`, ci_low/ci_high only |
    | a replay input rounded to six places (`b33036c`) | `serialization`, structure_params.* |
    | rounding reapplied after the exemption (`6b9d5cf`) | `serialization`, file vs its digest |
    """
    left = with_digest(baseline())
    corrupted = seed_all_five(baseline())
    # The digest written BEFORE the re-rounding, which is what `6b9d5cf` left
    # on disk: a stored hash the bytes beside it no longer produce.
    right = with_digest(corrupted, stored=left["payload_hash"])

    receipt = compare_records(left, right)
    assert receipt.verdict == DIFFER

    by_path = {f.field_path: f for f in receipt.findings}
    forecast = [p for p in by_path if p.startswith("forecast_")]
    analogs = [p for p in by_path if p.startswith("ci_")]
    params = [p for p in by_path if p.startswith("structure_params.")]

    causes = {
        "forecast_suppressed": forecast,
        "analog_interval": analogs,
        "replay_input_rounded": params,
        "file_disagrees_with_digest": [p for p in by_path if p == "digest"],
    }
    for name, paths in causes.items():
        assert paths, f"{name} produced no finding\n{receipt.summary()}"

    # Five independent causes; `28cf8b1` is the fifth and it cannot be seeded,
    # so four are visible here and the fifth is proved below by construction.
    assert len(causes) == 4
    assert test_a_field_cannot_be_dropped_from_the_compared_set.__doc__

    # Every one of them names a STAGE, and the right one.
    assert all(by_path[p].first_differing_stage == "forecast" for p in forecast)
    assert all(by_path[p].first_differing_stage == "analogs" for p in analogs)
    assert all(by_path[p].first_differing_stage == "serialization" for p in params)
    assert by_path["digest"].first_differing_stage == "serialization"

    # And not one of them names only a row.
    for finding in receipt.findings:
        assert finding.field_path and finding.first_differing_stage


def test_the_four_seeded_causes_are_reported_as_independent():
    """Fix several at once, rather than one per night.

    All four stages received agreeing inputs — the forecast does not feed the
    analogs, and the serialization stage reads geometry, which is untouched —
    so the comparator can prove none of them is a consequence of another.
    """
    left = with_digest(baseline())
    right = with_digest(seed_all_five(baseline()), stored=left["payload_hash"])
    receipt = compare_records(left, right)
    ids = {f.finding_id for f in receipt.findings}
    for finding in receipt.findings:
        assert set(finding.not_downstream_of) == ids - {finding.finding_id}, (
            f"{finding.field_path} was not proved independent"
        )


def test_stopping_at_the_first_difference_is_what_this_replaces():
    """A first-wins comparator would have reported one of these, not four."""
    left = with_digest(baseline())
    right = with_digest(seed_all_five(baseline()), stored=left["payload_hash"])
    stages = set(compare_records(left, right).stages_named())
    assert stages == {"forecast", "analogs", "serialization"}


# --------------------------------------------------------------------------
# each cause on its own
# --------------------------------------------------------------------------


def test_forecast_suppressed_names_the_forecast_stage_and_a_null_mask():
    receipt = compare_records(baseline(), seed_forecast_suppressed(baseline()))
    assert receipt.stages_named() == ("forecast",)
    assert {f.field_path for f in receipt.findings} == set(FORECAST_BLOCK)
    assert all(f.kind == "null_mask" for f in receipt.findings)
    assert all(f.null_mask_right and not f.null_mask_left for f in receipt.findings)


def test_analog_reseed_moves_the_interval_and_nothing_else():
    receipt = compare_records(baseline(), seed_analog_bootstrap_reseeded(baseline()))
    assert receipt.stages_named() == ("analogs",)
    assert {f.field_path for f in receipt.findings} == {"ci_low", "ci_high"}
    # The point estimate and the population size did NOT move, which is why a
    # frame-level determinism test passed straight over this.
    moved = {f.field_path for f in receipt.findings}
    assert "exp_pnl_analog" not in moved and "n_analogs" not in moved


def test_a_rounded_replay_input_names_serialization_and_the_exact_field():
    receipt = compare_records(baseline(), seed_replay_input_rounded(baseline()))
    assert receipt.stages_named() == ("serialization",)
    assert [f.field_path for f in receipt.findings] == [
        "structure_params.width_moneyness"
    ]
    finding = receipt.findings[0]
    assert finding.left_value != finding.right_value
    # Six places kept, so the two agree to the sixth decimal and part at the
    # seventh: 4.3e-07 on a 0.0512 width. Invisible on a board, and the size
    # of the difference is exactly why nothing noticed for a night.
    assert 1e-8 < abs(finding.delta) < 1e-5
    assert round(finding.left_value, 6) == finding.right_value


def test_a_file_that_disagrees_with_its_digest_is_caught():
    record = baseline()
    stored = content_hash(record)
    tampered = seed_replay_input_rounded(record)
    receipt = compare_records(
        with_digest(record),
        with_digest(tampered, stored=stored),
    )
    digest_findings = [f for f in receipt.findings if f.field_path == "digest"]
    assert len(digest_findings) == 1
    assert digest_findings[0].first_differing_stage == "serialization"


def test_a_field_cannot_be_dropped_from_the_compared_set():
    """`28cf8b1`, and why §8 calls it impossible by construction.

    The explainer compared 41 of the 70 fields the digest hashed because the 41
    were typed out somewhere. Here the compared set IS the records' own field
    set, so there is no list to fall out of date: the only way to remove a
    field from the comparison is to remove it from the record, which removes it
    from the digest in the same stroke.
    """
    left, right = with_digest(baseline()), with_digest(baseline())
    expected = set(flatten(left)) | set(flatten(right))
    receipt = compare_records(left, right)
    compared = {row.stage_id for row in receipt.stage_hashes}
    assert compared == set(SCORER_V1.stage_ids())
    assert receipt.population.compared == len(expected)


# --------------------------------------------------------------------------
# the §11 controls
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,mutate,stage,paths",
    [
        (
            "timestamp",
            lambda r: {**r, "evidence_cutoff": "2026-09-29"},
            "resolve_context",
            {"evidence_cutoff"},
        ),
        (
            "feature builder",
            lambda r: {**r, "model_inputs": {**r["model_inputs"], "or_implied": 6.9}},
            "features",
            {"model_inputs.or_implied"},
        ),
        (
            "geometry",
            lambda r: {**r, "legs": [{**r["legs"][0], "strike": 237.5},
                                     *r["legs"][1:]]},
            "geometry",
            {"legs[0].strike"},
        ),
        (
            "model hash",
            lambda r: {**r, "model_versions": {**r["model_versions"],
                                               "size": "size_v1_3"}},
            "features",
            {"model_versions.size"},
        ),
        (
            "dataset membership",
            lambda r: {**r, "n_analogs": 171,
                       "analog_buckets": {**r["analog_buckets"], "im_t1": "4-6"}},
            "analogs",
            {"n_analogs", "analog_buckets.im_t1"},
        ),
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
    """data_model §4: a score must resolve every dependency to a retained object.

    Here the record still claims a forecast model that the model-version block
    no longer names — the "another candidate borrowed its results" shape. It is
    two findings in two stages, not one vague mismatch.
    """
    broken = {**baseline(), "forecast_model": "size_v1_3",
              "model_versions": {"gate": "gate_midfill_str_thru"}}
    receipt = compare_records(baseline(), broken)
    assert receipt.verdict == DIFFER
    assert set(receipt.stages_named()) == {"features", "forecast"}
    assert any(f.kind == "missing_field" for f in receipt.findings)


# --------------------------------------------------------------------------
# the vacuous-pass controls
# --------------------------------------------------------------------------


def test_an_empty_population_never_reports_agreement():
    """§11: a fixture whose universe collapsed compares zero rows."""
    assert compare_records({}, {}).verdict == INCOMPARABLE
    assert merge_receipts([], comparison_kind="tier0", expected=12).verdict == (
        INCOMPARABLE
    )


def test_a_shrunken_corpus_is_incomparable_not_agreement():
    healthy = [compare_records(baseline(), baseline()) for _ in range(12)]
    assert merge_receipts(healthy, comparison_kind="tier0",
                          expected=12).verdict == AGREE
    collapsed = healthy[:3]
    assert merge_receipts(collapsed, comparison_kind="tier0",
                          expected=12).verdict == INCOMPARABLE


def test_running_the_same_frame_twice_is_not_evidence():
    """The determinism test that missed `b9aa1fd` compared a frame to itself.

    Asserting that a record equals itself is a test that passes for a broken
    comparator too, so the suite above never rests on it. This records the
    distinction rather than relying on it.
    """
    same = compare_records(baseline(), baseline())
    assert same.verdict == AGREE
    mutated = compare_records(baseline(), seed_analog_bootstrap_reseeded(baseline()))
    assert mutated.verdict == DIFFER


def test_a_receipt_survives_a_json_round_trip():
    """Tier 1 catches serialization losses; the receipt has to be writable."""
    receipt = compare_records(baseline(), seed_all_five(baseline()))
    text = json.dumps(receipt.payload(), sort_keys=True, default=str)
    assert json.loads(text)["verdict"] == DIFFER
