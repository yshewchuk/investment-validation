"""The tier-1 replay harness, without a scorer.

A full replay builds a real Scorer and takes minutes, so it runs as the gate's
evidence, not in the suite. What CAN be proved without it is proved here: the
request survives its own serialization, the seeded defects fire only for the
fixture they are armed for and leave the engine untouched afterwards, the
serialization seeds produce exactly the receipts their controls specify, and
the controls are judged by the shared spec.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.replay_identity import SEEDED_CONTROLS, check_control, pick_seed_targets  # noqa: E402
from checks.tier0_corpus import finding_dicts  # noqa: E402
from engine.v2.diagnosis import AGREE, DIFFER, INCOMPARABLE, merge_receipts  # noqa: E402
from tools import replay_tier1 as rt  # noqa: E402


@pytest.fixture(scope="module")
def cap():
    return rt._load_capture_module()


# --------------------------------------------------------------------------
# the request round trip
# --------------------------------------------------------------------------


def test_a_request_survives_its_own_serialization(cap):
    from engine import score

    request = score.ScoreRequest(
        ticker="MTN", strategy="TWIN-P5", as_of=pd.Timestamp("2026-09-11"),
        event_date=pd.Timestamp("2026-09-28"), strike=250.0,
        expiry=pd.Timestamp("2026-10-02"), fill=score.FillModel(alpha=0.5),
        variant=None, session="AMC", decision_offset=2, quote_max_age_sessions=5,
        chain_as_of=pd.Timestamp("2026-09-11"),
        structure_params={"width_moneyness": 0.05123456789012},
    )
    data = cap.request_to_dict(request)
    rebuilt = cap.request_from_dict(json.loads(json.dumps(data)))
    assert rebuilt.key() == request.key()
    assert cap.request_to_dict(rebuilt) == data


# --------------------------------------------------------------------------
# the engine seeds
# --------------------------------------------------------------------------


def _fake_engine():
    score_mod = types.SimpleNamespace(Scorer=type("Scorer", (), {
        "_size_from_forecast":
            lambda self, request, result, structure, *, size=True: ("recorded", size),
    }))
    analogs_mod = types.ModuleType("fake_analogs")
    analogs_mod.np = np

    def _summarize(self, matched, *args, request_key=None, **kwargs):
        return [float(x) for x in analogs_mod.np.sort(matched["ret"].to_numpy())]

    analogs_mod.AnalogMatcher = type("AnalogMatcher", (), {"_summarize": _summarize})
    return score_mod, analogs_mod


def test_engine_seeds_fire_only_for_the_armed_fixture_and_are_reverted():
    score_mod, analogs_mod = _fake_engine()
    real_size = score_mod.Scorer._size_from_forecast
    real_summarize = analogs_mod.AnalogMatcher._summarize
    seeds = rt.Seeds({"forecast_suppressed": "fid-a", "analog_bootstrap_reseeded": "fid-b"})
    matched = pd.DataFrame({"ret": [0.3, 0.1, 0.2]})

    with rt.seeded_engine(seeds, score_mod, analogs_mod, np):
        scorer, matcher = score_mod.Scorer(), analogs_mod.AnalogMatcher()
        assert scorer._size_from_forecast("req", "res", "st", size=False) == ("recorded", False)
        assert matcher._summarize(matched, request_key="k") == [0.1, 0.2, 0.3]

        seeds.active = "fid-a"
        # e845f3e: a pinned request (size=False) records nothing...
        assert scorer._size_from_forecast("req", "res", "st", size=False) == ("req", "st")
        # ...and an unpinned one is untouched.
        assert scorer._size_from_forecast("req", "res", "st", size=True) == ("recorded", True)
        assert matcher._summarize(matched, request_key="k") == [0.1, 0.2, 0.3]

        seeds.active = "fid-b"
        # b9aa1fd: rows in reversed arrival order, and no sort.
        assert matcher._summarize(matched, request_key="k") == [0.2, 0.1, 0.3]
        assert analogs_mod.np is np

    assert score_mod.Scorer._size_from_forecast is real_size
    assert analogs_mod.AnalogMatcher._summarize is real_summarize


# --------------------------------------------------------------------------
# the serialization seeds
# --------------------------------------------------------------------------


def _record() -> dict:
    return {"strategy": "TWIN-P5", "entry_cost": 3.45,
            "structure_params": {"width_moneyness": 0.05123456789012, "steps": 1}}


def _serialize(tmp_path: Path, armed: dict[str, str]):
    seeds = rt.Seeds(armed)
    seeds.active = "fid"
    ctx = rt.Context(None, None, None, None, None, pd, tmp_path, seeds)
    return rt._serialize(ctx, "fid", _record())


def test_an_unarmed_write_round_trips_cleanly(tmp_path):
    record, round_trip, integrity = _serialize(tmp_path, {})
    assert record == _record()
    assert round_trip.verdict == AGREE and integrity.verdict == AGREE
    assert not list(tmp_path.iterdir())


def test_rounding_before_the_digest_changes_the_record_but_not_the_file(tmp_path):
    record, round_trip, integrity = _serialize(tmp_path, {"replay_input_rounded": "fid"})
    assert record["structure_params"]["width_moneyness"] == 0.051235
    assert round_trip.verdict == AGREE and integrity.verdict == AGREE


def test_rounding_after_the_digest_is_caught_by_the_write_and_read_back(tmp_path):
    record, round_trip, integrity = _serialize(tmp_path, {"rounded_after_digest": "fid"})
    assert record == _record()
    found = {"round_trip": finding_dicts(round_trip), "integrity": finding_dicts(integrity),
             "record": []}
    assert check_control("rounded_after_digest", found) == []


# --------------------------------------------------------------------------
# judging the controls, and the population
# --------------------------------------------------------------------------


def _clean_result(receipt) -> dict:
    return {"record": receipt, "round_trip": receipt, "integrity": receipt,
            "frame_rows": [], "field_set_ok": True}


def test_untargeted_findings_are_collected_and_controls_without_a_target_fail():
    from engine.v2.diagnosis import compare_records

    clean = compare_records({"a": 1}, {"a": 1})
    moved = compare_records({"a": 1}, {"a": 2})
    results = {"000": _clean_result(clean), "001": _clean_result(clean) | {"record": moved}}
    controls, untargeted = rt._controls(results, targets={}, missing=list(SEEDED_CONTROLS))
    assert all(c["problems"] == ["no frozen pair can carry this control"]
               for c in controls.values())
    assert [f["fixture_id"] for f in untargeted] == ["001"]


def test_seed_targets_are_distinct_and_the_forecast_target_is_pinned():
    def pair(record: dict, request: dict | None = None) -> dict:
        return {"payload": {"record_kind": "score_result", "record": record,
                            "request": request or {}}}

    pairs = {
        "000": pair({"forecast_abs_move": 5.2, "ci_low": -0.1, "ci_high": 0.1,
                     "structure_params": {"w": 0.0512345678}}),
        "001": pair({"forecast_abs_move": 5.2, "structure_params": {"w": 0.0512345678}},
                    {"structure_params": {"w": 0.0512345678}}),
        "002": pair({"structure_params": {"w": 0.0612345678}}),
        "003": pair({"structure_params": {"w": 0.0712345678}}),
    }
    targets, missing = pick_seed_targets(pairs)
    assert missing == []
    assert targets["forecast_suppressed"] == "001"
    assert len(set(targets.values())) == len(targets)


def test_an_unreplayable_pair_makes_the_population_short():
    from engine.v2.diagnosis import compare_records

    replayed = [compare_records({"a": 1}, {"a": 1}) for _ in range(3)]
    merged = merge_receipts(replayed + [rt._unreplayable("002", "mystery")],
                            comparison_kind="tier1_real_replay", tier=1, expected=6)
    assert merged.verdict == INCOMPARABLE


def test_a_dyn_sv_tie_is_not_a_required_axis(cap):
    """Decision 2026-09-12: no real tie exists to capture; the rule is tested instead."""
    axes = cap.required_axes()
    assert "dyn_sv:tie" not in axes
    assert {"dyn_sv:full_menu", "dyn_sv:partial_menu", "dyn_sv:fallback"} <= set(axes)


def test_seeded_and_compatibility_receipts_are_written_apart():
    assert rt.receipt_path({"corpus_version": "v1", "seeded": False}).name == "v1.json"
    assert rt.receipt_path({"corpus_version": "v1", "seeded": True}).name == "v1.seeded.json"
    assert DIFFER != AGREE
