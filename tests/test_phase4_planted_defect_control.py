"""R4-15: the planted-defect control on the real saved-release comparator.

``checks/phase4_real.py::_native_parity`` is the record-by-record parity
comparison used for the Phase 4 saved-release evidence. It carries its own
inline self-test: for every compared pair it plants a defect on the NATIVE
side and re-runs the SAME comparator functions used for the real comparison
(``_record_checks``, ``_compare_numeric_outputs``) to prove each defect type
is actually catchable. ``native_stage_comparator_planted_defect`` in
``build_evidence``'s completion controls is exactly
``native_parity["planted_defect"]["detected"]``.

This file proves that self-test on a synthetic traced corpus, entirely
through the real comparator code (no control-only comparison path):

* a clean pair is complete and every planted-defect control fires True;
* a genuine native-side defect (dropped leg, wrong flag, extra key, flipped
  null mask) actually present in the scored record breaks the real
  comparison on that exact dimension;
* a comparator that ignores differences is itself caught: the self-test's
  own controls go False rather than rubber-stamping a blind pass (the
  control of the control).

Only ``_verified_trace_bundle``/``_replayed_member`` are stubbed here (the
hash-verification/replay boundary, covered on its own in
``tests/test_phase4_trace_verifier.py``); every comparator function this
file exercises is the real, unmodified one.
"""
from __future__ import annotations

from dataclasses import replace as dc_replace
from types import SimpleNamespace

import pytest

from checks import phase4_real
from engine.v2.contracts import ScoreRecord


def _clean_record():
    return {
        "strategy": "STR-THRU",
        "driver_prediction": 5.0,
        "forecast_abs_move": 5.0,
        "exp_pnl_sim": 0.2,
        "exp_pnl_model": 0.15,
        "win_sim": 0.6,
        "win_model": 0.55,
        "gate_score": 0.8,
        "gate_threshold": 0.5,
        "gate_pass": True,
        "ci_low": -0.01,
        "ci_high": 0.05,
        "n_analogs": 10,
        "spot": 100.0,
        "entry_cost": 5.0,
        "structure_width": 10.0,
        "driver_name": "abs_move",
        "implied_move": 4.0,
        "payoff": {"intercept": 0.0, "slope": 0.03},
        "flags": (),
        "model_inputs": {"x": 1.0, "y": None},
        "legs": (
            {"name": "call", "right": "C", "side": "long", "quantity": 1.0,
             "strike": 100.0, "expiry": "2026-09-18", "fill": 1.5,
             "cash_flow": -150.0},
        ),
        "entry_date": "2026-09-16", "exit_date": "2026-09-17",
        "quote_date": "2026-09-16",
    }


def _native_record(record):
    """A real ``ScoreRecord`` that agrees with ``_clean_record()`` on every
    comparison dimension ``_record_checks``/``_compare_numeric_outputs``
    check -- built the same way ``application.score_frozen`` builds one, so
    ``dataclasses.replace`` (what the production defect-planting code uses)
    works on it.
    """
    return ScoreRecord(
        score_id="native-score-1",
        canonical_request={},
        resolved_request=dict(record),
        event_ref={},
        clock_id="clock-1",
        snapshot_ref="snapshot-1",
        dependency_hash="dep-1",
        model_artifact_ids=(),
        selected_contracts=(),
        legs=tuple(record.get("legs") or ()),
        entry_exit_plan={
            "entry_date": record.get("entry_date"),
            "exit_date": record.get("exit_date"),
        },
        quote_provenance={"quote_date": record.get("quote_date")},
        forecasts={
            "driver_prediction": record.get("driver_prediction"),
            "forecast_abs_move": record.get("forecast_abs_move"),
            "exp_pnl_sim": record.get("exp_pnl_sim"),
        },
        uncertainty={},
        residual_state_ref=None,
        analog_state_ref=None,
        payoff_state_ref=None,
        feature_values={},
        null_masks={
            key: value is None
            for key, value in (record.get("model_inputs") or {}).items()
        },
        feature_lineage_refs=(),
        gate_terms={
            "gate_score": record.get("gate_score"),
            "gate_threshold": record.get("gate_threshold"),
            "gate_pass": record.get("gate_pass"),
        },
        chooser_candidates=(),
        chooser_selection=None,
        financial_diagnostics=phase4_real._expected_financial_diagnostics(record),
        requested_payoff_views=(),
        validation_status="scored",
        reason_codes=tuple(record.get("flags") or ()),
        warnings=(),
        evidence_refs=(),
    )


def _pair(fixture_id, record, kind="score_result"):
    return {
        "fixture_id": fixture_id,
        "payload_hash": f"hash-{fixture_id}",
        "payload": {"record_kind": kind, "record": record},
    }


def _corpus(tmp_path, *pairs):
    return SimpleNamespace(
        root=tmp_path,
        index={"pairs": {
            p["fixture_id"]: {"record_kind": p["payload"]["record_kind"]}
            for p in pairs
        }},
        ordered_ids=[p["fixture_id"] for p in pairs],
        pairs={p["fixture_id"]: p for p in pairs},
    )


def _make_stub_replay(records_by_fixture, natives_by_fixture=None):
    """Stub only the hash-verification/replay boundary. Every comparator
    function downstream of it (``_record_checks``, ``_compare_dimension``,
    ``compare_records``, ...) stays real."""
    natives_by_fixture = natives_by_fixture or {}

    def verified_fn(pair, _root):
        if pair["payload"]["record_kind"] != "score_result":
            raise phase4_real._TraceError("input_trace: missing")
        return {
            "same_input_receipt": "same", "trace_hash": "trace",
            "frozen_replay": None, "fixture_id": pair["fixture_id"],
        }

    def replayed_fn(verified):
        fixture_id = verified["fixture_id"]
        record = records_by_fixture[fixture_id]
        native = natives_by_fixture.get(fixture_id) or _native_record(record)
        receipts = tuple({"stage": s} for s in phase4_real._REQUIRED_TRACE_STAGES)
        return native, receipts, ()

    return verified_fn, replayed_fn


def test_clean_corpus_is_complete_and_self_test_detects_every_planted_defect(
        tmp_path, monkeypatch):
    record = _clean_record()
    pair = _pair("a", record)
    corpus = _corpus(tmp_path, pair)
    verified_fn, replayed_fn = _make_stub_replay({"a": record})
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", verified_fn)
    monkeypatch.setattr(phase4_real, "_replayed_member", replayed_fn)

    release, parity = phase4_real._native_parity(corpus)

    assert release["complete"] is True
    assert parity["complete"] is True
    assert parity["planted_defect"]["detected"] is True
    for name, ok in parity["planted_defect"]["controls"].items():
        assert ok is True, name


@pytest.mark.parametrize("dimension", ["flags", "contracts", "keys", "null_masks"])
def test_each_native_side_defect_actually_present_breaks_the_real_comparison(
        tmp_path, monkeypatch, dimension):
    record = _clean_record()
    clean_native = _native_record(record)
    defective_native = phase4_real._plant_structural_defect(
        record, clean_native, dimension)
    assert defective_native is not None

    pair = _pair("a", record)
    corpus = _corpus(tmp_path, pair)
    verified_fn, replayed_fn = _make_stub_replay(
        {"a": record}, natives_by_fixture={"a": defective_native})
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", verified_fn)
    monkeypatch.setattr(phase4_real, "_replayed_member", replayed_fn)

    release, parity = phase4_real._native_parity(corpus)

    assert parity["dimension_agreement"][dimension] is False
    assert release["complete"] is False
    assert parity["complete"] is False


def test_a_comparator_that_ignores_differences_is_itself_caught(
        tmp_path, monkeypatch):
    """The control of the control: a comparator rigged to agree on
    everything must not be able to make ``planted_defect.detected`` True."""
    record = _clean_record()
    pair = _pair("a", record)
    corpus = _corpus(tmp_path, pair)
    verified_fn, replayed_fn = _make_stub_replay({"a": record})
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", verified_fn)
    monkeypatch.setattr(phase4_real, "_replayed_member", replayed_fn)

    def _blind_compare_records(_expected, _actual, **_kwargs):
        return SimpleNamespace(verdict=phase4_real.AGREE, findings=())

    def _blind_record_checks(_record, _native):
        dims = ("keys", "contracts", "verdicts", "flags", "null_masks",
                "forecasts", "simulation", "financial_diagnostics", "analogs")
        return ({name: True for name in dims}, {})

    monkeypatch.setattr(phase4_real, "compare_records", _blind_compare_records)
    monkeypatch.setattr(phase4_real, "_record_checks", _blind_record_checks)

    release, parity = phase4_real._native_parity(corpus)

    # The blind comparator agrees on everything, including the corruption
    # the self-test tries to plant -- it would wrongly certify parity...
    assert release["population"]["agreed"] == release["population"]["compared"]
    # ...but the planted-defect self-test can no longer prove a single
    # corruption disagrees, so it refuses to certify the control itself.
    assert parity["planted_defect"]["detected"] is False
    assert not all(parity["planted_defect"]["controls"].values())


def test_plant_structural_defect_returns_none_when_nothing_to_corrupt():
    record = _clean_record()
    native = _native_record(record)
    no_legs_native = dc_replace(native, legs=())
    assert phase4_real._plant_structural_defect(
        record, no_legs_native, "contracts") is None
    no_masks_native = dc_replace(native, null_masks={})
    assert phase4_real._plant_structural_defect(
        record, no_masks_native, "null_masks") is None


def test_plant_structural_defect_rejects_unknown_dimension():
    record = _clean_record()
    native = _native_record(record)
    with pytest.raises(ValueError):
        phase4_real._plant_structural_defect(record, native, "mystery")