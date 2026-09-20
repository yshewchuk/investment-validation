"""R4-15 fix: the numeric negative control must report three distinct
states -- ``passed``, ``failed``, ``not_exercisable`` -- instead of
collapsing "never had anything to corrupt" onto the same False as "the
comparator saw a corruption and missed it".

Synthetic records only, built the same way
``tests/test_phase4_planted_defect_control.py`` builds them, driven straight
through the real ``_native_parity`` orchestration (no control-only path).
"""
from __future__ import annotations

from types import SimpleNamespace

from checks import phase4_real
from engine.v2.contracts import ScoreRecord


def _clean_record(**overrides):
    record = {
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
    record.update(overrides)
    return record


def _native_record(record):
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


def _stub_replay(records_by_fixture, natives_by_fixture=None):
    natives_by_fixture = natives_by_fixture or {}

    def verified_fn(pair, _root):
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


def _run(tmp_path, monkeypatch, record):
    pair = _pair("a", record)
    corpus = _corpus(tmp_path, pair)
    verified_fn, replayed_fn = _stub_replay({"a": record})
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", verified_fn)
    monkeypatch.setattr(phase4_real, "_replayed_member", replayed_fn)
    return phase4_real._native_parity(corpus)


def test_a_real_float_field_is_corrupted_and_noticed(tmp_path, monkeypatch):
    """simulation carries only real floats -- the ordinary case."""
    record = _clean_record()
    release, parity = _run(tmp_path, monkeypatch, record)

    entry = release["numeric_negative_controls"]["simulation"]
    assert entry["state"] == phase4_real._CONTROL_PASSED
    assert "exp_pnl_sim" in entry["reason"]
    assert "noticed" in entry["reason"]
    assert parity["planted_defect"]["controls"]["simulation"] is True
    assert release["numeric_coverage"]["simulation"] == 1


def test_a_bool_only_dimension_is_flipped_and_noticed(tmp_path, monkeypatch):
    """verdicts with gate_score/gate_threshold None and gate_pass a real
    bool -- exactly the shape that left the control permanently
    unexercised before this fix. The comparator is real (unmocked): this
    proves whether a flipped ``gate_pass`` is actually noticed."""
    record = _clean_record(gate_score=None, gate_threshold=None, gate_pass=True)
    release, parity = _run(tmp_path, monkeypatch, record)

    entry = release["numeric_negative_controls"]["verdicts"]
    assert entry["state"] == phase4_real._CONTROL_PASSED, entry["reason"]
    assert "gate_pass" in entry["reason"]
    assert "flipped bool" in entry["reason"]
    assert "noticed" in entry["reason"]
    assert parity["planted_defect"]["controls"]["verdicts"] is True
    assert release["numeric_coverage"]["verdicts"] == 1


def test_a_dimension_with_every_field_none_is_not_exercisable(tmp_path, monkeypatch):
    """analogs with every field None: nothing for the control to touch.
    Must be reported distinctly, never as a failed control."""
    record = _clean_record(ci_low=None, ci_high=None, n_analogs=None)
    release, parity = _run(tmp_path, monkeypatch, record)

    entry = release["numeric_negative_controls"]["analogs"]
    assert entry["state"] == phase4_real._CONTROL_NOT_EXERCISABLE
    assert entry["state"] != phase4_real._CONTROL_FAILED
    assert "analogs" in entry["reason"]
    assert "corruptible" in entry["reason"]
    # Not exercisable must not count as covered, and must not unlock the
    # completion gate the way a real pass does.
    assert release["numeric_coverage"]["analogs"] == 0
    assert parity["planted_defect"]["controls"]["analogs"] is False
    assert parity["planted_defect"]["detected"] is False


def test_a_blind_comparator_makes_the_flipped_bool_control_fail(tmp_path, monkeypatch):
    """The control of the control: a comparator that agrees on everything
    must show up as FAILED for verdicts (a real corruption went
    unnoticed), not as NOT_EXERCISABLE -- the control DID have a bool to
    flip; it is the comparator that is blind."""
    record = _clean_record(gate_score=None, gate_threshold=None, gate_pass=True)

    def _blind_compare_records(_expected, _actual, **_kwargs):
        return SimpleNamespace(verdict=phase4_real.AGREE, findings=())

    monkeypatch.setattr(phase4_real, "compare_records", _blind_compare_records)
    release, parity = _run(tmp_path, monkeypatch, record)

    entry = release["numeric_negative_controls"]["verdicts"]
    assert entry["state"] == phase4_real._CONTROL_FAILED
    assert "did NOT notice" in entry["reason"]
    assert parity["planted_defect"]["controls"]["verdicts"] is False
    assert parity["planted_defect"]["detected"] is False
    # Coverage still counts the row: the control HAD something to
    # corrupt here, it is the comparator that failed to see it.
    assert release["numeric_coverage"]["verdicts"] == 1
