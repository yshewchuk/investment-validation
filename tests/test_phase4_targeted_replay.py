"""Focused tests for the targeted strict Phase 4 replay diagnostic.

Three things these prove, on small synthetic fixtures (never the real corpus):

* **selection** — the named fixtures are the ones replayed, the repository
  loader is entered exactly once, and no acceptance battery is run;
* **shared-reference hydration/validation** — a pair that embeds a ``$shared``
  document is verified through ``checks.tier0_corpus.load`` hydration (a raw
  JSON decode would fail the digest, the exact bug that motivated the tool),
  and the trace/payload hashes are verified through repository code;
* **discrepancy / incomparability reporting** — value-free per-row checks,
  numeric finding field NAMES, key/flag/null-mask differences, and typed
  incomparability reasons — and that the diagnostic never claims sign-off.

Stub shapes follow ``tests/test_phase4_real_row_evidence.py`` (only
``_verified_trace_bundle``/``_replayed_member`` are stubbed; the real
``_record_checks`` runs). The real-trace pair is built by the same
``package_strict_trace`` recipe as ``tests/test_checks_phase5_phase4_replay.py``.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from checks import phase4_real
from checks import tier0_corpus as t0
from engine.v2.contracts import ScoreRecord
from engine.v2.foundation import content_hash, to_document
from engine.v2.scoring.stages import NativeScoreInputs, receipt
from tools import phase4_targeted_replay as targeted


# ---------------------------------------------------------------------------
# value-free per-row tests: helpers mirroring test_phase4_real_row_evidence
# ---------------------------------------------------------------------------


class _Sink:
    """A thread-safe line sink (the progress timer writes off-thread)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._chunks = []

    def write(self, text):
        with self._lock:
            self._chunks.append(text)

    def flush(self):
        pass

    @property
    def text(self):
        with self._lock:
            return "".join(self._chunks)


def _clean_record(**overrides):
    record = {
        "strategy": "STR-THRU", "driver_prediction": 5.0, "forecast_abs_move": 5.0,
        "exp_pnl_sim": 0.2, "exp_pnl_model": 0.15, "win_sim": 0.6, "win_model": 0.55,
        "gate_score": 0.8, "gate_threshold": 0.5, "gate_pass": True,
        "ci_low": -0.01, "ci_high": 0.05, "n_analogs": 10, "spot": 100.0,
        "entry_cost": 5.0, "structure_width": 10.0, "driver_name": "abs_move",
        "implied_move": 4.0, "payoff": {"intercept": 0.0, "slope": 0.03},
        "flags": (), "model_inputs": {"x": 1.0, "y": None},
        "legs": ({"name": "call", "right": "C", "side": "long", "quantity": 1.0,
                  "strike": 100.0, "expiry": "2026-09-18", "fill": 1.5,
                  "cash_flow": -150.0},),
        "entry_date": "2026-09-16", "exit_date": "2026-09-17", "quote_date": "2026-09-16",
    }
    record.update(overrides)
    return record


def _native_record(record):
    return ScoreRecord(
        score_id="native-score-1", canonical_request={}, resolved_request=dict(record),
        event_ref={}, clock_id="clock-1", snapshot_ref="snapshot-1",
        dependency_hash="dep-1", model_artifact_ids=(), selected_contracts=(),
        legs=tuple(record.get("legs") or ()),
        entry_exit_plan={"entry_date": record.get("entry_date"),
                         "exit_date": record.get("exit_date")},
        quote_provenance={"quote_date": record.get("quote_date")},
        forecasts={name: record.get(name) for name in (
            "driver_prediction", "forecast_abs_move", "exp_pnl_sim", "exp_pnl_model",
            "win_sim", "win_model")},
        uncertainty={}, residual_state_ref=None, analog_state_ref=None,
        payoff_state_ref=None, feature_values={},
        null_masks={key: value is None for key, value in (record.get("model_inputs") or {}).items()},
        feature_lineage_refs=(),
        gate_terms={name: record.get(name) for name in ("gate_score", "gate_threshold", "gate_pass")},
        chooser_candidates=(), chooser_selection=None,
        financial_diagnostics=phase4_real._expected_financial_diagnostics(record),
        requested_payoff_views=(), validation_status="scored",
        reason_codes=tuple(record.get("flags") or ()), warnings=(), evidence_refs=(),
    )


def _pair_doc(fixture_id, record, *, kind="score_result"):
    payload = {"request": {"ticker": "MTN", "strategy": record.get("strategy")},
               "record": record, "record_kind": kind}
    return {
        "schema_version": "tier0_pair.v1.1", "fixture_id": fixture_id, "covers": [],
        "notes": "", "payload": payload,
        "payload_hash": content_hash(payload), "request_hash": content_hash(payload["request"]),
        "envelope": {"captured_at": "2026-09-24T00:00:00.000000+00:00",
                     "worker_ref": "test:1", "duration_seconds": 0.0},
    }


def _write_corpus(root: Path, docs, *, declared=None):
    (root / "pairs").mkdir(parents=True, exist_ok=True)
    for doc in docs:
        (root / "pairs" / f"{doc['fixture_id']}.json").write_text(json.dumps(doc, sort_keys=True))
    index = {"corpus_hash": content_hash({d["fixture_id"]: d["payload_hash"] for d in docs})}
    if declared is not None:
        index["pairs"] = declared
    (root / "INDEX.json").write_text(json.dumps(index, sort_keys=True))
    return root


def _stub_replay(records_by_fixture, natives_by_fixture=None):
    natives_by_fixture = natives_by_fixture or {}

    def verified_fn(pair, _root):
        return {"same_input_receipt": "same", "trace_hash": "trace",
                "frozen_replay": None, "fixture_id": pair["fixture_id"]}

    def replayed_fn(verified):
        fid = verified["fixture_id"]
        native = natives_by_fixture.get(fid) or _native_record(records_by_fixture[fid])
        return native, tuple({"stage": s} for s in phase4_real._REQUIRED_TRACE_STAGES), ()

    return verified_fn, replayed_fn


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def test_selects_only_named_fixtures_loads_once_and_runs_no_battery(tmp_path, monkeypatch):
    rec = _clean_record()
    docs = [_pair_doc("001", rec), _pair_doc("002", rec), _pair_doc("003", rec)]
    root = _write_corpus(tmp_path / "corpus", docs)

    load_calls = []
    real_load = t0.load

    def counting_load(path):
        load_calls.append(path)
        return real_load(path)

    monkeypatch.setattr(t0, "load", counting_load)
    # If the full gate's battery were invoked, these would explode: the
    # diagnostic must replay, not re-verify the whole corpus.
    monkeypatch.setattr(t0, "run", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("battery must not run")))
    monkeypatch.setattr(phase4_real, "run_corpus", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("battery must not run")))
    verified_fn, replayed_fn = _stub_replay({f: rec for f in ("001", "002", "003")})
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", verified_fn)
    monkeypatch.setattr(phase4_real, "_replayed_member", replayed_fn)

    report = targeted.run_targeted(root, ["002", "003"], progress_stream=_Sink())

    assert load_calls == [root], "expected exactly one repository load() call"
    assert [r["fixture_id"] for r in report["rows"]] == ["002", "003"]
    assert report["summary"]["counts"]["compared"] == 2
    # Honest about the whole-corpus load's cost: a real high-water mark, not a claim.
    assert report["peak_rss_mb"] > 0
    assert "checks.tier0_corpus.load" in report["memory_note"]


def test_unknown_selection_is_a_usage_error_not_a_row(tmp_path, monkeypatch):
    docs = [_pair_doc("001", _clean_record())]
    root = _write_corpus(tmp_path / "corpus", docs)
    verified_fn, replayed_fn = _stub_replay({"001": _clean_record()})
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", verified_fn)
    monkeypatch.setattr(phase4_real, "_replayed_member", replayed_fn)

    with pytest.raises(targeted._SelectionError, match="neither declared nor loaded"):
        targeted.run_targeted(root, ["does-not-exist"], progress_stream=_Sink())


def test_declared_missing_undeclared_and_excluded_are_typed(tmp_path, monkeypatch):
    rec = _clean_record()
    kept = _pair_doc("kept", rec)
    excluded_doc = _pair_doc("dropped", _clean_record(), kind="research_replay")
    root = tmp_path / "corpus"
    (root / "pairs").mkdir(parents=True)
    for doc in (kept, excluded_doc):
        (root / "pairs" / f"{doc['fixture_id']}.json").write_text(json.dumps(doc, sort_keys=True))
    declared = {
        "kept": {"payload_hash": kept["payload_hash"], "request_hash": kept["request_hash"],
                 "record_kind": "score_result", "covers": []},
        # Declared in the manifest, but its file is absent on disk.
        "ghost": {"payload_hash": "sha256:" + "0" * 64, "request_hash": "sha256:" + "1" * 64,
                  "record_kind": "score_result", "covers": []},
        "dropped": {"payload_hash": excluded_doc["payload_hash"],
                    "request_hash": excluded_doc["request_hash"],
                    "record_kind": "research_replay", "covers": []},
    }
    (root / "INDEX.json").write_text(json.dumps({"pairs": declared}, sort_keys=True))
    # A pair file the manifest never declared.
    stray = _pair_doc("stray", rec)
    (root / "pairs" / "stray.json").write_text(json.dumps(stray, sort_keys=True))
    verified_fn, replayed_fn = _stub_replay({"kept": rec})
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", verified_fn)
    monkeypatch.setattr(phase4_real, "_replayed_member", replayed_fn)

    report = targeted.run_targeted(root, ["ghost", "kept", "dropped", "stray"],
                                   progress_stream=_Sink())
    rows = {r["fixture_id"]: r for r in report["rows"]}

    assert rows["ghost"]["disposition"] == "incomparable"
    assert "declared pair file missing" in rows["ghost"]["reason"]
    assert rows["stray"]["disposition"] == "incomparable"
    assert "undeclared pair file" in rows["stray"]["reason"]
    assert rows["kept"]["disposition"] == "compared"
    assert rows["dropped"]["disposition"] == "excluded"
    assert rows["dropped"]["reason"] == phase4_real.PHASE4_EXCLUDED_RECORD_KINDS["research_replay"]


# ---------------------------------------------------------------------------
# shared-reference hydration / validation
# ---------------------------------------------------------------------------


def _write_shared(root: Path, value) -> str:
    digest = content_hash(value)
    shared_dir = root / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": t0.SHARED_DOCUMENT_SCHEMA_VERSION, "digest": digest, "value": value}
    (shared_dir / f"{digest.split(':', 1)[-1]}.json").write_text(json.dumps(body, sort_keys=True))
    return digest


def test_shared_reference_pair_is_verified_through_load_hydration(tmp_path):
    """The motivating failure: a raw JSON decode of a ``$shared`` pair fails the
    digest, while the repository loader hydrates it back to the stored
    ``payload_hash``. The diagnostic uses the loader, so the row's payload hash
    verifies through repository code (and it honestly reports the missing trace
    rather than inventing a weaker hash)."""
    root = tmp_path / "corpus"
    (root / "pairs").mkdir(parents=True)
    pool = {"predictions": [0.1, 0.2, 0.3], "tag": "shared-pool"}
    digest = _write_shared(root, pool)
    ref = {t0.SHARED_REF_KEY: digest}
    stored_record = dict(_clean_record(), extra_field=ref)
    expanded_record = dict(_clean_record(), extra_field=pool)
    payload_ref = {"request": {"ticker": "MTN", "strategy": "STR-THRU"},
                   "record": stored_record, "record_kind": "score_result"}
    payload_expanded = {"request": payload_ref["request"], "record": expanded_record,
                        "record_kind": "score_result"}
    # The writer's contract: the digest is taken over the EXPANDED value.
    payload_hash = content_hash(payload_expanded)
    doc = {"schema_version": "tier0_pair.v1.1", "fixture_id": "shared-1", "covers": [],
           "notes": "", "payload": payload_ref, "payload_hash": payload_hash,
           "request_hash": content_hash(payload_ref["request"]),
           "envelope": {"captured_at": "2026-09-24T00:00:00+00:00", "worker_ref": "t:1",
                        "duration_seconds": 0.0}}
    (root / "pairs" / "shared-1.json").write_text(json.dumps(doc, sort_keys=True))
    (root / "INDEX.json").write_text(json.dumps({"corpus_hash": "sha256:" + "c" * 64},
                                                 sort_keys=True))

    # A naive raw decode (the pre-fix path) fails the digest ...
    raw = json.loads((root / "pairs" / "shared-1.json").read_text())
    assert content_hash(raw["payload"]) != payload_hash
    # ... while load() hydrates it back into agreement.
    hydrated = t0.load(root)
    assert content_hash(hydrated.pairs["shared-1"]["payload"],
                        fragments=hydrated.fragments) == payload_hash

    report = targeted.run_targeted(root, ["shared-1"], progress_stream=_Sink())
    row = report["rows"][0]
    assert row["payload_verified"] is True
    assert row["payload_hash_stored"] == payload_hash
    assert row["payload_hash_recomputed"] == payload_hash
    # No trace on this pair: an honest typed incomparability, never a weak pass.
    assert row["disposition"] == "incomparable"
    assert "input_trace: missing" in row["reason"]


def test_a_honestly_traced_gap_is_reported_verbatim(tmp_path):
    """A pair that records a strict_trace_gap names that reason; the diagnostic
    does not fabricate a trace to compare against."""
    root = tmp_path / "corpus"
    (root / "pairs").mkdir(parents=True)
    payload = {"request": {"ticker": "MTN", "strategy": "CAL-P"},
               "record": _clean_record(strategy="CAL-P"), "record_kind": "score_result",
               "trace_disposition": "gap", "strict_trace_gap": "strict probe does not support CAL-P"}
    doc = {"schema_version": "tier0_pair.v1.1", "fixture_id": "gap-1", "covers": [],
           "notes": "", "payload": payload, "payload_hash": content_hash(payload),
           "request_hash": content_hash(payload["request"]),
           "envelope": {"captured_at": "2026-09-24T00:00:00+00:00", "worker_ref": "t:1",
                        "duration_seconds": 0.0}}
    (root / "pairs" / "gap-1.json").write_text(json.dumps(doc, sort_keys=True))
    (root / "INDEX.json").write_text(json.dumps({"corpus_hash": "sha256:" + "d" * 64},
                                                 sort_keys=True))

    report = targeted.run_targeted(root, ["gap-1"], progress_stream=_Sink())
    row = report["rows"][0]
    assert row["disposition"] == "incomparable"
    assert row["trace_verified"] is False
    assert row["reason"] == "strict probe does not support CAL-P"


# ---------------------------------------------------------------------------
# discrepancy / incomparability reporting (value-free)
# ---------------------------------------------------------------------------


def _stub_corpus(tmp_path, monkeypatch, records, natives):
    docs = [_pair_doc(fid, records[fid]) for fid in records]
    root = _write_corpus(tmp_path / "corpus", docs)
    verified_fn, replayed_fn = _stub_replay(records, natives)
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", verified_fn)
    monkeypatch.setattr(phase4_real, "_replayed_member", replayed_fn)
    return root


def test_discrepancy_reports_finding_field_names_and_differences(tmp_path, monkeypatch):
    legacy_magic = 0.2
    native_magic = 0.987654321
    record = _clean_record(flags=("FLAG_A",), model_inputs={"x": 1.0, "y": None},
                           exp_pnl_sim=legacy_magic)
    base = _native_record(record)
    diverging = replace(
        base,
        # a decision key vanishes from native's own resolved_request
        resolved_request={k: v for k, v in record.items() if k != "gate_score"},
        forecasts={**base.forecasts, "exp_pnl_sim": native_magic},
        reason_codes=("FLAG_A", "NO_SCORE"),
        null_masks={"x": True, "y": True},
    )
    root = _stub_corpus(tmp_path, monkeypatch, {"a": record}, {"a": diverging})

    row = targeted.run_targeted(root, ["a"], progress_stream=_Sink())["rows"][0]

    assert row["disposition"] == "compared"
    assert set(row["checks_failed"]) >= {"keys", "simulation", "flags", "null_masks"}
    assert "exp_pnl_sim" in row["numeric_findings"]["simulation"]
    assert row["key_differences"]["legacy_only"] == ["gate_score"]
    assert row["flag_differences"]["native_only"] == ["NO_SCORE"]
    assert row["null_mask_differences"]["value_mismatch"] == ["x"]
    # Value-freedom: neither side's number leaks into the emitted evidence.
    dumped = json.dumps(row)
    assert str(native_magic) not in dumped
    assert "0.2" not in dumped


def test_agreeing_row_fails_nothing_and_records_no_differences(tmp_path, monkeypatch):
    record = _clean_record()
    root = _stub_corpus(tmp_path, monkeypatch, {"a": record}, {})
    row = targeted.run_targeted(root, ["a"], progress_stream=_Sink())["rows"][0]
    assert row["disposition"] == "compared"
    assert row["checks_failed"] == []
    assert not ({"key_differences", "flag_differences", "null_mask_differences"} & set(row))


def test_trace_verification_failure_is_incomparable(tmp_path, monkeypatch):
    record = _clean_record()
    root = tmp_path / "corpus"
    _write_corpus(root, [_pair_doc("a", record)])

    def raising(pair, _root):
        raise phase4_real._TraceError("input_trace.trace_hash: content hash mismatch")

    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", raising)
    row = targeted.run_targeted(root, ["a"], progress_stream=_Sink())["rows"][0]
    assert row["disposition"] == "incomparable"
    assert row["trace_verified"] is False
    assert "content hash mismatch" in row["reason"]


def test_trace_verified_but_replay_refused_keeps_trace_verified(tmp_path, monkeypatch):
    record = _clean_record()
    root = tmp_path / "corpus"
    _write_corpus(root, [_pair_doc("a", record)])
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle",
                        lambda pair, _root: {"trace_hash": "sha256:" + "a" * 64})

    def failing_replay(verified):
        raise phase4_real._TraceError("execution.simulation.output_hash: captured runtime mismatch")

    monkeypatch.setattr(phase4_real, "_replayed_member", failing_replay)
    row = targeted.run_targeted(root, ["a"], progress_stream=_Sink())["rows"][0]
    assert row["disposition"] == "incomparable"
    # The trace hash itself verified; only the replay downstream refused.
    assert row["trace_verified"] is True
    assert "captured runtime mismatch" in row["reason"]


# ---------------------------------------------------------------------------
# real trace: verify payload + trace hashes through repository code (no stubs)
# ---------------------------------------------------------------------------


def _strict_trace_corpus(tmp_path) -> Path:
    """One genuinely traced STR-THRU pair, built by Phase 4's own tooling
    (adapted from tests/test_checks_phase5_phase4_replay.py::_corpus)."""
    from checks.phase4_frozen_bridge import prepare_frozen_replay
    from tests.test_checks_phase5_acceptance import CLOCK, _linear
    from tools.capture_tier0_corpus import make_pair, package_strict_trace
    from tools.phase4_request_translation import canonical_request_from_legacy
    from tools.phase4_frozen_resources import package_frozen_resources

    root, source = tmp_path / "corpus", tmp_path / "corpus-src"
    source.mkdir(parents=True)
    data = _linear(1.0)
    (source / "size.json").write_bytes(data)
    package = package_frozen_resources(
        model_bindings=[{
            "model_id": "m-size-all", "artifact": "size.json",
            "artifact_sha256": hashlib.sha256(data).hexdigest(), "role": "size",
            "feature_order": ["x"], "output_names": ["y"], "strategy": "STR-THRU",
            "decision_clock": CLOCK, "adapter": "json-linear.v1",
        }],
        deployment_id="dep", release_root=root, source_root=source)
    legacy_request = {"ticker": "ABC", "strategy": "STR-THRU", "as_of": "2026-09-16",
                      "event_date": "2026-09-17", "session": "AMC",
                      "fill": {"policy_id": "legacy.fill_alpha.v1", "alpha": 0.5}}
    request = replace(
        canonical_request_from_legacy(legacy_request, event_id="event-1", snapshot="snapshot-1"),
        deployment_id="dep", decision_clock_id=CLOCK, model_artifact_refs=package.request_refs)
    blocks = {"context": {"strategy": "STR-THRU"}, "features": {"model_inputs": {"x": 3.0}},
              "forecast": {}, "geometry": None, "pricing": None, "analogs": {},
              "simulation": {}, "gate": {}, "chooser": {}, "diagnostics": {}, "model": {}}
    shared = {"request": to_document(request), "native_inputs": blocks}
    source_ref = content_hash(shared)
    receipts = tuple(receipt(stage, {"source_ref": source_ref}, {"execution": "native-runtime"})
                     for stage in phase4_real._REQUIRED_TRACE_STAGES)
    inputs = NativeScoreInputs(**blocks, source_ref=source_ref, stage_receipts=receipts)
    resources = list(package.resource_rows) + [
        {"resource_id": b["binding_id"], "ref": b["request_ref"], "kind": "sidecar",
         "document": b, "content_hash": content_hash(b)} for b in package.sidecar_document["bindings"]]
    metadata = {"frozen_inference": package.trace_declaration}
    plan = prepare_frozen_replay(
        release_root=root, resource_rows=resources,
        verified_documents=phase4_real._verified_resources(root, resources),
        metadata=metadata, request=request, inputs=inputs)
    trace, native = package_strict_trace(
        request, inputs, shared, resources=resources, metadata=metadata,
        frozen_runtime=(plan.inference, plan.release, plan.requests))
    pair = make_pair("pair-1", ["frozen"], legacy_request, {"score_id": native.score_id},
                     record_kind="score_result", duration=0.0, input_trace=trace,
                     legacy_input_hash=trace["shared_input_hash"])
    (root / "pairs").mkdir()
    (root / "pairs" / "pair-1.json").write_text(json.dumps(pair, sort_keys=True))
    (root / "INDEX.json").write_text(json.dumps({"corpus_hash": content_hash({"p": 1})},
                                                 sort_keys=True))
    return root


def test_real_traced_pair_verifies_payload_and_trace_hashes(tmp_path):
    root = _strict_trace_corpus(tmp_path)
    loaded = t0.load(root)
    # The repository verifier accepts the trace (this is where a raw decode dies).
    verified = phase4_real._verified_trace_bundle(loaded.pairs["pair-1"], loaded.root)
    assert verified["trace_hash"] == loaded.pairs["pair-1"]["payload"]["input_trace_hash"]

    report = targeted.run_targeted(root, ["pair-1"], progress_stream=_Sink())
    row = report["rows"][0]
    assert row["payload_verified"] is True
    assert row["trace_verified"] is True
    # The replay either compared cleanly or refused downstream; a hash-integrity
    # failure is the one outcome that must never be reported here.
    assert row["disposition"] in {"compared", "incomparable"}
    if row["disposition"] == "incomparable":
        assert "hash mismatch" not in row["reason"]


# ---------------------------------------------------------------------------
# progress / provisional ETA (a flushed heartbeat even while a fixture blocks)
# ---------------------------------------------------------------------------


def test_progress_reports_a_heartbeat_while_a_single_fixture_blocks(tmp_path, monkeypatch):
    record = _clean_record()
    root = tmp_path / "corpus"
    _write_corpus(root, [_pair_doc("slow", record)])
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle",
                        lambda pair, _root: {"trace_hash": "trace",
                                             "fixture_id": pair["fixture_id"]})

    def slow_replay(verified):
        time.sleep(0.2)
        native = _native_record(record)
        return native, tuple({"stage": s} for s in phase4_real._REQUIRED_TRACE_STAGES), ()

    monkeypatch.setattr(phase4_real, "_replayed_member", slow_replay)
    sink = _Sink()
    targeted.run_targeted(root, ["slow"], progress_stream=sink, progress_interval=0.03)

    text = sink.text
    assert "provisional ETA" in text
    # A heartbeat landed while the row had not finished (completed < total).
    assert "0/1" in text
    # The rate label switches from baseline to observed once a row completes.
    assert "baseline" in text
    assert "1/1" in text and "observed" in text


def test_progress_reporter_exposes_baseline_and_observed_rates():
    sink = _Sink()
    reporter = targeted._ProgressReporter(3, sink, interval=0.01, baseline=96.0)
    with reporter:
        reporter.begin_row("x")
        time.sleep(0.03)  # blocked on the first row: only the baseline exists
        reporter.end_row("x", 10.0)
    text = sink.text
    assert "0/3" in text          # heartbeat while blocked
    assert "baseline ~32m/20 rows" in text
    assert "1/3" in text          # final line after one of three completed
    # observed rate 10s/row -> 2 remaining rows -> ~20s
    assert "rate 10s/row, observed" in text


# ---------------------------------------------------------------------------
# never claims sign-off
# ---------------------------------------------------------------------------


def test_report_never_claims_sign_off_or_overall_status(tmp_path, monkeypatch):
    record = _clean_record()
    root = tmp_path / "corpus"
    _write_corpus(root, [_pair_doc("a", record)])
    verified_fn, replayed_fn = _stub_replay({"a": record})
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", verified_fn)
    monkeypatch.setattr(phase4_real, "_replayed_member", replayed_fn)

    report = targeted.run_targeted(root, ["a"], progress_stream=_Sink())
    assert report["claims_sign_off"] is False
    assert report["sign_off"] is False
    assert report["overall_phase4_status"] == "not_evaluated"
    assert report["full_phase4_gate_required"] is True
    dumped = json.dumps(report).lower()
    assert "pass" not in report["summary"]
    # A per-row "complete" would masquerade as the gate's population verdict.
    assert '"complete"' not in dumped


def test_main_writes_atomically_and_keeps_stdout_json_only(tmp_path, monkeypatch, capsys):
    record = _clean_record()
    root = tmp_path / "corpus"
    _write_corpus(root, [_pair_doc("a", record)])
    verified_fn, replayed_fn = _stub_replay({"a": record})
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", verified_fn)
    monkeypatch.setattr(phase4_real, "_replayed_member", replayed_fn)
    out = tmp_path / "nested" / "report.json"

    rc = targeted.main(["--corpus", str(root), "--fixture-id", "a", "--output", str(out)])

    assert rc == 0
    assert out.is_file()
    json.loads(out.read_text())  # valid JSON
    captured = capsys.readouterr()
    assert captured.out == ""      # JSON went to the file, not stdout
    assert "targeted-replay" in captured.err


# ---------------------------------------------------------------------------
# Blocker 1: a multi-member chooser must aggregate, then add the chooser check
# ---------------------------------------------------------------------------


def _chooser_pair_doc(fixture_id, summary_record):
    doc = _pair_doc(fixture_id, summary_record, kind="dyn_sv_choice")
    return doc


def _chooser_stub(members_specs, choice):
    def chooser(pair, _root):
        members = [
            (mrec, {"trace_hash": f"trace-{i}"}, native,
             tuple({"stage": s} for s in phase4_real._REQUIRED_TRACE_STAGES), ())
            for i, (mrec, native) in enumerate(members_specs)
        ]
        return members, choice
    return chooser


def _summary_record(**overrides):
    base = _clean_record(strategy="DYN-SV", chosen_strategy="STR-THRU", menu_size=2,
                         chosen_margin=0.05)
    base.update(overrides)
    return base


def _chooser_corpus(tmp_path, summary_record):
    root = tmp_path / "corpus"
    return _write_corpus(root, [_chooser_pair_doc("chooser", summary_record)])


def test_two_member_chooser_aggregates_all_members_then_adds_chooser(tmp_path, monkeypatch):
    """Reproduces the KeyError('chooser') crash: two ranked members, each with
    only ordinary checks, must aggregate across BOTH before the single chooser
    selection verdict is added — the row completes, not crashes."""
    summary = _summary_record()
    root = _chooser_corpus(tmp_path, summary)
    member = _clean_record()
    specs = [(member, _native_record(member)), (member, _native_record(member))]
    choice = replace(_native_record(summary), chooser_selection={
        "strategy": "STR-THRU", "menu_size": 2, "margin": 0.05})
    monkeypatch.setattr(phase4_real, "_replayed_chooser", _chooser_stub(specs, choice))

    row = targeted.run_targeted(root, ["chooser"], progress_stream=_Sink())["rows"][0]

    assert row["disposition"] == "compared"
    assert row["members"] == 2
    assert row["checks"]["chooser"] is True          # selection agrees
    assert row["checks_failed"] == []
    assert row["chooser_findings"] == []


def test_two_member_chooser_reports_disagreement(tmp_path, monkeypatch):
    summary = _summary_record()
    root = _chooser_corpus(tmp_path, summary)
    member = _clean_record()
    specs = [(member, _native_record(member)), (member, _native_record(member))]
    # Native picked a different winning margin than the legacy record claimed.
    choice = replace(_native_record(summary), chooser_selection={
        "strategy": "STR-THRU", "menu_size": 2, "margin": 0.99})
    monkeypatch.setattr(phase4_real, "_replayed_chooser", _chooser_stub(specs, choice))

    row = targeted.run_targeted(root, ["chooser"], progress_stream=_Sink())["rows"][0]

    assert row["disposition"] == "compared"
    assert row["checks"]["chooser"] is False
    assert "chooser" in row["checks_failed"]
    assert row["chooser_findings"] == ["chosen_margin"]


# ---------------------------------------------------------------------------
# Blocker 2: a manifest-bound row must agree on every declared field
# ---------------------------------------------------------------------------


def _manifest_corpus(tmp_path, declared_row):
    rec = _clean_record()
    doc = _pair_doc("m", rec)
    root = tmp_path / "corpus"
    return _write_corpus(root, [doc], declared={"m": declared_row}), rec


def test_declared_row_missing_payload_hash_is_incomparable(tmp_path):
    doc = _pair_doc("m", _clean_record())
    row = {"request_hash": doc["request_hash"], "record_kind": "score_result", "covers": []}
    root = _write_corpus(tmp_path / "corpus", [doc], declared={"m": row})

    out = targeted.run_targeted(root, ["m"], progress_stream=_Sink())["rows"][0]
    assert out["payload_verified"] is False
    assert out["disposition"] == "incomparable"
    assert "missing payload_hash" in out["reason"]


@pytest.mark.parametrize("bad_field", ["request_hash", "record_kind", "covers"])
def test_declared_row_field_mismatch_is_incomparable(tmp_path, bad_field):
    rec = _clean_record()
    doc = _pair_doc("m", rec)
    declared_row = {"payload_hash": doc["payload_hash"], "request_hash": doc["request_hash"],
                    "record_kind": "score_result", "covers": []}
    declared_row[bad_field] = {"request_hash": "sha256:" + "e" * 64,
                               "record_kind": "research_replay",
                               "covers": ["not:declared"]}[bad_field]
    root = _write_corpus(tmp_path / "corpus", [doc], declared={"m": declared_row})

    out = targeted.run_targeted(root, ["m"], progress_stream=_Sink())["rows"][0]
    assert out["payload_verified"] is False
    assert out["disposition"] == "incomparable"
    assert bad_field in out["reason"]


def test_declared_entry_that_is_not_an_object_is_incomparable(tmp_path):
    doc = _pair_doc("m", _clean_record())
    root = _write_corpus(tmp_path / "corpus", [doc], declared={"m": None})

    out = targeted.run_targeted(root, ["m"], progress_stream=_Sink())["rows"][0]
    assert out["payload_verified"] is False
    assert out["disposition"] == "incomparable"
    assert "not an object" in out["reason"]


def test_matching_declared_row_still_compares(tmp_path, monkeypatch):
    rec = _clean_record()
    doc = _pair_doc("m", rec)
    declared_row = {"payload_hash": doc["payload_hash"], "request_hash": doc["request_hash"],
                    "record_kind": "score_result", "covers": []}
    root = _write_corpus(tmp_path / "corpus", [doc], declared={"m": declared_row})
    verified_fn, replayed_fn = _stub_replay({"m": rec})
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", verified_fn)
    monkeypatch.setattr(phase4_real, "_replayed_member", replayed_fn)

    out = targeted.run_targeted(root, ["m"], progress_stream=_Sink())["rows"][0]
    assert out["payload_verified"] is True
    assert out["disposition"] == "compared"


# ---------------------------------------------------------------------------
# Blocker 3: a chooser failure cannot be mislabelled as a trace-hash failure
# ---------------------------------------------------------------------------


def test_chooser_replay_failure_reports_unknown_trace_status(tmp_path, monkeypatch):
    summary = _summary_record()
    root = _chooser_corpus(tmp_path, summary)

    def raising(pair, _root):
        raise phase4_real._TraceError(
            "chooser member 0: execution.simulation: captured runtime mismatch")

    monkeypatch.setattr(phase4_real, "_replayed_chooser", raising)
    out = targeted.run_targeted(root, ["chooser"], progress_stream=_Sink())["rows"][0]
    assert out["disposition"] == "incomparable"
    # The combined helper conflates trace verification and replay, so the tool
    # cannot claim the trace hash itself failed: unknown, not False.
    assert out["trace_verified"] is None


# ---------------------------------------------------------------------------
# Blocker 4: --output can never target the read-only corpus
# ---------------------------------------------------------------------------


def test_output_inside_corpus_is_rejected_before_loading(tmp_path, monkeypatch):
    rec = _clean_record()
    root = _write_corpus(tmp_path / "corpus", [_pair_doc("a", rec)])
    monkeypatch.setattr(t0, "load", lambda _p: (_ for _ in ()).throw(
        AssertionError("must not load before the output guard rejects")))

    with pytest.raises(targeted._SelectionError, match="must not write inside"):
        targeted.run_targeted(root, ["a"], output=root / "INDEX.json",
                              progress_stream=_Sink())


def test_output_in_corpus_subdirectory_is_rejected(tmp_path):
    root = _write_corpus(tmp_path / "corpus", [_pair_doc("a", _clean_record())])
    with pytest.raises(targeted._SelectionError, match="must not write inside"):
        targeted.run_targeted(root, ["a"], output=root / "sub" / "report.json",
                              progress_stream=_Sink())


def test_output_through_a_symlinked_parent_into_corpus_is_rejected(tmp_path):
    real = _write_corpus(tmp_path / "realcorpus", [_pair_doc("a", _clean_record())])
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(targeted._SelectionError, match="must not write inside"):
        targeted.run_targeted(real, ["a"], output=link / "INDEX.json",
                              progress_stream=_Sink())


def test_legitimate_outside_output_passes_and_writes_nothing(tmp_path, monkeypatch):
    rec = _clean_record()
    root = _write_corpus(tmp_path / "corpus", [_pair_doc("a", rec)])
    verified_fn, replayed_fn = _stub_replay({"a": rec})
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", verified_fn)
    monkeypatch.setattr(phase4_real, "_replayed_member", replayed_fn)
    outside = tmp_path / "elsewhere" / "report.json"

    report = targeted.run_targeted(root, ["a"], output=outside, progress_stream=_Sink())

    assert report["rows"][0]["disposition"] == "compared"
    assert not outside.exists()  # run_targeted is read-only; only main writes


def test_output_symlink_inside_corpus_pointing_outside_is_rejected(tmp_path):
    """A symlink that LIVES in the corpus but POINTS outside it must be refused:
    the atomic write's ``replace()`` swaps the symlink entry in the corpus dir,
    so resolving only the final path (to an outside target) would let the write
    clobber the read-only oracle."""
    root = _write_corpus(tmp_path / "corpus", [_pair_doc("a", _clean_record())])
    outside = tmp_path / "outside.json"
    outside.write_text('{"keep": "me"}')
    link = root / "report.json"
    link.symlink_to(outside)

    with pytest.raises(targeted._SelectionError, match="must not write inside"):
        targeted.run_targeted(root, ["a"], output=link, progress_stream=_Sink())

    assert link.is_symlink()                 # the corpus entry is untouched
    assert outside.read_text() == '{"keep": "me"}'  # its target too


# ---------------------------------------------------------------------------
# Blocker 2: load phase and active-row ETA (fake clock, no real sleeping)
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, d):
        self.t += d


def test_reporter_loading_phase_is_labeled_and_has_no_row_eta():
    clock = _FakeClock(100.0)
    reporter = targeted._ProgressReporter(2, _Sink(), baseline=10.0, clock=clock)
    reporter.begin_load()
    line = reporter._format(120.0)
    assert "phase=loading" in line
    assert "load ETA unknown" in line
    assert "2 row(s) queued" in line
    # A slow/gigabyte load must never masquerade as a replay parity ETA.
    assert "provisional ETA" not in line


def test_reporter_replay_eta_tracks_active_row_and_flags_overdue():
    clock = _FakeClock(1000.0)
    reporter = targeted._ProgressReporter(10, _Sink(), baseline=10.0, clock=clock)
    reporter.begin_load()
    clock.advance(5)
    reporter.end_load()          # load_seconds=5; phase=replay
    reporter.begin_row("x")      # row_started=1005; rate=baseline=10 (nothing done)

    at5 = reporter._format(1010.0)   # active row 5s in: ETA = (10-5)+9*10 = 95s
    at9 = reporter._format(1014.0)   # 9s in: (10-9)+9*10 = 91s
    at15 = reporter._format(1020.0)  # 15s in > prior 10s -> overdue

    assert "phase=replay" in at5 and "0/10" in at5
    assert "(replay, after load)" in at5 and "load 5s" in at5
    # The estimate moves with the in-flight row instead of a static whole-op 1.6m:
    # 95s -> 1.6m, 91s -> 1.5m, and the not-yet-overdue lines carry no flag.
    assert "1.6m" in at5 and "overdue" not in at5
    assert "1.5m" in at9 and "overdue" not in at9
    assert "overdue" in at15

    # Once a row has finished the rate switches to the observed mean.
    reporter.end_row("x", 15.0)
    reporter.begin_row("y")
    observed = reporter._format(clock.t + 1.0)
    assert "1/10" in observed and "observed" in observed and "15s/row" in observed


def test_reporter_eta_adds_no_spurious_row_when_nothing_is_active():
    """Between rows and after the run completes there is no in-flight row, so
    ``leftover_active`` must be zero — otherwise the ETA silently adds a whole
    extra row duration (the last heartbeat wrongly showed 1.6m on a done run)."""
    clock = _FakeClock(0.0)
    reporter = targeted._ProgressReporter(2, _Sink(), baseline=60.0, clock=clock)
    reporter.begin_load()
    reporter.end_load()

    reporter.begin_row("x")
    clock.advance(120)
    reporter.end_row("x", 120.0)        # completed=1; observed rate=120s/row
    between = reporter._format(clock.t)  # no active row -> ETA = 1*120 = 2.0m
    assert "1/2" in between
    assert "2.0m" in between and "4.0m" not in between

    reporter.begin_row("y")
    clock.advance(120)
    reporter.end_row("y", 120.0)         # completed=2; run is done
    done = reporter._format(clock.t)
    assert "2/2" in done and "0.0m" in done


def test_run_targeted_heartbeat_labels_loading_before_replay(tmp_path, monkeypatch):
    rec = _clean_record()
    root = _write_corpus(tmp_path / "corpus", [_pair_doc("a", rec)])
    real_load = t0.load

    def slow_load(path):
        time.sleep(0.12)
        return real_load(path)

    monkeypatch.setattr(t0, "load", slow_load)
    verified_fn, replayed_fn = _stub_replay({"a": rec})
    monkeypatch.setattr(phase4_real, "_verified_trace_bundle", verified_fn)
    monkeypatch.setattr(phase4_real, "_replayed_member", replayed_fn)

    sink = _Sink()
    targeted.run_targeted(root, ["a"], progress_stream=sink, progress_interval=0.02)
    text = sink.text
    assert "phase=loading" in text and "load ETA unknown" in text
    assert "phase=replay" in text


