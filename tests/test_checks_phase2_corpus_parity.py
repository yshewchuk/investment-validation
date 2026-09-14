"""D14: synthetic end-to-end test for ``checks/rearchitecture_phase2_corpus_parity``.

Real SQLite catalog, real ``ArtifactStore``, real ``Service`` under
``tests.ops_support.TEST_POLICY``, against a synthetic on-disk legacy tree
(``tests.test_v2_data_import.build_legacy_store``) and a tiny synthetic
tier-0 corpus (this module's own ``build_corpus``). ``snapshot_import`` and
``legacy_materialize`` run their real worker subprocess end to end; the
``legacy_score_requests`` worker is swapped for a canned-record stub via the
SAME argv-swap seam ``tests/test_v2_ops_snapshot_stages.py`` already uses for
``legacy_score`` -- the real legacy ``Scorer`` needs a real market panel this
synthetic tree does not carry, exactly the limitation that module's own
docstring records.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase2_corpus_parity import (  # noqa: E402
    build_receipt,
    control_localized_to_analogs,
    filtered_legacy_root,
    import_corpus,
    legacy_store_snapshot_hash,
    publish,
    run_corpus,
)
from checks.rearchitecture_phase2_evidence import (  # noqa: E402
    AUTHORITY_MODE,
    CORPUS_PARITY_KIND,
    PHASE2_EVIDENCE_V1,
    validate_evidence,
)
from engine.v2.data import reference_inputs  # noqa: E402
from engine.v2.diagnosis import AGREE, DIFFER  # noqa: E402
from engine.v2.foundation import SystemClock  # noqa: E402
from engine.v2.ops import executor  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.errors import OpsError  # noqa: E402
from engine.v2.ops.submission import NamespacePolicy  # noqa: E402
from tests.ops_support import TEST_POLICY  # noqa: E402
from tests.test_v2_data_import import build_legacy_store  # noqa: E402

POLICY = NamespacePolicy({"operator": frozenset({"shadow", "smoke"})})
YEAR = 2024
EVENT_DATE = f"{YEAR}-06-01"
#: The frozen legacy SNAPSHOT hash this module's synthetic store and corpus
#: agree on -- ``build_legacy_store`` writes every ``legacy_snapshot_metadata``
#: key (including ``snapshot``) as ``None``, so a matching value is stamped in
#: by :func:`_set_legacy_snapshot` after the store is built (D14 review: the
#: harness now refuses ``import``/``control-missing-analogs`` when a
#: ``source_root``'s SNAPSHOT hash does not equal the corpus's own).
SNAPSHOT_HASH = "sha256:test-snapshot-0001"


def _set_legacy_snapshot(store_root: Path, value: str | None) -> None:
    """Stamp the synthetic legacy store's SNAPSHOT ``snapshot`` key, preserving
    every other ``legacy_snapshot_metadata`` key ``build_legacy_store`` wrote."""
    path = Path(store_root) / reference_inputs.LEGACY_SNAPSHOT_PATH
    doc = json.loads(path.read_text())
    doc["snapshot"] = value
    path.write_text(json.dumps(doc))


# --------------------------------------------------------------------------
# synthetic corpus: a fake tier-0 corpus on disk, tier0_corpus.py's own layout
# --------------------------------------------------------------------------


def _request(ticker: str, strategy: str) -> dict:
    return {"ticker": ticker, "strategy": strategy, "event_date": EVENT_DATE,
           "as_of": EVENT_DATE, "expiry": EVENT_DATE, "session": "BMO",
           "fill": {"alpha": 0.5}}


def _pair(fid: str, kind: str, request: dict, record: dict) -> dict:
    return {"schema_version": "tier0_pair.v1.1", "fixture_id": fid, "payload_hash": "x",
           "request_hash": fid + "-req", "covers": [], "notes": "", "envelope": {},
           "payload": {"record_kind": kind, "request": request, "record": record}}


#: The synthetic trades table ``build_legacy_store`` writes has exactly one,
#: fixed ticker value ("fx_ticker" -- ``_synthetic_rows``'s own placeholder
#: for a non-primary-key, non-nullable string column). ``evidence_scope_
#: covers_trades`` requires the evidence ticker set to be a superset of
#: trades' REAL span, so every corpus request here uses that same ticker.
TICKER = "fx_ticker"


def build_corpus(root: Path, *, snapshot: str | None = SNAPSHOT_HASH) -> Path:
    """Two ``score_result`` + one ``research_replay`` + one ``dyn_sv_choice``
    (two frame rows) -- enough to exercise every population drop D14 names.
    ``snapshot`` defaults to matching :func:`_set_legacy_snapshot`'s stamp on
    the synthetic legacy store; a test wanting a mismatch passes a different
    value (or leaves the store's default ``None`` untouched)."""
    frame_rows = [{"request": _request(TICKER, f"F{i}"),
                  "record": {"gate_score": 0.3 + i * 0.01, "n_analogs": 5}} for i in range(2)]
    pairs = {
        "s1": _pair("s1", "score_result", _request(TICKER, "S1"),
                   {"gate_score": 0.5, "n_analogs": 10, "ci_low": 0.1, "ci_high": 0.2}),
        "s2": _pair("s2", "score_result", _request(TICKER, "S1b"),
                   {"gate_score": 0.7, "n_analogs": 8, "ci_low": 0.05, "ci_high": 0.15}),
        "r1": _pair("r1", "research_replay", _request(TICKER, "S2"), {"strategy": "S2"}),
        "d1": _pair("d1", "dyn_sv_choice", {"frame_rows": frame_rows}, {"chooser_score": 0.9}),
    }
    (root / "pairs").mkdir(parents=True)
    for fid, pair in pairs.items():
        (root / "pairs" / f"{fid}.json").write_text(json.dumps(pair))
    index = {"schema_version": "tier0_corpus.v1.1", "pairs": {}, "corpus_hash": "x",
            "snapshot": snapshot, "required_axes": [], "coverage": {}, "as_of": EVENT_DATE, "tier": 0,
            "stage_plan_ref": "scorer.v1", "tolerance_policy_ref": "score_record.exact.v1",
            "refusal_code_mapping": {}, "axis_inputs": {}}
    (root / "INDEX.json").write_text(json.dumps(index))
    return root


def expected_canned() -> dict:
    """The canned scorer's answer when it agrees with the frozen corpus."""
    return {
        "s1": {"gate_score": 0.5, "n_analogs": 10, "ci_low": 0.1, "ci_high": 0.2},
        "s2": {"gate_score": 0.7, "n_analogs": 8, "ci_low": 0.05, "ci_high": 0.15},
        "d1#frame-0": {"gate_score": 0.3, "n_analogs": 5},
        "d1#frame-1": {"gate_score": 0.31, "n_analogs": 5},
    }


# --------------------------------------------------------------------------
# worker-side seam: legacy_materialize runs for real; legacy_score_requests
# returns canned records (the existing test_v2_ops_snapshot_stages.py pattern)
# --------------------------------------------------------------------------

_STUB = r"""
import json, os, sys
from pathlib import Path
envelope = json.loads(sys.stdin.buffer.readline())
staging = envelope["staging"]
fd = int(envelope["result_fd"])
if envelope["worker"] == "legacy_score_requests":
    path = os.path.join(staging, envelope["parameters"]["requests_path"])
    entries = json.loads(open(path).read())
    rows = [{"request_id": e["canary_id"], "record": CANNED[e["canary_id"]]} for e in entries]
    with open(path, "w") as fh:
        json.dump({"rows": rows, "expected_population": len(entries)}, fh)
    result = {"outputs": [{"name": envelope["worker"],
                           "path": envelope["parameters"]["requests_path"],
                           "schema": "legacy_action.v1.0"}],
              "completed_ids": list(envelope["parameters"]["expected_ids"])}
else:
    from engine.v2.ops.worker import dispatch
    result = dispatch(envelope["worker"], envelope["parameters"], Path(staging), envelope=envelope)
result.update(schema_version="worker_result.v1.0", job_id=envelope["job_id"],
              attempt_id=envelope["attempt_id"], fence=envelope["fence"])
os.write(fd, json.dumps(result, default=str).encode() + b"\n")
os.close(fd)
"""


def install_stub(monkeypatch, canned: dict) -> None:
    real = subprocess.Popen
    source = "CANNED = " + json.dumps(canned) + "\n" + _STUB

    def popen(args, **kwargs):
        if list(args[-2:]) == ["-m", "engine.v2.ops.worker"]:
            args = [sys.executable, "-u", "-c", source]
        return real(args, **kwargs)

    monkeypatch.setattr(executor.subprocess, "Popen", popen)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def _setup_and_run(tmp_path, canned, monkeypatch, *, scope="corpus", key="t", drop_ticker=None):
    store_root = tmp_path / "legacy_store"
    build_legacy_store(store_root, year=YEAR, daily_market_parts=1, rows_per_part=2)
    _set_legacy_snapshot(store_root, SNAPSHOT_HASH)
    corpus_root = build_corpus(tmp_path / "corpus")
    ops_root = tmp_path / "ops"
    source = store_root
    if drop_ticker is not None:
        source = filtered_legacy_root(store_root, tmp_path / "filtered", drop_ticker)
    import_corpus(ops_root, source, scope, corpus_root, policy=POLICY, idempotency_key=f"{key}-import",
                 resource_policy=TEST_POLICY)
    install_stub(monkeypatch, canned)
    run_result = run_corpus(ops_root, store_root, scope, corpus_root, policy=POLICY,
                            idempotency_key=key, resource_policy=TEST_POLICY)
    return corpus_root, run_result


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_identical_results_agree(tmp_path, monkeypatch):
    corpus_root, run_result = _setup_and_run(tmp_path, expected_canned(), monkeypatch, key="agree")
    receipt = build_receipt(corpus_root, run_result, code_hash="c1", environment_hash="e1")
    assert receipt.verdict == AGREE, receipt.summary()
    assert receipt.comparison_kind == CORPUS_PARITY_KIND
    population = receipt.population
    assert (population.expected, population.supported, population.compared) == (6, 4, 4)
    assert len(population.excluded) == 2
    assert {e["stage"] for e in population.excluded} == {"expected_to_supported"}
    assert {e["key"] for e in population.excluded} == {"r1", "d1"}
    # Review item 1: compare binds the legacy SNAPSHOT hash the run actually
    # scored against, and shows it equals the corpus's own declared hash.
    assert receipt.envelope.diagnostic_ref == (
        f"legacy_snapshot_hash={SNAPSHOT_HASH};matches_corpus_snapshot=true")


def test_planted_field_difference_is_localized_and_redacted(tmp_path, monkeypatch):
    canned = expected_canned()
    canned["s2"] = dict(canned["s2"], gate_score=canned["s2"]["gate_score"] + 0.05)
    corpus_root, run_result = _setup_and_run(tmp_path, canned, monkeypatch, key="differ")
    receipt = build_receipt(corpus_root, run_result, code_hash="c1", environment_hash="e1")
    assert receipt.verdict == DIFFER
    assert [f.field_path for f in receipt.findings] == ["s2::gate_score"]
    finding = receipt.findings[0]
    assert finding.left_value is None and finding.right_value is None and finding.delta is None


def test_missing_analog_slice_control_localizes_to_analogs(tmp_path, monkeypatch):
    canned = expected_canned()
    for cid in ("s1", "s2"):
        canned[cid] = dict(canned[cid], n_analogs=canned[cid]["n_analogs"] - 1,
                           ci_low=canned[cid]["ci_low"] + 0.01)
    corpus_root, run_result = _setup_and_run(
        tmp_path, canned, monkeypatch, scope="corpus-control", key="control",
        drop_ticker="nonexistent_ticker")
    receipt = build_receipt(corpus_root, run_result, code_hash="c1", environment_hash="e1",
                            control_drop_ticker="nonexistent_ticker")
    ok, problems = control_localized_to_analogs(receipt)
    assert ok, problems
    assert receipt.verdict == DIFFER
    stages = {f.first_differing_stage for f in receipt.findings}
    assert stages == {"analogs"}
    # Review item 1: the control's filtered copy only rewrites trades rows,
    # so the SNAPSHOT hash alone would show a (misleading) match -- the
    # receipt must record the deliberate divergence explicitly instead.
    diagnostic = receipt.envelope.diagnostic_ref or ""
    assert f"legacy_snapshot_hash={SNAPSHOT_HASH}" in diagnostic
    assert "control_drop_ticker=nonexistent_ticker" in diagnostic


def test_missing_analog_control_fails_when_findings_are_not_localized(tmp_path, monkeypatch):
    """The control detector itself: a non-analog perturbation must NOT pass
    as a localized analog control."""
    canned = expected_canned()
    canned["s1"] = dict(canned["s1"], gate_score=canned["s1"]["gate_score"] + 0.2)
    corpus_root, run_result = _setup_and_run(
        tmp_path, canned, monkeypatch, scope="corpus-control2", key="control2",
        drop_ticker="nonexistent_ticker")
    receipt = build_receipt(corpus_root, run_result, code_hash="c1", environment_hash="e1")
    ok, problems = control_localized_to_analogs(receipt)
    assert not ok
    assert problems


def test_receipt_passes_evidence_validator_corpus_kind_and_binding(tmp_path, monkeypatch):
    corpus_root, run_result = _setup_and_run(tmp_path, expected_canned(), monkeypatch, key="valid")
    receipt = build_receipt(corpus_root, run_result, code_hash="deadbeef",
                            environment_hash="cafef00d")
    artifact_root = tmp_path / "evidence"
    ref = publish(receipt, artifact_root)
    population = receipt.population
    evidence = {
        "schema_version": PHASE2_EVIDENCE_V1, "code_hash": "deadbeef",
        "environment_hash": "cafef00d", "authority_mode": AUTHORITY_MODE,
        "corpus_comparison_receipt_ref": ref,
        "expected_population": population.expected, "supported_population": population.supported,
        "compared_population": population.compared,
    }
    findings, field_ok, _document_ok = validate_evidence(
        evidence, artifact_root=artifact_root, code_hash="deadbeef", environment_hash="cafef00d")
    corpus_findings = [f for f in findings if f.get("field") == "corpus_comparison_receipt_ref"]
    assert not corpus_findings, findings
    assert field_ok.get("corpus_comparison_receipt_ref") is True


def _jobs_count(ops_root: Path) -> int:
    conn = open_catalog(ops_root / "catalog.sqlite", clock=SystemClock())
    try:
        return conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    finally:
        conn.close()


def test_corpus_snapshot_mismatch_refused_before_any_job_submitted(tmp_path):
    store_root = tmp_path / "legacy_store"
    build_legacy_store(store_root, year=YEAR)
    _set_legacy_snapshot(store_root, "a-different-snapshot-hash")
    corpus_root = build_corpus(tmp_path / "corpus")  # declares SNAPSHOT_HASH
    ops_root = tmp_path / "ops"
    with pytest.raises(OpsError) as excinfo:
        import_corpus(ops_root, store_root, "corpus", corpus_root, policy=POLICY,
                      resource_policy=TEST_POLICY)
    assert excinfo.value.code == "INPUT_CHANGED"
    assert excinfo.value.problem.details["reason"] == "CORPUS_SNAPSHOT_MISMATCH"
    assert excinfo.value.problem.details["corpus_snapshot"] == SNAPSHOT_HASH
    assert excinfo.value.problem.details["source_snapshot"] == "a-different-snapshot-hash"
    # Refused before ``_open(root)`` ever runs: no catalog was created at
    # all, so there is nowhere a job could have been submitted.
    assert not (ops_root / "catalog.sqlite").exists()
    ops_root.mkdir(parents=True, exist_ok=True)
    assert _jobs_count(ops_root) == 0


def test_corpus_snapshot_match_allows_import_to_proceed(tmp_path):
    store_root = tmp_path / "legacy_store"
    build_legacy_store(store_root, year=YEAR)
    _set_legacy_snapshot(store_root, SNAPSHOT_HASH)
    corpus_root = build_corpus(tmp_path / "corpus")
    ops_root = tmp_path / "ops"
    result = import_corpus(ops_root, store_root, "corpus", corpus_root, policy=POLICY,
                           resource_policy=TEST_POLICY)
    assert result["scope"] == "corpus"
    assert _jobs_count(ops_root) > 0


def test_root_inside_source_root_refused(tmp_path):
    store_root = tmp_path / "legacy_store"
    build_legacy_store(store_root, year=YEAR)
    _set_legacy_snapshot(store_root, SNAPSHOT_HASH)
    corpus_root = build_corpus(tmp_path / "corpus")
    bad_root = store_root / "ops"  # resolves inside source_root
    with pytest.raises(OpsError) as excinfo:
        import_corpus(bad_root, store_root, "corpus", corpus_root, policy=POLICY,
                      resource_policy=TEST_POLICY)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert excinfo.value.problem.details["reason"] == "ROOT_INSIDE_LEGACY_TREE"
    assert not (bad_root / "catalog.sqlite").exists()


def test_legacy_store_snapshot_hash_reads_the_stamped_value(tmp_path):
    store_root = tmp_path / "legacy_store"
    build_legacy_store(store_root, year=YEAR)
    assert legacy_store_snapshot_hash(store_root) is None  # build_legacy_store's own default
    _set_legacy_snapshot(store_root, SNAPSHOT_HASH)
    assert legacy_store_snapshot_hash(store_root) == SNAPSHOT_HASH


def test_filtered_legacy_root_drops_only_the_named_tickers_trades_rows(tmp_path):
    """Unit-level proof of the real removal mechanism, kept separate from the
    full pipeline test above: the synthetic trades table this corpus scores
    against carries a single ticker ("fx_ticker") for every row, so dropping
    it there would empty the table outright rather than remove one slice
    among several -- exercised here instead, against a two-ticker table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    store_root = tmp_path / "legacy_store"
    build_legacy_store(store_root, year=YEAR)
    trades_path = store_root / "data" / "curated" / "trades" / f"year={YEAR}" / "part-0000.parquet"
    table = pq.read_table(trades_path)
    two_tickers = table.set_column(table.column_names.index("ticker"), "ticker",
                                   pa.array(["keep_me", "drop_me"]))
    pq.write_table(two_tickers, trades_path)

    filtered = filtered_legacy_root(store_root, tmp_path / "filtered", "drop_me")
    result = pq.read_table(filtered / "data" / "curated" / "trades" / f"year={YEAR}" /
                           "part-0000.parquet")
    assert result.column("ticker").to_pylist() == ["keep_me"]
    # Everything else is hard-linked, not rewritten.
    panel = filtered / "data" / "features" / "panel.parquet"
    assert panel.stat().st_ino == (store_root / "data" / "features" / "panel.parquet").stat().st_ino
