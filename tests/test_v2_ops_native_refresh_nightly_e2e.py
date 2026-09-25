"""S4B2 end-to-end: the production native RefreshPlan builder and cache-only reuse.

The worker subprocess is the real ``engine.v2.ops.worker`` (its argv swapped
for a stub -- the seam ``test_v2_ops_native_refresh_e2e.py`` uses) and the real
supervisor -> executor -> worker path runs end to end. Only the HTTP edge is
faked: the stub monkeypatches
``engine.v2.ops.providers.orats_daily_market.orats_daily_market_fetcher``, the
provider factory ``incremental_data._load_data_refresh_callback`` imports.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from engine.v2.data.catalog import commit_snapshot
from engine.v2.data.manifests import dataset_manifest, snapshot_ref
from engine.v2.foundation import ArtifactStore, content_hash, format_timestamp
from engine.v2.ops import cli, executor, refresh_staging
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly import NATIVE_DAILY_MARKET_ACCOUNT, build_legacy_job_requests
from engine.v2.ops.plans import nightly_plan
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.provider_budget import configure_account
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor, _reserve_provider, claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY, FakeClock, catalog, run_until, sample
from tests.test_v2_data_manifests import _DAILY_MARKET_CONTRACT, _DAILY_MARKET_REF
from tests.test_v2_ops_providers_orats import CORES_ROW, SUMMARIES_ROW

ROOT = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})
SCOPE = "shadow"
SESSION = "2026-05-01"
TICKER = "AAA"
_REAL_POPEN = subprocess.Popen


def _commit_parent(conn, store, clock):
    receipt_ref = content_hash({"s4b2": "parent-receipt"})
    manifest = dataset_manifest(
        _DAILY_MARKET_REF, (), knowledge_mode="reconstructed",
        coverage_receipt_refs=(receipt_ref,), availability_evidence_refs=())
    snapshot = snapshot_ref(
        {"daily_market": manifest}, calendar_version="cal.v1",
        source_priority_version="prio.v1", finality_receipt_refs=(receipt_ref,))
    commit_snapshot(
        conn, scope=SCOPE, request_hash=content_hash({"s4b2": "base"}),
        contracts=(_DAILY_MARKET_CONTRACT,), objects=(), records=(), manifests=(manifest,),
        snapshot=snapshot, expected_head_snapshot_id=None, expected_head_generation=0,
        receipt_id="s4b2-base", attempt_id="s4b2-base", fence=1,
        fence_check=lambda _conn: None, clock=clock, store=store)


def _head(conn):
    return conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = ?",
        (SCOPE,)).fetchone()


def _build_requests(conn, store, clock, root, catalog_path=None, tickers=(TICKER,),
                    context_tickers=None):
    """A native nightly plan with no pinned override, then its job requests."""
    catalog_path = catalog_path or str(root / "ops.sqlite")
    context_tickers = tuple(context_tickers) if context_tickers is not None else tuple(tickers)
    plan = nightly_plan(
        ROOT, SESSION, manifest_ref="artifact:parent",
        expected_population=(TICKER + "|x|" + SESSION,),
        tickers=tickers, context_tickers=context_tickers, refresh_mode="native",
        catalog_path=catalog_path, objects_root=str(root), clock=clock)
    requests = build_legacy_job_requests(
        plan, tickers=tickers, context_tickers=context_tickers,
        year_start=plan["year_start"], year_end=plan["year_end"],
        refresh_mode="native", catalog_path=catalog_path,
        objects_root=str(root), conn=conn, store=store, clock=clock)
    assert requests[0].job.kind == "incremental_refresh"
    return plan, requests


def _catalog_at(directory, name):
    """``tests.ops_support.catalog`` with an explicit catalog filename: the
    CLI always reads ``<root>/catalog.sqlite``, while the fixture uses
    ``ops.sqlite``."""
    clock = FakeClock()
    conn = open_catalog(directory / name, clock=clock)
    epoch = begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
    return conn, clock, Supervisor(epoch, "boot")


def _install_fake_http_fetcher(monkeypatch, *, explode, summaries=None, cores=None):
    """Swap the worker's argv for a stub that fakes only the ORATS HTTP edge."""
    summaries = [SUMMARIES_ROW] if summaries is None else list(summaries)
    cores = [CORES_ROW] if cores is None else list(cores)
    canned = (
        "def fake_http_get(url, *, timeout):\n"
        "    if '/hist/summaries?' in url:\n"
        "        return (200, {}, json.dumps({'data': SUMMARIES, 'message': 'ok'}).encode())\n"
        "    if '/hist/cores?' in url:\n"
        "        return (200, {}, json.dumps({'data': CORES, 'message': 'ok'}).encode())\n"
        "    raise AssertionError('unexpected url ' + url)\n"
    )
    exploding = (
        "def fake_http_get(url, *, timeout):\n"
        "    raise AssertionError('provider HTTP ran on a cache-only refresh: ' + url)\n"
    )
    stub = (
        "import functools, json, sys\n"
        "import engine.v2.ops.providers as providers\n"
        "from engine.v2.ops.providers import orats_daily_market\n"
        "from engine.v2.ops import worker\n"
        "SUMMARIES = json.loads(%r)\n"
        "CORES = json.loads(%r)\n"
    ) % (json.dumps(summaries), json.dumps(cores))
    stub += exploding if explode else canned
    stub += (
        "real = orats_daily_market.orats_daily_market_fetcher\n"
        "patched = functools.partial(real, http_get=fake_http_get)\n"
        "providers.orats_daily_market_fetcher = patched\n"
        "orats_daily_market.orats_daily_market_fetcher = patched\n"
        "raise SystemExit(worker.main())\n"
    )

    def popen(args, **kwargs):
        if list(args[-2:]) == ["-m", "engine.v2.ops.worker"]:
            args = [sys.executable, "-u", "-c", stub]
        return _REAL_POPEN(args, **kwargs)

    monkeypatch.setattr(executor.subprocess, "Popen", popen)


def _run_native(conn, clock, root, receipt, monkeypatch, *, explode, summaries=None, cores=None):
    monkeypatch.setenv("ORATS_API_KEY", "test-key")
    _install_fake_http_fetcher(monkeypatch, explode=explode, summaries=summaries, cores=cores)
    service = Service(conn, root, registry(), TEST_POLICY, clock=clock, code_source=ROOT)
    service.start()
    try:
        state = run_until(service, conn, receipt.job_id, timeout=120)
        if state != "succeeded":
            _raise_attempt_failure(conn, service, receipt.job_id, state)
    finally:
        service.close()


def _raise_attempt_failure(conn, service, job_id, state):
    attempt = conn.execute(
        "SELECT attempt_id FROM attempts WHERE job_id = ? "
        "ORDER BY attempt_number DESC LIMIT 1", (job_id,)).fetchone()
    stderr = ""
    if attempt is not None:
        path = service.store.staging_dir(attempt["attempt_id"]) / "diagnostics" / "worker.stderr"
        if path.is_file():
            stderr = path.read_text()
    failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id = ?",
                           (job_id,)).fetchone()[0]
    raise AssertionError(f"native refresh {state}: {failure}\n{stderr}")


def test_native_nightly_refresh_plan_is_built_and_commits_end_to_end(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)
    configure_account(conn, NATIVE_DAILY_MARKET_ACCOUNT, 1, remaining=10, live_reserve=1)

    _plan, requests = _build_requests(conn, store, clock, tmp_path)
    request = requests[0]
    assert request.job.parameters["provider_calls"] == 6
    assert request.job.provider_budget_ref == NATIVE_DAILY_MARKET_ACCOUNT
    receipt = submit(conn, registry(), POLICY, request, clock=clock)

    _run_native(conn, clock, tmp_path, receipt, monkeypatch, explode=False)

    new_head = _head(conn)
    assert new_head["generation"] == head["generation"] + 1
    assert new_head["snapshot_id"] != head["snapshot_id"]
    revisions = conn.execute(
        "SELECT ticker, session_date FROM data_daily_market_revisions").fetchall()
    assert [(row["ticker"], row["session_date"]) for row in revisions] == [(TICKER, SESSION)]
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 1


def test_missing_staged_refresh_identity_fails_the_job(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)
    configure_account(conn, NATIVE_DAILY_MARKET_ACCOUNT, 1, remaining=10, live_reserve=1)
    _plan, requests = _build_requests(conn, store, clock, tmp_path)
    receipt = submit(conn, registry(), POLICY, requests[0], clock=clock)

    no_op = lambda claim, staging: None
    monkeypatch.setattr(executor, "stage_refresh_input", no_op)
    monkeypatch.setattr(refresh_staging, "stage_refresh_input", no_op)
    _install_fake_http_fetcher(monkeypatch, explode=True)

    service = Service(conn, tmp_path, registry(), TEST_POLICY, clock=clock, code_source=ROOT)
    service.start()
    try:
        states = ("succeeded", "failed", "retry_wait")
        state = run_until(service, conn, receipt.job_id, timeout=60, states=states)
        for _ in range(4):
            if state != "retry_wait":
                break
            clock.advance(70)
            state = run_until(service, conn, receipt.job_id, timeout=60, states=states)
    finally:
        service.close()

    assert state == "failed"
    # The no-op staging leaves run_daily_market_refresh without
    # incremental_refresh_input.json, which returns the data layer's "failed"
    # status; run_refresh_worker maps that to this exact typed failure.
    failure = json.loads(conn.execute(
        "SELECT failure_json FROM jobs WHERE job_id = ?",
        (receipt.job_id,)).fetchone()[0])
    assert failure["code"] == "WORKER_FAILED"
    assert failure["message"] == "incremental refresh did not produce complete coverage"
    unchanged = _head(conn)
    assert (unchanged["snapshot_id"], unchanged["generation"]) == (
        head["snapshot_id"], head["generation"])
    assert conn.execute("SELECT COUNT(*) FROM data_daily_market_revisions").fetchone()[0] == 0


def test_extra_market_rows_are_dropped_and_missing_universe_rows_stay_empty(
        tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)
    configure_account(conn, NATIVE_DAILY_MARKET_ACCOUNT, 1, remaining=10, live_reserve=1)

    missing, extra = "BBB", "ZZZ"
    _plan, requests = _build_requests(conn, store, clock, tmp_path,
                                      context_tickers=(TICKER, missing))
    request = requests[0]
    assert request.job.parameters["provider_calls"] == 6
    receipt = submit(conn, registry(), POLICY, request, clock=clock)

    _run_native(conn, clock, tmp_path, receipt, monkeypatch, explode=False,
                summaries=(SUMMARIES_ROW, dict(SUMMARIES_ROW, ticker=extra)),
                cores=(CORES_ROW, dict(CORES_ROW, ticker=extra)))

    new_head = _head(conn)
    assert new_head["generation"] == head["generation"] + 1
    revisions = conn.execute(
        "SELECT ticker, session_date FROM data_daily_market_revisions").fetchall()
    assert [(row["ticker"], row["session_date"]) for row in revisions] == [(TICKER, SESSION)]
    assert missing not in {row["ticker"] for row in revisions}
    assert extra not in {row["ticker"] for row in revisions}
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 1


def test_second_native_refresh_for_the_same_session_is_cache_only(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    configure_account(conn, NATIVE_DAILY_MARKET_ACCOUNT, 1, remaining=10, live_reserve=1)

    _first_plan, first_requests = _build_requests(conn, store, clock, tmp_path)
    assert first_requests[0].job.parameters["provider_calls"] == 6
    first_receipt = submit(conn, registry(), POLICY, first_requests[0], clock=clock)
    _run_native(conn, clock, tmp_path, first_receipt, monkeypatch, explode=False)
    head_after_first = _head(conn)
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 1

    clock.advance(60)  # a fresh same-session generation, not a retry of plan one
    _second_plan, second_requests = _build_requests(conn, store, clock, tmp_path)
    assert second_requests[0].job.parameters["provider_calls"] == 0
    assert second_requests[0].job.provider_budget_ref is None
    second_receipt = submit(conn, registry(), POLICY, second_requests[0], clock=clock)
    _run_native(conn, clock, tmp_path, second_receipt, monkeypatch, explode=True)

    # Unchanged data is a correct no-op: the replayed coverage resolves to the
    # snapshot already at the head, and no provider call was made.
    head_after_second = _head(conn)
    assert (head_after_second["snapshot_id"], head_after_second["generation"]) == (
        head_after_first["snapshot_id"], head_after_first["generation"])
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM data_daily_market_revisions").fetchone()[0] == 1


def test_provider_account_cli_admits_native_refresh_and_absent_row_is_refused(tmp_path, capsys):
    conn, clock, supervisor = _catalog_at(tmp_path, "catalog.sqlite")
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    root = str(tmp_path)
    catalog_path = str(tmp_path / "catalog.sqlite")

    assert cli.main(["provider-account", "--root", root, "--account",
                     NATIVE_DAILY_MARKET_ACCOUNT, "--remaining", "10",
                     "--live-reserve", "1"]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created == {"account": NATIVE_DAILY_MARKET_ACCOUNT, "generation": "1",
                       "remaining": 10, "live_reserve": 1, "uncertain": 0,
                       "blocked_code": None, "next_eligible_at": None}

    assert cli.main(["provider-account", "--root", root, "--account",
                     NATIVE_DAILY_MARKET_ACCOUNT, "--remaining", "20",
                     "--live-reserve", "2"]) == 0
    updated = json.loads(capsys.readouterr().out)
    assert updated["generation"] == "2"
    assert (updated["remaining"], updated["live_reserve"]) == (20, 2)

    _plan, requests = _build_requests(conn, store, clock, tmp_path,
                                      catalog_path=catalog_path)
    assert requests[0].job.provider_budget_ref == NATIVE_DAILY_MARKET_ACCOUNT
    receipt = submit(conn, registry(), POLICY, requests[0], clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                       supervisor=supervisor, clock=clock, registry=registry())
    assert claim is not None and claim.job_id == receipt.job_id

    with pytest.raises(OpsError) as refused:
        with transaction(conn):
            _reserve_provider(conn, NATIVE_DAILY_MARKET_ACCOUNT + "-absent",
                              "attempt-absent", 1, {"provider_calls": 3},
                              format_timestamp(clock.now()))
    assert refused.value.code == "CREDENTIAL_INVALID"
