"""Price preparation keeps real snapshot fences and survives interrupted planning."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.v2.data import catalog as data_catalog
from engine.v2.data import manifests, reference_catalog
from engine.v2.data.repository import Repository
from engine.v2.ops import nightly_trigger, price_history_store
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.recovery import SupervisorLock
from tests.data_scan_support import RECEIPT, fake_hash, publish_and_inspect
from tests.test_v2_ops_price_history import _SEC, _SEC_REF, _securities_row
from tests.test_v2_ops_snapshot_stages import (
    SESSION as PLAN_SESSION,
)
from tests.test_v2_ops_snapshot_stages import (
    _plan_files,
)
from tests.test_v2_ops_snapshot_stages import (
    case as case,
)
from tests.v2.ops.test_planned_price_refresh import (
    SESSION,
    _assert_prices,
    _read_report,
)
from tests.v2.ops.test_planned_price_refresh import (
    install_fetcher as install_fetcher,
)
from tests.v2.ops.test_planned_price_refresh import (
    isolated_legacy_sources as isolated_legacy_sources,
)
from tests.v2.ops.test_planned_price_refresh import (
    shadow_root as shadow_root,
)


def _new_import(state, label, *, ticker="OTHER"):
    """Commit a synthetic shadow import while retaining the calendar reference."""
    record = publish_and_inspect(
        state.store, _SEC, _SEC_REF, [_securities_row(ticker, 2024)], "2024")
    manifest = manifests.dataset_manifest(
        _SEC_REF, [record], knowledge_mode="reconstructed", coverage_receipt_refs=(RECEIPT,),
        availability_evidence_refs=())
    snapshot = manifests.snapshot_ref(
        {"securities": manifest}, calendar_version=state.base.calendar_version,
        source_priority_version=state.base.source_priority_version,
        finality_receipt_refs=state.base.finality_receipt_refs)
    head = state.conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope='shadow'").fetchone()
    references = reference_catalog.reference_inputs_for_receipt(state.conn, receipt_id="base-r1")
    return data_catalog.commit_snapshot(
        state.conn, scope="shadow", request_hash=fake_hash(label), contracts=[_SEC],
        objects=[record.object_ref], records=[record], manifests=[manifest], snapshot=snapshot,
        expected_head_snapshot_id=head["snapshot_id"], expected_head_generation=head["generation"],
        receipt_id=label, attempt_id=label, fence=1, fence_check=lambda c: None,
        clock=state.clock, store=state.store, record_references=lambda c, receipt_id:
        reference_catalog.insert_reference_inputs(c, receipt_id, references)).snapshot_ref


def test_stale_base_refuses_before_fetch_or_capture(shadow_root, install_fetcher):
    """A stale plan base refuses without fetching or recording a capture."""
    state = shadow_root
    calls = install_fetcher(state.root)
    newer = _new_import(state, "other-import")

    with pytest.raises(OpsError, match="head moved") as error:
        nightly_trigger._refresh_plan_prices(
            state.root, SESSION, ("NEW",), state.clock, state.base.snapshot_id)

    assert error.value.code == "INPUT_CHANGED"
    assert calls == []
    assert not (state.root / "data" / "raw").exists()
    assert state.conn.execute("SELECT COUNT(*) FROM data_price_captures").fetchone()[0] == 0
    assert Repository(state.conn, state.store).resolve(newer.snapshot_id)


def test_capture_fence_rechecks_head_before_commit(shadow_root, install_fetcher, monkeypatch):
    """A concurrent import wins without recording capture rows or a report."""
    state = shadow_root
    install_fetcher(state.root)
    write = price_history_store._write_ticker_fragment
    changed = []

    def concurrent_import(*args, **kwargs):
        """Advance the head after the first synthetic fragment is written."""
        record = write(*args, **kwargs)
        if not changed:
            changed.append(_new_import(state, "during-capture"))
        return record

    monkeypatch.setattr(price_history_store, "_write_ticker_fragment", concurrent_import)
    with pytest.raises(OpsError) as error:
        nightly_trigger._refresh_plan_prices(
            state.root, SESSION, ("NEW",), state.clock, state.base.snapshot_id)

    assert error.value.code == "INPUT_CHANGED"
    assert state.conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'"
                              ).fetchone()[0] == changed[0].snapshot_id
    assert state.conn.execute("SELECT COUNT(*) FROM data_price_captures").fetchone()[0] == 0
    assert not state.root.joinpath(*nightly_trigger.STATE_DIR, SESSION + ".price_refresh.json").exists()


def test_real_capture_cas_failure_becomes_recordable_input_changed(
        shadow_root, install_fetcher, monkeypatch):
    """A lost capture CAS is translated into the trigger's retryable refusal."""
    state = shadow_root
    install_fetcher(state.root)
    commit = data_catalog.commit_snapshot
    changed = []

    def race_commit(*args, **kwargs):
        """Move the shadow head immediately before capture's real commit."""
        if kwargs["receipt_id"].startswith("receipt_ph_") and not changed:
            changed.append("changing")
            _new_import(state, "during-cas")
        return commit(*args, **kwargs)

    monkeypatch.setattr(data_catalog, "commit_snapshot", race_commit)
    with pytest.raises(OpsError) as error:
        nightly_trigger._refresh_plan_prices(
            state.root, SESSION, ("NEW",), state.clock, state.base.snapshot_id)

    assert error.value.code == "INPUT_CHANGED"
    assert error.value.problem.details == {"data_code": "SNAPSHOT_CONFLICT"}
    assert state.conn.execute("SELECT COUNT(*) FROM data_price_captures").fetchone()[0] == 0


def test_capture_then_crash_reimports_under_new_attempt_and_reuses_prices(
        shadow_root, install_fetcher, monkeypatch):
    """A crash after capture retries with a fresh import and cached prices."""
    state = shadow_root
    calls = install_fetcher(state.root)
    imports, plans = [], []

    def ensure(root, as_of, clock, attempt):
        """Return the original pin until the trigger advances its import attempt."""
        imports.append(attempt)
        snapshot = state.base if attempt == 0 else _new_import(state, "retry-import")
        return "ready", snapshot.snapshot_id

    def plan(root, as_of, tickers, context, clock, **kwargs):
        """Interrupt the first plan only after its captured prices are verified."""
        snapshot = kwargs["expected_shadow_snapshot_id"]
        _assert_prices(state, snapshot, "NEW")
        plans.append(snapshot)
        if len(plans) == 1:
            raise RuntimeError("simulated process interruption after capture")
        return "recovered-plan"

    monkeypatch.setattr(nightly_trigger, "_default_plan", plan)

    def attempt(prior=None, plan_ref=None):
        """Enter production price preparation with isolated terminal boundaries."""
        return nightly_trigger._submit_plan(
            state.root, SESSION, tickers=("NEW",), context_tickers=("NEW",),
            clock=state.clock, plan_fn=None, submit_fn=lambda *a: None,
            serve_fn=lambda *a: "completed", ensure_snapshot_fn=ensure,
            full_run=True, prior=prior, plan_ref=plan_ref)

    with pytest.raises(RuntimeError):
        attempt()
    first_capture = _read_report(state.root)["capture"]["result_snapshot_id"]
    stale = attempt()
    assert stale.status == "error" and stale.snapshot_attempt == 1
    assert "INPUT_CHANGED" in stale.detail
    assert nightly_trigger.load_state(state.root, SESSION) == stale
    recovered = attempt(stale)
    assert recovered.status == "completed" and recovered.snapshot_attempt == 1
    assert recovered.plan_ref == "recovered-plan"
    assert imports == [0, 0, 1]
    assert calls == ["NEW", "SPY"]
    _assert_prices(state, first_capture, "NEW")
    _assert_prices(state, plans[-1], "NEW")

    # A saved plan resumes directly, even if the current head later moves.
    _new_import(state, "after-plan", ticker="LATER")
    resumed = attempt(recovered, recovered.plan_ref)
    assert resumed.status == "completed"
    assert imports == [0, 0, 1] and len(plans) == 2
    assert calls == ["NEW", "SPY"]


@pytest.mark.parametrize("error_count,snapshot_attempt", [
    (0, 0),
    (nightly_trigger.MAX_CONSECUTIVE_ERRORS - 1, nightly_trigger.MAX_CONSECUTIVE_ERRORS - 1),
])
def test_capture_lock_contention_resumes_without_spending_retry_budget(
        shadow_root, install_fetcher, monkeypatch, error_count, snapshot_attempt):
    """Real lock contention stays resumable past both the error budget and window."""
    state = shadow_root
    calls = install_fetcher(state.root)
    state.clock.advance(6 * 3600)  # 02:00 ET, inside the initial retry window.
    imports, plans, submissions, probes = [], [], [], []
    if error_count:
        nightly_trigger.write_state(state.root, nightly_trigger.TriggerReceipt(
            as_of=SESSION, status="error", error_count=error_count,
            snapshot_attempt=snapshot_attempt))

    def ensure(root, as_of, clock, attempt):
        """Reuse the synthetic import pin and record its unchanged attempt key."""
        imports.append(attempt)
        return "ready", state.base.snapshot_id

    def plan(root, as_of, tickers, context, clock, **kwargs):
        """Accept the plan only after real capture makes both price series readable."""
        snapshot = kwargs["expected_shadow_snapshot_id"]
        for ticker in ("NEW", "SPY"):
            _assert_prices(state, snapshot, ticker)
        plans.append(snapshot)
        return "resumed-plan"

    def probe(as_of, tickers):
        """Record finality checks without contacting a provider."""
        probes.append((as_of, tuple(tickers)))
        return True, "final"

    def tick():
        """Run a trigger tick while preserving the real price-preparation path."""
        return nightly_trigger.run_trigger(
            state.root, SESSION, tickers=("NEW",), context_tickers=("NEW",),
            clock=state.clock, provider=probe,
            ensure_snapshot_fn=ensure, submit_fn=lambda *args: submissions.append(args),
            serve_fn=lambda *args: "completed")

    monkeypatch.setattr(nightly_trigger, "_default_plan", plan)
    lock = SupervisorLock(state.root / "data" / "operations" / "supervisor.lock")
    assert lock.acquire()
    retries = nightly_trigger.MAX_CONSECUTIVE_ERRORS + 1
    try:
        for index in range(retries):
            if index == 1:
                state.clock.advance(6 * 3600)  # 08:00 ET, after the window closes.
            busy = tick()
            assert busy.status == "busy_supervisor" and busy.plan_ref is None
            assert busy.error_count == error_count
            assert busy.snapshot_attempt == snapshot_attempt
            assert nightly_trigger.load_state(state.root, SESSION) == busy
            assert lock.held and plans == [] and submissions == []
        assert calls == ["NEW", "SPY"]
        assert state.conn.execute("SELECT COUNT(*) FROM data_price_captures").fetchone()[0] == 0
        assert state.conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'"
                                  ).fetchone()[0] == state.base.snapshot_id
        assert not state.root.joinpath(*nightly_trigger.STATE_DIR,
                                       SESSION + ".price_refresh.json").exists()
    finally:
        lock.release()

    recovered = tick()
    assert recovered.status == "completed" and recovered.plan_ref == "resumed-plan"
    assert recovered.snapshot_attempt == snapshot_attempt
    assert nightly_trigger.load_state(state.root, SESSION) == recovered
    assert imports == [snapshot_attempt] * (retries + 1)
    assert len(plans) == len(submissions) == 1
    assert probes == ([] if error_count else [(SESSION, ("NEW",))])
    assert calls == ["NEW", "SPY"]
    assert _read_report(state.root)["capture"]["result_snapshot_id"] == plans[0]
    assert plans[0] != state.base.snapshot_id


def test_completed_import_clears_timeout_budget_before_capture_lock_retry(
        shadow_root, install_fetcher, monkeypatch):
    """Capture busy preserves setup errors, not a completed import's timeout streak."""
    state = shadow_root
    calls = install_fetcher(state.root)
    state.clock.advance(12 * 3600)  # Resume at 08:00 ET, after the retry window.
    nightly_trigger.write_state(state.root, nightly_trigger.TriggerReceipt(
        as_of=SESSION, status="timed_out",
        error_count=nightly_trigger.MAX_CONSECUTIVE_ERRORS - 1, snapshot_attempt=1))
    imports, plans = [], []

    def ensure(root, as_of, clock, attempt):
        """Finish the previously timed-out import under the same attempt key."""
        imports.append(attempt)
        return "ready", state.base.snapshot_id

    def refused_plan(root, as_of, tickers, context, clock, **kwargs):
        """Raise the first setup error only after real capture succeeds."""
        snapshot = kwargs["expected_shadow_snapshot_id"]
        _assert_prices(state, snapshot, "NEW")
        plans.append(snapshot)
        raise OSError("synthetic plan write failure")

    def tick():
        """Resume through real price preparation without probing or submitting."""
        return nightly_trigger.run_trigger(
            state.root, SESSION, tickers=("NEW",), context_tickers=("NEW",),
            clock=state.clock, provider=lambda *args: pytest.fail("a resume re-probed"),
            ensure_snapshot_fn=ensure,
            submit_fn=lambda *args: pytest.fail("a refused plan submitted"),
            serve_fn=lambda *args: pytest.fail("a refused plan served"))

    monkeypatch.setattr(nightly_trigger, "_default_plan", refused_plan)
    lock = SupervisorLock(state.root / "data" / "operations" / "supervisor.lock")
    assert lock.acquire()
    try:
        busy = tick()
        assert busy.status == "busy_supervisor" and busy.error_count == 0
        assert busy.snapshot_attempt == 1 and busy.plan_ref is None
        assert nightly_trigger.load_state(state.root, SESSION) == busy
        assert plans == []
    finally:
        lock.release()

    error = tick()
    assert error.status == "error" and error.error_count == 1
    assert error.snapshot_attempt == 1 and error.plan_ref is None
    assert nightly_trigger.load_state(state.root, SESSION) == error
    assert imports == [1, 1] and len(plans) == 1
    assert calls == ["NEW", "SPY"]


@pytest.mark.parametrize("error_count", [0, nightly_trigger.MAX_CONSECUTIVE_ERRORS - 1])
@pytest.mark.parametrize("error,attempt_bump", [
    (fail("RESOURCE_UNAVAILABLE", "unclassified resource contention"), 0),
    (fail("RESOURCE_UNAVAILABLE", "another resource is unavailable",
          details={"resource": "another.lock"}), 0),
    (fail("INPUT_CHANGED", "the input changed", details={"resource": "supervisor.lock"}), 1),
    (OSError("synthetic catalog failure"), 0),
])
def test_other_setup_errors_consume_budget_after_capture_contention(
        tmp_path, error_count, error, attempt_bump):
    """Unrelated failures still spend retry budget without a busy receipt resetting it."""
    from tests.ops_support import FakeClock

    prior = nightly_trigger.TriggerReceipt(
        as_of=SESSION, status="busy_supervisor", error_count=error_count, snapshot_attempt=1)

    def refused_plan(*args, **kwargs):
        """Raise a specific non-contention setup failure through the real handler."""
        raise error

    receipt = nightly_trigger._submit_plan(
        tmp_path, SESSION, tickers=("NEW",), context_tickers=("NEW",), clock=FakeClock(),
        plan_fn=refused_plan, submit_fn=lambda *args: pytest.fail("a refused plan submitted"),
        serve_fn=lambda *args: pytest.fail("a refused plan served"),
        ensure_snapshot_fn=lambda *args: ("ready", "synthetic-snapshot"),
        full_run=True, prior=prior, plan_ref=None)

    assert receipt.status == ("error" if error_count == 0 else "failed_setup")
    assert receipt.error_count == error_count + 1
    assert receipt.snapshot_attempt == 1 + attempt_bump
    assert receipt.plan_ref is None
    assert nightly_trigger.load_state(tmp_path, SESSION) == receipt


def test_report_replace_failure_leaves_capture_and_prior_report(
        shadow_root, install_fetcher, monkeypatch):
    """A failed report replace leaves committed prices and the prior report intact."""
    state = shadow_root
    install_fetcher(state.root)
    report_path = state.root.joinpath(*nightly_trigger.STATE_DIR, SESSION + ".price_refresh.json")
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"prior": true}')
    replace = nightly_trigger.os.replace

    def fail_report(source, destination):
        """Fail only the price report replacement after successful capture."""
        if Path(destination) == report_path:
            raise OSError("injected report replacement failure")
        return replace(source, destination)

    monkeypatch.setattr(nightly_trigger.os, "replace", fail_report)
    with pytest.raises(OSError):
        nightly_trigger._refresh_plan_prices(
            state.root, SESSION, ("NEW",), state.clock, state.base.snapshot_id)

    snapshot_id = state.conn.execute(
        "SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'").fetchone()[0]
    assert snapshot_id != state.base.snapshot_id
    _assert_prices(state, snapshot_id, "NEW")
    assert json.loads(report_path.read_text()) == {"prior": True}


def test_missing_calendar_dependency_keeps_existing_refusal(shadow_root, install_fetcher):
    """Missing required calendar prices refuse without advancing the shadow head."""
    state = shadow_root
    calls = install_fetcher(state.root, failures={"SPY": 404})
    with pytest.raises(OpsError) as error:
        nightly_trigger._refresh_plan_prices(
            state.root, SESSION, ("NEW",), state.clock, state.base.snapshot_id)
    assert error.value.code == "SOURCE_NOT_FOUND"
    assert calls == ["NEW", "SPY"]
    assert state.conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'"
                              ).fetchone()[0] == state.base.snapshot_id


def test_real_nightly_planner_pins_refreshed_capture_before_input_manifest(
        case, install_fetcher, monkeypatch):
    """The real planner pins captured watchlist prices before reading its manifest."""
    from engine.v2.ops import bootstrap, cli

    args = cli.parser().parse_args(_plan_files(case))
    calls = install_fetcher(case.tmp)
    open_catalog = bootstrap.open_catalog
    monkeypatch.setattr(bootstrap, "open_catalog", lambda path, **kwargs:
                        open_catalog(case.tmp / "catalog.sqlite", **kwargs))
    monkeypatch.setattr(nightly_trigger, "_ops_root", lambda root: case.root)
    monkeypatch.setattr(nightly_trigger, "_qualification_path",
                        lambda root, name: args.expected_population)
    monkeypatch.setattr(nightly_trigger, "_derive_years", lambda as_of: (2020, 2021))
    captured = []

    def barrier_manifest(*args, **kwargs):
        """Verify the price report already exists at the input-manifest boundary."""
        report_path = case.tmp.joinpath(*nightly_trigger.STATE_DIR,
                                        PLAN_SESSION + ".price_refresh.json")
        report = json.loads(report_path.read_text())
        assert report["plan"]["daily"] == ["AAA", "BBB", "SPY"]
        assert report["missing_price_history"] == []
        captured.append(report["capture"]["result_snapshot_id"])
        return cli.parser().parse_args(_plan_files(case)).input_manifest

    monkeypatch.setattr(nightly_trigger, "_capture_input_manifest", barrier_manifest)
    plan_ref = nightly_trigger._prepare_default_plan(
        case.tmp, PLAN_SESSION, ("AAA", "BBB"), ("AAA", "BBB"), case.clock,
        expected_shadow_snapshot_id=case.snap.snapshot_id)
    saved = json.loads(case.store.read_verified(cli.artifact(case.conn, case.store, plan_ref)))
    assert calls == ["AAA", "BBB", "SPY"]
    assert captured == [saved["snapshot_inputs"]["snapshot_id"]]
    assert captured[0] != case.snap.snapshot_id
    assert saved["tickers"] == ["AAA", "BBB"]
    snapshot = Repository(case.conn, case.store).resolve(captured[0])
    assert "price_history" in snapshot.table_versions
