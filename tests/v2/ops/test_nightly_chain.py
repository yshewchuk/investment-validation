"""Real-file, real-catalog tests for the nightly snapshot chain (slice 4 of #564).

Receipts and session state are the real ones. The one seeded layer is the "snapshot store":
each step commits by atomically writing a marker file named for its predecessor, which is
what a crash-after-commit leaves behind; the import step uses a real catalog job whose
terminal state is set with SQL."""
from __future__ import annotations

import hashlib
import os

import pytest

from engine.v2.ops import nightly_chain as nc
from engine.v2.ops import nightly_session as ns
from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly_receipts import Effect
from engine.v2.ops.submission import job_id_for, submit
from tests.ops_support import POLICY, REGISTRY, catalog, request

IDENT = ns.SessionIdentity("2026-10-08", "shadow", "sel-1", "cat-1")


@pytest.fixture
def root(tmp_path):
    ns.ensure_session(tmp_path, IDENT)
    ns.mark_started(tmp_path, IDENT, 1)
    return tmp_path


class Store:
    """Markers under ``dir``; ``after`` is the snapshot a step commits from a predecessor."""

    def __init__(self, directory):
        self.dir = directory
        self.calls = []

    def after(self, name, pred):
        return f"{name}@{pred}"

    def marker(self, name, pred):
        return self.dir / f"{name}.{hashlib.sha256(pred.encode()).hexdigest()[:12]}"

    def commit(self, name, pred):
        path = self.marker(name, pred)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.after(name, pred))
        os.replace(tmp, path)

    def step(self, name, *, crash_after_commit=False, commit=True):
        def effect(pred):
            data = self.after(name, pred).encode()
            return Effect("artifact", str(self.marker(name, pred)),
                          hashlib.sha256(data).hexdigest())

        def run(pred):
            self.calls.append((name, pred))
            if commit:
                self.commit(name, pred)
            if crash_after_commit:
                raise RuntimeError("crash after commit")

        def successor(pred):
            path = self.marker(name, pred)
            return path.read_text() if path.exists() else None
        return nc.ChainStep(name, effect, run, successor)


def _steps(store, **per_step):
    return [store.step(name, **per_step.get(name, {})) for name in ("import", "moves", "capture")]


def _code(excinfo) -> str:
    return excinfo.value.code


def test_runs_each_step_on_its_predecessors_commit(root, tmp_path):
    store = Store(tmp_path)
    result = nc.run_snapshot_chain(root, IDENT, 1, "s0", _steps(store))
    assert store.calls == [("import", "s0"), ("moves", "import@s0"),
                           ("capture", "moves@import@s0")]
    assert result.final_snapshot == "capture@moves@import@s0"
    assert result.executed == ("import", "moves", "capture") and result.skipped == ()


def test_a_second_run_repeats_nothing(root, tmp_path):
    store = Store(tmp_path)
    first = nc.run_snapshot_chain(root, IDENT, 1, "s0", _steps(store))
    store.calls.clear()
    again = nc.run_snapshot_chain(root, IDENT, 1, "s0", _steps(store))
    assert store.calls == []
    assert again.final_snapshot == first.final_snapshot
    assert again.executed == () and again.skipped == ("import", "moves", "capture")


def test_a_crash_after_commit_is_adopted_not_repeated(root, tmp_path):
    store = Store(tmp_path)
    with pytest.raises(RuntimeError):
        nc.run_snapshot_chain(root, IDENT, 1, "s0",
                              _steps(store, moves={"crash_after_commit": True}))
    store.calls.clear()
    result = nc.run_snapshot_chain(root, IDENT, 1, "s0", _steps(store))
    assert store.calls == [("capture", "moves@import@s0")]
    assert result.skipped == ("import", "moves") and result.executed == ("capture",)


def test_a_step_that_commits_nothing_stops_the_chain(root, tmp_path):
    store = Store(tmp_path)
    with pytest.raises(OpsError) as exc:
        nc.run_snapshot_chain(root, IDENT, 1, "s0", _steps(store, moves={"commit": False}))
    assert _code(exc) == "DEPENDENCY_FAILED" and exc.value.problem.details == {"step": "moves"}
    assert [name for name, _ in store.calls] == ["import", "moves"]


def test_a_completed_step_whose_commit_vanished_is_integrity_failed(root, tmp_path):
    store = Store(tmp_path)
    nc.run_snapshot_chain(root, IDENT, 1, "s0", _steps(store))
    store.marker("import", "s0").unlink()
    with pytest.raises(OpsError) as exc:
        nc.run_snapshot_chain(root, IDENT, 1, "s0", _steps(store))
    assert _code(exc) == "INTEGRITY_FAILED"


def test_a_changed_start_snapshot_conflicts_with_the_recorded_step(root, tmp_path):
    store = Store(tmp_path)
    nc.run_snapshot_chain(root, IDENT, 1, "s0", _steps(store))
    with pytest.raises(OpsError) as exc:
        nc.run_snapshot_chain(root, IDENT, 1, "s1", _steps(store))
    assert _code(exc) == "IDEMPOTENCY_CONFLICT"


@pytest.mark.parametrize("start,names", [("", ["a"]), (" ", ["a"]), ("s0", []), ("s0", ["a", "a"])])
def test_malformed_chain_is_invalid_request(root, tmp_path, start, names):
    store = Store(tmp_path)
    with pytest.raises(OpsError) as exc:
        nc.run_snapshot_chain(root, IDENT, 1, start, [store.step(n) for n in names])
    assert _code(exc) == "INVALID_REQUEST"


def test_a_catalog_job_import_step_completes_only_when_the_job_succeeded(root, tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = Store(tmp_path)
    job_id = job_id_for("shadow", "import-1")

    def run(pred):
        store.calls.append(("import", pred))
        submit(conn, REGISTRY, POLICY, request("import-1"), clock=clock)

    step = nc.ChainStep("import", lambda pred: Effect("catalog_job", job_id), run,
                        lambda pred: "imported" if store.calls else None)
    with pytest.raises(OpsError) as exc:  # job submitted but still live: not complete
        nc.run_snapshot_chain(root, IDENT, 1, "s0", [step], conn=conn)
    assert _code(exc) == "INVALID_REQUEST"
    conn.execute("UPDATE jobs SET state = 'succeeded' WHERE job_id = ?", (job_id,))
    result = nc.run_snapshot_chain(root, IDENT, 1, "s0", [step], conn=conn)
    assert result.final_snapshot == "imported" and result.skipped == ("import",)
    assert store.calls == [("import", "s0")]
    conn.close()


def test_a_refresh_step_passes_its_predecessor_through_and_is_not_repeated(root, tmp_path):
    report = tmp_path / "refresh.report"
    calls = []

    def run(pred):
        calls.append(pred)
        report.write_text("done")

    refresh = nc.ChainStep("refresh", lambda pred: Effect("external", "refresh"), run,
                           lambda pred: pred if report.exists() else None)
    first = nc.run_snapshot_chain(root, IDENT, 1, "s0", [refresh])
    again = nc.run_snapshot_chain(root, IDENT, 1, "s0", [refresh])
    assert first.final_snapshot == again.final_snapshot == "s0"
    assert calls == ["s0"] and first.executed == ("refresh",) and again.skipped == ("refresh",)