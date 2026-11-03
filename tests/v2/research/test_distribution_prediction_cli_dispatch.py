"""Real operator routing with a narrow runner interface double, not fit proof."""
import json
import sqlite3
import sys
from types import ModuleType, SimpleNamespace

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import ArtifactStore, SystemClock, from_document, to_document
from experiments import distribution_prediction_cli as cli
from tests.v2.research.distribution_prediction_operator_support import request
from tests.v2.research.test_distribution_prediction_cli_errors import operator as operator


@pytest.fixture
def dispatched(operator, monkeypatch):
    calls, own, source = [], object(), object()
    def native(conn, store, spec, **kwargs):
        assert kwargs == {"max_evidence_bytes": cli.MAX_NATIVE_EVIDENCE_BYTES}
        calls.append("native:" + spec.experiment_id)
        return source if spec.experiment_id == "synthetic-source" else own
    def trials(conn, store, repository, actual_own, actual_source, source_request, ref, recipe, caps):
        assert actual_own is own and actual_source is source
        assert source_request == {} and isinstance(ref, ArtifactRef) and recipe["horizons"] == [5, 3, 1]
        calls.append("trials")
        return SimpleNamespace(conn=conn, store=store)
    class PredictionRun:
        def __init__(self, trials, policy):
            assert policy == request()
            self.trials = trials
            calls.append("runner")

        def preflight(self):
            calls.append("preflight")
            return {"action": "preflight"}

        def seal(self, *, clock):
            assert isinstance(clock, SystemClock)
            calls.append("seal")
            return {"action": "seal"}

        def record(self, slot):
            assert slot == "completed"
            calls.append("completed")
            row = self.trials.conn.execute("SELECT ref FROM synthetic_completion").fetchone()
            return json.loads(row[0]) if row else None

        def collect(self):
            calls.append("collect")
            row = self.trials.conn.execute("SELECT ref FROM synthetic_completion").fetchone()
            ref = from_document(ArtifactRef, json.loads(row[0]))
            return json.loads(self.trials.store.read_verified(ref))

        def _run_contained(self, clock, began):
            assert isinstance(clock, SystemClock) and began > 0
            calls.append("execute")
            return {"action": "run"}
    module = ModuleType("experiments.distribution_prediction_run")
    module.PredictionRun = PredictionRun
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cli, "read_native_registration", native)
    monkeypatch.setattr(cli, "DistributionTrials", trials)
    with sqlite3.connect(operator.paths["catalog"]) as conn:
        conn.execute("CREATE TABLE synthetic_completion(ref TEXT)")
    conn.close()
    operator.calls.clear()
    return SimpleNamespace(operator=operator, calls=calls, runner=PredictionRun)


@pytest.mark.parametrize("action", ["prepare", "register", "preflight", "seal", "run"])
def test_each_action_invokes_only_its_prediction_lifecycle_method(dispatched, monkeypatch, capsys, action):
    operator, calls = dispatched.operator, dispatched.calls
    operator.argv[0] = action
    def prepare(*args):
        calls.append("prepare")
        return object(), args[3], {"action": "prepare"}
    monkeypatch.setattr(cli, "_prepare", prepare)
    monkeypatch.setattr(cli, "reserve_native_registration", lambda *args, **kwargs: calls.append("reserve"))
    # Routing is tested over an existing ordinary SQLite catalog; native schema
    # migration and actual registration/runner authority have separate tests.
    monkeypatch.setattr(cli, "connect", lambda path, **kwargs: sqlite3.connect(path, isolation_level=None))
    assert cli.main(operator.argv) == 0
    result = json.loads(capsys.readouterr().out)
    native = ["native:synthetic-source", "native:synthetic-prediction", "trials", "runner"]
    if action in ("prepare", "register"):
        assert result == {"action": "prepare", "registered": action == "register"}
        assert calls == ["native:synthetic-source", "prepare"] + (["reserve"] if action == "register" else [])
    else:
        assert result == {"action": action}
        assert calls == (native + [action] if action != "run" else native + ["completed"] + native + ["execute"])


@pytest.mark.parametrize("mode", ["delete", "wal"])
@pytest.mark.parametrize("action", ["run", "collect", "report"])
def test_completed_operator_reads_real_catalog_and_cas_without_mutation(dispatched, monkeypatch, capsys, mode, action):
    operator, calls = dispatched.operator, dispatched.calls
    catalog, store = operator.paths["catalog"], ArtifactStore(operator.paths["store-root"])
    expected = {"schema_version": "synthetic_prediction_completion.v1", "final_read_authority": "none"}
    ref = store.publish_bytes(json.dumps(expected).encode(), schema_ref=expected["schema_version"])
    writer = sqlite3.connect(catalog, isolation_level=None)
    assert writer.execute("PRAGMA journal_mode=" + mode).fetchone()[0] == mode
    if mode == "wal":
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        checkpointed = catalog.read_bytes()
    writer.execute("INSERT INTO synthetic_completion VALUES(?)", (json.dumps(to_document(ref)),))
    if mode == "delete":
        writer.close()
    else:
        assert catalog.read_bytes() == checkpointed
        assert catalog.with_name(catalog.name + "-wal").stat().st_size > 0
    before = {p.name: p.read_bytes() for p in catalog.parent.glob("catalog.sqlite*") if not p.name.endswith("-shm")}
    objects = {str(p.relative_to(store.root)): p.read_bytes() for p in store.root.rglob("*") if p.is_file()}
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
    statements, opened = [], []
    def authorize(operation, *args):
        statements.append(operation)
        return sqlite3.SQLITE_OK if operation in allowed else sqlite3.SQLITE_DENY
    original_connect = sqlite3.connect
    def readonly(target, **kwargs):
        assert target.endswith("?mode=ro") and "immutable" not in target and kwargs["uri"] is True
        conn = original_connect(target, **kwargs)
        conn.set_authorizer(authorize)
        opened.append(conn)
        return conn
    def forbidden(*args, **kwargs):
        pytest.fail("completed prediction operator attempted writing or execution")
    monkeypatch.setattr(cli.sqlite3, "connect", readonly)
    monkeypatch.setattr(cli, "connect", forbidden)
    monkeypatch.setattr(ArtifactStore, "_commit", forbidden)
    monkeypatch.setattr(dispatched.runner, "_run_contained", forbidden)
    operator.argv[0] = action
    try:
        assert cli.main(operator.argv) == 0
        assert json.loads(capsys.readouterr().out) == expected
        assert calls[-1] == "collect" and len(opened) == 1 and statements and set(statements) <= allowed
        assert {p.name: p.read_bytes() for p in catalog.parent.glob("catalog.sqlite*") if not p.name.endswith("-shm")} == before
        assert {str(p.relative_to(store.root)): p.read_bytes() for p in store.root.rglob("*") if p.is_file()} == objects
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            opened[0].execute("SELECT 1")
    finally:
        if mode == "wal":
            writer.close()
