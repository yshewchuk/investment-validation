"""Real artifact and SQLite tests for generation-aware repair planning (#564 slice 6)."""
from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from engine.v2.ops import nightly_repair as nr
from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly_receipts import Effect
from engine.v2.ops.nightly_session import Generation


def _digest(char: str) -> str:
    return hashlib.sha256(char.encode()).hexdigest()


def _gen(number: int, invalidation=()) -> Generation:
    return Generation(number, f"run-{number}", "initial" if number == 1 else "rerun", "started", invalidation)


def _graph(*, score_request: str = "a", score_impl: str = "i",
           decision_impl: str = "i") -> tuple[nr.StageSpec, ...]:
    return (
        nr.StageSpec("import", (), _digest("a"), _digest("i")),
        nr.StageSpec("score", ("import",), _digest(score_request), _digest(score_impl)),
        nr.StageSpec("decision_commit", ("score",), _digest("d"), _digest(decision_impl)),
        nr.StageSpec("publication", ("decision_commit",), _digest("p"), _digest("i")),
        nr.StageSpec("audit", ("import",), _digest("z"), _digest("i")),
    )


def _outputs(stages, directory, run_id: str) -> tuple[nr.OutputBinding, ...]:
    outputs = []
    for stage in stages:
        path = directory / f"{run_id}-{stage.name}.bin"
        data = f"committed:{run_id}:{stage.name}".encode()
        path.write_bytes(data)
        outputs.append(nr.OutputBinding(
            stage.name, run_id, stage.dependencies, stage.request_digest,
            stage.implementation_digest,
            Effect("artifact", str(path), hashlib.sha256(data).hexdigest())))
    return tuple(outputs)


def _plan(generation, stages, outputs=(), *, previous_run_id=None, invalidation=(), conn=None):
    return nr.build_repair_plan(generation, stages, previous=outputs,
                                previous_run_id=previous_run_id,
                                invalidation=invalidation, conn=conn)


def _code(excinfo) -> str:
    return excinfo.value.code


def test_first_generation_executes_every_stage(tmp_path):
    graph = _graph()
    plan = _plan(_gen(1), graph)
    assert (plan.run_id, plan.execute, plan.inherited) == (
        "run-1", tuple(stage.name for stage in graph), ())
    assert len(plan.plan_id) == 64


def test_plan_identity_changes_for_each_generation_even_with_same_graph():
    graph = _graph()
    first = _plan(_gen(1), graph)
    second = _plan(_gen(2), graph)
    assert first.plan_id != second.plan_id
    assert first.plan_id == _plan(_gen(1), graph).plan_id


def test_plan_identity_is_canonical_for_independent_graph_order():
    graph = _graph()
    reordered = (graph[0], graph[4], graph[1], graph[2], graph[3])
    assert _plan(_gen(2), graph).plan_id == _plan(_gen(2), reordered).plan_id


def test_dependency_order_does_not_change_identity_or_block_inheritance(tmp_path):
    left = nr.StageSpec("left", (), _digest("l"), _digest("i"))
    right = nr.StageSpec("right", (), _digest("r"), _digest("i"))
    join = nr.StageSpec("join", ("left", "right"), _digest("j"), _digest("i"))
    old = (left, right, join)
    current = (left, right, replace(join, dependencies=("right", "left")))
    prior = _outputs(old, tmp_path, "run-1")
    original = _plan(_gen(2), old)
    repaired = _plan(_gen(2), current, prior, previous_run_id="run-1")
    assert repaired.plan_id == original.plan_id
    assert repaired.execute == () and tuple(x.stage for x in repaired.inherited) == (
        "left", "right", "join")
    invalidated = _plan(_gen(2), current, prior, previous_run_id="run-1", invalidation=("right",))
    assert invalidated.execute == ("right", "join")
    assert tuple(x.stage for x in invalidated.inherited) == ("left",)


def test_outputs_for_stages_removed_from_current_graph_are_dropped(tmp_path):
    graph = _graph()
    obsolete = nr.StageSpec("obsolete", ("import",), _digest("o"), _digest("i"))
    prior = _outputs(graph + (obsolete,), tmp_path, "run-1")
    plan = _plan(_gen(2), graph, prior, previous_run_id="run-1")
    assert plan.execute == ()
    assert tuple(item.stage for item in plan.inherited) == tuple(x.name for x in graph)
    stale = _plan(_gen(2, ("obsolete",)), graph, prior, previous_run_id="run-1")
    assert stale.execute == () and stale.inherited == prior[:len(graph)]


def test_matching_verified_outputs_inherit_without_rewriting_source_bindings(tmp_path):
    graph = _graph()
    prior = _outputs(graph, tmp_path, "run-1")
    plan = _plan(_gen(2), graph, prior, previous_run_id="run-1")
    assert plan.execute == ()
    assert plan.inherited == prior
    assert {binding.source_run_id for binding in plan.inherited} == {"run-1"}
    assert all(binding.succeeded for binding in plan.inherited)
    with pytest.raises(FrozenInstanceError):
        plan.inherited[0].source_run_id = "run-2"


def test_changed_request_digest_invalidates_its_dependent_cone_only(tmp_path):
    old = _graph()
    prior = _outputs(old, tmp_path, "run-1")
    current = _graph(score_request="changed")
    plan = _plan(_gen(2), current, prior, previous_run_id="run-1")
    assert plan.execute == ("score", "decision_commit", "publication")
    assert tuple(item.stage for item in plan.inherited) == ("import", "audit")


def test_changed_implementation_closure_invalidates_dependents(tmp_path):
    old = _graph()
    prior = _outputs(old, tmp_path, "run-1")
    current = _graph(score_impl="new-closure")
    plan = _plan(_gen(2), current, prior, previous_run_id="run-1")
    assert plan.execute == ("score", "decision_commit", "publication")
    assert tuple(item.stage for item in plan.inherited) == ("import", "audit")


def test_changed_decision_implementation_invalidates_publication(tmp_path):
    old = _graph()
    prior = _outputs(old, tmp_path, "run-1")
    current = _graph(decision_impl="new-closure")
    plan = _plan(_gen(2), current, prior, previous_run_id="run-1")
    assert plan.execute == ("decision_commit", "publication")
    assert tuple(item.stage for item in plan.inherited) == ("import", "score", "audit")


def test_explicit_invalidation_propagates_only_to_dependents(tmp_path):
    graph = _graph()
    prior = _outputs(graph, tmp_path, "run-1")
    plan = _plan(_gen(2, ("score",)), graph, prior, previous_run_id="run-1")
    assert plan.execute == ("score", "decision_commit", "publication")
    assert tuple(item.stage for item in plan.inherited) == ("import", "audit")


def test_dependency_list_change_invalidates_the_stage_and_its_dependents(tmp_path):
    old = _graph()
    prior = _outputs(old, tmp_path, "run-1")
    current = old[:3] + (nr.StageSpec("publication", ("score",), _digest("p"),
                                      _digest("i")), old[4])
    plan = _plan(_gen(2), current, prior, previous_run_id="run-1")
    assert plan.execute == ("publication",)


@pytest.mark.parametrize("kind", ("tampered", "missing", "oversized", "fifo"))
def test_unprovable_artifact_invalidates_its_stage_and_descendants(tmp_path, kind):
    graph = _graph()
    prior = _outputs(graph, tmp_path, "run-1")
    path = Path(prior[0].effect.ref)
    if kind == "tampered":
        path.write_bytes(b"tampered")
    elif kind == "missing":
        path.unlink()
    elif kind == "oversized":
        with path.open("wb") as artifact:
            artifact.truncate(nr._MAX_ARTIFACT_BYTES + 1)
    else:
        path.unlink()
        os.mkfifo(path)
    plan = _plan(_gen(2), graph, prior, previous_run_id="run-1")
    assert plan.execute == ("import", "score", "decision_commit", "publication", "audit")
    assert plan.inherited == ()


def test_incomplete_prior_receipt_is_not_inherited(tmp_path):
    graph = _graph()
    prior = list(_outputs(graph, tmp_path, "run-1"))
    prior[1] = replace(prior[1], succeeded=False)
    plan = _plan(_gen(2), graph, tuple(prior), previous_run_id="run-1")
    assert plan.execute == ("score", "decision_commit", "publication")
    assert tuple(item.stage for item in plan.inherited) == ("import", "audit")


def test_external_effect_is_never_inherited(tmp_path):
    graph = _graph()
    prior = list(_outputs(graph, tmp_path, "run-1"))
    prior[1] = nr.OutputBinding("score", "run-1", graph[1].dependencies,
                                graph[1].request_digest, graph[1].implementation_digest,
                                Effect("external", "refresh"))
    plan = _plan(_gen(2), graph, tuple(prior), previous_run_id="run-1")
    assert plan.execute == ("score", "decision_commit", "publication")
    assert tuple(item.stage for item in plan.inherited) == ("import", "audit")


def test_succeeded_catalog_job_can_be_inherited_from_real_sqlite_state():
    stage = nr.StageSpec("import", (), _digest("request"), _digest("code"))
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE jobs (job_id TEXT PRIMARY KEY, state TEXT NOT NULL)")
    conn.execute("INSERT INTO jobs VALUES (?, ?)", ("job-1", "succeeded"))
    binding = nr.OutputBinding("import", "run-1", (), stage.request_digest,
                               stage.implementation_digest, Effect("catalog_job", "job-1"))
    plan = _plan(_gen(2), (stage,), (binding,), previous_run_id="run-1", conn=conn)
    assert plan.execute == () and plan.inherited == (binding,)
    conn.close()


@pytest.mark.parametrize("state, schema", [("queued", "jobs"), ("failed", "jobs"),
                                           (None, "jobs"), ("succeeded", "none"),
                                           ("succeeded", "wrong")])
def test_unproven_catalog_job_is_recomputed(state, schema):
    stage = nr.StageSpec("import", (), _digest("request"), _digest("code"))
    conn = None if schema == "none" else sqlite3.connect(":memory:")
    if conn is not None:
        conn.execute("CREATE TABLE jobs (job_id TEXT PRIMARY KEY, state TEXT NOT NULL)" if schema == "jobs" else "CREATE TABLE unrelated (value TEXT)")
        if schema == "jobs" and state is not None:
            conn.execute("INSERT INTO jobs VALUES (?, ?)", ("job-1", state))
    binding = nr.OutputBinding("import", "run-1", (), stage.request_digest,
                               stage.implementation_digest, Effect("catalog_job", "job-1"))
    plan = _plan(_gen(2), (stage,), (binding,), previous_run_id="run-1", conn=conn)
    assert plan.execute == ("import",) and plan.inherited == ()
    if conn is not None:
        conn.close()


def test_invalid_generation_and_unknown_invalidation_are_refused():
    graph = _graph()
    allocated = Generation(2, "run-2", "rerun", "allocated", ())
    with pytest.raises(OpsError) as exc:
        _plan(allocated, graph)
    assert _code(exc) == "INVALID_REQUEST"
    for malformed in (Generation(0, "run-0", "rerun", "started", ()),
                      Generation(2, "", "rerun", "started", ())):
        with pytest.raises(OpsError) as exc:
            _plan(malformed, graph)
        assert _code(exc) == "INVALID_REQUEST"
    with pytest.raises(OpsError) as exc:
        _plan(_gen(2), graph, invalidation=("missing",))
    assert _code(exc) == "INVALID_REQUEST"
    with pytest.raises(OpsError) as exc:
        _plan(_gen(2), graph, invalidation=(" ",))
    assert _code(exc) == "INVALID_REQUEST"


@pytest.mark.parametrize("source_id,prior_id", [("run-0", "run-1"), (None, "run-1"),
                                               ("", "run-1"), ("run-2", "run-2")])
def test_prior_binding_needs_the_caller_selected_source_generation(tmp_path, source_id, prior_id):
    graph = _graph()
    prior = _outputs(graph, tmp_path, prior_id)
    with pytest.raises(OpsError) as exc:
        _plan(_gen(2), graph, prior, previous_run_id=source_id)
    assert _code(exc) == "INVALID_REQUEST"


def test_non_sequence_prior_bindings_are_invalid_request():
    with pytest.raises(OpsError) as exc:
        _plan(_gen(2), _graph(), object(), previous_run_id="run-1")
    assert _code(exc) == "INVALID_REQUEST"


def test_malformed_or_non_topological_graph_is_refused():
    graph = _graph()
    malformed = ((), (nr.StageSpec("score", ("import",), _digest("a"), _digest("i")),),
                 graph + (graph[0],),
                 (nr.StageSpec("self", ("self",), _digest("a"), _digest("i")),),
                 (object(),))
    for candidate in malformed:
        with pytest.raises(OpsError) as exc:
            _plan(_gen(2), candidate)
        assert _code(exc) == "INVALID_REQUEST"


@pytest.mark.parametrize("changes", [
    {"stage": 1}, {"source_run_id": 1},
    {"effect": Effect("artifact", "relative", "a" * 64)},
    {"effect": Effect("artifact", "/absolute/output", "not-hex")},
    {"effect": Effect("catalog_job", "bad\0id")}, {"effect": Effect("external", "")},
    {"dependencies": (1,)},
])
def test_malformed_prior_binding_or_effect_is_invalid_request(tmp_path, changes):
    graph = _graph()
    binding = replace(_outputs(graph, tmp_path, "run-1")[0], **changes)
    with pytest.raises(OpsError) as exc:
        _plan(_gen(2), graph, (binding,), previous_run_id="run-1")
    assert _code(exc) == "INVALID_REQUEST"
    valid = _outputs(graph, tmp_path, "run-1")[0]
    with pytest.raises(OpsError) as exc:
        _plan(_gen(2), graph, (valid, valid), previous_run_id="run-1")
    assert _code(exc) == "INVALID_REQUEST"


def test_repair_planning_is_read_only_for_decisions_publications_and_artifacts(tmp_path):
    graph = _graph()
    prior = _outputs(graph, tmp_path, "run-1")
    paths = tuple(Path(item.effect.ref) for item in prior)
    files = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    conn = sqlite3.connect(":memory:")
    conn.executescript("CREATE TABLE decisions (id TEXT, value TEXT); CREATE TABLE publications (id TEXT, state TEXT); INSERT INTO decisions VALUES ('occ-1','first'); INSERT INTO publications VALUES ('release-1','published');")
    before = (conn.execute("SELECT * FROM decisions").fetchall(), conn.execute("SELECT * FROM publications").fetchall())
    denied = []
    allowed = (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION)
    conn.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in allowed else (denied.append(action) or sqlite3.SQLITE_DENY))
    plan = _plan(_gen(2), graph, prior, previous_run_id="run-1", invalidation=("score",), conn=conn)
    after = (conn.execute("SELECT * FROM decisions").fetchall(), conn.execute("SELECT * FROM publications").fetchall())
    assert plan.execute == ("score", "decision_commit", "publication")
    assert not denied and before == after
    assert files == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    conn.close()
