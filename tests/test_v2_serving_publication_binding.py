"""P3-1c: binding a Phase 3 projection candidate to the existing fenced ops
publisher -- rearchitecture phase-3 guide §5.4/§5.5/§6, §8 P3-1 step 3.

Exercises the full chain end to end: a real synthetic serving candidate
(``projections.build_candidate``, the same fixtures ``tests/test_v2_serving_
projections.py`` uses), a real synthetic ops catalog (the same low-level
``stage_release``/``publish_local`` technique ``tests/test_v2_ops_same_
session_replan.py`` uses for its own generation/rollback proofs), and a real
``engine.v2.serving.api`` app served over real HTTP resolving "current"
through the published pointer's bound ``projection_binding.json``.

No ``ui/``, no ``engine/v2/serving/legacy_bundle.py``, no real
``/root/phase2-shadow-ops`` data -- entirely synthetic, tmp-path scoped.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import Problem, PreviewRelease  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore, content_hash, to_document  # noqa: E402
from engine.v2.ops.effects_graph import publication_effect  # noqa: E402
from engine.v2.ops.errors import OpsError  # noqa: E402
from engine.v2.ops.input_bindings import resolve_and_record  # noqa: E402
from engine.v2.ops.publication import current as release_current  # noqa: E402
from engine.v2.ops.publication import publish_local, stage_release  # noqa: E402
from engine.v2.serving import projections  # noqa: E402
from engine.v2.serving.api import create_app  # noqa: E402
from tests.ops_support import catalog, enqueue_claim  # noqa: E402
from tests.test_v2_ops_effects_graph import FAKE_STORE_ROOT  # noqa: E402
from tests.test_v2_ops_effects_graph import REPO  # noqa: E402
from tests.test_v2_ops_effects_graph import _bundle_tar  # noqa: E402
from tests.test_v2_ops_effects_graph import _commit as _ops_commit  # noqa: E402
from tests.test_v2_ops_effects_graph import _open as _ops_open  # noqa: E402
from tests.test_v2_ops_effects_graph import _params as _ops_params  # noqa: E402
from tests.test_v2_ops_effects_graph import _publish as _ops_publish_json  # noqa: E402
from tests.test_v2_ops_effects_graph import _publish_raw as _ops_publish_raw  # noqa: E402
from tests.test_v2_ops_effects_graph import _seed_decisions  # noqa: E402
from tests.test_v2_ops_effects_graph import _submit_and_claim  # noqa: E402
from tests.test_v2_ops_effects_graph import _succeed_parent  # noqa: E402
from tests.test_v2_serving_api import _FORBIDDEN_MODULE_SUBSTRINGS, _get, _start, _stop  # noqa: E402
from tests.test_v2_serving_projections import (  # noqa: E402
    _bundle,
    _compact,
    _event_row,
    _events_snapshot,
    _preview_input,
    _row,
    _score_doc,
    _serving,
)

TOKEN = "test-token-binding-7a1f"

# --------------------------------------------------------------------------
# synthetic serving candidates: two distinct releases from one shared
# Phase-2 snapshot (AAA/2024-01-05, BBB/2024-01-05)
# --------------------------------------------------------------------------


def _releases(tmp_path):
    (tmp_path / "phase2").mkdir()
    conn, store, snap = _events_snapshot(tmp_path / "phase2", [
        _event_row("e1", "AAA", datetime(2024, 1, 5)),
        _event_row("e2", "BBB", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    row_a, row_b = _row(ticker="AAA"), _row(ticker="BBB")
    release_a = projections.build_candidate(
        _preview_input(), _score_doc(rows=[row_a]), _bundle(_compact(row_a)),
        repository=repo, snapshot_ref=snap, store=serving_store, conn=serving_conn,
        requested_as_of="2024-01-04", resolved_as_of="2024-01-04")
    release_b = projections.build_candidate(
        _preview_input(), _score_doc(rows=[row_b]), _bundle(_compact(row_b)),
        repository=repo, snapshot_ref=snap, store=serving_store, conn=serving_conn,
        requested_as_of="2024-01-05", resolved_as_of="2024-01-05")
    assert isinstance(release_a, PreviewRelease) and isinstance(release_b, PreviewRelease)
    return repo, snap, serving_conn, serving_store, release_a, release_b


def _app(tmp_path, ops_target):
    return create_app(serving_db=str(tmp_path / "serving" / "serving.sqlite"),
                      store_root=str(tmp_path / "serving" / "objects"),
                      serving_root=str(tmp_path / "serving"),
                      publication_root=str(ops_target), token=TOKEN)


# --------------------------------------------------------------------------
# low-level ops publish: mirrors tests/test_v2_ops_same_session_replan.py's
# own _files/_binding/_gate_receipt/_gate/_all_gates/_publish_and_stage
# technique, with an optional projection_binding.json file added
# --------------------------------------------------------------------------


def _stage_files(store, tag, binding_doc):
    files = {"bundle.tar": store.publish_bytes(("<html>" + tag + "</html>").encode(),
                                               schema_ref="release_file.v1.0")}
    if binding_doc is not None:
        files["projection_binding.json"] = store.publish_bytes(
            json.dumps(binding_doc, sort_keys=True).encode(), schema_ref=projections.PROJECTION_BINDING_V1)
    return files


def _binding_hash(release_id, occurrence, files):
    return content_hash({"release_id": release_id, "occurrence": occurrence, "files": files})


def _gate_receipt(store, kind, binding_hash):
    return store.publish_bytes(json.dumps({"kind": kind, "status": "passed",
                                           "input_hash": binding_hash}).encode(),
                               schema_ref="gate_receipt.v1.0")


def _gate(ref, binding_hash):
    return {"ok": True, "receipt_ref": ref.content_hash, "input_hash": binding_hash,
            "receipt_artifact": to_document(ref)}


def _all_gates(store, release_id, occurrence, files):
    binding = _binding_hash(release_id, occurrence, files)
    return {kind: _gate(_gate_receipt(store, kind, binding), binding)
            for kind in ("decision", "projection", "security", "engineering")}


def _stage_and_publish(conn, store, claim, release_id, occurrence, files, *, expected_current, target,
                       scope, clock, generation=""):
    gates = _all_gates(store, release_id, occurrence, files)
    staged = stage_release(conn, store, release_id, occurrence, files,
                           expected_current=expected_current, gates=gates, clock=clock, claim=claim)
    assert staged["eligible"] is True
    result = publish_local(conn, claim, store, target, release_id, scope=scope, clock=clock,
                           generation=generation)
    assert result["delivered"] is True
    return result


# --------------------------------------------------------------------------
# 1. candidate -> publish -> API current names that projection; detail reads
# --------------------------------------------------------------------------


def test_candidate_publish_api_current_names_that_projection_and_detail_reads_work(tmp_path):
    repo, snap, serving_conn, serving_store, release_a, release_b = _releases(tmp_path)
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "opsstore")
    target = tmp_path / "opsrelease"
    scope = "shadow"
    try:
        binding = projections.projection_binding(serving_conn, release_a.release_id)
        assert binding["projection_release_id"] == release_a.release_id
        files = _stage_files(store, "release-a", binding)
        claim = enqueue_claim(conn, clock, supervisor, key="pub-a")
        _stage_and_publish(conn, store, claim, "rel-a", "2026-09-14", files,
                           expected_current=None, target=target, scope=scope, clock=clock)

        server, thread, base = _start(_app(tmp_path, target))
        try:
            code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
            assert code == 200
            document = json.loads(body)
            assert document["release_id"] == release_a.release_id

            code, body, _ = _get(base, "/api/v1/events", token=TOKEN,
                                 params={"release_id": release_a.release_id})
            assert code == 200
            items = json.loads(body)["items"]
            assert len(items) == 1
            score_id = items[0]["scores"][0]["score_id"]

            code, body, _ = _get(base, "/api/v1/scores/" + score_id, token=TOKEN,
                                 params={"release_id": release_a.release_id})
            assert code == 200
            assert json.loads(body)["score_id"] == score_id
        finally:
            _stop(server, thread)
    finally:
        conn.close()
        serving_conn.close()


# --------------------------------------------------------------------------
# 2. a findings failure never reaches publication; CURRENT unchanged
# --------------------------------------------------------------------------


def test_findings_failure_never_reaches_publication_and_current_stays_unchanged(tmp_path):
    """The chosen operator entry point (§5.4/P3-1c, "a publication-effect
    input") only ever binds a projection_binding.json for a CANDIDATE that
    actually committed -- projections.build_candidate returning a Problem
    means there is no release_id to bind at all, so the publish step this
    task adds is simply never invoked. This proves the previously-published
    pointer is untouched by that refusal."""
    repo, snap, serving_conn, serving_store, release_a, release_b = _releases(tmp_path)
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "opsstore")
    target = tmp_path / "opsrelease"
    scope = "shadow"
    try:
        binding = projections.projection_binding(serving_conn, release_a.release_id)
        files = _stage_files(store, "release-a", binding)
        claim = enqueue_claim(conn, clock, supervisor, key="pub-a")
        _stage_and_publish(conn, store, claim, "rel-a", "2026-09-14", files,
                           expected_current=None, target=target, scope=scope, clock=clock)
        pointer_before = release_current(target)
        assert pointer_before == "rel-a"

        # A row scored but never rendered -- ProjectionFindings.ok is False,
        # the same technique test_v2_serving_projections.py's own findings-
        # failure test uses.
        row = _row(ticker="AAA", strike=999.0)
        failing = projections.build_candidate(
            _preview_input(), _score_doc(rows=[row]), {}, repository=repo, snapshot_ref=snap,
            store=serving_store, conn=serving_conn, requested_as_of="2024-01-06",
            resolved_as_of="2024-01-06")
        assert isinstance(failing, Problem)
        assert failing.code == "PROJECTION_REFUSED"

        assert release_current(target) == pointer_before
    finally:
        conn.close()
        serving_conn.close()


# --------------------------------------------------------------------------
# 3/4. binding manifest hash mismatch -- tampered doc, or a changed index
# --------------------------------------------------------------------------


def test_tampered_binding_document_is_a_typed_refusal_from_the_resolver(tmp_path):
    repo, snap, serving_conn, serving_store, release_a, release_b = _releases(tmp_path)
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "opsstore")
    target = tmp_path / "opsrelease"
    scope = "shadow"
    try:
        binding = dict(projections.projection_binding(serving_conn, release_a.release_id))
        binding["projection_manifest_hash"] = "sha256:" + "f" * 64  # tampered
        files = _stage_files(store, "release-a", binding)
        claim = enqueue_claim(conn, clock, supervisor, key="pub-a")
        _stage_and_publish(conn, store, claim, "rel-a", "2026-09-14", files,
                           expected_current=None, target=target, scope=scope, clock=clock)

        server, thread, base = _start(_app(tmp_path, target))
        try:
            code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
            assert code == 500
            assert json.loads(body)["code"] == "CURRENT_BINDING_INVALID"
        finally:
            _stop(server, thread)
    finally:
        conn.close()
        serving_conn.close()


def test_a_changed_index_is_a_typed_refusal_from_the_resolver(tmp_path):
    """A binding that matched at publish time, followed by a direct edit to
    the serving index row it was bound to (never through ``build_candidate``
    -- simulating disk corruption or a hand-edit), fails the SAME check:
    ``serving_index_identity`` no longer matches the live row."""
    repo, snap, serving_conn, serving_store, release_a, release_b = _releases(tmp_path)
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "opsstore")
    target = tmp_path / "opsrelease"
    scope = "shadow"
    try:
        binding = projections.projection_binding(serving_conn, release_a.release_id)
        files = _stage_files(store, "release-a", binding)
        claim = enqueue_claim(conn, clock, supervisor, key="pub-a")
        _stage_and_publish(conn, store, claim, "rel-a", "2026-09-14", files,
                           expected_current=None, target=target, scope=scope, clock=clock)

        serving_conn.execute("UPDATE serving_release SET findings_json = ? WHERE release_id = ?",
                             (json.dumps({"tampered": True}), release_a.release_id))

        server, thread, base = _start(_app(tmp_path, target))
        try:
            code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
            assert code == 500
            assert json.loads(body)["code"] == "CURRENT_BINDING_INVALID"
        finally:
            _stop(server, thread)
    finally:
        conn.close()
        serving_conn.close()


# --------------------------------------------------------------------------
# 5. crash after index insert, before publish: API still serves the
#    previous release; a rerun publishes cleanly
# --------------------------------------------------------------------------


def test_crash_before_publish_serves_previous_release_then_rerun_publishes_cleanly(tmp_path):
    repo, snap, serving_conn, serving_store, release_a, release_b = _releases(tmp_path)
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "opsstore")
    target = tmp_path / "opsrelease"
    scope = "shadow"
    try:
        binding_a = projections.projection_binding(serving_conn, release_a.release_id)
        files_a = _stage_files(store, "release-a", binding_a)
        claim_a = enqueue_claim(conn, clock, supervisor, key="pub-a")
        _stage_and_publish(conn, store, claim_a, "rel-a", "2026-09-14", files_a,
                           expected_current=None, target=target, scope=scope, clock=clock)

        # release_b's candidate is already fully committed in serving.sqlite
        # (build_candidate succeeded in `_releases` above) -- but the
        # operator "crashes" here, before ever building/staging its binding
        # or publishing it. The API must keep serving release_a untouched.
        server, thread, base = _start(_app(tmp_path, target))
        try:
            code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
            assert code == 200
            assert json.loads(body)["release_id"] == release_a.release_id

            # A rerun: complete the interrupted publish for release_b, the
            # next occurrence (a distinct session -- not a same-session
            # retry, which is §5.5 item 1's own separately-proven case).
            binding_b = projections.projection_binding(serving_conn, release_b.release_id)
            files_b = _stage_files(store, "release-b", binding_b)
            claim_b = enqueue_claim(conn, clock, supervisor, key="pub-b")
            _stage_and_publish(conn, store, claim_b, "rel-b", "2026-09-15", files_b,
                               expected_current="rel-a", target=target, scope=scope, clock=clock)

            code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
            assert code == 200
            assert json.loads(body)["release_id"] == release_b.release_id

            # Both generations stay readable by explicit release id.
            for release in (release_a, release_b):
                code, body, _ = _get(base, "/api/v1/releases/" + release.release_id, token=TOKEN)
                assert code == 200
                assert json.loads(body)["release_id"] == release.release_id
        finally:
            _stop(server, thread)
    finally:
        conn.close()
        serving_conn.close()


# --------------------------------------------------------------------------
# 6. second generation publish, then rollback: API current follows each
#    step; both readable by explicit id
# --------------------------------------------------------------------------


def test_second_generation_publish_then_rollback_api_current_follows_each_step(tmp_path):
    repo, snap, serving_conn, serving_store, release_a, release_b = _releases(tmp_path)
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "opsstore")
    target = tmp_path / "opsrelease"
    scope, occurrence = "shadow", "2026-09-14"
    try:
        binding_a = projections.projection_binding(serving_conn, release_a.release_id)
        files_a = _stage_files(store, "release-a", binding_a)
        claim1 = enqueue_claim(conn, clock, supervisor, key="pub-gen1")
        _stage_and_publish(conn, store, claim1, "rel-gen1", occurrence, files_a,
                           expected_current=None, target=target, scope=scope, clock=clock,
                           generation="gen1")

        server, thread, base = _start(_app(tmp_path, target))
        try:
            code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
            assert json.loads(body)["release_id"] == release_a.release_id

            binding_b = projections.projection_binding(serving_conn, release_b.release_id)
            files_b = _stage_files(store, "release-b", binding_b)
            claim2 = enqueue_claim(conn, clock, supervisor, key="pub-gen2")
            _stage_and_publish(conn, store, claim2, "rel-gen2", occurrence, files_b,
                               expected_current="rel-gen1", target=target, scope=scope, clock=clock,
                               generation="gen2")

            code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
            assert json.loads(body)["release_id"] == release_b.release_id

            # Rollback: restage generation 1's own content (release_a's
            # binding) under a fresh release id -- the only existing
            # release-pointer-moving path (§5.5) -- and publish it.
            claim3 = enqueue_claim(conn, clock, supervisor, key="pub-rollback")
            _stage_and_publish(conn, store, claim3, "rel-rollback", occurrence, files_a,
                               expected_current="rel-gen2", target=target, scope=scope, clock=clock,
                               generation="gen3-rollback")

            code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
            assert json.loads(body)["release_id"] == release_a.release_id

            for release in (release_a, release_b):
                code, body, _ = _get(base, "/api/v1/releases/" + release.release_id, token=TOKEN)
                assert code == 200
                assert json.loads(body)["release_id"] == release.release_id
        finally:
            _stop(server, thread)
    finally:
        conn.close()
        serving_conn.close()


# --------------------------------------------------------------------------
# 7. gate receipts bound to generation 1 cannot publish generation 2's
#    candidate
# --------------------------------------------------------------------------


def test_gate_receipts_bound_to_generation_1_cannot_publish_generation_2s_candidate(tmp_path):
    """§5.4: "adding files changes the publication binding" -- gate receipts
    computed against generation 1's ``binding_hash`` (its own release id and
    files, including release_a's projection_binding.json) do not validate
    for generation 2's DIFFERENT release id/files (release_b's), so
    ``stage_release`` reports it ineligible and ``publish_local`` refuses."""
    repo, snap, serving_conn, serving_store, release_a, release_b = _releases(tmp_path)
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "opsstore")
    target = tmp_path / "opsrelease"
    scope = "shadow"
    try:
        binding_a = projections.projection_binding(serving_conn, release_a.release_id)
        files_1 = _stage_files(store, "release-a", binding_a)
        gates_1 = _all_gates(store, "rel-gen1", "2026-09-14", files_1)

        binding_b = projections.projection_binding(serving_conn, release_b.release_id)
        files_2 = _stage_files(store, "release-b", binding_b)

        claim = enqueue_claim(conn, clock, supervisor, key="gate-swap")
        staged = stage_release(conn, store, "rel-gen2", "2026-09-14", files_2,
                               expected_current=None, gates=gates_1, clock=clock, claim=claim)
        assert staged["eligible"] is False

        with pytest.raises(OpsError, match="PUBLICATION_REFUSED"):
            publish_local(conn, claim, store, target, "rel-gen2", scope=scope, clock=clock)
        assert release_current(target) is None
    finally:
        conn.close()
        serving_conn.close()


# --------------------------------------------------------------------------
# operator entry point: publication_effect accepts projection_binding.json
# as one more named input binding -- the smallest of the guide's two
# options, no new job kind or DAG wiring
# --------------------------------------------------------------------------


def test_publication_effect_binds_the_operator_supplied_projection_and_publishes(tmp_path):
    repo, snap, serving_conn, serving_store, release_a, release_b = _releases(tmp_path)
    conn, clock, supervisor, store, root = _ops_open(tmp_path)
    try:
        scope, session = "shadow", "2026-09-14"
        _seed_decisions(conn, clock, scope, session, predictions=[
            {"row_id": "evt-1-pred", "event_id": "evt-1", "ticker": "FAKE", "strategy": "TWIN-P",
             "event_date": session, "status": "resolved", "resolved_at": session + "T21:00:00+00:00"}])
        tag = scope + ":" + session

        finality_doc = {"date": session, "is_final": True, "market_wide": True, "daily_share": 1.0,
                        "chain_share": 1.0, "covered": 1, "detail": "final"}
        finality_ref = _ops_publish_json(store, conn, clock, finality_doc, "legacy_action.v1.0")
        finality_job = _succeed_parent(conn, clock, supervisor, key="fin-" + tag,
                                       output_name="legacy_finality", ref=finality_ref)

        bundle_ref = _ops_publish_raw(store, conn, clock, _bundle_tar(), "legacy_action.v1.0")
        projection_job = _succeed_parent(conn, clock, supervisor, key="proj-" + tag,
                                         output_name="legacy_render", ref=bundle_ref)

        binding_doc = projections.projection_binding(serving_conn, release_a.release_id)
        binding_ref = _ops_publish_json(store, conn, clock, binding_doc, projections.PROJECTION_BINDING_V1)
        binding_job = _succeed_parent(conn, clock, supervisor, key="bind-" + tag,
                                      output_name="projection_binding", ref=binding_ref)

        selfcheck_ref = _ops_publish_json(store, conn, clock, {"ok": True}, "legacy_action.v1.0")
        selfcheck_job = _succeed_parent(conn, clock, supervisor, key="self-" + tag,
                                        output_name="legacy_selfcheck", ref=selfcheck_ref)

        engineering_ref = _ops_publish_json(store, conn, clock, {"ok": True}, "engineering_gate.v1.0")
        engineering_job = _succeed_parent(conn, clock, supervisor, key="eng-" + tag,
                                          output_name="engineering_gate", ref=engineering_ref)

        input_bindings = {
            "bundle.tar": projection_job + "#legacy_render",
            "finality.json": finality_job + "#legacy_finality",
            "selfcheck.json": selfcheck_job + "#legacy_selfcheck",
            "engineering_gate.json": engineering_job + "#engineering_gate",
            "projection_binding.json": binding_job + "#projection_binding",
        }
        claim = _submit_and_claim(
            conn, clock, supervisor, kind="publication", key="pub-" + tag,
            parameters=_ops_params("publication", session, scope, input_bindings=input_bindings),
            dependency_job_ids=(finality_job, projection_job, binding_job, selfcheck_job, engineering_job))
        resolve_and_record(conn, store, claim)

        result = publication_effect(conn, store, claim, root, REPO, clock=clock,
                                    store_root=FAKE_STORE_ROOT)
        _ops_commit(conn, clock, claim, result)

        target = root / "releases" / scope
        current_id = release_current(target)
        assert current_id is not None
        published_binding = json.loads(
            (target / "releases" / current_id / "projection_binding.json").read_text())
        assert published_binding["projection_release_id"] == release_a.release_id

        server, thread, base = _start(_app(tmp_path, target))
        try:
            code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
            assert code == 200
            assert json.loads(body)["release_id"] == release_a.release_id
        finally:
            _stop(server, thread)
    finally:
        conn.close()
        serving_conn.close()


# --------------------------------------------------------------------------
# 8. the API process never imports ops, even resolving a real bound pointer
#    (extends tests/test_v2_serving_api.py's own no-import guard)
# --------------------------------------------------------------------------


_GUARD_SCRIPT = """
import json, sys, threading, time, urllib.error, urllib.request
sys.path.insert(0, {root!r})
from engine.v2.serving.api import create_app
app = create_app(serving_db={serving_db!r}, store_root={store_root!r},
                 serving_root={serving_root!r}, publication_root={publication_root!r}, token={token!r})
after_create = sorted(sys.modules)
import uvicorn
config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
server = uvicorn.Server(config)
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
deadline = time.monotonic() + 10
while not server.started and time.monotonic() < deadline:
    time.sleep(0.01)
port = server.servers[0].sockets[0].getsockname()[1]
req = urllib.request.Request("http://127.0.0.1:" + str(port) + "/api/v1/releases/current",
                             headers={{"Authorization": "Bearer " + {token!r}}})
try:
    urllib.request.urlopen(req, timeout=10).read()
except urllib.error.HTTPError:
    pass
after_request = sorted(sys.modules)
server.should_exit = True
thread.join(timeout=5)
print(json.dumps({{"after_create": after_create, "after_request": after_request}}))
"""


def test_api_process_never_imports_ops_even_resolving_a_real_bound_pointer(tmp_path):
    import subprocess

    repo, snap, serving_conn, serving_store, release_a, release_b = _releases(tmp_path)
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "opsstore")
    target = tmp_path / "opsrelease"
    scope = "shadow"
    try:
        binding = projections.projection_binding(serving_conn, release_a.release_id)
        files = _stage_files(store, "release-a", binding)
        claim = enqueue_claim(conn, clock, supervisor, key="pub-a")
        _stage_and_publish(conn, store, claim, "rel-a", "2026-09-14", files,
                           expected_current=None, target=target, scope=scope, clock=clock)
    finally:
        conn.close()
        serving_conn.close()

    script = _GUARD_SCRIPT.format(
        root=str(ROOT), serving_db=str(tmp_path / "serving" / "serving.sqlite"),
        store_root=str(tmp_path / "serving" / "objects"), serving_root=str(tmp_path / "serving"),
        publication_root=str(target), token=TOKEN)
    script_path = tmp_path / "guard.py"
    script_path.write_text(script)
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    for key in ("after_create", "after_request"):
        modules = payload[key]
        for forbidden in _FORBIDDEN_MODULE_SUBSTRINGS:
            hit = [m for m in modules if forbidden in m]
            assert not hit, f"{key}: forbidden module(s) imported: {hit}"
