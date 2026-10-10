"""Real-file tests for the nightly session identity and CAS state (slice 1 of #564)."""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from engine.v2.ops import nightly_session as ns
from engine.v2.ops.errors import OpsError

IDENT = ns.SessionIdentity("2026-10-08", "shadow", "sel-1", "cat-1")


def _code(excinfo) -> str:
    return excinfo.value.code


def test_identity_is_deterministic_and_field_sensitive():
    other = ns.SessionIdentity("2026-10-08", "shadow", "sel-2", "cat-1")
    assert IDENT.session_key == ns.SessionIdentity("2026-10-08", "shadow", "sel-1", "cat-1").session_key
    assert IDENT.session_key != other.session_key


@pytest.mark.parametrize("args", [("2026-10-8", "s", "a", "b"), ("2026-10-08", "", "a", "b"),
                                  ("2026-10-08", "s", " ", "b"), ("2026-10-08", "s", "a", "")])
def test_malformed_identity_is_invalid_request(args):
    with pytest.raises(OpsError) as exc:
        ns.SessionIdentity(*args)
    assert _code(exc) == "INVALID_REQUEST"


def test_ordinary_calls_resume_one_session_without_new_generation(tmp_path):
    first = ns.ensure_session(tmp_path, IDENT)
    again = ns.ensure_session(tmp_path, IDENT)
    assert first == again
    assert first.revision == 1 and len(first.generations) == 1
    assert first.active.generation == 1 and first.active.status == "allocated"
    assert first.active.run_id == ns._digest(IDENT.session_key, 1)
    assert ns.load_session(tmp_path, IDENT) == first


def test_mark_started_is_idempotent_and_ordinary_call_keeps_generation(tmp_path):
    ns.ensure_session(tmp_path, IDENT)
    started = ns.mark_started(tmp_path, IDENT, 1)
    assert started.active.status == "started" and started.revision == 2
    assert ns.mark_started(tmp_path, IDENT, 1) == started
    assert ns.ensure_session(tmp_path, IDENT) == started


def test_rerun_reattaches_to_allocated_then_allocates_after_start(tmp_path):
    ns.ensure_session(tmp_path, IDENT)
    assert ns.request_rerun(tmp_path, IDENT, ("b", "a")).revision == 1  # reattach to gen 1
    ns.mark_started(tmp_path, IDENT, 1)
    second = ns.request_rerun(tmp_path, IDENT, ("b", "a", "a"))
    assert second.active.generation == 2 and second.active.reason == "rerun"
    assert second.active.invalidation == ("a", "b")
    assert second.active.run_id != second.generations[0].run_id
    # a repeated request while generation 2 is still allocated does not allocate a third
    assert ns.request_rerun(tmp_path, IDENT, ("c",)) == second
    assert len(ns.load_session(tmp_path, IDENT).generations) == 2


def test_superseded_generation_cannot_mark_started(tmp_path):
    ns.ensure_session(tmp_path, IDENT)
    ns.mark_started(tmp_path, IDENT, 1)
    ns.request_rerun(tmp_path, IDENT)
    with pytest.raises(OpsError) as exc:
        ns.mark_started(tmp_path, IDENT, 1)
    assert _code(exc) == "CHECKPOINT_INCOMPATIBLE"


def test_crash_before_the_swap_records_nothing(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("crash")
    monkeypatch.setattr(ns, "compare_and_swap", boom)
    with pytest.raises(RuntimeError):
        ns.ensure_session(tmp_path, IDENT)
    assert ns.load_session(tmp_path, IDENT) is None
    monkeypatch.undo()
    assert ns.ensure_session(tmp_path, IDENT).active.generation == 1


def test_stale_swap_loses_and_exhausted_retries_are_lease_lost(tmp_path, monkeypatch):
    state = ns.ensure_session(tmp_path, IDENT)
    gens = state.generations
    assert ns.compare_and_swap(tmp_path, IDENT, 0, gens) is None
    assert ns.compare_and_swap(tmp_path, IDENT, 1, gens).revision == 2
    monkeypatch.setattr(ns, "compare_and_swap", lambda *a, **k: None)
    with pytest.raises(OpsError) as exc:
        ns.mark_started(tmp_path, IDENT, 1)
    assert _code(exc) == "LEASE_LOST" and exc.value.problem.retryable


def _ensure(root):
    return ns.ensure_session(Path(root), IDENT).active.run_id


def _rerun(root):
    return ns.request_rerun(Path(root), IDENT).active.run_id


def test_concurrent_first_calls_have_one_winner(tmp_path):
    with multiprocessing.get_context("fork").Pool(6) as pool:
        run_ids = pool.map(_ensure, [str(tmp_path)] * 12)
    assert len(set(run_ids)) == 1
    assert ns.load_session(tmp_path, IDENT).revision == 1


def test_concurrent_reruns_allocate_exactly_one_generation(tmp_path):
    ns.ensure_session(tmp_path, IDENT)
    ns.mark_started(tmp_path, IDENT, 1)
    with multiprocessing.get_context("fork").Pool(6) as pool:
        run_ids = pool.map(_rerun, [str(tmp_path)] * 12)
    state = ns.load_session(tmp_path, IDENT)
    assert len(state.generations) == 2 and set(run_ids) == {state.active.run_id}


def test_corrupt_state_is_refused_and_left_untouched(tmp_path):
    path = ns.session_path(tmp_path, IDENT)
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    with pytest.raises(OpsError) as exc:
        ns.ensure_session(tmp_path, IDENT)
    assert _code(exc) == "INTEGRITY_FAILED" and path.read_text() == "{not json"


def test_state_of_another_session_key_is_integrity_failed(tmp_path):
    ns.ensure_session(tmp_path, IDENT)
    path = ns.session_path(tmp_path, IDENT)
    doc = json.loads(path.read_text())
    doc["session_key"] = "f" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(OpsError) as exc:
        ns.load_session(tmp_path, IDENT)
    assert _code(exc) == "INTEGRITY_FAILED"


def test_rerun_without_a_session_and_non_string_identity_are_invalid_request(tmp_path):
    with pytest.raises(OpsError) as exc:
        ns.request_rerun(tmp_path, IDENT)
    assert _code(exc) == "INVALID_REQUEST"
    assert ns.load_session(tmp_path, IDENT) is None
    with pytest.raises(OpsError) as exc:
        ns.SessionIdentity(20261008, "s", "a", "b")
    assert _code(exc) == "INVALID_REQUEST"


def test_changed_stored_identity_is_integrity_failed(tmp_path):
    ns.ensure_session(tmp_path, IDENT)
    path = ns.session_path(tmp_path, IDENT)
    doc = json.loads(path.read_text())
    doc["identity"]["scope"] = "other"
    path.write_text(json.dumps(doc))
    with pytest.raises(OpsError) as exc:
        ns.load_session(tmp_path, IDENT)
    assert _code(exc) == "INTEGRITY_FAILED"


def test_impossible_calendar_date_is_invalid_request():
    with pytest.raises(OpsError) as exc:
        ns.SessionIdentity("2026-02-30", "s", "a", "b")
    assert _code(exc) == "INVALID_REQUEST"


def test_non_utf8_state_is_integrity_failed(tmp_path):
    path = ns.session_path(tmp_path, IDENT)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(OpsError) as exc:
        ns.ensure_session(tmp_path, IDENT)
    assert _code(exc) == "INTEGRITY_FAILED" and path.read_bytes() == b"\xff\xfe\x00bad"


@pytest.mark.parametrize("bad", [1.0, True])
def test_non_int_generation_is_refused_and_state_unchanged(tmp_path, bad):
    ns.ensure_session(tmp_path, IDENT)
    path = ns.session_path(tmp_path, IDENT)
    before = path.read_bytes()
    with pytest.raises(OpsError) as exc:
        ns.mark_started(tmp_path, IDENT, bad)
    assert _code(exc) == "INVALID_REQUEST" and path.read_bytes() == before
    assert ns.mark_started(tmp_path, IDENT, 1).active.status == "started"
    assert ns.request_rerun(tmp_path, IDENT).active.generation == 2


def test_float_generation_in_stored_file_is_integrity_failed(tmp_path):
    ns.ensure_session(tmp_path, IDENT)
    path = ns.session_path(tmp_path, IDENT)
    path.write_text(path.read_text().replace('"generation": 1,', '"generation": 1.0,'))
    with pytest.raises(OpsError) as exc:
        ns.load_session(tmp_path, IDENT)
    assert _code(exc) == "INTEGRITY_FAILED"


def test_cas_refuses_a_state_the_loader_would_reject_and_keeps_prior(tmp_path):
    ns.ensure_session(tmp_path, IDENT)
    path = ns.session_path(tmp_path, IDENT)
    before = path.read_bytes()
    with pytest.raises(OpsError) as exc:
        ns.compare_and_swap(tmp_path, IDENT, 1, ())
    assert _code(exc) == "INTEGRITY_FAILED" and path.read_bytes() == before
    assert ns.load_session(tmp_path, IDENT).revision == 1


def test_reason_must_match_generation_position(tmp_path):
    ns.ensure_session(tmp_path, IDENT)
    path = ns.session_path(tmp_path, IDENT)
    doc = json.loads(path.read_text())
    doc["generations"][0]["reason"] = "rerun"
    path.write_text(json.dumps(doc))
    with pytest.raises(OpsError) as exc:
        ns.load_session(tmp_path, IDENT)
    assert _code(exc) == "INTEGRITY_FAILED"


def test_stored_invalidation_must_be_a_list_of_strings(tmp_path):
    ns.ensure_session(tmp_path, IDENT)
    path = ns.session_path(tmp_path, IDENT)
    for bad in ("abc", [1]):
        doc = json.loads(path.read_text())
        doc["generations"][0]["invalidation"] = bad
        path.write_text(json.dumps(doc))
        with pytest.raises(OpsError) as exc:
            ns.load_session(tmp_path, IDENT)
        assert _code(exc) == "INTEGRITY_FAILED"


def test_tampered_run_id_is_integrity_failed(tmp_path):
    ns.ensure_session(tmp_path, IDENT)
    path = ns.session_path(tmp_path, IDENT)
    doc = json.loads(path.read_text())
    doc["generations"][0]["run_id"] = "0" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(OpsError) as exc:
        ns.load_session(tmp_path, IDENT)
    assert _code(exc) == "INTEGRITY_FAILED"


def test_old_format_receipt_at_session_path_is_refused(tmp_path):
    path = ns.session_path(tmp_path, IDENT)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": "nightly_trigger.v1.0", "as_of": "2026-10-08",
                                "status": "completed"}))
    with pytest.raises(OpsError) as exc:
        ns.ensure_session(tmp_path, IDENT)
    assert _code(exc) == "CHECKPOINT_INCOMPATIBLE"


def test_old_per_date_state_is_ignored_and_infers_nothing(tmp_path):
    old = tmp_path / "reports" / "phase6" / "nightly_trigger" / "2026-10-08.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"schema_version": "nightly_trigger.v1.0", "as_of": "2026-10-08",
                               "status": "completed"}))
    before = old.read_bytes()
    state = ns.ensure_session(tmp_path, IDENT)
    assert state.active.generation == 1 and state.active.status == "allocated"
    assert old.read_bytes() == before