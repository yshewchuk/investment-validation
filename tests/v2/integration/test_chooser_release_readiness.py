"""Real release-reader paths for STR-THRU; only synthetic temporary state."""
# packages: engine.v2.scoring, engine.v2.models, engine.v2.ops
import json

import pytest

from engine.v2.models import deployment
from engine.v2.models.frozen_state import serialize_frozen_state
from engine.v2.ops import health as health_module
from engine.v2.ops.health import health
from engine.v2.ops.native_score_batch import run_native_score_batch_worker
from engine.v2.ops.supervisor import Service
from engine.v2.scoring import release_bindings
from engine.v2.scoring.release_bindings import ModelNotReady
from tests.ops_support import POLICY, REGISTRY, catalog
from tests.test_v2_ops_native_score_batch import _event_doc, _stage_release, _worker_parameters
from tests.v2.scoring.test_v2_scoring_release_bindings import _row, _write_catalog
from tests.v2.scoring.test_release_chooser_states import _artifact, _stored


@pytest.mark.parametrize("member", ["chooser_analog_pool", "admissible_table:dyn_sv"])
@pytest.mark.parametrize("status", [None, "VALID", "AT_LIMIT", "BELOW_LIMIT", "OVERSIZE",
                                    "MISSING", "PENDING", "STAGED"])
def test_str_thru_release_readiness_and_worker_follow_declared_chooser_state(
        tmp_path, monkeypatch, capsys, member, status):
    _stage_release(tmp_path)
    if status in ("VALID", "AT_LIMIT", "BELOW_LIMIT", "OVERSIZE"):
        artifact = _artifact(member)
        _write_catalog(tmp_path, rows=[_row(member, [_stored(tmp_path, artifact)])])
        if status != "VALID":
            headroom = {"AT_LIMIT": 0, "BELOW_LIMIT": 1, "OVERSIZE": -1}[status]
            monkeypatch.setattr(release_bindings, "_CHOOSER_MAX_ARTIFACT_BYTES",
                                len(serialize_frozen_state(artifact)) + headroom)
    elif status is not None:
        # Empty STAGED objects are invalid too. There is no DYN-SV request.
        _write_catalog(tmp_path, rows=[_row(member, [], status=status)])
    monkeypatch.setenv(deployment.MODEL_RELEASE_ROOT_ENV, str(tmp_path))
    conn, clock, _ = catalog(tmp_path)
    service = Service(conn, tmp_path / "ops", REGISTRY, POLICY,
                      clock=clock, code_source=tmp_path, store_root=tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    (work / "events.json").write_text(json.dumps([_event_doc(strategy="STR-THRU")]))
    parameters = _worker_parameters(tmp_path, expected_ids=["evt-1"])
    try:
        root = service._native_release_root_or_none(
            occurrence="2026-01-15", scope_hash="test-scope-hash")
        if status in (None, "VALID", "AT_LIMIT", "BELOW_LIMIT"):
            assert root == str(tmp_path)
            assert service._native_release_memo["ok"] is True
            assert capsys.readouterr().out == ""
            run_native_score_batch_worker(parameters, work)
            records = json.loads((work / "records.json").read_text())["records"]
            assert set(records) == {"TEST|STR-THRU|2026-01-15|am"}
            assert records["TEST|STR-THRU|2026-01-15|am"]["canonical_request"][
                "strategy_version"] == "STR-THRU"
            assert records["TEST|STR-THRU|2026-01-15|am"]["chooser_candidates"] == []
            assert records["TEST|STR-THRU|2026-01-15|am"]["chooser_selection"] is None
            assert json.loads((work / "refusals.json").read_text())["refusals"] == {}
        else:
            assert root is None
            assert service._native_release_memo["ok"] is False
            problem = json.loads(capsys.readouterr().out)
            assert problem["event"] == "native_score_batch_release_unavailable"
            assert problem["problem"]["code"] == "MODEL_NOT_READY"
            assert member not in problem["problem"]["message"]
            assert str(tmp_path) not in problem["problem"]["message"]
            failures = conn.execute(
                "SELECT occurrence, scope_hash, status, error_code, problem_json "
                "FROM native_stage_failures WHERE occurrence = ? AND scope_hash = ?",
                ("2026-01-15", "test-scope-hash")).fetchall()
            assert len(failures) == 1
            failure = failures[0]
            assert failure["status"] == "FAILED"
            assert failure["error_code"] == "MODEL_NOT_READY"
            assert json.loads(failure["problem_json"])["code"] == "MODEL_NOT_READY"
            monkeypatch.setattr(
                health_module, "_current_delivered_release_with_scope",
                lambda conn: (None, None))
            monkeypatch.setattr(
                health_module, "_withheld_release",
                lambda conn, current: None)
            monkeypatch.setattr(
                health_module, "_release_session_evidence",
                lambda conn, current, expected_scope: (None, None))
            document = health(conn, clock=clock)
            assert any(
                row["occurrence"] == "2026-01-15"
                and row["status"] == "FAILED"
                and row["error_code"] == "MODEL_NOT_READY"
                for row in document["native_stage_failures"])
            with pytest.raises(ModelNotReady) as error:
                run_native_score_batch_worker(parameters, work)
            assert error.value.member_id == member
            assert not (work / "records.json").exists()
            assert not (work / "refusals.json").exists()
    finally:
        conn.close()
