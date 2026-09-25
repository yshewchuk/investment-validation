"""P6-3 ``decisions-supersede``: the ``ops decisions supersede`` operator path.

Covers the whole route end to end over a tmp catalog: the CLI submits a
supervised ``decisions_supersede`` job (validating the payload before
submission), the claimed attempt's coordinator commit
(``decision_commit.commit_supersede``) appends the superseding decision under
the fence, and every refusal leaves the ledger byte-for-byte unchanged.
No real ledger is ever opened.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from engine.v2.ledger.decisions import insert, rows, set_authority
from engine.v2.ops import worker
from engine.v2.ops.catalog import transaction
from engine.v2.ops.cli import decisions_command, parser
from engine.v2.ops.decision_commit import commit_supersede, validated_supersede_payload
from engine.v2.ops.errors import OpsError
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.stages import registry
from tests.ops_support import catalog, sample

STAMP = "2026-09-12T00:00:00.000000Z"
OLD = "prediction:r1"
OLD_PAYLOAD = {"row_id": "r1", "event_id": "e1", "ticker": "r1"}


def _authority(conn):
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)


def _seed(conn, decision_id=OLD, payload=None):
    with transaction(conn):
        return insert(conn, logical_key=decision_id, decision_id=decision_id,
                      payload=payload if payload is not None else OLD_PAYLOAD,
                      purpose="shadow", kind="prediction", validations={}, created_at=STAMP)


def _submit(tmp_path, conn, clock, *, row_id=OLD, reason="restated by operator",
            payload=None, json_text=None):
    argv = ["decisions", "supersede", "--row-id", row_id, "--reason", reason]
    if payload is not None or json_text is not None:
        path = tmp_path / "new_payload.json"
        path.write_text(json_text if json_text is not None
                        else json.dumps(payload, sort_keys=True))
        argv += ["--from-json", str(path)]
    return decisions_command(parser().parse_args(argv), tmp_path, conn, clock)


def _claim(conn, clock, supervisor):
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                       supervisor=supervisor, clock=clock, registry=registry())
    assert claim is not None
    return claim


def _by_id(conn):
    return {row["decision_id"]: row for row in rows(conn)}


def _counts(conn):
    return {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("jobs", "attempts", "decisions")}


def test_happy_path_appends_a_superseding_decision(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    _authority(conn)
    _seed(conn)

    receipt = _submit(tmp_path, conn, clock, payload={"row_id": "r1", "restated": True})
    assert receipt["kind"] == "decisions_supersede"
    assert receipt["state"] == "queued"
    claim = _claim(conn, clock, supervisor)
    assert claim.spec.kind == "decisions_supersede"

    assert commit_supersede(conn, claim, clock=clock) == (None, ())

    committed = _by_id(conn)
    assert set(committed) == {OLD, "supersede:" + OLD}
    new = committed["supersede:" + OLD]
    assert new["supersedes"] == OLD
    assert new["kind"] == "supersede"
    assert new["purpose"] == "decision_supersede"
    payload = json.loads(new["payload_json"])
    assert payload["supersede_reason"] == "restated by operator"
    assert payload["restated"] is True
    # append-only: the old row is still present, unchanged.
    old = committed[OLD]
    assert json.loads(old["payload_json"]) == OLD_PAYLOAD
    assert old["supersedes"] is None


def test_identical_retry_resolves_to_the_same_row(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    _authority(conn)
    _seed(conn)
    _submit(tmp_path, conn, clock)
    claim = _claim(conn, clock, supervisor)

    commit_supersede(conn, claim, clock=clock)
    before = _by_id(conn)
    # The identical supersession again: the already-superseded refusal exempts
    # this exact (decision_id, payload hash), so insert's own idempotency
    # returns the existing row rather than appending a second one.
    commit_supersede(conn, claim, clock=clock)
    assert _by_id(conn) == before
    assert len(before) == 2


def test_empty_reason_is_refused_before_submission(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    _authority(conn)
    with pytest.raises(OpsError) as excinfo:
        _submit(tmp_path, conn, clock, reason="")
    assert excinfo.value.code == "INVALID_REQUEST"
    assert _counts(conn) == {"jobs": 0, "attempts": 0, "decisions": 0}


def test_unknown_target_is_refused_and_writes_nothing(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    _authority(conn)
    _submit(tmp_path, conn, clock, row_id="prediction:missing")
    claim = _claim(conn, clock, supervisor)

    before = _counts(conn)
    with pytest.raises(OpsError) as excinfo:
        commit_supersede(conn, claim, clock=clock)
    assert excinfo.value.code == "VALIDATION_FAILED"
    assert "unknown decision_id" in str(excinfo.value)
    assert _counts(conn) == before
    assert rows(conn) == []


def test_already_superseded_target_is_refused_and_the_first_row_stands(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    _authority(conn)
    _seed(conn)
    _submit(tmp_path, conn, clock, reason="first reason")
    commit_supersede(conn, _claim(conn, clock, supervisor), clock=clock)
    first = rows(conn)

    _submit(tmp_path, conn, clock, reason="second reason")
    claim = _claim(conn, clock, supervisor)
    with pytest.raises(OpsError) as excinfo:
        commit_supersede(conn, claim, clock=clock)
    assert excinfo.value.code == "VALIDATION_FAILED"
    assert "already superseded" in str(excinfo.value)
    assert rows(conn) == first
    assert json.loads(first[1]["payload_json"])["supersede_reason"] == "first reason"


def test_invalid_payload_is_refused_before_submission(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    _authority(conn)
    with pytest.raises(OpsError) as excinfo:
        _submit(tmp_path, conn, clock, json_text="[1, 2, 3]")
    assert excinfo.value.code == "VALIDATION_FAILED"
    assert _counts(conn) == {"jobs": 0, "attempts": 0, "decisions": 0}
    with pytest.raises(OpsError):
        validated_supersede_payload("not an object")


def test_invalid_payload_is_refused_again_at_commit(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    _authority(conn)
    _seed(conn)
    _submit(tmp_path, conn, clock)
    claim = _claim(conn, clock, supervisor)
    tampered = dataclasses.replace(claim, spec=dataclasses.replace(
        claim.spec, parameters={**claim.spec.parameters, "new_payload": ["not", "an", "object"]}))

    before = _counts(conn)
    with pytest.raises(OpsError) as excinfo:
        commit_supersede(conn, tampered, clock=clock)
    assert excinfo.value.code == "VALIDATION_FAILED"
    assert _counts(conn) == before
    assert [row["decision_id"] for row in rows(conn)] == [OLD]


def test_the_worker_only_proves_the_attempt(tmp_path):
    result = worker.dispatch("decisions_supersede",
                             {"expected_ids": ["decisions_supersede"]}, tmp_path)
    assert result == {
        "outputs": [{"name": "decisions_supersede_receipt", "path": "receipt.json",
                     "schema": "effect_receipt.v1.0"}],
        "completed_ids": ["decisions_supersede"], "no_work": False}
    assert (tmp_path / "receipt.json").is_file()
