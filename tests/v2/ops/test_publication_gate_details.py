"""``PUBLICATION_REFUSED`` names the gates that failed.

The refusal itself was always correct -- an ineligible release never reaches
materialization -- but it told an operator only that "release gates are not all
valid", so the first fix step was re-running the gate suite. These tests pin
the detail the refusal now carries: ``details.failed_gates``, the manifest gate
names whose recorded ``ok`` is the boolean ``False``, sorted.

Only the refusal path is exercised: the fake connection answers nothing but the
release lookup, and the store/target/claim/clock are tripwires, so a publish
that got past the refusal fails loudly instead of writing.
"""
from __future__ import annotations

import json

import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.ops.publication import publish_local

MESSAGE = "release gates are not all valid"


class _Tripwire:
    """Any use means the refusal did not happen first."""

    def __getattr__(self, name):
        raise AssertionError(f"publication proceeded past the refusal and used {name!r}")


class _Result:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _Catalog:
    """Answers the release lookup and nothing else."""

    def __init__(self, row):
        self._row = row
        self.statements = []

    def execute(self, sql, params=()):
        self.statements.append(sql)
        if sql == "SELECT * FROM releases WHERE release_id=?":
            return _Result(self._row)
        raise AssertionError(f"publication proceeded past the refusal: {sql}")


def _manifest(gates):
    return {"schema_version": "release_manifest.v1.0", "release_id": "rel-1",
            "occurrence": "2026-10-09", "files": {}, "gates": gates}


def _ineligible_row(gates):
    return {"release_id": "rel-1", "occurrence": "2026-10-09", "expected_current": None,
            "eligible": 0, "manifest_json": json.dumps(_manifest(gates))}


def _publish(conn):
    with pytest.raises(OpsError) as excinfo:
        publish_local(conn, _Tripwire(), _Tripwire(), _Tripwire(), "rel-1",
                      scope="shadow", clock=_Tripwire())
    return excinfo.value


def test_refusal_lists_the_failed_gates_sorted():
    conn = _Catalog(_ineligible_row({
        "zeta": {"ok": False, "receipt_ref": "sha256:" + "0" * 64},
        "alpha": {"ok": False},
        "decision": {"ok": True},
    }))
    problem = _publish(conn).problem
    assert problem.code == "PUBLICATION_REFUSED"
    assert problem.message == MESSAGE
    assert problem.details == {"failed_gates": ["alpha", "zeta"]}
    assert conn.statements == ["SELECT * FROM releases WHERE release_id=?"]


def test_a_missing_gate_row_refuses_with_no_failed_gates():
    problem = _publish(_Catalog(None)).problem
    assert problem.code == "PUBLICATION_REFUSED"
    assert problem.message == MESSAGE
    assert problem.details == {"failed_gates": []}


def test_entries_without_a_false_verdict_are_not_listed_as_failed():
    # A non-mapping entry and a mapping with no ``ok`` say nothing about the
    # gate's verdict; only ``ok is False`` is a failure. ``ok: 0`` is not the
    # boolean either -- an invented falsy status is not evidence of a failed
    # gate.
    conn = _Catalog(_ineligible_row({
        "not_a_mapping": "failed",
        "no_verdict": {"receipt_ref": "sha256:" + "0" * 64},
        "zeroish": {"ok": 0},
        "beta": {"ok": False},
    }))
    assert _publish(conn).problem.details == {"failed_gates": ["beta"]}
