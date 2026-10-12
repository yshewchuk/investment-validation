"""Explicit, redacted completion reports for one nightly session generation."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from engine.v2.ops.errors import OpsError, make_problem
from engine.v2.ops.nightly_session import (
    MAX_SESSION_BYTES,
    STATE_DIR,
    SessionIdentity,
    atomic_write,
    load_session,
    session_transition_lock,
)

__all__ = ["Disposition", "SCHEMA", "build_completion_report", "record_completion"]

SCHEMA = "nightly_completion.v1"
_ROLES = {"required", "optional", "qualification"}
_STATES = {"pending", "succeeded", "degraded", "failed", "timed_out", "recovery_required"}
_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_RUN_ID = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_MAX_DISPOSITION_COUNT = 2**63 - 1
_TERMINAL = {"succeeded", "degraded", "failed", "timed_out"}


@dataclass(frozen=True)
class Disposition:
    """Safe public summary for one explicitly selected step or sidecar."""

    status: str
    reason_code: str | None = None
    count: int = 0
    evidence_valid: bool = True


def _valid_expected(expected: object) -> bool:
    return (isinstance(expected, Mapping) and bool(expected)
            and all(isinstance(name, str) and _NAME.fullmatch(name)
                    and isinstance(role, str) and role in _ROLES
                    for name, role in expected.items()))


def _valid_disposition(item: object) -> bool:
    if not isinstance(item, Disposition):
        return False
    if not isinstance(item.status, str) or item.status not in _STATES:
        return False
    if item.reason_code is not None and (
            not isinstance(item.reason_code, str) or not _CODE.fullmatch(item.reason_code)):
        return False
    if type(item.count) is not int or not 0 <= item.count <= _MAX_DISPOSITION_COUNT:
        return False
    return type(item.evidence_valid) is bool


def _validate(run_id: object, expected: Mapping[str, str],
              observed: Mapping[str, Disposition]) -> dict[str, Disposition]:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise OpsError(make_problem("INVALID_REQUEST", "completion needs a run ID"))
    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
        raise OpsError(make_problem("INVALID_REQUEST", "completion dispositions must be mappings"))
    if not _valid_expected(expected):
        raise OpsError(make_problem("INVALID_REQUEST", "completion needs named expected work"))
    if any(name not in expected for name in observed):
        raise OpsError(make_problem("INVALID_REQUEST", "completion has an unselected disposition"))
    result: dict[str, Disposition] = {}
    for name, role in expected.items():
        item = observed.get(name, Disposition("pending"))
        if not _valid_disposition(item):
            raise OpsError(make_problem("INVALID_REQUEST", "completion disposition is malformed"))
        result[name] = item
    return result


def _required_outcome(required: Mapping[str, Disposition]) -> str:
    states = {item.status for item in required.values()}
    if "recovery_required" in states or any(not item.evidence_valid for item in required.values()):
        return "recovery_required"
    if "timed_out" in states:
        return "timed_out"
    if "failed" in states or "degraded" in states:
        return "failed"
    if "pending" in states:
        return "waiting"
    return "completed"


def _qualification_outcome(items: list[Disposition]) -> str:
    if not items:
        return "not_requested"
    if any(not item.evidence_valid or item.status == "recovery_required" for item in items):
        return "invalid"
    if any(item.status == "pending" for item in items):
        return "pending"
    if any(item.status in {"failed", "timed_out"} for item in items):
        return "failed"
    if any(item.status == "degraded" or item.count for item in items):
        return "degraded"
    return "qualified"


def _is_degraded(items: list[Disposition]) -> bool:
    return any(item.status in {"degraded", "failed", "timed_out", "recovery_required"}
               or item.count for item in items)


def _overall_outcome(required_status: str, sidecars: Mapping[str, Disposition],
                     degraded: bool) -> str:
    if required_status == "recovery_required" or any(
            item.status == "recovery_required" or not item.evidence_valid
            for item in sidecars.values()):
        return "recovery_required"
    if required_status != "completed":
        return required_status
    if any(item.status == "pending" for item in sidecars.values()):
        return "waiting"
    return "completed_degraded" if degraded else "completed"


def _report_rows(expected: Mapping[str, str],
                 dispositions: Mapping[str, Disposition]) -> list[dict]:
    return [{"name": name, "role": expected[name], "status": dispositions[name].status,
             "reason_code": dispositions[name].reason_code, "count": dispositions[name].count,
             "evidence_valid": dispositions[name].evidence_valid}
            for name in sorted(expected)]


def build_completion_report(run_id: str, expected: Mapping[str, str],
                            observed: Mapping[str, Disposition]) -> dict:
    """Build stable outcomes from explicit run-scoped work; never infer from supervisor idleness."""
    dispositions = _validate(run_id, expected, observed)
    required = {name: dispositions[name] for name, role in expected.items() if role == "required"}
    sidecars = {name: dispositions[name] for name, role in expected.items() if role != "required"}
    if not required:
        raise OpsError(make_problem("INVALID_REQUEST", "completion needs required work"))
    optional = [dispositions[name] for name, role in expected.items() if role == "optional"]
    qualification = [dispositions[name] for name, role in expected.items()
                     if role == "qualification"]
    required_status = _required_outcome(required)
    degraded = _is_degraded(optional + qualification)
    return {"schema_version": SCHEMA, "run_id": run_id,
            "status": _overall_outcome(required_status, sidecars, degraded),
            "required_outcome": required_status,
            "degraded_outcome": "degraded" if degraded else "clean",
            "qualification_outcome": _qualification_outcome(qualification),
            "required_count": len(required), "sidecar_count": len(sidecars),
            "sidecars_terminal": sum(item.status in _TERMINAL and item.evidence_valid
                                      for item in sidecars.values()),
            "dispositions": _report_rows(expected, dispositions)}


def record_completion(root: Path, identity: SessionIdentity, generation: int,
                      expected: Mapping[str, str], observed: Mapping[str, Disposition]) -> dict:
    """Write only for the active started generation, under the session-transition lock."""
    if not isinstance(identity, SessionIdentity):
        raise OpsError(make_problem("INVALID_REQUEST", "completion needs a session identity"))
    with session_transition_lock(root, identity):
        state = load_session(root, identity)
        if (type(generation) is not int or state is None
                or state.active.generation != generation or state.active.status != "started"):
            raise OpsError(make_problem("CHECKPOINT_INCOMPATIBLE",
                                        "completion generation is not the active started run"))
        report = build_completion_report(state.active.run_id, expected, observed)
        path = Path(root).joinpath(*STATE_DIR, f"{state.active.run_id}.completion.json")
        atomic_write(path, json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
                     max_bytes=MAX_SESSION_BYTES,
                     size_error="completion report is too large")
        return report
