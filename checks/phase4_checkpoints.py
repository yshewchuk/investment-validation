"""Bounded Phase 4 diagnostic-checkpoint contract.

This does not alter the existing completion gate or execute scoring.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from engine.v2.foundation import content_hash

SCHEMA_VERSION = "phase4_diagnostic_checkpoints.v1.0"
REQUIRED_BRANCHES = frozenset((
    "pre_expiry", "multi_expiry", "debit_credit", "missing_inputs",
    "overrides", "ties", "fallback",
    "frozen_inference_canonical_application",
))
REQUIRED_CHECKPOINT_GROUPS = frozenset((
    "features", "selection_pricing", "simulation", "gate_inputs",
))
OPTIONAL_CHECKPOINT_GROUPS = frozenset(("dyn_sv",))
_DISPOSITIONS = frozenset(("compared", "refused_as_expected", "incomparable"))


class CheckpointError(ValueError):
    pass


def _fail(label: str, reason: str) -> None:
    raise CheckpointError(label + ": " + reason)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(label, "expected nonempty string")
    return value


def _hash(value: Any, label: str) -> str:
    value = _string(value, label)
    if not value.startswith("sha256:") or len(value) != 71:
        _fail(label, "expected sha256 content hash")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise CheckpointError(label + ": expected sha256 content hash") from exc
    return value


def _json(value: Any, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(label, "non-finite value")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json(item, label + "[" + str(index) + "]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(label, "object keys must be strings")
            _json(item, label + "." + key)
        return
    _fail(label, "value is not JSON-safe")


def _hashed(row: Any, label: str) -> Any:
    if not isinstance(row, Mapping) or set(row) != {"value", "content_hash"}:
        _fail(label, "expected value and content_hash")
    _json(row["value"], label + ".value")
    if content_hash(row["value"]) != _hash(row["content_hash"], label + ".content_hash"):
        _fail(label, "content hash mismatch")
    return row["value"]


def _checkpoint(group: str, value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        _fail(label, "expected object")
    if group == "features":
        valid = (
            set(value) == {"feature_vector", "missing_mask", "model_identity"}
            and isinstance(value["feature_vector"], (Mapping, list))
            and isinstance(value["missing_mask"], (Mapping, list))
            and isinstance(value["model_identity"], Mapping)
        )
    elif group == "selection_pricing":
        valid = (
            set(value) == {"selected_legs", "entry_cost"}
            and isinstance(value["selected_legs"], list)
            and (value["entry_cost"] is None or isinstance(value["entry_cost"], (int, float)))
        )
    elif group == "simulation":
        valid = (
            set(value) == {
                "horizon", "capital_denominator", "residual_population_identity",
                "draw_count", "seed",
            }
            and isinstance(value["horizon"], (str, Mapping))
            and (value["capital_denominator"] is None or isinstance(value["capital_denominator"], (int, float)))
            and isinstance(value["residual_population_identity"], Mapping)
            and type(value["draw_count"]) is int and value["draw_count"] >= 0
            and isinstance(value["seed"], (str, int))
        )
    elif group == "gate_inputs":
        valid = isinstance(value, Mapping)
    else:
        valid = (
            set(value) == {"eligibility", "ranking"}
            and isinstance(value["eligibility"], (Mapping, list))
            and isinstance(value["ranking"], (Mapping, list))
        )
    if not valid:
        _fail(label, "malformed checkpoint")


def _resource(root: Path, row: Any, index: int, ids: set[str]) -> None:
    label = "resources[" + str(index) + "]"
    if not isinstance(row, Mapping) or set(row) != {"resource_id", "path", "sha256"}:
        _fail(label, "expected resource_id, path, sha256")
    resource_id = _string(row["resource_id"], label + ".resource_id")
    if resource_id in ids:
        _fail(label + ".resource_id", "duplicate")
    candidate = (root / _string(row["path"], label + ".path")).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise CheckpointError(label + ".path: escapes release root") from exc
    if not candidate.is_file():
        _fail(label + ".path", "missing")
    actual = "sha256:" + hashlib.sha256(candidate.read_bytes()).hexdigest()
    if actual != _hash(row["sha256"], label + ".sha256"):
        _fail(label + ".sha256", "mismatch")
    ids.add(resource_id)


def _case(row: Any, index: int, resource_ids: set[str]) -> tuple[str, str, set[str]]:
    label = "cases[" + str(index) + "]"
    expected = {
        "case_id", "request", "request_hash", "strategy", "branches",
        "resource_refs", "executable_inputs", "disposition", "checkpoints",
        "case_hash",
    }
    if not isinstance(row, Mapping) or set(row) != expected:
        _fail(label, "unexpected or missing fields")
    case_id = _string(row["case_id"], label + ".case_id")
    strategy = _string(row["strategy"], label + ".strategy")
    if not isinstance(row["request"], Mapping):
        _fail(label + ".request", "expected exact saved request object")
    _json(row["request"], label + ".request")
    if content_hash(row["request"]) != _hash(row["request_hash"], label + ".request_hash"):
        _fail(label + ".request_hash", "mismatch")
    branches = row["branches"]
    if (
        not isinstance(branches, list) or not branches
        or any(not isinstance(item, str) or not item.strip() for item in branches)
        or len(set(branches)) != len(branches)
    ):
        _fail(label + ".branches", "expected unique nonempty labels")
    refs = row["resource_refs"]
    if (
        not isinstance(refs, list) or any(not isinstance(item, str) for item in refs)
        or len(set(refs)) != len(refs) or set(refs) - resource_ids
    ):
        _fail(label + ".resource_refs", "unknown or duplicate resource")
    if row["executable_inputs"] not in {"available", "missing"}:
        _fail(label + ".executable_inputs", "expected available or missing")
    if row["disposition"] not in _DISPOSITIONS:
        _fail(label + ".disposition", "unsupported")
    if row["executable_inputs"] == "missing" and row["disposition"] != "incomparable":
        _fail(label, "missing executable inputs must be incomparable")
    checkpoints = row["checkpoints"]
    if not isinstance(checkpoints, Mapping):
        _fail(label + ".checkpoints", "expected object")
    if set(checkpoints) - (REQUIRED_CHECKPOINT_GROUPS | OPTIONAL_CHECKPOINT_GROUPS):
        _fail(label + ".checkpoints", "unsupported group")
    missing = REQUIRED_CHECKPOINT_GROUPS - set(checkpoints)
    if missing:
        _fail(label + ".checkpoints", "missing required group")
    dyn_sv = strategy == "DYN-SV" or "dyn_sv" in branches
    if dyn_sv != ("dyn_sv" in checkpoints):
        _fail(label + ".checkpoints.dyn_sv", "required exactly for DYN-SV")
    for group, checkpoint in checkpoints.items():
        _checkpoint(group, _hashed(checkpoint, label + ".checkpoints." + group),
                    label + ".checkpoints." + group + ".value")
    body = {key: value for key, value in row.items() if key != "case_hash"}
    if content_hash(body) != _hash(row["case_hash"], label + ".case_hash"):
        _fail(label + ".case_hash", "mismatch")
    return case_id, strategy, set(branches)


def validate_bundle(bundle: Any, release_root: Path) -> dict[str, Any]:
    """Validate release resources, cases, coverage, and content hashes."""
    expected = {
        "schema_version", "release_id", "resources", "coverage", "cases",
        "manifest_hash",
    }
    if not isinstance(bundle, Mapping) or set(bundle) != expected:
        _fail("bundle", "unexpected or missing fields")
    if bundle["schema_version"] != SCHEMA_VERSION:
        _fail("bundle.schema_version", "unsupported")
    _string(bundle["release_id"], "bundle.release_id")
    _json(bundle, "bundle")
    if not isinstance(bundle["resources"], list):
        _fail("bundle.resources", "expected list")
    resource_ids: set[str] = set()
    for index, row in enumerate(bundle["resources"]):
        _resource(release_root, row, index, resource_ids)
    if not isinstance(bundle["cases"], list) or not bundle["cases"]:
        _fail("bundle.cases", "expected nonempty list")
    case_ids: set[str] = set()
    strategy_cases: dict[str, set[str]] = {}
    branch_cases: dict[str, set[str]] = {}
    for index, row in enumerate(bundle["cases"]):
        case_id, strategy, branches = _case(row, index, resource_ids)
        if case_id in case_ids:
            _fail("cases[" + str(index) + "].case_id", "duplicate")
        case_ids.add(case_id)
        strategy_cases.setdefault(strategy, set()).add(case_id)
        for branch in branches:
            branch_cases.setdefault(branch, set()).add(case_id)
    coverage = bundle["coverage"]
    if not isinstance(coverage, Mapping) or set(coverage) != {"strategies", "branches"}:
        _fail("bundle.coverage", "expected strategies and branches")
    strategies, branches = coverage["strategies"], coverage["branches"]
    if not isinstance(strategies, Mapping) or not strategies:
        _fail("bundle.coverage.strategies", "expected nonempty object")
    if not isinstance(branches, Mapping) or set(branches) != REQUIRED_BRANCHES:
        _fail("bundle.coverage.branches", "incomplete critical branches")
    for strategy, ids in strategies.items():
        _string(strategy, "bundle.coverage.strategies key")
        if not isinstance(ids, list) or not ids or set(ids) != strategy_cases.get(strategy, set()):
            _fail("bundle.coverage.strategies." + strategy, "case coverage mismatch")
    if set(strategies) != set(strategy_cases):
        _fail("bundle.coverage.strategies", "every case strategy must be declared")
    for branch, ids in branches.items():
        if not isinstance(ids, list) or not ids or set(ids) != branch_cases.get(branch, set()):
            _fail("bundle.coverage.branches." + branch, "case coverage mismatch")
    body = {key: value for key, value in bundle.items() if key != "manifest_hash"}
    if content_hash(body) != _hash(bundle["manifest_hash"], "bundle.manifest_hash"):
        _fail("bundle.manifest_hash", "mismatch")
    return {
        "release_id": bundle["release_id"],
        "case_ids": tuple(sorted(case_ids)),
        "strategies": tuple(sorted(strategy_cases)),
        "branches": tuple(sorted(branch_cases)),
        "manifest_hash": bundle["manifest_hash"],
    }


def load_bundle(path: Path, release_root: Path | None = None) -> dict[str, Any]:
    """Load strict JSON then validate a diagnostic checkpoint bundle."""
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError("bundle: invalid JSON at " + str(path)) from exc
    return validate_bundle(bundle, release_root or path.parent)
