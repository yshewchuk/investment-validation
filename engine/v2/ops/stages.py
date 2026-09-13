"""Allowlisted stage contracts and strict result validation."""
from __future__ import annotations

from dataclasses import dataclass

from engine.v2.foundation import ArtifactError, safe_relative_path
from engine.v2.ops.errors import fail
from engine.v2.ops.submission import JobKind, KindRegistry, RetryPolicy


@dataclass(frozen=True)
class CheckParameters:
    expected_ids: tuple[str, ...]
    #: Exercises the same launch-time input-binding resolution as the legacy
    #: kinds (P2-5/B1a); the ``artifact_check`` worker never reads a bound
    #: file, so this is only ever used to test resolution and cache identity.
    input_bindings: dict[str, str] | None = None


@dataclass(frozen=True)
class LegacyParameters:
    expected_ids: tuple[str, ...]
    session: str = ""
    tickers: tuple[str, ...] = ()
    year_start: int = 0
    year_end: int = 0
    horizon_days: int = 35
    sample: int = 10
    force: bool = False
    alt_strikes: int = 1
    expected_population: tuple[str, ...] = ()
    input_bindings: dict[str, str] | None = None
    #: A6: where ``legacy_score_requests`` finds its request batch, relative
    #: to the staging root — under ``legacy/`` once the read set is copied in.
    requests_path: str = ""


def registry():
    kinds = [JobKind(
        name="artifact_check", worker="artifact_check", parameters=CheckParameters,
        resource_classes=frozenset({"delivery"}), effects=("staged",),
        retry=RetryPolicy("bounded", 3, (1, 5)), checkpoint_contract="receipt.v1.0",
        namespaces=frozenset({"shadow", "smoke"}))]
    profiles = {"legacy_score": "legacy_score", "legacy_score_requests": "legacy_score",
                "legacy_finality": "validation",
                "legacy_decisions": "validation", "legacy_settlement": "legacy_rebuild",
                "legacy_model_evidence": "model_evidence", "legacy_render": "projection",
                "legacy_selfcheck": "validation"}
    for action in ("legacy_finality", "legacy_score", "legacy_decisions",
                   "legacy_settlement", "legacy_model_evidence", "legacy_render",
                   "legacy_selfcheck", "legacy_score_requests"):
        kinds.append(JobKind(
            name=action, worker=action, parameters=LegacyParameters,
            resource_classes=frozenset({profiles[action]}),
            effects=("staged",), retry=RetryPolicy("bounded", 2, (5, 30)),
            checkpoint_contract="legacy_action.v1.0",
            namespaces=frozenset({"shadow", "smoke"}),
            store_domains=(("legacy_store", "read"),)))
    return KindRegistry(kinds)


def validate_result(claim, result):
    if result.get("schema_version") != "worker_result.v1.0":
        raise fail("VALIDATION_FAILED", "missing or unsupported worker result")
    if (result.get("job_id"), result.get("attempt_id"), result.get("fence")) != (
            claim.job_id, claim.attempt_id, claim.fence):
        raise fail("LEASE_LOST", "worker result belongs to another attempt")
    expected = claim.spec.parameters["expected_ids"]
    actual = result.get("completed_ids", [])
    if actual != list(expected) or len(actual) != len(set(actual)):
        raise fail("VALIDATION_FAILED", "worker coverage differs", details={"field": "completed_ids"})
    if not actual and result.get("no_work") is not True:
        raise fail("VALIDATION_FAILED", "no-work receipt missing")
    outputs = result.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise fail("VALIDATION_FAILED", "worker output manifest missing")
    names = set()
    for output in outputs:
        if not isinstance(output, dict) or not all(
                isinstance(output.get(field), str) for field in ("name", "path", "schema")):
            raise fail("VALIDATION_FAILED", "worker output manifest is malformed")
        if output["name"] in names:
            raise fail("VALIDATION_FAILED", "worker output names are duplicated")
        names.add(output["name"])
        try:
            safe_relative_path(output["path"])
        except ArtifactError:
            raise fail("VALIDATION_FAILED", "worker output path escapes staging") from None
    return outputs
