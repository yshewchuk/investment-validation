"""Scalar provider requirements without changing persisted job documents."""
from __future__ import annotations

from dataclasses import dataclass

from engine.v2.contracts import JobSpec
from engine.v2.ops.errors import fail


@dataclass(frozen=True)
class ProviderRequirement:
    """An account and its uncoerced estimate; scheduler guards validate counts."""

    account: str
    calls: object


def validate_scalar_provider_shape(parameters: dict) -> None:
    """Do not admit a per-account declaration before it can be enforced."""
    if "provider_calls_by_account" in parameters:
        raise fail("INVALID_REQUEST", "per-account provider requirements are not supported",
                   details={"field": "job.parameters.provider_calls_by_account"})


def provider_call_count(parameters: object) -> object:
    """Preserve the scalar estimate and historical missing-count default."""
    return parameters.get("provider_calls", 1) if isinstance(parameters, dict) else 1


def provider_requirements(job: JobSpec) -> tuple[ProviderRequirement, ...]:
    """Resolve zero or one scalar requirement without rewriting the job."""
    validate_scalar_provider_shape(job.parameters)
    if not job.provider_budget_ref:
        return ()
    return (ProviderRequirement(job.provider_budget_ref, provider_call_count(job.parameters)),)
