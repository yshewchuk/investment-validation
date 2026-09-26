"""S4C: the shared provider-receipt cache and failure classification.

Extracted from tests/test_v2_ops_calendar_moves_jobs.py (P6 slice-4c split,
Part 0) alongside the module it tests.
"""
from __future__ import annotations

from engine.v2.ops.unit_receipts import provider_failure_code


def test_provider_failure_code_orders_mixed_kinds():
    assert provider_failure_code(("complete", "legitimate_empty")) is None
    assert provider_failure_code(("complete", "transient")) == "TRANSIENT_SOURCE"
    assert provider_failure_code(("transient", "not_final")) == "SOURCE_NOT_FINAL"
    # An unparseable body (refused) is bad source data, never a credential
    # problem; only the provider's own 401/403 kind is CREDENTIAL_INVALID.
    assert provider_failure_code(("transient", "refused")) == "SOURCE_INVALID"
    assert provider_failure_code(("refused", "not_final")) == "SOURCE_INVALID"
    assert provider_failure_code(("transient", "credential_invalid")) == "CREDENTIAL_INVALID"
