"""Changed refusal pins conflict across attempt receipt identities."""
from __future__ import annotations

import csv
from types import SimpleNamespace

import pytest

from engine.v2.ops import effects_graph, experiments
from engine.v2.ops.errors import OpsError, make_problem

_EXPERIMENT_ID = "EXP-489-pin-identity"
_VARIANT_ID = "resolved-variant-identity"
_PINS = {
    "snapshot_id": "snap-pin-identity",
    "holdout_as_of_month": "2025-01",
    "random_membership_version": "canonical-event-sha256.v1",
    "rolling_membership_version": "calendar-months.v1",
}


def test_changed_receipt_pins_conflict_across_attempt_effects(tmp_path):
    claim = SimpleNamespace(spec=SimpleNamespace(
        kind="experiment", parameters={"no_ledger": False}))
    problem = make_problem("HOLDOUT_ACCESS_DENIED", "holdout refused")

    def refusal_receipt(pins):
        return {"schema_version": experiments.REFUSAL_RECEIPT_SCHEMA,
                "status": "refused", "failure_code": "HOLDOUT_ACCESS_DENIED",
                "experiment_id": _EXPERIMENT_ID, "variant_id": _VARIANT_ID,
                **pins}

    def replay(pins):
        effect = effects_graph._experiment_refusal_failure_effect(
            claim, problem, code_source=tmp_path, store_root=None,
            refusal_receipt=refusal_receipt(pins))
        assert effect is not None
        effect(None)

    replay(dict(_PINS))
    replay(dict(_PINS))
    with pytest.raises(OpsError) as conflict:
        replay({**_PINS, "holdout_as_of_month": "2025-05"})
    assert conflict.value.code == "IDEMPOTENCY_CONFLICT"

    ledger = experiments.experiments_ledger_path(tmp_path)
    with open(ledger, newline="") as fh:
        refused = [row for row in csv.DictReader(fh) if row["stage"] == "refused"]
    assert len(refused) == 1
    assert refused[0]["id"] == _EXPERIMENT_ID
    assert refused[0]["spec_hash"] == _VARIANT_ID
