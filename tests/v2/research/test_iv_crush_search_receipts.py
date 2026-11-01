"""Independent signed receipt oracle; rehash corruptions to avoid stale-hash refusals."""
from copy import deepcopy

import pytest

from engine.v2.foundation import content_hash
from engine.v2.research.iv_crush_inputs import validate_iv_crush_receipt


def evidence():
    context = {"snapshot_id": "target-snapshot", "holdout_as_of_month": "2025-01",
        "random_membership_version": "random-v1", "rolling_membership_version": "rolling-v1",
        "population_use": "post-release selection", "purpose": "selection",
        "target_version": {"dataset_version_id": "target-version", "contract": "IV30-relative"},
        "availability_basis": "modeled_post_close_plus_one_calendar_day",
        "source_publication": "unverified", "source_finality": "unverified"}
    planned = [{"event_id": event, "fragment_id": "fragment-" + event,
        "object_id": "object-" + event, "content_hash": "sha256:" + digit * 64}
        for event, digit in (("train", "a"), ("first", "b"), ("second", "c"))]
    preflight = {**context, "schema_version": "research_iv_crush_preflight.v1",
        "event_ids": [row["event_id"] for row in planned], "planned_inputs": planned}
    preflight["preflight_id"] = content_hash(preflight)
    identity = {**context, "schema_version": "research_iv_crush_targets.v1",
        "labels_before": "2025-01-01", "excluded_future_events": [],
        "consumed_inputs": [{**row, "source_id": "declared-caller", "input_hash": "sha256:" + "d" * 64}
                            for row in planned[1:]]}
    receipt = {**identity, "inputs_id": content_hash(identity), "holdout_exclusions": []}
    fold = {"evaluation_event_ids": ["second", "first"], "evaluation_labels_before": "2025-01-01"}
    targets = [{"event_id": "first", "crush_pct_iv30": -140.}, {"event_id": "second", "crush_pct_iv30": 25.}]
    return preflight, fold, receipt, targets


def test_signed_receipt_matches_complete_independent_oracle_without_mutation():
    values = evidence()
    before = deepcopy(values)
    assert validate_iv_crush_receipt(*values) is None
    assert values == before


@pytest.mark.parametrize("change", ["snapshot", "cutoff", "purpose", "publication", "version", "extra",
    "future", "holdout", "object", "source", "hash", "consumed_order", "target_order", "duplicate",
    "missing", "target_extra", "nan", "infinity", "boolean", "string"])
def test_self_consistent_receipt_corruptions_refuse(change):
    preflight, fold, receipt, targets = evidence()
    fields = {"snapshot": "snapshot_id", "cutoff": "labels_before", "purpose": "purpose",
              "publication": "source_publication", "version": "target_version", "extra": "unknown"}
    if change in fields:
        receipt[fields[change]] = "changed"
    elif change in {"future", "holdout"}:
        receipt["excluded_future_events" if change == "future" else "holdout_exclusions"] = ["first"]
    elif change in {"object", "source", "hash"}:
        receipt["consumed_inputs"][0][{"object": "object_id", "source": "source_id", "hash": "input_hash"}[change]] = " "
    elif change == "consumed_order":
        receipt["consumed_inputs"].reverse()
    elif change == "target_order":
        targets.reverse()
    elif change == "duplicate":
        targets[1] = dict(targets[0])
    elif change == "missing":
        targets.pop()
    elif change == "target_extra":
        targets[0]["ignored"] = 1
    else:
        targets[0]["crush_pct_iv30"] = {"nan": float("nan"), "infinity": float("inf"),
                                       "boolean": True, "string": "-140"}[change]
    receipt["inputs_id"] = content_hash({key: value for key, value in receipt.items()
                                       if key not in {"inputs_id", "holdout_exclusions"}})
    with pytest.raises(ValueError):
        validate_iv_crush_receipt(preflight, fold, receipt, targets)
