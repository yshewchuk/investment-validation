"""Synthetic tests of the EXP-188 outer (2025) prediction binder. No store; the fake read refuses any target slot."""
import copy

import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments.magnitude_search_refits import refit_key
from experiments.size_pnl_outer_inputs import bind_outer

FORBIDDEN = ("outer-terminal", "outer-validation", "outer-inputs", "outer-worker-result")
IDS = ["e1", "e2", "e3"]
PLAN = {
    "plan_id": "plan-x",
    "selection": {"winner_rules": {"rmse": "min"}},
    "folds": [{"outer": {"year": 2025, "evaluation_event_ids": IDS}, "inner": []}],
    "blends": [{"blend_id": "b1", "component_candidate_ids": ["c1", "c2"], "name": "b"}],
    "horizons": [1],
}
ROSTER = {"models": [
    {"model_id": "m_c1", "kind": "candidate", "candidate_id": "c1"},
    {"model_id": "m_b1", "kind": "blend", "candidate_id": "b1"},
    {"model_id": "m_prod", "kind": "production", "candidate_id": "size_v1_4"},
]}


def admit(ids):
    return {"admitted": sorted(ids), "dropped": []}


def make_read(slots):
    reads = {}

    def read(slot):
        assert not slot.startswith(FORBIDDEN), "a 2025 target slot was read: " + slot
        reads[slot] = reads.get(slot, 0) + 1
        return slots.get(slot)

    return read, reads


def forecasts(ids, base):
    return [{"event_id": e, "horizon_sessions": 1, "predicted_absolute_move_pct": base + i} for i, e in enumerate(ids)]


def put(slots, cid, *, ids=IDS, base=1.0, plan=PLAN):
    """Store one refit unit's start, execution and forecasts slots, keyed by the real refit_key."""
    key = refit_key(plan, 2025, cid)
    rows = forecasts(ids, base)
    slots["outer-start:" + key] = {"plan_id": plan["plan_id"], "outer_year": 2025,
                                   "candidate": {"candidate_id": cid}, "outer_id": key}
    slots["outer-execution:" + key] = {"status": "succeeded", "forecasts_id": content_hash(rows)}
    slots["outer-forecasts:" + key] = rows
    return key


def cand(cid):
    return {"status": "bound", "years": {"2024": {"pred": [0.5, 0.6, 0.7], "event_ids": IDS, "binding": "bind-" + cid}},
            "binding": "bind0-" + cid}


def blend_entry():
    return {"status": "bound", "kind": "blend", "components": {"c1": cand("c1"), "c2": cand("c2")}, "binding": "blend0"}


def bound_models():
    return {"m_c1": cand("c1"), "m_b1": blend_entry(),
            "m_prod": {"status": "deferred", "reason": "production_bound_by_s2b"}}


def run(slots, plan=PLAN, bound=None, roster=ROSTER):
    read, reads = make_read(slots)
    return bind_outer(plan, read, bound_models() if bound is None else bound, roster, admit, outer_year=2025), reads


def test_candidate_binds_outer_year_and_input_not_mutated():
    slots = {}
    put(slots, "c1", base=1.0)
    bound = bound_models()
    before = copy.deepcopy(bound)
    out, _ = run(slots, bound=bound)
    assert bound == before
    year = out["m_c1"]["years"]["2025"]
    assert list(year["pred"]) == [1.0, 2.0, 3.0]
    assert list(year["event_ids"]) == IDS
    assert out["m_c1"]["binding"] != bound["m_c1"]["binding"]
    assert out["m_c1"]["binding"] == content_hash([[y, out["m_c1"]["years"][y]["binding"]] for y in ["2024", "2025"]])


def test_blend_binds_both_components_and_recomputes_binding():
    slots = {}
    put(slots, "c1", base=1.0)
    put(slots, "c2", base=10.0)
    out, _ = run(slots)
    blend = out["m_b1"]
    assert list(blend["components"]["c1"]["years"]["2025"]["pred"]) == [1.0, 2.0, 3.0]
    assert list(blend["components"]["c2"]["years"]["2025"]["pred"]) == [10.0, 11.0, 12.0]
    assert blend["binding"] != blend_entry()["binding"]
    assert blend["binding"] == content_hash([[cid, blend["components"][cid]["binding"]] for cid in ["c1", "c2"]])


def test_missing_forecasts_exclude_model_and_blend_component():
    slots = {}
    key = put(slots, "c1")
    put(slots, "c2")
    del slots["outer-forecasts:" + key]
    out, _ = run(slots)
    assert out["m_c1"] == {"status": "excluded", "reason": "outer_missing"}
    assert out["m_b1"] == {"status": "excluded", "reason": "component_outer_missing"}


def test_blend_with_missing_component_is_excluded():
    slots = {}
    put(slots, "c1")
    out, _ = run(slots)
    assert out["m_b1"] == {"status": "excluded", "reason": "component_outer_missing"}


def test_failed_execution_excludes_model():
    slots = {}
    key = put(slots, "c1")
    slots["outer-execution:" + key] = {"status": "failed", "forecasts_id": None}
    del slots["outer-forecasts:" + key]
    out, _ = run(slots)
    assert out["m_c1"] == {"status": "excluded", "reason": "outer_failed"}


def test_failed_execution_with_matching_start_is_excluded():
    slots = {}
    key = put(slots, "c1")
    slots["outer-execution:" + key] = {"status": "failed", "forecasts_id": None}
    del slots["outer-forecasts:" + key]
    out, _ = run(slots)
    assert out["m_c1"] == {"status": "excluded", "reason": "outer_failed"}


def test_failed_execution_with_non_null_forecasts_id_refuses():
    slots = {}
    key = put(slots, "c1")
    slots["outer-execution:" + key] = {"status": "failed", "forecasts_id": "sha256:x"}
    del slots["outer-forecasts:" + key]
    assert_identity_mismatch(slots)


def test_failed_execution_with_forecasts_list_refuses():
    slots = {}
    key = put(slots, "c1")
    slots["outer-execution:" + key] = {"status": "failed", "forecasts_id": None}
    assert_identity_mismatch(slots)


def test_failed_execution_with_another_plans_start_refuses():
    slots = {}
    key = put(slots, "c1")
    slots["outer-start:" + key]["plan_id"] = "other-plan"
    slots["outer-execution:" + key] = {"status": "failed", "forecasts_id": None}
    with pytest.raises(OpsError) as exc:
        run(slots)
    assert exc.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.problem.details["reason"] == "outer_identity_mismatch"


@pytest.mark.parametrize("field, value", [("outer_year", 2024), ("candidate", {"candidate_id": "c9"}), ("outer_id", "wrong-key")])
def test_start_with_wrong_identity_field_refuses(field, value):
    slots = {}
    key = put(slots, "c1")
    slots["outer-start:" + key][field] = value
    with pytest.raises(OpsError) as exc:
        run(slots)
    assert exc.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.problem.details["reason"] == "outer_identity_mismatch"


def test_execution_status_other_than_succeeded_or_failed_refuses():
    slots = {}
    key = put(slots, "c1")
    slots["outer-execution:" + key]["status"] = "running"
    with pytest.raises(OpsError) as exc:
        run(slots)
    assert exc.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.problem.details["reason"] == "outer_identity_mismatch"


def test_forecasts_hash_mismatch_refuses():
    slots = {}
    key = put(slots, "c1")
    slots["outer-execution:" + key]["forecasts_id"] = "sha256:wrong"
    with pytest.raises(OpsError) as exc:
        run(slots)
    assert exc.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.problem.details["reason"] == "outer_forecasts_hash_mismatch"


def test_wrong_plan_id_in_start_refuses():
    slots = {}
    key = put(slots, "c1")
    slots["outer-start:" + key]["plan_id"] = "other-plan"
    with pytest.raises(OpsError) as exc:
        run(slots)
    assert exc.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.problem.details["reason"] == "outer_identity_mismatch"


def test_partial_cohort_excludes_model():
    slots = {}
    put(slots, "c1", ids=IDS[:2])
    out, _ = run(slots)
    assert out["m_c1"] == {"status": "excluded", "reason": "partial"}


def test_non_per_candidate_plan_refuses():
    plan = dict(PLAN, selection={})
    bound = {"m_c1": cand("c1")}
    with pytest.raises(OpsError) as exc:
        run({}, plan=plan, bound=bound)
    assert exc.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.problem.details["reason"] == "outer_plan_unsupported"


def test_non_per_candidate_plan_with_only_production_returns_unchanged():
    plan = dict(PLAN, selection={})
    bound = {"m_prod": {"status": "deferred", "reason": "production_bound_by_s2b"}}
    out, reads = run({}, plan=plan, bound=bound)
    assert out == bound
    assert out["m_prod"] is bound["m_prod"]
    assert reads == {}


def test_production_and_excluded_entries_pass_through():
    slots = {}
    put(slots, "c1")
    put(slots, "c2")
    roster = {"models": ROSTER["models"] + [{"model_id": "m_old", "kind": "candidate", "candidate_id": "c3"}]}
    bound = bound_models()
    bound["m_old"] = {"status": "excluded", "reason": "missing"}
    out, reads = run(slots, bound=bound, roster=roster)
    assert out["m_prod"] is bound["m_prod"]
    assert out["m_old"] is bound["m_old"]
    assert refit_key(PLAN, 2025, "c3") not in str(reads)


def test_shared_component_is_read_once():
    plan = dict(PLAN, blends=PLAN["blends"] + [{"blend_id": "b2", "component_candidate_ids": ["c1", "c2"], "name": "b2"}])
    roster = {"models": [
        {"model_id": "m_c1", "kind": "candidate", "candidate_id": "c1"},
        {"model_id": "m_b1", "kind": "blend", "candidate_id": "b1"},
        {"model_id": "m_b2", "kind": "blend", "candidate_id": "b2"},
    ]}
    bound = bound_models()
    bound["m_b2"] = blend_entry()
    slots = {}
    key1 = put(slots, "c1", plan=plan)
    key2 = put(slots, "c2", plan=plan)
    out, reads = run(slots, plan=plan, bound=bound, roster=roster)
    assert reads["outer-forecasts:" + key1] == 1
    assert reads["outer-forecasts:" + key2] == 1
    assert out["m_b2"]["components"]["c1"]["years"]["2025"] is out["m_c1"]["years"]["2025"]


@pytest.mark.parametrize("slot, value", [
    ("outer-start", ["not", "a", "dict"]),
    ("outer-execution", "succeeded"),
    ("outer-forecasts", {"event_id": "e1", "horizon_sessions": 1, "predicted_absolute_move_pct": 1.0}),
])
def test_wrong_shape_slot_refuses(slot, value):
    slots = {}
    key = put(slots, "c1")
    slots[slot + ":" + key] = value
    with pytest.raises(OpsError) as exc:
        run(slots)
    assert exc.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.problem.details["reason"] == "outer_identity_mismatch"


def test_forecast_row_without_predicted_move_refuses():
    slots = {}
    key = put(slots, "c1")
    rows = forecasts(IDS, 1.0)
    del rows[1]["predicted_absolute_move_pct"]
    slots["outer-forecasts:" + key] = rows
    slots["outer-execution:" + key]["forecasts_id"] = content_hash(rows)
    with pytest.raises(OpsError) as exc:
        run(slots)
    assert exc.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.problem.details["reason"] == "outer_identity_mismatch"


def test_all_three_outer_slots_missing_excludes_model():
    out, _ = run({})
    assert out["m_c1"] == {"status": "excluded", "reason": "outer_missing"}


def assert_identity_mismatch(slots):
    with pytest.raises(OpsError) as exc:
        run(slots)
    assert exc.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.problem.details["reason"] == "outer_identity_mismatch"


def test_empty_execution_with_valid_start_refuses():
    slots = {}
    key = put(slots, "c1")
    slots["outer-execution:" + key] = {}
    del slots["outer-forecasts:" + key]
    assert_identity_mismatch(slots)


def test_succeeded_execution_without_forecasts_is_outer_missing():
    slots = {}
    key = put(slots, "c1")
    del slots["outer-forecasts:" + key]
    out, _ = run(slots)
    assert out["m_c1"] == {"status": "excluded", "reason": "outer_missing"}


def test_bad_forecast_row_with_missing_execution_refuses():
    slots = {}
    key = put(slots, "c1")
    del slots["outer-execution:" + key]
    del slots["outer-forecasts:" + key][1]["predicted_absolute_move_pct"]
    assert_identity_mismatch(slots)


def test_forecast_row_carrying_target_field_refuses():
    slots = {}
    key = put(slots, "c1")
    rows = forecasts(IDS, 1.0)
    rows[0]["absolute_move_pct"] = 1.5
    slots["outer-forecasts:" + key] = rows
    slots["outer-execution:" + key]["forecasts_id"] = content_hash(rows)
    assert_identity_mismatch(slots)


def test_nonfinite_prediction_in_partial_cohort_refuses():
    slots = {}
    key = put(slots, "c1", ids=IDS[:2])
    rows = forecasts(IDS[:2], 1.0)
    rows[0]["predicted_absolute_move_pct"] = float("nan")
    slots["outer-forecasts:" + key] = rows
    slots["outer-execution:" + key]["forecasts_id"] = "sha256:not-checked"  # shape check runs before the hash check
    assert_identity_mismatch(slots)


def test_execution_status_running_refuses():
    slots = {}
    key = put(slots, "c1")
    slots["outer-execution:" + key]["status"] = "running"
    assert_identity_mismatch(slots)


def test_execution_and_forecasts_without_start_refuses_before_hash_check():
    slots = {}
    key = put(slots, "c1")
    del slots["outer-start:" + key]
    slots["outer-execution:" + key]["forecasts_id"] = "sha256:corrupt"
    assert_identity_mismatch(slots)


def test_forecasts_without_execution_refuses():
    slots = {}
    key = put(slots, "c1")
    del slots["outer-execution:" + key]
    assert_identity_mismatch(slots)


def test_execution_without_start_refuses():
    slots = {}
    key = put(slots, "c1")
    del slots["outer-start:" + key]
    assert_identity_mismatch(slots)


def test_start_only_is_outer_missing():
    slots = {}
    key = put(slots, "c1")
    del slots["outer-execution:" + key]
    del slots["outer-forecasts:" + key]
    out, _ = run(slots)
    assert out["m_c1"] == {"status": "excluded", "reason": "outer_missing"}
