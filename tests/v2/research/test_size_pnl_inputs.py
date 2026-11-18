"""EXP-188 S2a: binding of synthetic v14 trial predictions. No store, no repository."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import size_pnl_inputs as sp
from experiments.magnitude_search_plan import PLAN_V3, trial_key

CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
PRED = "predicted_absolute_move_pct"
E23 = ["e1", "e2", "e3", "e4"]
E24 = ["e5", "e6", "e7", "e8"]
SEAL = {"seal": "s1"}
PANEL = "panel-1"
RUN_READ = {"driver-seal": SEAL}.get
SELECTION = {"selections": [{"outer_year": 2025, "selected_candidate_id": "c1",
                             "winners": {"rmse": {"candidate_id": "c2"},
                                         "mae": {"candidate_id": "c1"}}}]}
PAIR_MAP = {"H-sq": "c2", "H-abs": "c2", "E-raw": "c1", "E-log": "c1", "N-raw": "c2", "N-log": "c2"}
BLEND_MAP = {"M4": "b1", "M5": "b1"}


def _plan(**extra):
    plan = {
        "plan_id": "p1", "anchor": {"run_id": "r1"}, "horizons": [5, 3, 1],
        "candidates": [
            {"candidate_id": "c1", "recipe": {"schema_version": "magnitude_elasticnet_recipe.v2"}},
            {"candidate_id": "c2", "recipe": {"schema_version": "magnitude_raw_recipe.v1"}},
        ],
        "blends": [{"blend_id": "b1", "component_candidate_ids": ["c1", "c2"]}],
        "selection": {"extra_refits": ["c1", "c2"]},
        "folds": [{"outer": {"year": 2025}, "inner": [
            {"year": 2023, "evaluation_event_ids": E23},
            {"year": 2024, "evaluation_event_ids": E24}]}],
    }
    plan.update(extra)
    return plan


def _record(event_ids, *, y=0.5, horizons=(5, 3, 1), status="succeeded"):
    preds = [{"event_id": eid, "horizon_sessions": h, PRED: 1.0 + h, "absolute_move_pct": y}
             for eid in event_ids for h in horizons]
    return {"status": status, "predictions": preds}


def _key(plan, inner_year, candidate_id):
    return "terminal:" + trial_key(plan, 2025, inner_year, candidate_id)


def _store(plan, cids=("c1", "c2")):
    store = {"driver-seal": SEAL}
    for group in plan["folds"]:
        for inner in group["inner"]:
            for cid in cids:
                key = "terminal:" + trial_key(plan, group["outer"]["year"], inner["year"], cid)
                store[key] = _record(inner["evaluation_event_ids"])
    return store


def _admit_all(ids):
    return {"admitted": sorted(ids), "dropped": []}


def _expected(**over):
    base = {"plan_id": "p1", "registration_id": "r1", "seal_id": content_hash(SEAL), "panel_id": PANEL}
    base.update(over)
    return base


def _refusal(exc):
    return exc.value.problem.code, exc.value.problem.details.get("reason")


def _roster(plan=None, selection=SELECTION, outer=2025, pairs=PAIR_MAP, blends=BLEND_MAP):
    return sp.resolve_roster(plan or _plan(), selection, outer_year=outer,
                             pair_candidates=pairs, blend_ids=blends)


def test_bind_run_ok():
    assert sp.bind_run(_plan(), RUN_READ, expected=_expected(), panel_id=PANEL) == _expected()


@pytest.mark.parametrize("key", ["plan_id", "registration_id", "seal_id"])
def test_bind_run_mismatch_refuses(key):
    with pytest.raises(OpsError) as exc:
        sp.bind_run(_plan(), RUN_READ, expected=_expected(**{key: "wrong"}), panel_id=PANEL)
    assert _refusal(exc) == (CONFLICT, key + "_mismatch")


def test_bind_run_panel_mismatch_refuses():
    with pytest.raises(OpsError) as exc:
        sp.bind_run(_plan(), RUN_READ, expected=_expected(), panel_id="other")
    assert _refusal(exc) == (CONFLICT, "panel_id_mismatch")


def test_bind_run_missing_expected_key_refuses():
    expected = _expected()
    del expected["seal_id"]
    with pytest.raises(OpsError) as exc:
        sp.bind_run(_plan(), RUN_READ, expected=expected, panel_id=PANEL)
    assert _refusal(exc) == (CONFLICT, "seal_id_mismatch")


def test_bind_run_lead_absent_refuses():
    with pytest.raises(OpsError) as exc:
        sp.bind_run(_plan(horizons=[5, 3]), RUN_READ, expected=_expected(), panel_id=PANEL)
    assert _refusal(exc) == (CONFLICT, "lead_unavailable")


def test_bind_run_missing_seal_refuses():
    with pytest.raises(OpsError) as exc:
        sp.bind_run(_plan(), {}.get, expected=_expected(), panel_id=PANEL)
    assert _refusal(exc) == (CONFLICT, "seal_unavailable")


def test_bind_array_happy_path_and_horizon_filter():
    out = sp.bind_array(_record(E23), E23, _admit_all, horizon=1)
    assert list(out["event_ids"]) == E23
    assert out["event_ids"].dtype == object
    assert np.allclose(out["pred"], 2.0)
    assert np.allclose(sp.bind_array(_record(E23), E23, _admit_all, horizon=5)["pred"], 6.0)


def test_outcome_values_are_never_read():
    clean = sp.bind_array(_record(E23, y=0.5), E23, _admit_all)
    poison = sp.bind_array(_record(E23, y="POISON"), E23, _admit_all)
    assert np.array_equal(clean["pred"], poison["pred"])
    assert "POISON" not in repr(poison)


def test_binding_changes_with_any_stored_byte_or_admission():
    base = sp.bind_array(_record(E23), E23, _admit_all)["binding"]
    assert sp.bind_array(_record(E23, y=0.7), E23, _admit_all)["binding"] != base
    record = _record(E23)
    record["predictions"][0]["note"] = "x"
    assert sp.bind_array(record, E23, _admit_all)["binding"] != base

    def admit_v2(ids):
        return {"admitted": sorted(ids), "dropped": [], "rev": 2}

    assert sp.bind_array(_record(E23), E23, admit_v2)["binding"] != base


def test_outside_id_refused_before_admission_is_called():
    calls = []

    def recording(ids):
        calls.append(list(ids))
        return _admit_all(ids)

    with pytest.raises(OpsError) as exc:
        sp.bind_array(_record(E23[:3] + ["zz"]), E23, recording)
    assert _refusal(exc) == (CONFLICT, "event_ids_outside_population")
    assert calls == []


def test_partial_subset_refused():
    with pytest.raises(OpsError) as exc:
        sp.bind_array(_record(E23[:2]), E23, _admit_all)
    assert _refusal(exc) == (CONFLICT, "partial_event_ids")


def test_duplicate_ids_refused():
    record = _record(E23)
    record["predictions"].append(dict(record["predictions"][-1]))
    with pytest.raises(OpsError) as exc:
        sp.bind_array(record, E23, _admit_all)
    assert _refusal(exc) == (CONFLICT, "duplicate_event_id")


def test_nonfinite_prediction_refused():
    record = _record(E23)
    for row in record["predictions"]:
        if row["horizon_sessions"] == 1:
            row[PRED] = float("nan")
    with pytest.raises(OpsError) as exc:
        sp.bind_array(record, E23, _admit_all)
    assert _refusal(exc) == (CONFLICT, "nonfinite_prediction")


def test_admission_dropping_an_id_refused():
    def admit_drop(ids):
        return {"admitted": sorted(ids)[:-1], "dropped": sorted(ids)[-1:]}

    with pytest.raises(OpsError) as exc:
        sp.bind_array(_record(E23), E23, admit_drop)
    assert _refusal(exc) == (CONFLICT, "admission_dropped")


def test_generic_synthetic_record_binds():
    record = {"status": "succeeded", "predictions": [
        {"event_id": "g1", "horizon_sessions": 1, PRED: 0.25},
        {"event_id": "g2", "horizon_sessions": 1, PRED: 0.5}]}
    out = sp.bind_array(record, ["g1", "g2"], _admit_all)
    assert list(out["event_ids"]) == ["g1", "g2"]
    assert np.allclose(out["pred"], [0.25, 0.5])


def test_bind_candidate_bound():
    plan = _plan()
    out = sp.bind_candidate(plan, _store(plan).get, "c1", _admit_all)
    assert out["status"] == "bound"
    assert sorted(out["years"]) == ["2023", "2024"]
    assert out["binding"] == sp.bind_candidate(plan, _store(plan).get, "c1", _admit_all)["binding"]


def test_bind_candidate_missing_failed_partial_are_excluded():
    plan = _plan()
    store = _store(plan)
    del store[_key(plan, 2024, "c1")]
    assert sp.bind_candidate(plan, store.get, "c1", _admit_all) == {"status": "excluded", "reason": "missing"}
    for body, reason in ((_record(E23, status="failed"), "failed"), (["x"], "failed"),
                         (_record(E23[:2]), "partial")):
        store = _store(plan)
        store[_key(plan, 2023, "c1")] = body
        assert sp.bind_candidate(plan, store.get, "c1", _admit_all) == {"status": "excluded", "reason": reason}


def test_bind_candidate_conflict_propagates():
    plan = _plan()

    def admit_drop(ids):
        return {"admitted": [], "dropped": list(ids)}

    with pytest.raises(OpsError) as exc:
        sp.bind_candidate(plan, _store(plan).get, "c1", admit_drop)
    assert _refusal(exc) == (CONFLICT, "admission_dropped")


def test_v3_shared_trials_are_read_once():
    inner = _plan()["folds"][0]["inner"]
    plan = _plan(schema_version=PLAN_V3,
                 folds=[{"outer": {"year": y}, "inner": inner} for y in (2025, 2026)])
    store = _store(plan)
    calls = []

    def read(slot):
        calls.append(slot)
        return store.get(slot)

    assert sp.bind_candidate(plan, read, "c1", _admit_all)["status"] == "bound"
    assert len(calls) == 2


def test_population_admit_reads_frame_with_selection_purpose():
    seen = {}

    def load(repository, snapshot, *, as_of_month, purpose, event_ids):
        seen.update(as_of_month=as_of_month, purpose=purpose, event_ids=event_ids)
        return pd.DataFrame({"event_id": ["e2", "e1"]})

    admit = sp.population_admit("repo", "snap", "2025-01", load=load)
    assert admit(["e2", "e1"]) == {"admitted": ["e1", "e2"], "dropped": []}
    assert seen == {"as_of_month": "2025-01", "purpose": "selection", "event_ids": ["e1", "e2"]}


def test_population_admit_maps_ops_refusal_to_conflict():
    def load(*args, **kwargs):
        raise fail("SNAPSHOT_UNRESOLVED", "unresolved")

    admit = sp.population_admit("repo", "snap", "2025-01", load=load)
    with pytest.raises(OpsError) as exc:
        admit(["e1"])
    assert _refusal(exc) == (CONFLICT, "admission_refused")


def test_resolve_roster_roles_collapse_and_log_forms():
    roster = _roster()
    by_role = {role: m for m in roster["models"] for role in m["roles"]}
    assert by_role["M1"] is by_role["M3"]
    assert by_role["M1"]["roles"] == ["M1", "M3", "E-raw", "E-log"]
    assert by_role["M1"]["target_transform"] == "log1p"
    assert by_role["M2"]["candidate_id"] == "c2"
    assert by_role["M2"]["target_transform"] == "none"
    assert by_role["M0"]["source"] == "production"
    assert by_role["M4"]["target_transform"] == "mixed"
    assert len(roster["models"]) == 4
    assert roster["not_testable"] == []
    assert roster["pairs"] == {"H": "testable", "E": "testable", "N": "testable"}


def test_unresolved_winners_and_unknown_candidates_are_not_testable():
    reasons = {r["role"]: r["reason"] for r in _roster(outer=2026)["not_testable"]}
    assert reasons["M1"] == reasons["M2"] == reasons["M3"] == "winner_unresolved"
    unknown = {"selections": [{"outer_year": 2025, "selected_candidate_id": "zz",
                               "winners": {"rmse": {"candidate_id": "c2"}, "mae": {"candidate_id": "c1"}}}]}
    assert {r["role"]: r["reason"] for r in _roster(selection=unknown)["not_testable"]} == {"M1": "not_in_plan"}


def test_pair_not_in_extra_refits_or_unmapped_is_not_testable():
    roster = _roster(_plan(selection={"extra_refits": ["c2"]}))
    reasons = {r["role"]: r["reason"] for r in roster["not_testable"]}
    assert reasons["E-raw"] == reasons["E-log"] == "not_in_extra_refits"
    assert roster["pairs"] == {"H": "testable", "E": "not_testable", "N": "testable"}
    partial = {k: v for k, v in PAIR_MAP.items() if k != "N-log"}
    roster = _roster(pairs=partial, blends={"M4": "b1"})
    reasons = {r["role"]: r["reason"] for r in roster["not_testable"]}
    assert reasons == {"N-log": "unmapped", "M5": "unmapped"}
    assert roster["pairs"]["N"] == "not_testable"


def test_too_many_models_refused(monkeypatch):
    monkeypatch.setattr(sp, "MAX_MODELS", 3)
    with pytest.raises(OpsError) as exc:
        _roster()
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "too_many_models")


def test_bind_models_counts_and_defers_production(capsys):
    plan = _plan()
    roster = _roster(plan)
    out = sp.bind_models(plan, _store(plan).get, roster, _admit_all)
    production = [m["model_id"] for m in roster["models"] if m["source"] == "production"]
    assert out[production[0]] == {"status": "deferred", "reason": "production_bound_by_s2b"}
    assert sorted(r["status"] for r in out.values()) == ["bound", "bound", "deferred", "excluded"]
    assert "models=4 bound=2 excluded=1 deferred=1" in capsys.readouterr().err


def test_module_source_never_names_outcome_key():
    text = Path(sp.__file__).read_text(encoding="utf-8")
    assert '"absolute_move_pct"' not in text
    assert "'absolute_move_pct'" not in text
