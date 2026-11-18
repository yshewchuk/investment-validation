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
PAIR_MAP = {"H-sq": "c-hsq", "H-abs": "c-habs", "E-raw": "c-eraw", "E-log": "c-elog",
            "N-raw": "c-nraw", "N-log": "c-nlog"}
PAIR_IDS = tuple(PAIR_MAP.values())
BLEND_MAP = {"M4": "b1", "M5": "b1"}


def _hgb_recipe(loss):
    return {"schema_version": "magnitude_ridge_hgb_recipe.v2", "model": "history_hgb",
            "estimator": {"class": "HistGradientBoostingRegressor", "params": {
                "loss": loss, "learning_rate": 0.03, "l2_regularization": 10.0,
                "max_iter": 3000, "max_leaf_nodes": 15, "min_samples_leaf": 300}},
            "target_transform": "none"}


def _en_recipe(schema, alpha=0.03):
    return {"schema_version": schema, "model": "elasticnet",
            "estimator": {"class": "ElasticNet", "params": {"alpha": alpha, "l1_ratio": 0.75}},
            "preprocessing": {"spline": {"n_knots": 6, "degree": 3}}}


def _mlp_recipe(schema, profile="chrono15_relu64x32_l2_1"):
    return {"schema_version": schema, "model": "mlp", "profile": profile,
            "estimator": {"class": "MLPRegressor", "params": {"hidden_layer_sizes": [64, 32]}}}


def _pair_candidates():
    """Six pair members with the real recipe shapes for their roles (H-sq, H-abs, E-raw, E-log, N-raw, N-log)."""
    return [
        {"candidate_id": "c-hsq", "recipe": _hgb_recipe("squared_error")},
        {"candidate_id": "c-habs", "recipe": _hgb_recipe("absolute_error")},
        {"candidate_id": "c-eraw", "recipe": _en_recipe("magnitude_elasticnet_spline_recipe.v1")},
        {"candidate_id": "c-elog", "recipe": _en_recipe("magnitude_elasticnet_spline_recipe.v2")},
        {"candidate_id": "c-nraw", "recipe": _mlp_recipe("magnitude_mlp_recipe.v1")},
        {"candidate_id": "c-nlog", "recipe": _mlp_recipe("magnitude_mlp_recipe.v2")},
    ]


def _plan(**extra):
    plan = {
        "plan_id": "p1", "anchor": {"run_id": "r1"}, "horizons": [5, 3, 1],
        "candidates": [
            {"candidate_id": "c1", "recipe": {"schema_version": "magnitude_elasticnet_recipe.v2"}},
            {"candidate_id": "c2", "recipe": {"schema_version": "magnitude_raw_recipe.v1"}},
            *_pair_candidates(),
        ],
        "blends": [{"blend_id": "b1", "component_candidate_ids": ["c1", "c2"]}],
        "selection": {"extra_refits": ["c1", "c2", *PAIR_IDS]},
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


def _store(plan, cids=("c1", "c2") + PAIR_IDS):
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


def _twelve_args():
    """Plan and roster inputs that resolve to exactly 12 distinct models (all roles distinct)."""
    plan = _plan(
        candidates=[{"candidate_id": f"w{i}", "recipe": c["recipe"]}
                    for i, c in enumerate(_pair_candidates())],
        blends=[{"blend_id": f"bl{i}", "component_candidate_ids": ["w0", "w1"]} for i in range(1, 6)],
        selection={"extra_refits": [f"w{i}" for i in range(6)]})
    selection = {"selections": [{"outer_year": 2025, "selected_candidate_id": "bl1",
                                 "winners": {"rmse": {"candidate_id": "bl2"},
                                             "mae": {"candidate_id": "bl3"}}}]}
    return {"plan": plan, "selection": selection, "outer_year": 2025,
            "pair_candidates": {"H-sq": "w0", "H-abs": "w1", "E-raw": "w2",
                                "E-log": "w3", "N-raw": "w4", "N-log": "w5"},
            "blend_ids": {"M4": "bl4", "M5": "bl5"}}


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
    out = sp.bind_array(_record(E23), E23, _admit_all)
    assert list(out["event_ids"]) == E23
    assert out["event_ids"].dtype == object
    assert np.allclose(out["pred"], 2.0)


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
    assert list(out["years"]["2023"]["pred"]) == [2.0] * len(E23)
    assert list(out["years"]["2024"]["pred"]) == [2.0] * len(E24)
    assert list(out["years"]["2023"]["event_ids"]) == E23
    assert list(out["years"]["2024"]["event_ids"]) == E24
    assert out["binding"] == content_hash(
        [[year, out["years"][year]["binding"]] for year in sorted(out["years"])])


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


def test_repeated_inner_year_under_distinct_trials_refused(monkeypatch):
    monkeypatch.setattr(sp, "trial_key",
                        lambda plan, outer, inner, cid: f"{outer}:{trial_key(plan, outer, inner, cid)}")
    inner = _plan()["folds"][0]["inner"]
    plan = _plan(folds=[{"outer": {"year": y}, "inner": inner} for y in (2025, 2026)])
    store = {"driver-seal": SEAL}
    for group in plan["folds"]:
        for item in group["inner"]:
            key = "terminal:" + sp.trial_key(plan, group["outer"]["year"], item["year"], "c1")
            store[key] = _record(item["evaluation_event_ids"])
    with pytest.raises(OpsError) as exc:
        sp.bind_candidate(plan, store.get, "c1", _admit_all)
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "repeated_inner_year")


def test_population_admit_reads_frame_with_selection_purpose():
    seen = {}

    def load(repository, snapshot, *, as_of_month, purpose, event_ids, max_object_bytes):
        seen.update(as_of_month=as_of_month, purpose=purpose, event_ids=event_ids,
                    max_object_bytes=max_object_bytes)
        return pd.DataFrame({"event_id": ["e1", "e3"]})

    admit = sp.population_admit("repo", "snap", "2025-01", max_object_bytes=4096, load=load)
    assert admit(["e2", "e1"]) == {"admitted": ["e1", "e3"], "dropped": []}
    assert seen == {"as_of_month": "2025-01", "purpose": "selection", "event_ids": ["e1", "e2"],
                    "max_object_bytes": 4096}
    with pytest.raises(OpsError) as exc:
        sp.bind_array(_record(["e1", "e2"]), ["e1", "e2"], admit)
    assert _refusal(exc) == (CONFLICT, "admission_dropped")


@pytest.mark.parametrize("bad", [0, -1, True, None])
def test_population_admit_refuses_invalid_object_cap(bad):
    with pytest.raises(OpsError) as exc:
        sp.population_admit("repo", "snap", "2025-01", max_object_bytes=bad,
                            load=lambda *args, **kwargs: None)
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "max_object_bytes_invalid")


def test_population_admit_propagates_ops_refusal_unchanged():
    original = fail("SNAPSHOT_UNRESOLVED", "unresolved", details={"reason": "snapshot_missing"})

    def load(*args, **kwargs):
        raise original

    admit = sp.population_admit("repo", "snap", "2025-01", max_object_bytes=4096, load=load)
    with pytest.raises(OpsError) as exc:
        admit(["e1"])
    assert exc.value.problem == original.problem


def test_resolve_roster_roles_collapse_and_log_forms():
    roster = _roster()
    by_role = {role: m for m in roster["models"] for role in m["roles"]}
    assert by_role["M1"] is by_role["M3"]
    assert by_role["M1"]["roles"] == ["M1", "M3"]
    assert by_role["M1"]["target_transform"] == "log1p"
    assert by_role["M2"]["candidate_id"] == "c2"
    assert by_role["M2"]["target_transform"] == "none"
    assert by_role["M2"]["roles"] == ["M2"]
    assert by_role["H-abs"]["candidate_id"] == "c-habs"
    assert by_role["N-log"]["target_transform"] == "log1p"
    assert by_role["M0"]["source"] == "production"
    assert by_role["M4"]["target_transform"] == "mixed"
    assert len(roster["models"]) == 10
    assert roster["not_testable"] == []
    assert roster["pairs"] == {"H": "testable", "E": "testable", "N": "testable"}


def test_unresolved_winners_and_unknown_candidates_are_not_testable():
    reasons = {r["role"]: r["reason"] for r in _roster(outer=2026)["not_testable"]}
    assert reasons["M1"] == reasons["M2"] == reasons["M3"] == "winner_unresolved"
    unknown = {"selections": [{"outer_year": 2025, "selected_candidate_id": "zz",
                               "winners": {"rmse": {"candidate_id": "c2"}, "mae": {"candidate_id": "c1"}}}]}
    assert {r["role"]: r["reason"] for r in _roster(selection=unknown)["not_testable"]} == {"M1": "not_in_plan"}


def test_blend_winner_resolves_as_mixed_model():
    blend_winner = {"selections": [{"outer_year": 2025, "selected_candidate_id": "b1",
                                    "winners": {"rmse": {"candidate_id": "c2"},
                                                "mae": {"candidate_id": "c1"}}}]}
    roster = _roster(selection=blend_winner)
    by_role = {role: m for m in roster["models"] for role in m["roles"]}
    assert by_role["M1"]["candidate_id"] == "b1"
    assert by_role["M1"]["target_transform"] == "mixed"
    assert by_role["M1"]["kind"] == "blend"
    assert "M1" not in {r["role"] for r in roster["not_testable"]}


def test_winner_neither_candidate_nor_blend_is_not_in_plan():
    neither = {"selections": [{"outer_year": 2025, "selected_candidate_id": "c1",
                               "winners": {"rmse": {"candidate_id": "zz"},
                                           "mae": {"candidate_id": "c1"}}}]}
    reasons = {r["role"]: r["reason"] for r in _roster(selection=neither)["not_testable"]}
    assert reasons == {"M2": "not_in_plan"}


def test_pair_not_in_extra_refits_or_unmapped_is_not_testable():
    roster = _roster(_plan(selection={"extra_refits": ["c1", "c2", "c-hsq", "c-eraw", "c-nraw"]}))
    reasons = {r["role"]: r["reason"] for r in roster["not_testable"]}
    assert reasons == {"H-abs": "not_in_extra_refits", "E-log": "not_in_extra_refits",
                       "N-log": "not_in_extra_refits"}
    assert roster["pairs"] == {"H": "not_testable", "E": "not_testable", "N": "not_testable"}
    partial = {k: v for k, v in PAIR_MAP.items() if k != "N-log"}
    roster = _roster(pairs=partial, blends={"M4": "b1"})
    reasons = {r["role"]: r["reason"] for r in roster["not_testable"]}
    assert reasons == {"N-log": "unmapped", "M5": "unmapped"}
    assert roster["pairs"]["N"] == "not_testable"


@pytest.mark.parametrize("role, mapped, pair", [("E-raw", "c1", "E"), ("N-log", "c2", "N")])
def test_transform_mismatch_makes_role_and_pair_not_testable(role, mapped, pair):
    roster = _roster(pairs={**PAIR_MAP, role: mapped})
    assert {r["role"]: r["reason"] for r in roster["not_testable"]} == {role: "transform_mismatch"}
    assert roster["pairs"] == {p: ("not_testable" if p == pair else "testable") for p in sp.PAIRS}


def test_pair_on_one_candidate_cannot_satisfy_both_losses():
    roster = _roster(pairs={**PAIR_MAP, "H-sq": "c1", "H-abs": "c1"})
    assert {r["role"]: r["reason"] for r in roster["not_testable"]} == {
        "H-sq": "recipe_mismatch", "H-abs": "recipe_mismatch"}
    assert roster["pairs"] == {"H": "not_testable", "E": "testable", "N": "testable"}


def test_pair_mapped_to_wrong_loss_is_recipe_mismatch():
    roster = _roster(pairs={**PAIR_MAP, "H-sq": "c-habs"})
    assert {r["role"]: r["reason"] for r in roster["not_testable"]} == {"H-sq": "recipe_mismatch"}
    assert roster["pairs"] == {"H": "not_testable", "E": "testable", "N": "testable"}


def test_old_unrelated_mapping_is_recipe_mismatch():
    roster = _roster(pairs={**PAIR_MAP, "H-sq": "c1"})
    assert {r["role"]: r["reason"] for r in roster["not_testable"]} == {"H-sq": "recipe_mismatch"}
    assert roster["pairs"]["H"] == "not_testable"


def test_e_raw_wrong_alpha_is_recipe_mismatch():
    plan = _plan()
    plan["candidates"].append({"candidate_id": "c-ewrong",
                               "recipe": _en_recipe("magnitude_elasticnet_spline_recipe.v1", alpha=0.01)})
    plan["selection"]["extra_refits"].append("c-ewrong")
    roster = _roster(plan, pairs={**PAIR_MAP, "E-raw": "c-ewrong"})
    assert {r["role"]: r["reason"] for r in roster["not_testable"]} == {"E-raw": "recipe_mismatch"}
    assert roster["pairs"]["E"] == "not_testable"


def test_n_role_with_other_mlp_profile_is_recipe_mismatch():
    plan = _plan()
    plan["candidates"].append({"candidate_id": "c-nother", "recipe": _mlp_recipe(
        "magnitude_mlp_recipe.v1", profile="chrono10_relu64x32_l2_1")})
    plan["selection"]["extra_refits"].append("c-nother")
    roster = _roster(plan, pairs={**PAIR_MAP, "N-raw": "c-nother"})
    assert {r["role"]: r["reason"] for r in roster["not_testable"]} == {"N-raw": "recipe_mismatch"}
    assert roster["pairs"]["N"] == "not_testable"


def test_exactly_max_models_resolves():
    roster = sp.resolve_roster(**_twelve_args())
    assert len(roster["models"]) == sp.MAX_MODELS == 12
    assert len({m["model_id"] for m in roster["models"]}) == 12
    assert roster["not_testable"] == []


def test_too_many_models_refused(monkeypatch):
    monkeypatch.setattr(sp, "MAX_MODELS", 11)
    with pytest.raises(OpsError) as exc:
        sp.resolve_roster(**_twelve_args())
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "too_many_models")


def test_bind_models_counts_and_defers_production(capsys):
    plan = _plan()
    roster = _roster(plan)
    out = sp.bind_models(plan, _store(plan).get, roster, _admit_all)
    production = [m["model_id"] for m in roster["models"] if m["source"] == "production"]
    assert out[production[0]] == {"status": "deferred", "reason": "production_bound_by_s2b"}
    assert sorted(r["status"] for r in out.values()) == ["bound"] * 9 + ["deferred"]
    assert "models=10 bound=9 excluded=0 deferred=1" in capsys.readouterr().err


def test_model_ids_are_content_hashes_of_source_candidate_and_transform():
    by_role = {role: m for m in _roster()["models"] for role in m["roles"]}
    assert by_role["M2"]["model_id"] == content_hash(
        {"source": "v14", "candidate_id": "c2", "target_transform": "none"})
    assert by_role["M0"]["model_id"] == content_hash(
        {"source": "production", "candidate_id": "size_v1_4", "target_transform": "none"})


def test_roster_kinds_mark_production_candidate_and_blend():
    roster = _roster()
    kinds = {m["source"] + ":" + m["candidate_id"]: m["kind"] for m in roster["models"]}
    assert kinds == {"production:size_v1_4": "production", "v14:c1": "candidate",
                     "v14:c2": "candidate", "v14:b1": "blend",
                     **{f"v14:{cid}": "candidate" for cid in PAIR_IDS}}


def test_blend_binds_its_components():
    plan = _plan()
    store = _store(plan)
    out = sp.bind_models(plan, store.get, _roster(plan), _admit_all)
    model_id = [m["model_id"] for m in _roster(plan)["models"] if m["kind"] == "blend"][0]
    blend = out[model_id]
    assert blend["status"] == "bound"
    assert blend["kind"] == "blend"
    assert sorted(blend["components"]) == ["c1", "c2"]
    assert all(blend["components"][cid]["status"] == "bound" for cid in ("c1", "c2"))
    again = sp.bind_models(plan, _store(plan).get, _roster(plan), _admit_all)[model_id]
    assert blend["binding"] == again["binding"]


def test_blend_binding_changes_when_a_component_record_changes():
    plan = _plan()
    roster = _roster(plan)
    model_id = [m["model_id"] for m in roster["models"] if m["kind"] == "blend"][0]
    base = sp.bind_models(plan, _store(plan).get, roster, _admit_all)[model_id]["binding"]
    store = _store(plan)
    store[_key(plan, 2023, "c1")] = _record(E23, y=0.9)
    changed = sp.bind_models(plan, store.get, roster, _admit_all)[model_id]["binding"]
    assert changed != base


def test_blend_with_missing_component_is_excluded_with_reason():
    plan = _plan()
    store = _store(plan)
    del store[_key(plan, 2024, "c2")]
    roster = _roster(plan)
    model_id = [m["model_id"] for m in roster["models"] if m["kind"] == "blend"][0]
    out = sp.bind_models(plan, store.get, roster, _admit_all)
    assert out[model_id] == {"status": "excluded", "reason": "component_missing"}


def test_unknown_blend_id_is_excluded_with_reason():
    plan = _plan()
    roster = {"models": [{"model_id": "m-unknown", "source": "v14", "candidate_id": "zz",
                          "target_transform": "mixed", "kind": "blend", "roles": ["M4"]}]}
    out = sp.bind_models(plan, _store(plan).get, roster, _admit_all)
    assert out["m-unknown"] == {"status": "excluded", "reason": "blend_unknown"}


def test_module_source_never_names_outcome_key():
    text = Path(sp.__file__).read_text(encoding="utf-8")
    assert '"absolute_move_pct"' not in text
    assert "'absolute_move_pct'" not in text
