"""Tests for EXP-190 Q4b baseline B3 (experiments/size_quantile_b3.py). Synthetic fixtures only."""
import numpy as np
import pandas as pd
import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments.magnitude_search_plan import trial_key
from experiments.magnitude_search_refits import refit_key
from experiments.size_quantile_b3 import b3_quantiles, bind_series
from experiments.size_quantile_baselines import pool_quantiles
from experiments.size_quantile_metrics import Rules

PRED = "predicted_absolute_move_pct"
QCOLS = [f"q{round(a * 100)}" for a in Rules().levels]
E23, E24, E25 = ["a1", "a2", "a3"], ["b1", "b2"], ["c1", "c2"]
BASE = {"c1": 1.0, "c2": 3.0}
ALL_BOUND = {"2023": "bound", "2024": "bound", "2025": "bound"}


def _vals(ids, base):
    return {eid: base + 0.1 * i for i, eid in enumerate(ids)}


def _plan(**over):
    plan = {"plan_id": "p1", "anchor": {"run_id": "r1"}, "horizons": [5, 3, 1],
            "candidates": [{"candidate_id": "c1"}, {"candidate_id": "c2"}],
            "blends": [{"blend_id": "b1", "component_candidate_ids": ["c1", "c2"]}],
            "selection": {"winner_rules": {"mae": "normalized.mae"}, "extra_refits": ["c1", "c2"]},
            "folds": [{"outer": {"year": 2025, "evaluation_event_ids": E25},
                       "inner": [{"year": 2023, "evaluation_event_ids": E23},
                                 {"year": 2024, "evaluation_event_ids": E24}]}]}
    plan.update(over)
    return plan


def _record(values, status="succeeded"):
    """Lead-1 rows carry ``values``; horizon-5 rows hold 99.0 and an outcome field is present: neither is used."""
    lead = [{"event_id": e, "horizon_sessions": 1, PRED: v, "absolute_move_pct": "POISON"}
            for e, v in values.items()]
    far = [{"event_id": e, "horizon_sessions": 5, PRED: 99.0} for e in values]
    return {"status": status, "predictions": lead + far}


def _store(plan, cids=("c1", "c2")):
    """Sealed trial records (one per candidate and inner year) and outer refit slots for each candidate."""
    store, outer = {}, plan["folds"][0]["outer"]
    for cid in cids:
        for inner in plan["folds"][0]["inner"]:
            key = trial_key(plan, outer["year"], inner["year"], cid)
            store["terminal:" + key] = _record(_vals(inner["evaluation_event_ids"], BASE[cid]))
        key = refit_key(plan, outer["year"], cid)
        forecasts = [{"event_id": e, "horizon_sessions": 1, PRED: v}
                     for e, v in _vals(outer["evaluation_event_ids"], BASE[cid]).items()]
        store["outer-start:" + key] = {"plan_id": plan["plan_id"], "outer_year": outer["year"],
                                       "candidate": {"candidate_id": cid}, "outer_id": key}
        store["outer-execution:" + key] = {"status": "succeeded", "forecasts_id": content_hash(forecasts)}
        store["outer-forecasts:" + key] = forecasts
    return store


def _key(plan, inner_year, cid="c1"):
    return "terminal:" + trial_key(plan, 2025, inner_year, cid)


def _outer_key(plan, cid="c1"):
    return refit_key(plan, 2025, cid)


def _binding(kind="candidate", cid="c1", plan_id="p1"):
    return {"plan_id": plan_id, "b3_candidate_id": cid, "kind": kind}


def _admit(ids):
    return {"admitted": sorted(ids), "dropped": []}


def _run(plan, store, kind="candidate", cid="c1"):
    return bind_series(plan, store.get, _binding(kind, cid), _admit)


def _reason(exc):
    return exc.value.problem.details.get("reason")


def test_candidate_binds_all_years_from_lead_one_predictions():
    plan = _plan()
    out = _run(plan, _store(plan))
    assert out["years"] == ALL_BOUND
    assert out["b3_candidate_id"] == "c1"
    assert out["series"].index.name == "event_id"
    # The horizon-5 rows (99.0) and the outcome field ("POISON") never reach the series.
    assert out["series"].to_dict() == {**_vals(E23, 1.0), **_vals(E24, 1.0), **_vals(E25, 1.0)}


def test_binding_hash_is_deterministic_for_identical_sealed_inputs():
    plan = _plan()
    assert _run(plan, _store(plan))["binding"] == _run(plan, _store(plan))["binding"]


def test_missing_failed_and_partial_inner_trials_affect_only_their_year():
    plan = _plan()
    cases = {"missing": None,
             "failed": _record(_vals(E24, 1.0), status="failed"),
             "partial": _record(_vals(E24[:1], 1.0))}
    for reason, body in cases.items():
        store = _store(plan)
        if body is None:
            del store[_key(plan, 2024)]
        else:
            store[_key(plan, 2024)] = body
        out = _run(plan, store)
        assert out["years"] == {"2023": "bound", "2024": reason, "2025": "bound"}, reason
        assert set(out["series"].index) == set(E23 + E25)


def test_no_sealed_inner_fold_for_a_year_is_missing_and_others_stay_bound():
    plan = _plan(folds=[{"outer": {"year": 2025, "evaluation_event_ids": E25},
                         "inner": [{"year": 2023, "evaluation_event_ids": E23}]}])
    out = _run(plan, _store(plan))
    assert out["years"] == {"2023": "bound", "2024": "missing", "2025": "bound"}


def test_outer_missing_and_failed_affect_2025_only():
    plan = _plan()
    store = _store(plan)
    del store["outer-forecasts:" + _outer_key(plan)]
    out = _run(plan, store)
    assert out["years"] == {"2023": "bound", "2024": "bound", "2025": "outer_missing"}
    store = _store(plan)
    store["outer-execution:" + _outer_key(plan)] = {"status": "failed", "forecasts_id": None}
    del store["outer-forecasts:" + _outer_key(plan)]
    out = _run(plan, store)
    assert out["years"] == {"2023": "bound", "2024": "bound", "2025": "outer_failed"}
    assert set(out["series"].index) == set(E23 + E24)


def test_plan_without_a_2025_outer_fold_marks_2025_missing_only():
    plan = _plan(folds=[{"outer": {"year": 2026, "evaluation_event_ids": E25},
                         "inner": [{"year": 2023, "evaluation_event_ids": E23},
                                   {"year": 2024, "evaluation_event_ids": E24}]}])
    out = _run(plan, _store(plan))
    assert out["years"] == {"2023": "bound", "2024": "bound", "2025": "missing"}
    assert set(out["series"].index) == set(E23 + E24)


def test_outer_path_reads_only_the_outer_refit_slots():
    plan = _plan()
    store, calls = _store(plan), []

    def read(slot):
        calls.append(slot)
        return store.get(slot)

    bind_series(plan, read, _binding(), _admit)
    assert any(slot.startswith("outer-start:") for slot in calls)
    assert not any(slot.startswith(("outer-terminal", "outer-validation", "outer-inputs")) for slot in calls)


def test_blend_series_is_the_mean_of_its_components_for_inner_and_outer():
    plan = _plan()
    out = _run(plan, _store(plan), kind="blend", cid="b1")
    assert out["years"] == ALL_BOUND
    assert out["series"].to_dict() == pytest.approx({**_vals(E23, 2.0), **_vals(E24, 2.0), **_vals(E25, 2.0)})


def test_blend_inner_component_missing_is_named():
    plan = _plan()
    store = _store(plan)
    del store[_key(plan, 2024, "c2")]
    out = _run(plan, store, kind="blend", cid="b1")
    assert out["years"] == {"2023": "bound", "2024": "component_missing", "2025": "bound"}


def test_blend_outer_component_missing_is_named():
    plan = _plan()
    store = _store(plan)
    del store["outer-forecasts:" + _outer_key(plan, "c2")]
    out = _run(plan, store, kind="blend", cid="b1")
    assert out["years"] == {"2023": "bound", "2024": "bound", "2025": "component_outer_missing"}


def test_plan_mismatch_refuses():
    plan = _plan()
    with pytest.raises(OpsError) as exc:
        bind_series(plan, _store(plan).get, _binding(plan_id="other"), _admit)
    assert _reason(exc) == "b3_plan_mismatch"


@pytest.mark.parametrize("binding", [_binding(kind="other"), _binding(kind="blend", cid="zz")])
def test_unknown_kind_or_unknown_blend_refuses_as_unbound(binding):
    plan = _plan()
    with pytest.raises(OpsError) as exc:
        bind_series(plan, _store(plan).get, binding, _admit)
    assert _reason(exc) == "b3_series_unbound"


def test_blend_with_no_components_refuses_as_unbound():
    plan = _plan(blends=[{"blend_id": "b1", "component_candidate_ids": []}])
    with pytest.raises(OpsError) as exc:
        bind_series(plan, _store(plan).get, _binding(kind="blend", cid="b1"), _admit)
    assert _reason(exc) == "b3_series_unbound"


def test_malformed_trial_record_refuses_as_unbound():
    plan = _plan()
    store = _store(plan)
    store[_key(plan, 2023)] = {"status": "succeeded"}
    with pytest.raises(OpsError) as exc:
        bind_series(plan, store.get, _binding(), _admit)
    assert _reason(exc) == "b3_series_unbound"


def test_event_id_repeated_across_years_refuses():
    plan = _plan(folds=[{"outer": {"year": 2025, "evaluation_event_ids": E25},
                         "inner": [{"year": 2023, "evaluation_event_ids": E23},
                                   {"year": 2024, "evaluation_event_ids": ["a1", "b2"]}]}])
    with pytest.raises(OpsError) as exc:
        bind_series(plan, _store(plan).get, _binding(), _admit)
    assert _reason(exc) == "duplicate_event_id"


def test_nonfinite_prediction_propagates_bind_conflict_unchanged():
    plan = _plan()
    store = _store(plan)
    store[_key(plan, 2023)] = _record({**_vals(E23, 1.0), "a1": float("nan")})
    with pytest.raises(OpsError) as exc:
        bind_series(plan, store.get, _binding(), _admit)
    assert exc.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert _reason(exc) == "nonfinite_prediction"


# b3_quantiles: B0's construction over donors whose pred_move is the bound M3 series.
DONOR_IDS = [f"d{i:03d}" for i in range(600)]
OLD_IDS = [f"o{i:03d}" for i in range(300)]
M3 = 1.0 + 0.01 * np.arange(600)
ERR = ((np.arange(600) * 37) % 101) / 10 - 5.0
SCORED = ["d518", "d550"]


def _world():
    """Donors d000-d599 from 2023-01-01 (series members), 2022 donors o000-o299 (not in the series), and x_missing."""
    dates = np.concatenate([
        (pd.Timestamp("2023-01-01") + pd.to_timedelta(np.arange(600), unit="D")).to_numpy(),
        (pd.Timestamp("2022-01-01") + pd.to_timedelta(np.arange(300), unit="D")).to_numpy(),
        [np.datetime64("2024-05-01")]])
    ids = DONOR_IDS + OLD_IDS + ["x_missing"]
    table = pd.DataFrame({"event_date": pd.to_datetime(dates), "pred_move": 50.0, "pred_crush": 0.0}, index=ids)
    abs_move = np.concatenate([M3 + ERR, np.full(300, 2.0), [1.0]])
    labels = pd.DataFrame({"abs_move": abs_move, "crush_pct_iv30": 0.0}, index=ids)
    return table, labels


def _bound(years=None):
    series = pd.Series(M3, index=pd.Index(DONOR_IDS, name="event_id"))
    return {"series": series, "years": years or dict(ALL_BOUND)}


def _events(table):
    ids = SCORED + ["x_missing", "o010"]
    return pd.DataFrame({"event_date": table.loc[ids, "event_date"].to_numpy(), "pred_move": 0.0}, index=ids)


def test_b3_matches_pool_quantiles_on_m3_predictions_not_production():
    table, labels = _world()
    out = b3_quantiles(table, labels, _events(table), _bound())
    m3_table = table.loc[DONOR_IDS].assign(pred_move=M3)
    scored = _events(table).loc[SCORED].assign(pred_move=M3[[518, 550]])
    expected = pool_quantiles(m3_table, labels, scored, name="B3")
    pd.testing.assert_frame_equal(out.loc[SCORED], expected)


def test_same_day_and_later_donor_labels_do_not_change_an_event():
    table, labels = _world()
    base = b3_quantiles(table, labels, _events(table), _bound())
    changed = labels.copy()
    changed.loc[["d518", "d599"], "abs_move"] = 1e6
    out = b3_quantiles(table, changed, _events(table), _bound())
    pd.testing.assert_frame_equal(out.loc[["d518"]], base.loc[["d518"]])


def test_donors_before_2023_outside_the_series_do_not_enter_the_pool():
    table, labels = _world()
    out = b3_quantiles(table, labels, _events(table), _bound())
    trimmed = b3_quantiles(table.drop(OLD_IDS), labels.drop(OLD_IDS), _events(table).drop("o010"), _bound())
    pd.testing.assert_frame_equal(out.loc[SCORED], trimmed.loc[SCORED])


def test_event_without_a_prediction_in_a_bound_year_is_forecast_missing():
    table, labels = _world()
    out = b3_quantiles(table, labels, _events(table), _bound())
    assert out.loc["x_missing", "reason"] == "forecast_missing"
    assert out.loc["x_missing", QCOLS].isna().all()
    assert out.loc["x_missing", "n_rows"] == 0


def test_unbound_year_and_2022_events_are_not_testable_with_nan_quantiles():
    table, labels = _world()
    ids = ["d518", "d550", "o010"]
    out = b3_quantiles(table, labels, _events(table), _bound({"2023": "bound", "2024": "missing", "2025": "bound"}))
    assert (out.loc[ids, "reason"] == "not_testable").all()
    assert out.loc[ids, QCOLS].isna().all().all()
    assert out.loc[ids, "bucket"].isna().all()
    assert not out.loc[ids, "fallback"].any()
    assert (out.loc[ids, "n_rows"] == 0).all()


def test_no_bound_year_makes_every_event_not_testable():
    table, labels = _world()
    empty = pd.Series([], dtype=float, index=pd.Index([], name="event_id"))
    bound = {"series": empty, "years": {"2023": "missing", "2024": "missing", "2025": "missing"}}
    out = b3_quantiles(table, labels, _events(table), bound)
    assert (out["reason"] == "not_testable").all()
    assert out[QCOLS].isna().all().all()


def test_duplicate_table_index_still_raises():
    table, labels = _world()
    dup = pd.concat([table, table.iloc[[0]]])
    with pytest.raises(ValueError):
        b3_quantiles(dup, labels, _events(table), _bound())
