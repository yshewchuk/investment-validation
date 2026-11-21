"""EXP-188 S4: per-model pools, the linear put fast path, and its equivalence check (synthetic only)."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from engine.pnl_sim import MIN_POOL, expected_pnl
from engine.structures import ResolvedLeg
from engine.v2.ops.errors import OpsError
from experiments import size_pnl_pool as pool_mod
from experiments.size_pnl_pool import (build_pool, check_equivalence, event_seed, paired_events,
                                       put_means, structure_pnl, structure_value,
                                       undetermined_reason)

DRAW = 200
STRIKES = np.array([90.0, 95.0, 100.0, 105.0])
LATE = pd.Timestamp("2024-11-01")
EARLY = pd.Timestamp("2022-03-01")
SINGLE = {"strike": 100.0, "side": "sell", "qty": 1}
SPREAD = [{"strike": 100.0, "side": "sell", "qty": 1}, {"strike": 95.0, "side": "buy", "qty": 1}]
CONDOR = [{"strike": 90.0, "side": "buy", "qty": 1}, {"strike": 95.0, "side": "sell", "qty": 1},
          {"strike": 100.0, "side": "buy", "qty": 2}, {"strike": 105.0, "side": "sell", "qty": 1}]


def _tables(seed=7, n=600):
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1096, n), unit="D")
    pred_move = rng.uniform(2, 8, n)
    pred_crush = rng.normal(-15, 3, n)
    index = pd.Index([f"E{i}" for i in range(n)], name="event_id")
    table = pd.DataFrame({"event_date": dates, "pred_move": pred_move, "pred_crush": pred_crush},
                         index=index)
    labels = pd.DataFrame({"abs_move": pred_move + rng.normal(0, 1.5, n),
                           "crush_pct_iv30": pred_crush + rng.normal(0, 5, n)}, index=index)
    return table, labels


@pytest.fixture
def data():
    return _tables()


@pytest.fixture
def pool(data):
    return build_pool(*data, floor=None)


def _item(legs, entry, key="K", date=LATE):
    return {"key": key, "event_date": date, "spot": 100.0, "pre_iv30": 40.0, "pred_move": 5.0,
            "pred_crush": -15.0, "dte_exit": 9, "strikes": STRIKES, "legs": legs,
            "entry_cost": entry}


def _fast(pool, item):
    means = put_means(pool, key=item["key"], event_date=item["event_date"], spot=item["spot"],
                      pre_iv30=item["pre_iv30"], pred_move=item["pred_move"],
                      pred_crush=item["pred_crush"], dte_exit=item["dte_exit"],
                      strikes=item["strikes"], draws=DRAW)
    return structure_pnl(means, item["strikes"], item["legs"], item["entry_cost"])


def _direct(pool, item):
    return expected_pnl(exit_legs=item["legs"], spot=item["spot"], entry_cost=item["entry_cost"],
                        pre_iv30=item["pre_iv30"], pred_abs_move=item["pred_move"],
                        pred_iv_crush=item["pred_crush"], dte_exit=item["dte_exit"],
                        event_date=item["event_date"], pool=pool, key=item["key"], draws=DRAW)


def _means(pool, key="K"):
    return put_means(pool, key=key, event_date=LATE, spot=100.0, pre_iv30=40.0, pred_move=5.0,
                     pred_crush=-15.0, dte_exit=9, strikes=STRIKES, draws=DRAW)


def _history(n):
    return pd.DataFrame({"event_date": pd.date_range("2024-06-01", periods=n, freq="D"),
                         "exp_pnl_sim": np.linspace(-0.5, 0.5, n)})


def _candidates():
    dates = pd.to_datetime(["2023-02-10", "2023-05-10", "2023-08-01", "2023-11-01", "2024-01-15",
                            "2024-04-01"])
    return pd.Series(dates, index=[f"E{i}" for i in range(6)])


def test_build_pool_residual_columns(data):
    table, labels = data
    pool = build_pool(table, labels, floor=None)
    assert len(pool) == len(table)
    assert np.allclose(np.sort(pool._move), np.sort(labels.abs_move - table.pred_move))
    assert np.allclose(np.sort(pool._crush), np.sort(labels.crush_pct_iv30 - table.pred_crush))


def test_build_pool_residuals_pair_by_event_id():
    # Labels are listed in reverse order; the join must pair each event by id, and the pool keeps date order.
    index = pd.Index(["E0", "E1", "E2"], name="event_id")
    table = pd.DataFrame({"event_date": pd.to_datetime(["2023-03-01", "2023-06-01", "2023-09-01"]),
                          "pred_move": [5.0, 4.0, 6.0],
                          "pred_crush": [-15.0, -10.0, -20.0]}, index=index)
    labels = pd.DataFrame({"abs_move": [6.5, 3.0, 7.5],
                           "crush_pct_iv30": [-18.5, -14.0, -12.0]},
                          index=pd.Index(["E2", "E1", "E0"], name="event_id"))
    pool = build_pool(table, labels, floor=None)
    # E0: move 7.5 - 5.0 = 2.5, crush -12.0 - (-15.0) = 3.0
    # E1: move 3.0 - 4.0 = -1.0, crush -14.0 - (-10.0) = -4.0
    # E2: move 6.5 - 6.0 = 0.5, crush -18.5 - (-20.0) = 1.5
    assert np.allclose(pool._move, [2.5, -1.0, 0.5])
    assert np.allclose(pool._crush, [3.0, -4.0, 1.5])


def test_build_pool_is_independent_of_row_order():
    # E1, E2, E3 share 2023-06-01; every residual is distinct so any reordering shows up.
    index = pd.Index([f"E{i}" for i in range(6)], name="event_id")
    table = pd.DataFrame({"event_date": pd.to_datetime(["2023-03-01", "2023-06-01", "2023-06-01",
                                                        "2023-06-01", "2023-09-01", "2023-11-01"]),
                          "pred_move": [5.0, 4.0, 6.0, 3.0, 7.0, 2.5],
                          "pred_crush": [-15.0, -10.0, -20.0, -12.0, -16.0, -8.0]}, index=index)
    labels = pd.DataFrame({"abs_move": [6.5, 3.0, 7.0, 9.0, 4.5, 4.25],
                           "crush_pct_iv30": [-18.5, -14.0, -12.0, -17.0, -13.0, -9.5]},
                          index=index)
    base = build_pool(table, labels, floor=None)
    reversed_rows = build_pool(table.iloc[::-1], labels, floor=None)
    shuffled = build_pool(table, labels.iloc[[3, 0, 5, 1, 4, 2]], floor=None)
    for other in (reversed_rows, shuffled):
        assert np.array_equal(base._move, other._move)
        assert np.array_equal(base._crush, other._crush)
        assert np.array_equal(base._pred, other._pred)
        assert np.array_equal(base._dates, other._dates)
    # Pool order: E0, then the tied block E1, E2, E3 in id order, then E4, E5.
    # Residuals: E0 1.5, E1 -1.0, E2 1.0, E3 6.0, E4 -2.5, E5 1.75.
    assert np.allclose(base._move, [1.5, -1.0, 1.0, 6.0, -2.5, 1.75])
    assert np.allclose(base._move[1:4], [-1.0, 1.0, 6.0])


def test_floor_drops_pre_2023_rows():
    index = pd.Index(["before", "boundary"], name="event_id")
    table = pd.DataFrame({"event_date": [pd.Timestamp("2022-12-31"), pd.Timestamp("2023-01-01")],
                          "pred_move": [5.0, 5.0], "pred_crush": [-15.0, -15.0]}, index=index)
    labels = pd.DataFrame({"abs_move": [7.0, 9.0], "crush_pct_iv30": [-12.0, -10.0]}, index=index)
    kept = build_pool(table, labels)
    assert len(kept) == 1
    # Only the 2023-01-01 row survives: move 9.0 - 5.0 = 4.0.
    assert np.allclose(kept._move, [4.0])
    assert len(build_pool(table, labels, floor=None)) == 2


def test_missing_column_refuses(data):
    table, labels = data
    with pytest.raises(OpsError) as caught:
        build_pool(table.drop(columns="pred_crush"), labels, floor=None)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details["reason"] == "pool_columns_missing"


def test_pools_are_per_model(data):
    table, labels = data
    other = table.assign(pred_move=table.pred_move + 1.0)
    a = build_pool(table, labels, floor=None)
    b = build_pool(other, labels, floor=None)
    draw_a = a.draw(LATE, 5.0, DRAW, np.random.default_rng(1))[0]
    draw_b = b.draw(LATE, 5.0, DRAW, np.random.default_rng(1))[0]
    assert not np.allclose(draw_a, draw_b)
    assert not np.allclose(_means(a), _means(b))


@pytest.mark.parametrize("legs,entry", [(SPREAD, 1.5), (CONDOR, 2.0), ([SINGLE], 2.0)])
def test_fast_path_matches_expected_pnl(pool, legs, entry):
    item = _item(legs, entry)
    assert abs(_fast(pool, item) - _direct(pool, item)["exp_pnl_sim"]) <= 1e-9


def test_same_key_gives_same_vector_for_any_structure(pool):
    first = _means(pool)
    assert np.array_equal(first, _means(pool))
    assert structure_value(first, STRIKES, SPREAD) == pytest.approx(first[2] - first[1])
    assert structure_value(first, STRIKES, [SINGLE]) == pytest.approx(first[2])


def test_seed_matches_expected_pnl_and_depends_on_key(pool):
    item = _item([SINGLE], 1.0)
    means = _means(pool)
    # With entry cost 1 and one long put, expected_pnl's mean return is the put mean minus 1.
    assert means[2] == pytest.approx(_direct(pool, item)["exp_pnl_sim"] + 1.0, abs=1e-9)
    digest = hashlib.sha256(f"K|{LATE}".encode()).digest()[:8]
    assert event_seed("K", LATE) == int.from_bytes(digest, "big")
    assert not np.array_equal(means, _means(pool, key="J"))


@pytest.mark.parametrize("override", [{"spot": np.nan}, {"spot": np.inf}, {"pre_iv30": 0.0},
                                      {"pre_iv30": -5.0}, {"dte_exit": -1}])
def test_unsimulable_inputs_return_none(pool, override):
    kwargs = dict(key="K", event_date=LATE, spot=100.0, pre_iv30=40.0, pred_move=5.0,
                  pred_crush=-15.0, dte_exit=9, strikes=STRIKES, draws=DRAW)
    kwargs.update(override)
    assert put_means(pool, **kwargs) is None


def test_thin_pool_is_undetermined(data):
    pool = build_pool(*data, floor=None)
    assert pool.before(EARLY) < MIN_POOL
    assert _means_at(pool, EARLY) is None


def _means_at(pool, date):
    return put_means(pool, key="K", event_date=date, spot=100.0, pre_iv30=40.0, pred_move=5.0,
                     pred_crush=-15.0, dte_exit=9, strikes=STRIKES, draws=DRAW)


def test_structure_value_convention_and_guards():
    means = np.array([1.0, 2.0, 4.0, 8.0])
    legs = [SINGLE, {"strike": 90.0, "side": "buy", "qty": 2},
            {"strike": 105.0, "side": "buy", "qty": 0}]
    assert structure_value(means, STRIKES, legs) == pytest.approx(4.0 - 2.0)
    assert structure_pnl(means, STRIKES, legs, 2.0) == pytest.approx(0.0)
    with pytest.raises(OpsError) as caught:
        structure_value(means, STRIKES, [{"strike": 97.0, "side": "sell", "qty": 1}])
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details["reason"] == "leg_off_ladder"
    for bad in (0.0, -1.0, np.nan):
        assert structure_pnl(means, STRIKES, legs, bad) is None


def test_leg_with_no_strike_is_skipped():
    means = np.array([1.0, 2.0, 4.0, 8.0])
    no_strike = SPREAD + [{"strike": None, "side": "buy", "qty": 1}]
    nan_strike = SPREAD + [{"strike": np.nan, "side": "buy", "qty": 1}]
    base = structure_value(means, STRIKES, SPREAD)
    assert structure_value(means, STRIKES, no_strike) == pytest.approx(base)
    assert structure_value(means, STRIKES, nan_strike) == pytest.approx(base)


def test_nan_leg_quantity_refuses():
    means = np.array([1.0, 2.0, 4.0, 8.0])
    with pytest.raises(OpsError) as caught:
        structure_value(means, STRIKES, [{"strike": 100.0, "side": "sell", "qty": np.nan}])
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details["reason"] == "leg_invalid"


def test_put_means_increase_with_strike(pool):
    ladder = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    means = put_means(pool, key="K", event_date=LATE, spot=100.0, pre_iv30=40.0, pred_move=5.0,
                      pred_crush=-15.0, dte_exit=9, strikes=ladder, draws=DRAW)
    assert means.shape == (5,)
    assert np.all(np.diff(means) > 0)


def test_equivalence_refuses_nan_means(pool, monkeypatch):
    monkeypatch.setattr(pool_mod, "put_means",
                        lambda *args, **kwargs: np.full(len(STRIKES), np.nan))
    with pytest.raises(OpsError) as caught:
        check_equivalence(pool, [_item([SINGLE], 2.0)], draws=DRAW)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details["reason"] == "fast_path_mismatch"


@pytest.mark.parametrize("tolerance", [0.0, np.nan])
def test_tolerance_must_be_positive_and_finite(pool, tolerance):
    with pytest.raises(ValueError, match="tolerance must be a positive finite number"):
        check_equivalence(pool, [_item([SINGLE], 2.0)], tolerance=tolerance, draws=DRAW)


def test_resolved_legs_feed_the_check(pool):
    # A put spread as price_structure resolves it: sell the 100 put, buy the 95 put.
    expiry = pd.Timestamp("2024-11-10")
    common = dict(name="put", right="P", expiry=expiry, dte=9, bid=1.0, ask=1.2, price=1.1,
                  cash_flow=0.0, wide_market=False)
    resolved = [ResolvedLeg(side="sell", qty=1.0, strike=100.0, **common),
                ResolvedLeg(side="buy", qty=1.0, strike=95.0, **common)]
    legs = [{"strike": leg.strike, "side": leg.side, "qty": leg.qty} for leg in resolved]
    out = check_equivalence(pool, [_item(legs, 1.5)], draws=DRAW)
    assert out["checked"] == 1
    assert out["max_abs_diff"] <= 1e-9


def test_undetermined_reasons(pool):
    assert undetermined_reason(pool, _history(100), EARLY) == "thin_pool"
    assert undetermined_reason(pool, _history(99), LATE) == "thin_window"
    assert undetermined_reason(pool, _history(100), LATE) is None


def test_paired_events_drop_for_every_model():
    kept, dropped = paired_events({"A": {"E1"}, "B": {"E4"}}, _candidates())
    assert kept == ["E0", "E2", "E3", "E5"]
    assert dropped == {"2023Q2": 1, "2024Q1": 1}
    kept, dropped = paired_events({"A": {"E1"}, "B": {"E1", "E3"}}, _candidates())
    assert kept == ["E0", "E2", "E4", "E5"]
    assert dropped == {"2023Q2": 1, "2023Q4": 1}


def test_paired_events_refuses_unknown_id():
    with pytest.raises(ValueError, match="not in the candidate population"):
        paired_events({"A": {"ZZ"}}, _candidates())


def test_equivalence_passes_on_good_sample(pool):
    sample = [_item(SPREAD, 1.5), _item(CONDOR, 2.0), _item([SINGLE], 2.0)]
    out = check_equivalence(pool, sample, draws=DRAW)
    assert out["checked"] == 3
    assert out["max_abs_diff"] <= 1e-9


def test_equivalence_refuses_perturbed_fast_path(pool, monkeypatch):
    real = pool_mod.put_means

    def shifted(*args, **kwargs):
        means = real(*args, **kwargs)
        return None if means is None else means + 1e-6

    monkeypatch.setattr(pool_mod, "put_means", shifted)
    with pytest.raises(OpsError) as caught:
        check_equivalence(pool, [_item([SINGLE], 2.0)], draws=DRAW)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details["reason"] == "fast_path_mismatch"


def test_equivalence_refuses_empty_sample(pool):
    with pytest.raises(ValueError, match="non-empty sample"):
        check_equivalence(pool, [], draws=DRAW)


@pytest.mark.parametrize("shift,passes", [(1e-6, False), (1e-12, True)])
def test_tolerance_pinned_both_sides(pool, monkeypatch, shift, passes):
    real = pool_mod.put_means

    def shifted(*args, **kwargs):
        means = real(*args, **kwargs)
        return None if means is None else means + shift

    monkeypatch.setattr(pool_mod, "put_means", shifted)
    item = [_item([SINGLE], 2.0)]
    if passes:
        assert check_equivalence(pool, item, draws=DRAW)["checked"] == 1
    else:
        with pytest.raises(OpsError) as caught:
            check_equivalence(pool, item, draws=DRAW)
        assert caught.value.problem.details["reason"] == "fast_path_mismatch"


def test_one_sided_none_fast_path_refuses(pool, monkeypatch):
    # The direct path prices this good item, but the fast path returns None: a mismatch, not a skip.
    monkeypatch.setattr(pool_mod, "put_means", lambda *args, **kwargs: None)
    with pytest.raises(OpsError) as caught:
        check_equivalence(pool, [_item([SINGLE], 2.0)], draws=DRAW)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details["reason"] == "fast_path_mismatch"


def test_put_means_retains_nothing(pool):
    means = _means(pool)
    assert means.shape == (len(STRIKES),)
    assert np.isfinite(means).all()
    assert means.base is None


def test_equivalence_all_unsimulable_refuses(pool):
    item = _item([SINGLE], 2.0)
    item["spot"] = np.nan
    with pytest.raises(OpsError) as caught:
        check_equivalence(pool, [item], draws=DRAW)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details["reason"] == "equivalence_empty"


def test_equivalence_mixed_sample_counts_compared_and_skipped(pool):
    bad = _item([SINGLE], 2.0)
    bad["spot"] = np.nan
    out = check_equivalence(pool, [bad, _item(SPREAD, 1.5)], draws=DRAW)
    assert out["checked"] == 1
    assert out["skipped"] == 1
    assert out["max_abs_diff"] <= 1e-9


def test_put_means_rejects_non_increasing_strikes(pool):
    with pytest.raises(ValueError, match="strictly increasing"):
        put_means(pool, key="K", event_date=LATE, spot=100.0, pre_iv30=40.0, pred_move=5.0,
                  pred_crush=-15.0, dte_exit=9, strikes=np.array([100.0, 95.0]), draws=DRAW)


def test_build_pool_missing_crush_label_refuses(data):
    table, labels = data
    with pytest.raises(OpsError) as caught:
        build_pool(table, labels.drop(columns="crush_pct_iv30"), floor=None)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details["reason"] == "pool_columns_missing"


def _pool_of(n):
    """A pool of exactly ``n`` rows, all dated before LATE."""
    index = pd.Index([f"R{i}" for i in range(n)], name="event_id")
    table = pd.DataFrame({"event_date": pd.Timestamp("2024-01-01"), "pred_move": 5.0,
                          "pred_crush": -15.0}, index=index)
    labels = pd.DataFrame({"abs_move": 6.0, "crush_pct_iv30": -14.0}, index=index)
    return build_pool(table, labels, floor=None)


def test_min_pool_boundary_is_not_thin_pool():
    # Exactly MIN_POOL rows before the date: the pool check passes, so the window check decides.
    at_min = _pool_of(MIN_POOL)
    assert at_min.before(LATE) == MIN_POOL
    assert undetermined_reason(at_min, _history(99), LATE) == "thin_window"
    assert undetermined_reason(at_min, _history(100), LATE) is None


def test_one_below_min_pool_is_thin_pool():
    below = _pool_of(MIN_POOL - 1)
    assert below.before(LATE) == MIN_POOL - 1
    assert undetermined_reason(below, _history(100), LATE) == "thin_pool"
