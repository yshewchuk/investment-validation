"""EXP-188 S4: per-model pools, the linear put fast path, and its equivalence check (synthetic only)."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from engine.pnl_sim import MIN_POOL, expected_pnl
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


def test_floor_drops_pre_2023_rows(data):
    table, labels = data
    kept = int((table.event_date >= pool_mod.COMMON_START).sum())
    assert 0 < kept < len(table)
    assert len(build_pool(table, labels)) == kept
    assert len(build_pool(table, labels, floor=None)) == len(table)


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
    with pytest.raises(ValueError, match="off the ladder"):
        structure_value(means, STRIKES, [{"strike": 97.0, "side": "sell", "qty": 1}])
    for bad in (0.0, -1.0, np.nan):
        assert structure_pnl(means, STRIKES, legs, bad) is None


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
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
        check_equivalence(pool, [], draws=DRAW)


def test_put_means_retains_nothing(pool):
    means = _means(pool)
    assert means.shape == (len(STRIKES),)
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
