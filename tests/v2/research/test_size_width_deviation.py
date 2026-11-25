import json

import numpy as np
import pandas as pd
import pytest

from experiments import size_pnl_shortvol as sv
from experiments import size_width_deviation as dv
from experiments.size_pnl_pool import build_pool
from experiments.size_width_band import tier
from experiments.size_width_deviation import (DeviationSweep, bucket, bucket_table, decline_table, quantile_table,
                                              realised_ratio, tables)
from experiments.size_width_sweep import WINDOWS, Sweep
from tests.v2.research.test_size_pnl_shortvol import _tables, chunk_of, events_for, scored_frame, scored_row

DRAW = 200
NAME = "CND-PS"
PRED = 6.0
JUNE = (pd.Timestamp("2024-06-01"), pd.Timestamp("2024-07-01"))
POOL = build_pool(*_tables(), floor=None)
PRIOR = [f"P{i}" for i in range(6)]
CUR = [f"C{i}" for i in range(6)]


def _small_window(monkeypatch):
    real = sv.trailing_cutoff
    monkeypatch.setattr(sv, "trailing_cutoff",
                        lambda history, as_of, **kw: real(history, as_of, **{**kw, "min_window": 1}))


def _scene():
    chunk = chunk_of(*[(e, {}) for e in PRIOR + CUR])
    prior = events_for(*PRIOR, date="2024-05-10")
    cur = events_for(*CUR, date="2024-06-10")
    sessions = {e: (pd.Timestamp("2024-05-15"), pd.Timestamp("2024-05-25")) for e in PRIOR}
    sessions.update({e: (pd.Timestamp("2024-06-11"), pd.Timestamp("2024-06-21")) for e in CUR})
    sims = sv.simulate(chunk, pd.concat([prior, cur], ignore_index=True), POOL, draws=DRAW)
    return chunk, prior, cur, sessions, sims


def _run(sweep, scene):
    chunk, prior, cur, sessions, sims = scene
    sweep.add_month(chunk, prior, POOL, output=False, sessions=sessions, draws=DRAW, sims=sims)
    sweep.add_month(chunk, cur, POOL, output=True, sessions=sessions, draws=DRAW, sims=sims)
    return sweep


def _dev(eid, date, ratio, *, mcap=20e9, ret=0.1, name=NAME):
    return {"structure": name, "event_id": eid, "event_date": pd.Timestamp(date), "mcap_usd": mcap,
            "ratio": ratio, "ret": ret}


# ---- tiers, buckets, the realised ratio ------------------------------------

def test_tier_boundaries_are_lower_inclusive_and_nan_is_unknown():
    assert tier(10e9) == "ge_10B" and tier(9.99e9) == "1B_10B"
    assert tier(1e9) == "1B_10B" and tier(999_999_999.0) == "lt_1B"
    assert tier(float("nan")) == "unknown" and tier(None) == "unknown"


def test_bucket_boundaries_put_0_9_and_1_1_in_the_middle():
    assert bucket(0.9) == "0.9_1.1" and bucket(1.1) == "0.9_1.1" and bucket(1.0) == "0.9_1.1"
    assert bucket(0.89) == "lt_0.9" and bucket(1.11) == "gt_1.1"


def test_realised_ratio_on_hand_derived_ladders():
    dense = chunk_of(("A", {}))  # CND-PS width 0.03 at PRED 6: up1 target 103 is listed, so 3 / (100 * 0.03) = 1.0
    assert realised_ratio(dense, 0, NAME, PRED) == pytest.approx(1.0)
    sparse = chunk_of(("A", {}), strikes=range(60, 141, 5))  # TWIN-P width 0.04: up1 target 104 snaps to 105, 5 / 4 = 1.25
    assert realised_ratio(sparse, 0, "TWIN-P", PRED) == pytest.approx(1.25)
    coarse = chunk_of(("A", {}), strikes=range(60, 141, 10))  # up1 snaps onto atm: ladder_too_coarse, no ratio
    assert realised_ratio(coarse, 0, NAME, PRED) is None


def test_realised_ratio_resolves_through_the_pricing_it_is_given(monkeypatch):
    calls = []

    class Stub:
        def resolve(self, chunk, i, name, pred):
            calls.append((i, name, pred))
            return None, "structure_unresolved"

    assert realised_ratio(chunk_of(("A", {})), 0, NAME, PRED, Stub()) is None
    assert calls == [(0, NAME, PRED)]


# ---- the sweep records T8's rows ------------------------------------------

def test_deviation_sweep_takes_what_a_plain_sweep_takes_and_records_arm_1_only(monkeypatch):
    _small_window(monkeypatch)
    scene = _scene()
    plain, dev = _run(Sweep(), scene), _run(DeviationSweep(), scene)
    pd.testing.assert_frame_equal(dev.rows([(NAME, 1.0)], JUNE), plain.rows([(NAME, 1.0)], JUNE))
    expected = sum(len(plain.rows([(n, 1.0)], JUNE)) for n in sv.NAMES)
    assert len(dev.deviation) == expected > 0
    mine = [d for d in dev.deviation if d["structure"] == NAME]
    assert len(mine) == len(plain.rows([(NAME, 1.0)], JUNE))
    assert all(d["ratio"] == pytest.approx(1.0) for d in mine)
    assert dev.unavailable == 0


def test_burn_in_records_no_deviation_rows(monkeypatch):
    _small_window(monkeypatch)
    chunk, prior, _cur, sessions, sims = _scene()
    dev = DeviationSweep()
    dev.add_month(chunk, prior, POOL, output=False, sessions=sessions, draws=DRAW, sims=sims)
    assert dev.deviation == [] and dev.taken == {}


def test_unavailable_ratios_are_counted_and_kept_as_nan(monkeypatch):
    _small_window(monkeypatch)
    monkeypatch.setattr(dv, "realised_ratio", lambda *args: None)
    dev = _run(DeviationSweep(), _scene())
    assert dev.unavailable == len(dev.deviation) > 0
    assert all(np.isnan(d["ratio"]) for d in dev.deviation)


# ---- tables ---------------------------------------------------------------

def test_quantile_table_percentiles_window_and_nan_exclusion():
    sweep = DeviationSweep()
    sweep.deviation = [_dev(f"e{i}", "2024-06-10", r) for i, r in enumerate([0.8, 0.9, 1.0, 1.1, 1.2])]
    sweep.deviation += [_dev("nan", "2024-06-10", float("nan")), _dev("late", "2025-06-10", 9.0)]
    rows = quantile_table(sweep, {"2024": WINDOWS["2024"]})
    cell = [r for r in rows if r["structure"] == NAME and r["tier"] == "ge_10B"]
    assert len(cell) == 1 and cell[0]["n"] == 5 and cell[0]["window"] == "2024"
    assert [cell[0][k] for k in ("q10", "q25", "q50", "q75", "q90")] == pytest.approx([0.84, 0.9, 1.0, 1.1, 1.16])
    late = quantile_table(sweep, {"2025": WINDOWS["2025"]})
    assert [(r["n"], r["q50"]) for r in late if r["structure"] == NAME] == [(1, 9.0)]


def test_bucket_table_means_by_hand():
    sweep = DeviationSweep()
    sweep.deviation = [_dev("a", "2024-06-10", 0.5, ret=0.1), _dev("b", "2024-06-10", 0.95, ret=0.2),
                       _dev("c", "2024-06-10", 1.0, ret=0.3), _dev("d", "2024-06-10", 1.2, ret=-0.1),
                       _dev("n", "2024-06-10", float("nan"), ret=9.0)]
    rows = {r["bucket"]: r for r in bucket_table(sweep, {"2024": WINDOWS["2024"]}) if r["structure"] == NAME}
    assert rows["lt_0.9"]["n"] == 1 and rows["lt_0.9"]["mean_ret"] == pytest.approx(0.1)
    assert rows["0.9_1.1"]["n"] == 2 and rows["0.9_1.1"]["mean_ret"] == pytest.approx(0.25)
    assert rows["gt_1.1"]["mean_ret"] == pytest.approx(-0.1)


def _cur(eid, *, pnl=100.0, status="scored", reason=None):
    row = scored_row(eid, pnl=pnl, date="2024-06-10", name=NAME)
    row.update(status=status, reason=reason)
    if status != "scored":
        row["exp_pnl_sim"] = float("nan")
    return row


def test_decline_table_shares_by_hand():
    prior = [scored_row(f"P{i}", pnl=i / 100.0, date="2024-01-15", name=NAME) for i in range(120)]
    now = [_cur("t1"), _cur("t2"), _cur("b1", pnl=-100.0),
           _cur("c1", status="undetermined", reason="ladder_too_coarse"),
           _cur("c2", status="undetermined", reason="ladder_too_coarse"),
           _cur("s1", status="undetermined", reason="structure_unresolved")]
    sweep = Sweep()
    sweep.scored[(NAME, 1.0)] = [scored_frame(*prior), scored_frame(*now)]
    rows = decline_table(sweep, {"jun": JUNE})
    cell = {r["reason"]: r for r in rows if r["structure"] == NAME and r["tier"] == "ge_10B"}
    assert {k: v["n"] for k, v in cell.items()} == {"taken": 2, "below_cutoff": 1, "ladder_too_coarse": 2,
                                                    "structure_unresolved": 1}
    assert cell["taken"]["share"] == pytest.approx(2 / 6) and cell["ladder_too_coarse"]["share"] == pytest.approx(2 / 6)
    assert sum(r["share"] for r in rows if r["structure"] == NAME) == pytest.approx(1.0)
    assert {r["structure"] for r in rows} == {NAME}


def test_tables_bundles_the_three_tables_and_the_unavailable_count_as_plain_json():
    sweep = DeviationSweep()
    sweep.deviation = [_dev("a", "2024-06-10", 1.0), _dev("b", "2024-06-10", float("nan"))]
    sweep.unavailable = 1
    got = tables(sweep, {"2024": WINDOWS["2024"]})
    assert set(got) == {"quantiles", "buckets", "declines", "deviation_unavailable"}
    assert got["deviation_unavailable"] == 1 and type(got["deviation_unavailable"]) is int
    assert [r["n"] for r in got["quantiles"] if r["structure"] == NAME] == [1]
    assert [r["bucket"] for r in got["buckets"] if r["structure"] == NAME] == ["0.9_1.1"]
    assert got["declines"] == []  # no scored history, so no decline shares
    assert json.loads(json.dumps(got)) == got
