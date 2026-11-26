"""EXP-189 fill-model seam: the default marks are today's, the seamed sweep reproduces the unseamed one, and a custom model is live."""
import re

import numpy as np
import pandas as pd
import pytest

from engine.forecast_sizing import forecast_params
from engine.structures import STRUCTURES
from experiments import size_pnl_decisions as decisions
from experiments import size_pnl_shortvol as sv
from experiments import size_width_fills as fills
from experiments import size_width_pricing as seam
from experiments import size_width_tuning as tuning
from experiments.size_pnl_pool import build_pool
from experiments.size_pnl_straddle import EXIT_UNQUOTED
from experiments.size_width_sweep import Sweep
from tests.v2.research.test_size_pnl_shortvol import PRED, _tables, chunk_of, events_for
from tests.v2.research.test_size_width_sweep import DRAW, NAME, _two_months

PERIOD = pd.Period("2024-06", "M")
SIMS_GRID = np.arange(60.0, 141.0)


def _small_on(monkeypatch, module):
    """``_small_window`` for any module holding ``trailing_cutoff``: the real module, or a copy whose name is looked up in its own globals."""
    real = module.trailing_cutoff
    monkeypatch.setattr(module, "trailing_cutoff", lambda history, as_of, **kw: real(history, as_of, **{**kw, "min_window": 1}))


def _files(directory):
    return sorted(p.name for p in directory.iterdir() if p.is_file())


def _run(pricing, out, **kwargs):
    """The two-month chain on a ladder sweep over ``pricing``, the output month frozen and the inner tuning written under ``out``."""
    sweep = Sweep(ladder=True, pricing=pricing, **kwargs)
    pool = build_pool(*_tables(), floor=None)
    chunk, prior, cur, sessions = _two_months()
    sweep.add_month(chunk, prior, pool, output=False, sessions=sessions, draws=DRAW)
    fresh = sweep.add_month(chunk, cur, pool, output=True, sessions=sessions, draws=DRAW)
    taken = fresh.assign(model_id="M0", book="short_vol", arm_or_rule=fresh["arm"], status="taken", reason=None)
    dec = sweep.declined_rows().assign(model_id="M0", book="short_vol", arm_or_rule=lambda f: f["arm"])
    out.mkdir()
    decisions.freeze_month(out, PERIOD, [taken, dec])
    tuning.run_inner(sweep, out, "H")
    return sweep, fresh


def test_the_seamed_default_reproduces_the_unseamed_module_exactly(tmp_path, monkeypatch):
    seamed = seam.load()
    _small_on(monkeypatch, sv)
    _small_on(monkeypatch, seamed)
    plain, plain_fresh = _run(sv, tmp_path / "unseamed")
    seam_sweep, seam_fresh = _run(seamed, tmp_path / "seamed", choices=True)
    assert len(plain_fresh) > 0  # at least one taken row

    pd.testing.assert_frame_equal(plain_fresh, seam_fresh, check_exact=True)
    pd.testing.assert_frame_equal(plain.declined_rows(), seam_sweep.declined_rows(), check_exact=True)
    assert plain.taken.keys() == seam_sweep.taken.keys() and plain.scored.keys() == seam_sweep.scored.keys()
    for store in ("taken", "scored"):
        for key, frames in getattr(plain, store).items():
            other = getattr(seam_sweep, store)[key]
            assert len(frames) == len(other)
            for got, want in zip(other, frames):
                pd.testing.assert_frame_equal(got, want, check_exact=True)

    rel = f"{PERIOD}.parquet"
    assert (tmp_path / "unseamed" / "months" / rel).read_bytes() == (tmp_path / "seamed" / "months" / rel).read_bytes()
    left, right = tmp_path / "unseamed" / "receipts", tmp_path / "seamed" / "receipts"
    assert _files(left) == _files(right) and _files(left)
    for name in _files(left):
        assert (left / name).read_bytes() == (right / name).read_bytes(), name


def _sims(ids):
    return {e: (SIMS_GRID, 0.001 * (SIMS_GRID - 100.0) ** 2) for e in ids}


def test_a_custom_fill_model_moves_the_seamed_marks_and_leaves_the_real_module_alone():
    real_entry, real_exit = sv.entry, sv._exit_value

    class Custom(fills.CurrentFills):
        def entry(self, trade, priced):
            got = super().entry(trade, priced)
            return None if got is None else (got[0] + 0.10, got[1])

        def exit_value(self, trade, traded, alpha, mid=False):
            return super().exit_value(trade, traded, alpha, mid=mid) + 0.05

    base, moved = seam.load(), seam.load(fills=Custom())
    # R's exit rung is 3 (exit unquoted): score marks it at zero, the defined-risk floor, without reading the quotes
    chunk = chunk_of(("A", {}), ("B", {}), ("R", {k: (1.1, 0.9, 3) for k in range(80, 121)}))
    events = events_for("A", "B", "R")
    sims = _sims(["A", "B", "R"])
    b_scores = base.score_events(chunk, events, sims, NAME, alpha=0.5)
    m_scores = moved.score_events(chunk, events, sims, NAME, alpha=0.5)
    ok = (b_scores["status"] == "scored").to_numpy()
    assert ok.any()
    np.testing.assert_allclose((m_scores["entry_cost"] - b_scores["entry_cost"])[ok].to_numpy(), 0.10)
    assert (m_scores["exp_pnl_sim"][ok].to_numpy() != b_scores["exp_pnl_sim"][ok].to_numpy()).all()
    base_exp, base_cost = b_scores["exp_pnl_sim"][ok].to_numpy(), b_scores["entry_cost"][ok].to_numpy()
    value = base_exp * base_cost + base_cost  # the sims' expected payoff, independent of the fill model
    np.testing.assert_allclose(m_scores["exp_pnl_sim"][ok].to_numpy(), (value - (base_cost + 0.10)) / (base_cost + 0.10))

    taken = b_scores[ok].assign(status="taken")
    b_rows = base.score(chunk, events, taken, NAME, 0.5, repriced=True)
    m_rows = moved.score(chunk, events, taken, NAME, 0.5, repriced=True)
    np.testing.assert_allclose((m_rows["entry_cost"] - b_rows["entry_cost"]).to_numpy(), 0.10)
    rung3 = (b_rows["rung"] == EXIT_UNQUOTED).to_numpy()
    assert rung3.any() and (~rung3).any()
    assert (m_rows.loc[rung3, "exit_value"] == 0.0).all() and (b_rows.loc[rung3, "exit_value"] == 0.0).all()
    np.testing.assert_allclose((m_rows["exit_value"] - b_rows["exit_value"])[~rung3].to_numpy(), 0.05)
    pnl_per_share = lambda rows: (rows["exit_value"] - rows["entry_cost"]).to_numpy()
    change = pnl_per_share(m_rows) - pnl_per_share(b_rows)
    np.testing.assert_allclose(change[~rung3], -0.05)  # +0.05 exit - 0.10 entry
    np.testing.assert_allclose(change[rung3], -0.10)   # rung 3 exits at zero: only the entry moves

    assert sv.entry is real_entry and sv._exit_value is real_exit
    assert type(seam.PRICING.fills) is fills.CurrentFills and type(moved.fills) is Custom
    pd.testing.assert_frame_equal(seam.PRICING.score_events(chunk, events, sims, NAME, alpha=0.5), b_scores, check_exact=True)


FLAGS = ("exit_fair_mid", "exit_unpriceable")


def test_current_fills_is_bit_identical_to_the_real_module_on_every_rung():
    chunk = chunk_of(("A", {}), ("B", {k: (0.9, 1.1, 2) for k in range(80, 121)}),
                     ("R", {k: (1.1, 0.9, 3) for k in range(80, 121)}))
    events = events_for("A", "B", "R")
    sims = _sims(["A", "B", "R"])
    seamed = seam.load()
    rungs = set()
    for name in sv.NAMES:
        for alpha in (0.0, 0.5, 1.0):
            want = sv.score_events(chunk, events, sims, name, alpha=alpha)
            pd.testing.assert_frame_equal(seamed.score_events(chunk, events, sims, name, alpha=alpha), want, check_exact=True)
            taken = want[want["status"] == "scored"].assign(status="taken")
            for decided in (taken, taken.iloc[0:0]):
                for drop in (False, True):
                    for mid in (False, True):
                        for rep in (False, True):
                            kw = dict(drop_unquoted=drop, mid_marks=mid, repriced=rep)
                            want_rows = sv.score(chunk, events, decided, name, alpha, **kw)
                            got_rows = seamed.score(chunk, events, decided, name, alpha, **kw)
                            pd.testing.assert_frame_equal(got_rows, want_rows, check_exact=True)
                            rungs |= set(want_rows["rung"])
    assert "rung2" in rungs and EXIT_UNQUOTED in rungs


class Recorder(fills.CurrentFills):
    def __init__(self):
        self.entries, self.exits = [], []

    def entry(self, trade, priced):
        self.entries.append(trade)
        return super().entry(trade, priced)

    def exit_value(self, trade, traded, alpha, mid=False):
        self.exits.append((trade, list(traded)))
        return super().exit_value(trade, traded, alpha, mid=mid)


def test_the_fills_receive_the_trade_of_the_event_being_priced():
    chunk = chunk_of(("A", {}), ("B", {}))
    events = events_for("A", "B")
    sims = _sims(["A", "B"])
    rec = Recorder()
    model = seam.load(fills=rec)
    scored = model.score_events(chunk, events, sims, NAME, alpha=0.5)
    model.score(chunk, events, scored.loc[scored["status"] == "scored"].assign(status="taken"), NAME, 0.5)
    assert {t.event_id for t in rec.entries} == {"A", "B"}
    assert {t.event_id for t, _ in rec.exits} == {"A", "B"}
    for trade in rec.entries:
        assert trade.chunk is chunk and trade.structure == NAME
        assert str(chunk["event_id"][trade.i]) == trade.event_id
        pred = events.loc[events["event_id"] == trade.event_id, "pred_move"].iloc[0]
        want_legs, _ = seam.load().resolve(chunk, trade.i, NAME, pred)
        assert trade.legs == tuple(want_legs)
    for trade, traded in rec.exits:
        assert str(chunk["event_id"][trade.i]) == trade.event_id
        lo, hi = int(chunk["offsets"][trade.i]), int(chunk["offsets"][trade.i + 1])
        assert all(lo <= leg["idx"] < hi for leg in traded)
        assert trade.chunk is chunk and trade.structure == NAME
        pred = events.loc[events["event_id"] == trade.event_id, "pred_move"].iloc[0]
        want_legs, _ = seam.load().resolve(chunk, trade.i, NAME, pred)
        assert trade.legs == tuple(want_legs)
        assert traded == [leg for leg in trade.legs if leg["qty"] > 0]


def test_an_exit_flag_named_like_a_score_column_is_refused():
    class Clashing(fills.CurrentFills):
        exit_flags = ("exit_value",)

    with pytest.raises(ValueError, match="exit_flags would overwrite score columns"):
        seam.load(fills=Clashing())


class SkipB(fills.CurrentFills):
    skip = True
    unusable = False

    def entry(self, trade, priced):
        if self.skip and trade.event_id == "B":
            return fills.Skip("f_not_a_debit")
        if self.unusable and trade.event_id == "B":
            return None  # an unusable quote, not a Skip
        return super().entry(trade, priced)


def test_a_skip_reaches_the_score_events_row_and_leaves_no_stale_reason():
    chunk = chunk_of(("A", {}), ("B", {}))
    events = events_for("A", "B")
    sims = _sims(["A", "B"])
    default = seam.load().score_events(chunk, events, sims, NAME, alpha=0.5)
    fill = SkipB()
    model = seam.load(fills=fill)
    rows = model.score_events(chunk, events, sims, NAME, alpha=0.5)
    b = rows.loc[rows["event_id"] == "B"].iloc[0]
    assert b["status"] == "undetermined" and b["reason"] == "f_not_a_debit"
    assert b[["entry_cost", "rel_spread", "exp_pnl_sim"]].isna().all()
    pd.testing.assert_frame_equal(rows.loc[rows["event_id"] == "A"].drop(columns="reason"), default.loc[default["event_id"] == "A"].drop(columns="reason"), check_exact=True)
    assert pd.isna(rows.loc[rows["event_id"] == "A", "reason"]).all()
    fill.skip = False
    again = model.score_events(chunk, events, sims, NAME, alpha=0.5)
    pd.testing.assert_frame_equal(again, default, check_exact=True)

    fill.unusable = True  # a later call where B's quote is unusable: the earlier skip must not leak into its reason
    later = model.score_events(chunk, events, sims, NAME, alpha=0.5)
    assert later.loc[later["event_id"] == "B", "reason"].iloc[0] == "entry_quote_unusable"


def _taken_ab():
    chunk = chunk_of(("A", {}), ("B", {}))
    events = events_for("A", "B")
    sims = _sims(["A", "B"])
    scored = seam.load().score_events(chunk, events, sims, NAME, alpha=0.5)
    return chunk, events, sims, scored.loc[scored["status"] == "scored"].assign(status="taken")


def test_a_skip_switched_on_after_the_decisions_are_built_is_named_at_repricing():
    chunk, events, _, taken = _taken_ab()
    assert set(taken["event_id"]) == {"A", "B"}
    model = seam.load(fills=SkipB())  # the decisions above were built with no skip; the skip is on for this model only
    with pytest.raises(ValueError, match="skipped by the fill model at re-pricing: f_not_a_debit") as err:
        model.score(chunk, events, taken, NAME, 0.5)
    assert "'B'" in str(err.value)


def test_an_unusable_quote_at_repricing_keeps_its_own_message_and_no_skip_reason_leaks():
    chunk, events, sims, taken = _taken_ab()
    fill = SkipB()
    fresh = SkipB()
    fresh.skip, fresh.unusable = False, True
    with pytest.raises(ValueError, match="has an unusable entry quote") as err:
        seam.load(fills=fresh).score(chunk, events, taken, NAME, 0.5)
    assert "fill model" not in str(err.value)

    model = seam.load(fills=fill)
    model.score_events(chunk, events, sims, NAME, alpha=0.5)  # skips B: leaves its reason behind
    fill.skip, fill.unusable = False, True  # B's quote is now unusable, returned as None rather than a Skip
    with pytest.raises(ValueError, match="has an unusable entry quote") as err:
        model.score(chunk, events, taken, NAME, 0.5)
    assert "fill model" not in str(err.value)


def test_trade_legs_keep_the_zero_quantity_reference_legs():
    chunk = chunk_of(("A", {}))
    events = events_for("A")
    sims = _sims(["A"])
    expected, got = {}, {}
    for name in sv.NAMES:
        expected[name] = [(leg.name, leg.qty) for leg in STRUCTURES[name](**forecast_params(name, PRED)).legs]
        rec = Recorder()
        seam.load(fills=rec).score_events(chunk, events, sims, name, alpha=0.5)
        assert len(rec.entries) == 1, name
        got[name] = [(leg["name"], leg["qty"]) for leg in rec.entries[0].legs]
        assert got[name] == expected[name], name
    zero = [name for name in sv.NAMES if any(qty == 0 for _, qty in expected[name])]
    assert zero, "no sized structure has a zero-quantity leg"  # CND-PS: its anchor is a reference leg
    for name in zero:
        assert any(qty == 0 for _, qty in got[name]), name


class Flagged(fills.CurrentFills):
    exit_flags = FLAGS
    on = True

    def exit_value(self, trade, traded, alpha, mid=False):
        if not self.on:
            return super().exit_value(trade, traded, alpha, mid=mid)
        if trade.event_id == "A":
            return fills.Exit(super().exit_value(trade, traded, alpha, mid=mid), frozenset({"exit_fair_mid"}))
        if trade.event_id == "B":
            return fills.Exit(0.0, frozenset({"exit_unpriceable"}), unquoted=True)
        return super().exit_value(trade, traded, alpha, mid=mid)


def test_exit_flags_and_an_unquoted_exit_reach_the_score_rows():
    curved = {k: (1.0 + 0.01 * (k - 100) ** 2, 1.2 + 0.01 * (k - 100) ** 2, 1) for k in range(80, 121)}  # (bid, ask, rung)
    chunk = chunk_of(("A", curved), ("B", curved), ("C", curved), ("R", {k: (1.1, 0.9, 3) for k in range(80, 121)}))
    events = events_for("A", "B", "C", "R")
    sims = _sims(["A", "B", "C", "R"])
    flagged = Flagged()
    model = seam.load(fills=flagged)
    scored = model.score_events(chunk, events, sims, NAME, alpha=0.5)
    taken = scored.loc[scored["status"] == "scored"].assign(status="taken")
    rows = model.score(chunk, events, taken, NAME, 0.5)
    for flag in FLAGS:
        assert rows[flag].dtype == bool
    by = rows.set_index("event_id")
    assert by.loc["A", "exit_fair_mid"] and not by.loc["A", "exit_unpriceable"]
    assert by.loc["B", "rung"] == EXIT_UNQUOTED and by.loc["B", "exit_value"] == 0.0
    np.testing.assert_allclose(by.loc["B", "pnl_per_share"], -by.loc["B", "entry_cost"])
    assert not by.loc["R", list(FLAGS)].any() and not by.loc["C", list(FLAGS)].any()
    assert by.loc["R", "rung"] == EXIT_UNQUOTED

    base = seam.load().score(chunk, events, taken, NAME, 0.5).set_index("event_id")
    cols = ["exit_value", "pnl_per_share", "pnl_dollars", "rung"]
    for eid in ("A", "C"):  # a flag does not change the value or the rung
        for col in cols:
            assert by.loc[eid, col] == base.loc[eid, col], (eid, col)
    assert base.loc["A", "exit_value"] != 0.0
    assert base.loc["B", "exit_value"] != 0.0 and base.loc["B", "rung"] != EXIT_UNQUOTED  # the zero comes from the unquoted Exit
    assert not by.loc["B", "exit_fair_mid"] and by.loc["B", "exit_unpriceable"]

    dropped = model.score(chunk, events, taken, NAME, 0.5, drop_unquoted=True)
    assert list(dropped["event_id"]) == ["A", "C"] and list(dropped.index) == [0, 1]
    assert all(dropped[f].dtype == bool for f in FLAGS)
    empty = model.score(chunk, events, taken.iloc[0:0], NAME, 0.5)
    assert len(empty) == 0 and all(empty[f].dtype == bool for f in FLAGS)

    flagged.on = False  # state resets between score calls on the same model
    plain = model.score(chunk, events, taken, NAME, 0.5)
    assert all(plain[f].dtype == bool and not plain[f].any() for f in FLAGS)
    pd.testing.assert_frame_equal(plain.drop(columns=list(FLAGS)), seam.load().score(chunk, events, taken, NAME, 0.5),
                                  check_exact=True)


class Undeclared(fills.CurrentFills):
    def exit_value(self, trade, traded, alpha, mid=False):
        return fills.Exit(super().exit_value(trade, traded, alpha, mid=mid), frozenset({"exit_fair_mid"}))


class UnquotedNonzero(fills.CurrentFills):
    def exit_value(self, trade, traded, alpha, mid=False):
        return fills.Exit(0.5, unquoted=True)


@pytest.mark.parametrize(
    "bad, message",
    [
        (Undeclared, "are not declared in exit_flags"),
        (UnquotedNonzero, "an unquoted exit must have value 0.0"),
    ],
)
def test_an_undeclared_flag_or_a_nonzero_unquoted_exit_is_refused(bad, message):
    chunk = chunk_of(("A", {}))
    events = events_for("A")
    scored = seam.load().score_events(chunk, events, _sims(["A"]), NAME, alpha=0.5)
    taken = scored.loc[scored["status"] == "scored"].assign(status="taken")
    with pytest.raises(ValueError, match=re.escape(message)):
        seam.load(fills=bad()).score(chunk, events, taken, NAME, 0.5)
