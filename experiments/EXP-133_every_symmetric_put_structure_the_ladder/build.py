#!/usr/bin/env python3
"""Price 12,600 candidate structures per event, and pick one per arm.

The whole experiment lives or dies on one observation, so it is stated first.

**Expected P&L is LINEAR in the contract vector.** A candidate's simulated exit
value is ``sum_i q_i * BS(S_d, K_i, T, v_d)`` summed over draws ``d``, so its
expected value is ``sum_i q_i * m_i`` where ``m_i`` is the mean simulated exit
value of a SINGLE PUT at strike ``K_i``. The 4,000 paired draws therefore have
to be evaluated once per event per LADDER STRIKE — about 45 of them — and every
one of the 12,600 candidates is then a dot product over at most seven of those
numbers. Simulating each candidate separately would compute the same number
50,000 times slower. ``check_equivalence`` asserts it IS the same number,
against ``engine.pnl_sim.expected_pnl``, and the run aborts if it is not.

The same linearity holds for the real money: entry cost and realized exit value
are sums of per-strike quotes, so pricing every candidate against real ORATS
quotes is two gathers and a sum. That is what makes a search this wide
affordable at all, and it is also the risk — ``check_equivalence`` prices a
sample of candidates through ``engine.structures.price_structure``, the
program's one pricing path, and aborts on any disagreement beyond 1e-9.

**What is deliberately NOT vectorised away.** The zero-floor property is
re-checked per event on the strikes the chain actually lists, in dollars, never
on ladder positions — see ``family.patterns``. A candidate whose floor is
negative is not a defined-risk structure and is dropped before it can be
chosen.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import pnl_sim, replay as replay_mod  # noqa: E402
from engine.build_trades import event_universe  # noqa: E402
from engine.data import store  # noqa: E402
from engine.data.features import tier4  # noqa: E402
from engine.features import load_panel  # noqa: E402
from engine.fills import MIN_MEANINGFUL_COST, FillModel  # noqa: E402
from engine.models.training import iv_crush  # noqa: E402
from engine.replay import ALPHA_GRID  # noqa: E402
from engine.structures import ChainSnapshot, ExpirySelector  # noqa: E402

import family as fam  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MID = 0.5
#: Registered entry filters. The arithmetic reward term is absent on purpose:
#: EXP-129/131 replaced it with the P&L gate and this experiment inherits that.
MAX_REL_SPREAD = 0.25
MCAP_FLOOR = 10e9
#: The incumbent arm's own sizing bounds, from EXP-125/126, on the SPACING.
INCUMBENT_WIDTH_MIN, INCUMBENT_WIDTH_MAX = 0.005, 0.15
#: Draws 1..HALF select; draws HALF..DRAWS supply the gated number.
DRAWS = pnl_sim.DRAWS
HALF = DRAWS // 2

#: Arms. `best_all` is the registered primary; the rest isolate one thing it
#: confounds. `oracle_realized` reads the outcome and can never be promoted.
ARMS = ("best_all", "best_twin_only", "best_twin_p5", "best_n5_n7",
        "random_pick", "incumbent", "oracle_realized")

TWIN_P5_KEY = "N5q2_-2_1"


# --------------------------------------------------------------------------
# the pattern grid, flattened once
# --------------------------------------------------------------------------


class Grid:
    """The 12,600 patterns as flat arrays, so an event is array arithmetic.

    Seven leg slots: the anchor, then up/down for each of at most three tiers.
    A slot a family does not use carries ``qty = 0`` and is masked out of every
    sum, which keeps one rectangular array instead of three ragged ones.
    """

    N_SLOTS = 7

    def __init__(self, patterns: tuple[fam.Pattern, ...]):
        self.patterns = patterns
        n = len(patterns)
        self.anchor = np.array([p.anchor_offset for p in patterns], dtype=np.int64)
        self.tiers = np.array([p.family.tiers for p in patterns], dtype=np.int64)
        # Offsets padded with 0. Position 0 is the anchor's own slot, which is
        # always resolvable, so a padded lookup is harmless rather than a
        # sentinel everything downstream has to remember.
        self.offs = np.zeros((n, 3), dtype=np.int64)
        self.qty = np.zeros((n, self.N_SLOTS), dtype=np.float64)
        for i, p in enumerate(patterns):
            self.qty[i, 0] = p.family.q0
            for t, d in enumerate(p.offsets):
                self.offs[i, t] = d
                self.qty[i, 1 + 2 * t] = p.family.tail[t]
                self.qty[i, 2 + 2 * t] = p.family.tail[t]
        self.outer = self.offs.max(axis=1)
        self.n_strikes = np.array([p.family.n_strikes for p in patterns], dtype=np.int64)
        self.twin = np.array([p.family.twin_peaked for p in patterns], dtype=bool)
        self.is_twin_p5 = np.array([p.family.key == TWIN_P5_KEY for p in patterns], dtype=bool)
        self.family_key = np.array([p.family.key for p in patterns], dtype=object)
        self.shape_key = np.array([p.shape_key for p in patterns], dtype=object)
        self.key = np.array([p.key for p in patterns], dtype=object)
        self.slot_used = self.qty != 0
        self.contracts = np.abs(self.qty).sum(axis=1)

    def __len__(self) -> int:
        return len(self.patterns)


GRID = Grid(fam.patterns())

#: Arm masks that depend only on the pattern, not on the event.
ARM_PATTERN_MASK = {
    "best_all": np.ones(len(GRID), dtype=bool),
    "best_twin_only": GRID.twin,
    "best_twin_p5": GRID.is_twin_p5,
    "best_n5_n7": np.isin(GRID.n_strikes, (5, 7)),
    "random_pick": np.ones(len(GRID), dtype=bool),
    "oracle_realized": np.ones(len(GRID), dtype=bool),
}


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def forecasts() -> pd.DataFrame:
    """Tier-4 walk-forward forecasts plus the pre-print vol the crush multiplies."""
    f = tier4.load_forecasts()[
        ["ticker", "event_date", "pred_abs_move", "pred_abs_move_sd", "pred_iv_crush_30"]
    ]
    crush = iv_crush.crush_frame()[["ticker", "event_date", "pre_iv30"]]
    out = f.merge(crush, on=["ticker", "event_date"], how="left")
    out["event_date"] = pd.to_datetime(out["event_date"])
    need = ["pred_abs_move", "pred_abs_move_sd", "pred_iv_crush_30", "pre_iv30"]
    out = out.dropna(subset=need)
    return out[(out["pred_abs_move"] > 0) & (out["pred_abs_move_sd"] > 0) & (out["pre_iv30"] > 0)]


def residual_history() -> pd.DataFrame:
    """EXP-129's paired out-of-sample errors, rebuilt from the same three tables."""
    panel = load_panel()[["ticker", "date", "abs_move"]].rename(columns={"date": "event_date"})
    f = tier4.load_forecasts()[["ticker", "event_date", "pred_abs_move", "pred_iv_crush_30"]]
    crush = iv_crush.crush_frame()[["ticker", "event_date", "crush_pct_iv30"]]
    h = f.merge(panel, on=["ticker", "event_date"], how="inner")
    h = h.merge(crush, on=["ticker", "event_date"], how="inner")
    h["err_move"] = h["abs_move"] - h["pred_abs_move"]
    h["err_crush"] = h["crush_pct_iv30"] - h["pred_iv_crush_30"]
    h["event_date"] = pd.to_datetime(h["event_date"])
    return h.dropna(subset=["err_move", "err_crush", "pred_abs_move"])


def market_caps() -> pd.DataFrame:
    sec = store.read_table("securities", years=range(2017, 2027),
                           columns=["ticker", "year", "mcap_usd"])
    return sec


# --------------------------------------------------------------------------
# one event
# --------------------------------------------------------------------------


class EventSkip(Exception):
    """No candidate could be priced for this event, with the replay's own reason."""


def _ladder(rows: pd.DataFrame, expiry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(strikes, bid, ask)`` for the puts at one expiry, ascending, cleaned."""
    puts = rows[(rows["right"] == "P") & (rows["expiry"] == expiry)]
    puts = puts.sort_values("strike")
    # A strike listed twice at one expiry is a data defect; keeping the first
    # matches price_structure, which takes `hit.iloc[0]`.
    puts = puts[~puts["strike"].duplicated()]
    return (puts["strike"].to_numpy(dtype=float),
            puts["bid"].to_numpy(dtype=float),
            puts["ask"].to_numpy(dtype=float))


def _mirror_index(strikes: np.ndarray, anchor_idx: int, up_idx: np.ndarray) -> np.ndarray:
    """Index of the exact dollar mirror of each ``up_idx`` about the anchor, or −1.

    "Exact" is the whole point: ``StrikeSelector('mirror')`` requires the
    mirrored strike to be LISTED rather than snapping to the nearest one,
    because an approximately symmetric structure does not cancel its deep-ITM
    tail and its max loss is then not the debit. The tolerance below is a
    float-representation tolerance on strike prices, not a snapping window.
    """
    target = 2.0 * strikes[anchor_idx] - strikes[up_idx]
    pos = np.searchsorted(strikes, target)
    out = np.full(up_idx.shape, -1, dtype=np.int64)
    for cand in (pos - 1, pos):
        ok = (cand >= 0) & (cand < strikes.size)
        idx = np.clip(cand, 0, strikes.size - 1)
        hit = ok & (np.abs(strikes[idx] - target) <= 1e-6 * np.maximum(1.0, np.abs(target)))
        out = np.where(hit & (out < 0), idx, out)
    return out


def _lookup_tables(strikes: np.ndarray, spot: float, quoted: np.ndarray):
    """Per (anchor offset, ladder offset): the up index, its mirror, and validity.

    Rows are the three anchor offsets, columns are ladder offsets 0..MAX. Offset
    0 is the anchor's own slot and is always valid where the anchor is, which
    is what lets a padded offset be looked up instead of branched around.
    """
    n = strikes.size
    below = np.flatnonzero(strikes <= spot)
    if below.size == 0:
        raise EventSkip("structure_unresolved")
    a0 = int(below[-1])
    max_pos = fam.MAX_LADDER_POS
    anchors = np.array(fam.ANCHOR_OFFSETS, dtype=np.int64)
    a_idx = a0 + anchors
    up = np.full((anchors.size, max_pos + 1), -1, dtype=np.int64)
    dn = np.full((anchors.size, max_pos + 1), -1, dtype=np.int64)
    # An anchor that walks off the top of the ladder, or lands on a strike the
    # exit chain does not quote, is marked -1 HERE rather than being left as an
    # out-of-range index for a later gather to trip over. It cost eleven events
    # an IndexError in the first smoke run.
    a_idx = np.where((a_idx >= 0) & (a_idx < n)
                     & quoted[np.clip(a_idx, 0, n - 1)], a_idx, -1)
    for r, a in enumerate(a_idx):
        if a < 0:
            continue
        up[r, 0] = dn[r, 0] = a
        cand = a + np.arange(1, max_pos + 1)
        ok = (cand < n) & quoted[np.clip(cand, 0, n - 1)]
        cand_ok = np.where(ok, cand, a)
        mirrored = _mirror_index(strikes, a, cand_ok)
        mirrored = np.where(ok & (mirrored >= 0) & quoted[np.clip(mirrored, 0, n - 1)],
                            mirrored, -1)
        up[r, 1:] = np.where(ok & (mirrored >= 0), cand, -1)
        dn[r, 1:] = mirrored
    return a0, a_idx, up, dn


def _leg_indices(a_idx, up, dn, n_strikes):
    """The (P, 7) strike-index matrix for the whole grid, and which rows resolve.

    Every unresolved slot is clipped to a valid index AFTER ``ok`` has recorded
    that it was unresolved, so downstream gathers are unconditional array
    arithmetic and the mask is the only thing that decides what counts.
    """
    rows = GRID.anchor  # anchor offsets are 0/1/2, which index the tables directly
    idx = np.zeros((len(GRID), Grid.N_SLOTS), dtype=np.int64)
    idx[:, 0] = a_idx[rows]
    ok = idx[:, 0] >= 0
    for t in range(3):
        d = GRID.offs[:, t]
        u = up[rows, d]
        m = dn[rows, d]
        idx[:, 1 + 2 * t] = u
        idx[:, 2 + 2 * t] = m
        used = GRID.slot_used[:, 1 + 2 * t]
        ok &= ~used | ((u >= 0) & (m >= 0))
    return np.clip(idx, 0, n_strikes - 1), ok


def _floor_ok(strikes: np.ndarray, idx: np.ndarray, qty: np.ndarray) -> np.ndarray:
    """Payoff >= 0 at every strike, in DOLLARS, for each candidate.

    The payoff is piecewise linear with breakpoints only at the strikes and is
    zero outside them (the contracts sum to zero and the strikes mirror), so
    nonnegativity at the strikes is nonnegativity everywhere. Slots a family
    does not use carry qty 0 and contribute nothing.
    """
    k = strikes[idx]                                   # (P, 7)
    payoff = (qty[:, None, :] * np.maximum(k[:, None, :] - k[:, :, None], 0.0)).sum(axis=2)
    # Which slots exist is read off `qty`, not off the module-level grid: this
    # function is also called on a single hand-built candidate, and reaching
    # for a global that happened to line up with the argument is how a helper
    # becomes correct only at its one call site.
    used = qty != 0
    return ~((payoff < -1e-9) & used).any(axis=1)


def _prices(bid, ask, alpha):
    """``(buy price, sell price)`` per strike at one fill alpha."""
    fill = FillModel(float(alpha))
    return fill.buy(bid, ask), fill.sell(bid, ask)


def _net(qty, idx, px_long, px_short):
    """``sum_i q_i * px_i`` with the price chosen by the SIGN of the position.

    Opening: a long leg pays ``buy`` and a short leg receives ``sell``, and
    ``cost`` is the net of the two. Closing: the sides flip, so the caller
    passes ``sell`` for longs and ``buy`` for shorts and gets ``exit_value``.
    This is ``FillModel.cash_flow`` summed over legs, sign-for-sign.
    """
    px = np.where(qty > 0, px_long[idx], px_short[idx])
    return (qty * px).sum(axis=1)


def _sim_means(strikes, spot, dte_exit, pred_move, pred_crush, pre_iv,
               event_date, pool, key):
    """Mean simulated exit value of ONE put at each ladder strike, per draw half.

    Reproduces ``engine.pnl_sim.expected_pnl``'s draw sequence exactly — same
    SHA-256 seeding, same pool draw, same sign draw, same floors — and stops
    one step earlier, at the per-strike means, because expectation is linear in
    the contract vector and the candidates differ only in that vector.
    """
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(f"{key}|{event_date}".encode()).digest()[:8], "big")
    )
    err_move, err_crush = pool.draw(event_date, pred_move, DRAWS, rng)
    if err_move.size == 0:
        return None, None, 0
    move = np.maximum(pred_move + err_move, 0.0)
    crush = pred_crush + err_crush
    sign = rng.choice((-1.0, 1.0), size=move.size)
    spot_exit = spot * np.maximum(1.0 + sign * move / 100.0, pnl_sim.MIN_SPOT_FRACTION)
    vol_exit = (pre_iv / 100.0) * (1.0 + crush / 100.0)
    values = pnl_sim.black_scholes_put(
        spot_exit[:, None], strikes[None, :], dte_exit / 365.0, vol_exit[:, None]
    )
    return (values[:HALF].mean(axis=0), values[HALF:].mean(axis=0),
            int(pool.before(event_date)))


def _incumbent_candidate(strikes, spot, a0, quoted, pred_move):
    """TWIN-P5 wing 3 sized ``a = pred_abs_move / 100`` — the rule as it stands.

    Not drawn from the pattern grid: its spacing comes from a target share of
    spot snapped to the ladder (``StrikeSelector('offset_from')``), not from a
    ladder position, so it is rebuilt here leg for leg. Returns ``(idx, qty)``
    or ``None`` when the ladder cannot carry it — which is exactly what
    EXP-126 counted as ``structure_unresolved``.
    """
    target_frac = pred_move / 100.0
    if not INCUMBENT_WIDTH_MIN <= target_frac <= INCUMBENT_WIDTH_MAX:
        return None
    a = strikes[a0]
    above = np.flatnonzero((strikes > a) & quoted)
    if above.size == 0:
        return None
    up1 = int(above[np.abs(strikes[above] - (a + target_frac * spot)).argmin()])
    step = strikes[up1] - a
    want = np.array([a - step, a + 3.0 * step, a - 3.0 * step])
    idx = []
    for target in want:
        pos = np.searchsorted(strikes, target)
        hit = -1
        for cand in (pos - 1, pos):
            if 0 <= cand < strikes.size and abs(strikes[cand] - target) <= 1e-6 * max(1.0, abs(target)):
                hit = int(cand)
        if hit < 0 or not quoted[hit]:
            return None
        idx.append(hit)
    dn1, up_w, dn_w = idx
    # TWIN-P5: +2 at A, -2 at A±a, +1 at A±3a. Slot order matches the grid's.
    return (np.array([a0, up1, dn1, up_w, dn_w, a0, a0], dtype=np.int64),
            np.array([2.0, -2.0, -2.0, 1.0, 1.0, 0.0, 0.0]))


def price_event(row, entry_rows, exit_rows, pool, *, rng_pick) -> dict | None:
    """Every candidate for one event: geometry, real P&L, simulated P&L, choices.

    Returns ``None`` with a reason recorded when the event yields no candidate
    at all. Nothing here decides whether to TRADE — the gate is applied in
    ``run.py`` over the whole history, because its bar is a trailing quantile
    and no single event can see it.
    """
    entry_rows = replay_mod._clean(entry_rows)
    exit_rows = replay_mod._clean(exit_rows)
    if entry_rows.empty or exit_rows.empty:
        raise EventSkip("bad_quote")

    entry_snap = ChainSnapshot(ticker=row["ticker"], obs_date=row["entry_date"],
                               event_date=row["event_date"], rows=entry_rows,
                               session=row["session"])
    spot_entry = entry_snap.spot_price
    expiry = ExpirySelector(kind="first_post_event").select(
        entry_rows, row["event_date"], row["session"])

    strikes, bid_e, ask_e = _ladder(entry_rows, expiry)
    if strikes.size < 3:
        raise EventSkip("structure_unresolved")
    exit_at_expiry = exit_rows[(exit_rows["right"] == "P") & (exit_rows["expiry"] == expiry)]
    if exit_at_expiry.empty:
        raise EventSkip("expiry_gone_at_exit")

    # The exit prices the SAME contracts, so it is looked up on the entry
    # ladder rather than rebuilt: a strike the exit chain does not quote is a
    # strike no candidate may use, which is `structure_unresolved` under
    # pinning and not a silently different structure.
    ex = exit_at_expiry.drop_duplicates("strike").set_index("strike")
    bid_x = ex["bid"].reindex(strikes).to_numpy(dtype=float)
    ask_x = ex["ask"].reindex(strikes).to_numpy(dtype=float)
    quoted = np.isfinite(bid_x) & np.isfinite(ask_x)
    bid_x = np.where(quoted, bid_x, 0.0)
    ask_x = np.where(quoted, ask_x, 0.0)
    if quoted.sum() < 3:
        raise EventSkip("structure_unresolved")

    exit_snap = ChainSnapshot(ticker=row["ticker"], obs_date=row["exit_date"],
                              event_date=row["event_date"], rows=exit_rows,
                              session=row["session"])
    spot_exit = exit_snap.spot_price
    dte_entry = int(entry_rows[entry_rows["expiry"] == expiry]["dte"].iloc[0])
    dte_exit = int(exit_at_expiry["dte"].iloc[0])

    a0, a_idx, up, dn = _lookup_tables(strikes, spot_entry, quoted)
    idx, resolves = _leg_indices(a_idx, up, dn, strikes.size)
    if not resolves.any():
        raise EventSkip("structure_unresolved")

    # --- the two width rules, on the OUTER strikes, in dollars --------------
    anchor_px = strikes[idx[:, 0]]
    outer_up = np.where(GRID.outer > 0, up[GRID.anchor, GRID.outer], -1)
    have_outer = resolves & (outer_up >= 0)
    half = np.where(have_outer, strikes[np.clip(outer_up, 0, strikes.size - 1)] - anchor_px, np.nan)
    lo, hi = anchor_px - half, anchor_px + half
    m, s = row["pred_abs_move"], row["pred_abs_move_sd"]
    spans = (lo < spot_entry * (1 - m / 100.0)) & (hi > spot_entry * (1 + m / 100.0))
    within = ((lo >= spot_entry * (1 - (m + 3 * s) / 100.0))
              & (hi <= spot_entry * (1 + (m + 3 * s) / 100.0)))
    listed = resolves & have_outer
    floor_ok = _floor_ok(strikes, idx, GRID.qty)
    admissible = listed & spans & within & floor_ok

    # --- real money, every alpha -------------------------------------------
    priced = {}
    for alpha in ALPHA_GRID:
        buy_e, sell_e = _prices(bid_e, ask_e, alpha)
        buy_x, sell_x = _prices(bid_x, ask_x, alpha)
        cost = _net(GRID.qty, idx, buy_e, sell_e)
        value = _net(GRID.qty, idx, sell_x, buy_x)
        priced[alpha] = (cost, value)
    cost_mid, value_mid = priced[MID]
    # `engine.replay.replay_one` refuses an event outright — across EVERY alpha —
    # as soon as one alpha prices it at or below MIN_MEANINGFUL_COST, because a
    # structure that opens for a credit at one fill assumption has no
    # return-on-debit and its alpha sweep would otherwise be computed on a
    # different sample at every alpha. The same rule has to hold per CANDIDATE
    # here, since a candidate is what an event offers. The first build applied
    # it at mid only, which left the sweep comparing 12,437 rows at alpha 0.5
    # against 5,188 at alpha 1.0 and made `breakeven_alpha` an interpolation
    # between two different books.
    priceable = np.ones(len(GRID), dtype=bool)
    for alpha in ALPHA_GRID:
        priceable &= priced[alpha][0] > MIN_MEANINGFUL_COST
    admissible &= priceable

    # Mean relative spread across the legs that exist — EXP-126's filter, and
    # the one thing standing between a width search and the widest quotes on
    # the board.
    mid_e = 0.5 * (bid_e + ask_e)
    rel = np.where(mid_e > 0, (ask_e - bid_e) / np.where(mid_e > 0, mid_e, 1.0), np.inf)
    leg_rel = rel[idx]
    n_legs = GRID.slot_used.sum(axis=1)
    rel_spread = np.where(GRID.slot_used, leg_rel, 0.0).sum(axis=1) / np.maximum(n_legs, 1)
    tradeable = admissible & (rel_spread <= MAX_REL_SPREAD)

    # --- simulated expectation, linear in the contract vector ---------------
    m_sel, m_gate, pool_n = _sim_means(
        strikes, spot_entry, dte_exit, row["pred_abs_move"], row["pred_iv_crush_30"],
        row["pre_iv30"], row["event_date"], pool, key=f"EXP-133|{row['ticker']}")
    # A structure whose long and short legs offset can net to a debit at or
    # below zero. That is a real state, not an error — replay_one calls it
    # `zero_cost` — and every such candidate is already masked out by
    # `cost_mid > MIN_MEANINGFUL_COST` above, so the division is allowed to
    # produce the NaN it is going to produce rather than warn about it.
    with np.errstate(divide="ignore", invalid="ignore"):
        if m_sel is None:
            sim_sel = sim_gate = np.full(len(GRID), np.nan)
            m_sel = m_gate = None
        else:
            sim_sel = ((GRID.qty * m_sel[idx]).sum(axis=1) - cost_mid) / cost_mid
            sim_gate = ((GRID.qty * m_gate[idx]).sum(axis=1) - cost_mid) / cost_mid
        ret_mid = (value_mid - cost_mid) / cost_mid

    # --- the choices --------------------------------------------------------
    picks: dict[str, int | None] = {}
    for arm, pattern_mask in ARM_PATTERN_MASK.items():
        pool_mask = tradeable & pattern_mask
        if arm != "oracle_realized":
            pool_mask = pool_mask & np.isfinite(sim_sel)
        rows_ok = np.flatnonzero(pool_mask)
        if rows_ok.size == 0:
            picks[arm] = None
            continue
        if arm == "random_pick":
            picks[arm] = int(rows_ok[rng_pick.integers(0, rows_ok.size)])
        elif arm == "oracle_realized":
            picks[arm] = int(rows_ok[np.argmax(ret_mid[rows_ok])])
        else:
            picks[arm] = int(rows_ok[np.argmax(sim_sel[rows_ok])])

    return {
        "strikes": strikes, "bid_e": bid_e, "ask_e": ask_e,
        "bid_x": bid_x, "ask_x": ask_x, "quoted": quoted,
        "idx": idx, "priced": priced, "sim_sel": sim_sel, "sim_gate": sim_gate,
        "ret_mid": ret_mid, "rel_spread": rel_spread, "n_legs": n_legs,
        "listed": listed, "admissible": admissible, "tradeable": tradeable,
        "picks": picks, "spot_entry": spot_entry, "spot_exit": spot_exit,
        "expiry": expiry, "dte_entry": dte_entry, "dte_exit": dte_exit,
        "pool_n": pool_n, "a0": a0, "half": half, "anchor_px": anchor_px,
        "outer_up": outer_up, "m_sel": m_sel, "m_gate": m_gate,
        "spans": spans, "within": within, "floor_ok": floor_ok,
        "incumbent": _incumbent_candidate(strikes, spot_entry, a0, quoted,
                                          row["pred_abs_move"]),
    }


# --------------------------------------------------------------------------
# turning a chosen candidate back into a trade row
# --------------------------------------------------------------------------


def _legs_blob(ev, leg_idx, leg_qty, alpha):
    """The ``legs`` JSON ``engine.replay`` writes, for one candidate at one alpha.

    Reproduced rather than referenced because the tail-injection stress reads
    it: ``experiments.common._tail_shock`` reprices the exit from the stored
    quotes, and a blob missing a leg would silently shock a different
    structure than the one that traded.
    """
    fill = FillModel(float(alpha))
    strikes = ev["strikes"]
    entry, exits = [], []
    for slot in range(Grid.N_SLOTS):
        q = float(leg_qty[slot])
        if q == 0:
            continue
        i = int(leg_idx[slot])
        side = "buy" if q > 0 else "sell"
        opp = "sell" if q > 0 else "buy"
        be, ae = float(ev["bid_e"][i]), float(ev["ask_e"][i])
        bx, ax = float(ev["bid_x"][i]), float(ev["ask_x"][i])
        name = ("atm" if slot == 0 else
                f"{'up' if slot % 2 else 'dn'}{(slot + 1) // 2}")
        entry.append({"name": name, "right": "P", "side": side, "qty": abs(q),
                      "strike": float(strikes[i]), "expiry": str(pd.Timestamp(ev["expiry"]).date()),
                      "dte": ev["dte_entry"], "bid": be, "ask": ae,
                      "price": float(fill.price(side, be, ae)),
                      "cash_flow": float(fill.cash_flow(side, be, ae, abs(q))),
                      "wide_market": bool(FillModel.is_wide(be, ae))})
        exits.append({"name": name, "right": "P", "side": opp, "qty": abs(q),
                      "strike": float(strikes[i]), "expiry": str(pd.Timestamp(ev["expiry"]).date()),
                      "dte": ev["dte_exit"], "bid": bx, "ask": ax,
                      "price": float(fill.price(opp, bx, ax)),
                      "cash_flow": float(fill.cash_flow(opp, bx, ax, abs(q))),
                      "wide_market": bool(FillModel.is_wide(bx, ax))})
    return json.dumps({"spot_entry": ev["spot_entry"], "spot_exit": ev["spot_exit"],
                       "dte_entry": ev["dte_entry"], "entry": entry, "exit": exits})


def _score_one(ev, leg_idx, leg_qty):
    """Cost, exit value, expected P&L and spread for ONE candidate, every alpha."""
    out = {"cost": {}, "value": {}}
    for alpha in ALPHA_GRID:
        buy_e, sell_e = _prices(ev["bid_e"], ev["ask_e"], alpha)
        buy_x, sell_x = _prices(ev["bid_x"], ev["ask_x"], alpha)
        out["cost"][alpha] = float(_net(leg_qty[None, :], leg_idx[None, :], buy_e, sell_e)[0])
        out["value"][alpha] = float(_net(leg_qty[None, :], leg_idx[None, :], sell_x, buy_x)[0])
    mid_e = 0.5 * (ev["bid_e"] + ev["ask_e"])
    rel = np.where(mid_e > 0, (ev["ask_e"] - ev["bid_e"]) / np.where(mid_e > 0, mid_e, 1.0), np.inf)
    used = leg_qty != 0
    out["rel_spread"] = float(rel[leg_idx][used].mean())
    out["n_legs"] = int(used.sum())
    out["contracts"] = float(np.abs(leg_qty).sum())
    return out


def _geometry(ev, leg_idx, leg_qty, row):
    """The numbers the report reads the chosen shape off: width, wings, landing."""
    strikes, used = ev["strikes"], leg_qty != 0
    k = strikes[leg_idx][used]
    anchor = float(strikes[leg_idx[0]])
    half = float(k.max() - anchor)
    return {
        "anchor": anchor,
        "half_width": half,
        "half_width_pct_spot": 100.0 * half / ev["spot_entry"],
        # How wide the structure is against the move it must span. 1.0 is the
        # binding edge of the first width rule; the second caps it at
        # (m + 3s)/m. The distribution between them is the search's answer to
        # "how wide should this be", which no formula in the program has ever
        # been asked.
        "width_over_forecast": (100.0 * half / ev["spot_entry"]) / float(row["pred_abs_move"]),
        "anchor_over_spot": anchor / ev["spot_entry"],
        "landed_pct_spot": 100.0 * abs(ev["spot_exit"] - ev["spot_entry"]) / ev["spot_entry"],
        # Where the print landed relative to the wings: >= 1 is outside the
        # structure entirely, where every family pays exactly zero.
        "landed_over_half": abs(ev["spot_exit"] - anchor) / half if half > 0 else np.nan,
    }


def _emit(ev, row, arm, leg_idx, leg_qty, pattern_index, sim_sel, sim_gate, label):
    """One candidate as five trade rows — the shape ``engine.evaluate`` requires."""
    scored = _score_one(ev, leg_idx, leg_qty)
    if scored["cost"][MID] <= MIN_MEANINGFUL_COST:
        return []
    geo = _geometry(ev, leg_idx, leg_qty, row)
    rows = []
    for alpha in ALPHA_GRID:
        cost, value = scored["cost"][alpha], scored["value"][alpha]
        rows.append({
            "arm": arm, "strategy": "EXP133", "variant": label,
            "event_id": row["event_id"], "ticker": row["ticker"],
            "event_date": row["event_date"], "session": row["session"],
            "entry_date": row["entry_date"], "exit_date": row["exit_date"],
            "fill_alpha": float(alpha),
            "entry_cost": cost, "exit_value": value, "pnl": value - cost,
            "ret": (value - cost) / cost if cost > MIN_MEANINGFUL_COST else np.nan,
            "spot_entry": ev["spot_entry"], "spot_exit": ev["spot_exit"],
            "strike": geo["anchor"], "expiry": pd.Timestamp(ev["expiry"]),
            "dte_entry": ev["dte_entry"], "dte_exit": ev["dte_exit"],
            "n_legs": scored["n_legs"], "contracts": scored["contracts"],
            "rel_spread": scored["rel_spread"],
            "wide_market": bool(FillModel.is_wide(ev["bid_e"], ev["ask_e"])[leg_idx][leg_qty != 0].any()),
            "quote_repaired": False,
            "pattern_index": pattern_index,
            "exp_pnl_sim": sim_gate, "exp_pnl_sim_select": sim_sel,
            "pool_n": ev["pool_n"],
            "pred_abs_move": float(row["pred_abs_move"]),
            "pred_abs_move_sd": float(row["pred_abs_move_sd"]),
            "n_admissible": int(ev["tradeable"].sum()),
            **geo,
            "legs": _legs_blob(ev, leg_idx, leg_qty, alpha),
        })
    return rows


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def _pattern_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "pattern_index": np.arange(len(GRID)),
        "key": GRID.key, "shape_key": GRID.shape_key, "family_key": GRID.family_key,
        "n_strikes": GRID.n_strikes, "twin_peaked": GRID.twin,
        "anchor_offset": GRID.anchor, "contracts": GRID.contracts,
        "offsets": [":".join(str(d) for d in p.offsets) for p in GRID.patterns],
        "label": [p.family.label for p in GRID.patterns],
    })


def build_all(*, force: bool = False, limit_years=None) -> dict:
    """Price every candidate on every event, one year of chains at a time."""
    out_trades = RESULTS / "candidates.parquet"
    out_tally = RESULTS / "tallies.parquet"
    out_events = RESULTS / "event_summary.parquet"
    if out_trades.exists() and out_tally.exists() and not force:
        print("[build] cached", flush=True)
        return {"trades": pd.read_parquet(out_trades),
                "tallies": pd.read_parquet(out_tally),
                "events": pd.read_parquet(out_events),
                "meta": json.loads((RESULTS / "build_meta.json").read_text())}

    started = time.time()
    events = event_universe()
    events["event_date"] = pd.to_datetime(events["event_date"])
    events = events.merge(forecasts(), on=["ticker", "event_date"], how="inner")
    print(f"[build] {len(events):,} events carry every forecast the sim needs", flush=True)

    from engine.structures import twin_peak_5
    plan = replay_mod.filter_plan_by_availability(
        replay_mod.plan_events(twin_peak_5(wing_multiple=3, width_moneyness=0.05), events))
    keyframe = plan.frame.set_index("event_id")
    events = events[events["event_id"].isin(set(plan.frame["event_id"]))].copy()
    events["_year"] = events["event_date"].dt.year
    if limit_years is not None:
        events = events[events["_year"].isin(set(limit_years))]
    print(f"[build] {len(events):,} of those have both chains "
          f"({events['_year'].nunique()} years)", flush=True)

    pool = pnl_sim.ResidualPool(residual_history())
    print(f"[build] residual pool: {len(pool):,} paired errors", flush=True)

    n = len(GRID)
    tally = {name: np.zeros(n, dtype=np.int64)
             for name in ("listed", "admissible", "tradeable")}
    for arm in ARM_PATTERN_MASK:
        tally[f"argmax_{arm}"] = np.zeros(n, dtype=np.int64)
    skips: dict[str, int] = {}
    parts, ev_rows = [], []
    equivalence: list[dict] = []

    for year, block in events.groupby("_year", sort=True):
        rows = keyframe.reindex(block["event_id"]).dropna(subset=["entry_date"])
        keys = set()
        for col in ("entry_date", "exit_date"):
            keys |= {(t, d) for t, d in zip(rows["ticker"], rows[col])}
        index = replay_mod.load_chain_index(keys, progress_every=0)

        merged = block.merge(
            plan.frame[["event_id", "entry_date", "exit_date"]], on="event_id", how="inner")
        for row in merged.to_dict("records"):
            entry_rows = index.get(row["ticker"], row["entry_date"])
            exit_rows = index.get(row["ticker"], row["exit_date"])
            if entry_rows is None or entry_rows.empty:
                skips["no_entry_chain"] = skips.get("no_entry_chain", 0) + 1
                continue
            if exit_rows is None or exit_rows.empty:
                skips["no_exit_chain"] = skips.get("no_exit_chain", 0) + 1
                continue
            # Seeded per event so `random_pick` is a fixed null rather than a
            # different null every time the build is rerun.
            rng_pick = np.random.default_rng(
                int.from_bytes(hashlib.sha256(
                    f"pick|{row['event_id']}".encode()).digest()[:8], "big"))
            try:
                ev = price_event(row, entry_rows, exit_rows, pool, rng_pick=rng_pick)
            except EventSkip as exc:
                skips[str(exc)] = skips.get(str(exc), 0) + 1
                continue
            except Exception as exc:                      # noqa: BLE001
                skips[f"error:{type(exc).__name__}"] = skips.get(
                    f"error:{type(exc).__name__}", 0) + 1
                continue

            # Sample the cross-check across the WHOLE run, not just the first
            # event that offered a candidate. The first build checked one event
            # and the receipt described itself as "a sampled cross-check",
            # which overstated what had been verified.
            if ev["tradeable"].any() and len(equivalence) < EQUIV_EVENTS:
                if rng_pick.random() < EQUIV_RATE or len(equivalence) == 0:
                    equivalence.append(
                        check_equivalence(ev, row, entry_rows, exit_rows, pool))

            tally["listed"] += ev["listed"]
            tally["admissible"] += ev["admissible"]
            tally["tradeable"] += ev["tradeable"]
            for arm, pick in ev["picks"].items():
                if pick is not None:
                    tally[f"argmax_{arm}"][pick] += 1
                    parts.extend(_emit(
                        ev, row, arm, ev["idx"][pick], GRID.qty[pick], int(pick),
                        float(ev["sim_sel"][pick]), float(ev["sim_gate"][pick]),
                        str(GRID.key[pick])))
            inc = ev["incumbent"]
            if inc is not None:
                inc_idx, inc_qty = inc
                sc = _score_one(ev, inc_idx, inc_qty)
                if sc["cost"][MID] > MIN_MEANINGFUL_COST and sc["rel_spread"] <= MAX_REL_SPREAD:
                    sim_s, sim_g = _incumbent_sim(ev, inc_idx, inc_qty, sc["cost"][MID])
                    parts.extend(_emit(ev, row, "incumbent", inc_idx, inc_qty, -1,
                                       sim_s, sim_g, "TWIN-P5-w3-pred100"))

            ev_rows.append({
                "event_id": row["event_id"], "ticker": row["ticker"],
                "event_date": row["event_date"], "year": int(year),
                "n_listed": int(ev["listed"].sum()),
                "n_admissible": int(ev["admissible"].sum()),
                "n_tradeable": int(ev["tradeable"].sum()),
                # Which rule actually stopped each listed candidate. A search
                # whose universe is set by the ladder is a different experiment
                # from one whose universe is set by the width rules, and the
                # two are told apart only here.
                "n_too_narrow": int((ev["listed"] & ~ev["spans"]).sum()),
                "n_too_wide": int((ev["listed"] & ev["spans"] & ~ev["within"]).sum()),
                "n_floor_fail": int((ev["listed"] & ev["spans"] & ev["within"]
                                     & ~ev["floor_ok"]).sum()),
                "n_spread_fail": int((ev["admissible"] & ~ev["tradeable"]).sum()),
                "ladder_bound_binds": bool(
                    (ev["listed"] & (GRID.outer >= fam.MAX_LADDER_POS)).any()),
                "pool_n": ev["pool_n"], "spot_entry": ev["spot_entry"],
                "ladder_steps": int(ev["strikes"].size),
                "pred_abs_move": float(row["pred_abs_move"]),
                "pred_abs_move_sd": float(row["pred_abs_move_sd"]),
                "incumbent_resolved": bool(ev["incumbent"] is not None),
            })
        del index
        print(f"[build] {year}: {len(ev_rows):,} events priced, "
              f"{len(parts):,} trade rows, {time.time() - started:.0f}s", flush=True)

    trades = pd.DataFrame(parts)
    tallies = _pattern_frame()
    for name, counts in tally.items():
        tallies[name] = counts
    event_summary = pd.DataFrame(ev_rows)
    meta = {
        "events_priced": int(len(event_summary)),
        "skips": skips,
        "patterns": int(n),
        "equivalence": _equiv_summary(equivalence),
        "equivalence_events": equivalence,
        "elapsed_s": round(time.time() - started, 1),
    }
    trades.to_parquet(out_trades, index=False)
    tallies.to_parquet(out_tally, index=False)
    event_summary.to_parquet(out_events, index=False)
    (RESULTS / "build_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[build] done: {len(event_summary):,} events, {len(trades):,} rows, "
          f"skips {skips}", flush=True)
    return {"trades": trades, "tallies": tallies, "events": event_summary, "meta": meta}


def _incumbent_sim(ev, leg_idx, leg_qty, cost_mid):
    """The incumbent's expected P&L, off the same per-strike means as everyone else.

    The incumbent is not in the pattern grid — its spacing comes from a share
    of spot rather than a ladder position — but it is priced against the same
    ladder, the same draws and the same per-strike means, so the arm comparison
    is a comparison of contract vectors and nothing else.
    """
    if ev["m_sel"] is None:
        return float("nan"), float("nan")
    sel = float((leg_qty * ev["m_sel"][leg_idx]).sum())
    gate = float((leg_qty * ev["m_gate"][leg_idx]).sum())
    return (sel - cost_mid) / cost_mid, (gate - cost_mid) / cost_mid


# --------------------------------------------------------------------------
# the equivalence receipt
# --------------------------------------------------------------------------


#: How far the fast pricer may differ from ``price_structure`` before the run
#: refuses to continue. Dollar prices here are O(1)-O(100), so 1e-9 is
#: float64 summation noise and nothing else.
EQUIV_TOL = 1e-9
#: Candidates cross-checked per sampled event, and how many events to sample.
#: Spread across the whole run rather than taken from the front, so the receipt
#: covers every year, every family and every ladder shape the run actually met.
EQUIV_SAMPLE = 12
EQUIV_EVENTS = 150
EQUIV_RATE = 0.01


def _equiv_summary(receipts: list[dict]) -> dict:
    """Roll the per-event receipts into the one line the report prints."""
    if not receipts:
        return {}
    return {
        "events_checked": len(receipts),
        "candidates_sampled": sum(r["candidates_sampled"] for r in receipts),
        "price_comparisons": sum(r["price_comparisons"] for r in receipts),
        "price_max_abs_diff": max(r["price_max_abs_diff"] for r in receipts),
        "price_skipped_no_structure": sum(r["price_skipped_no_structure"] for r in receipts),
        "sim_comparisons": sum(r["sim_comparisons"] for r in receipts),
        "sim_max_abs_diff": max(r["sim_max_abs_diff"] for r in receipts),
        "tolerance": EQUIV_TOL,
        "first_event": receipts[0]["ticker"] + " " + receipts[0]["event_date"][:10],
        "last_event": receipts[-1]["ticker"] + " " + receipts[-1]["event_date"][:10],
    }


def check_equivalence(ev, row, entry_rows, exit_rows, pool) -> dict:
    """Price a sample of candidates BOTH ways and refuse to continue if they differ.

    The program's load-bearing rule is that there is exactly one pricing path,
    ``engine.structures.price_structure``, so the live scorer and the research
    code cannot drift apart. This module adds a second one — a linear algebra
    shortcut that prices 12,600 candidates in the time the first prices one —
    and the only thing that makes that tolerable is proving, on real chains,
    that it computes the same numbers. Two claims are checked:

    1. **entry cost and exit value at every alpha** against ``price_structure``
       resolving the same candidate through its own strike selectors. This
       tests the strike arithmetic (grid steps, listed mirrors) as well as the
       fill arithmetic, because the selectors resolve the strikes independently
       from the ladder rather than being handed the indices.
    2. **expected P&L** against ``engine.pnl_sim.expected_pnl``, the live gate's
       own function, on the same seed. This is where the linearity claim is
       tested: the shortcut averages 4,000 draws once per STRIKE and dots them
       into the contract vector, and ``expected_pnl`` averages 4,000 draws of
       the whole structure. They are the same number or the shortcut is wrong.

    Four-strike candidates cannot take path 1 — ``LegSpec`` requires a positive
    qty, so a family with no contract at its own axis has no ``Structure`` to
    mirror about — and the receipt reports the share skipped rather than
    quietly narrowing what was checked.
    """
    from engine.structures import price_structure, structure_return

    entry_snap = ChainSnapshot(ticker=row["ticker"], obs_date=row["entry_date"],
                               event_date=row["event_date"], rows=entry_rows,
                               session=row["session"])
    exit_snap = ChainSnapshot(ticker=row["ticker"], obs_date=row["exit_date"],
                              event_date=row["event_date"], rows=exit_rows,
                              session=row["session"])
    rows_ok = np.flatnonzero(ev["tradeable"])
    rng = np.random.default_rng(0)
    sample = rows_ok if rows_ok.size <= EQUIV_SAMPLE else rng.choice(
        rows_ok, EQUIV_SAMPLE, replace=False)

    worst, checked, skipped = 0.0, 0, 0
    for p in sample:
        structure = fam.to_structure(GRID.patterns[int(p)])
        if structure is None:
            skipped += 1
            continue
        pinned = None
        for alpha in ALPHA_GRID:
            fill = FillModel(float(alpha))
            entry = price_structure(structure, entry_snap, fill, pin=pinned)
            pinned = pinned or entry.legs
            exit_ = price_structure(structure, exit_snap, fill, pin=pinned, closing=True)
            got = structure_return(entry, exit_)
            mine_cost = ev["priced"][alpha][0][p]
            mine_value = ev["priced"][alpha][1][p]
            worst = max(worst, abs(got["cost"] - mine_cost),
                        abs(got["exit_value"] - mine_value))
            checked += 1
        if worst > EQUIV_TOL:
            raise RuntimeError(
                f"pricer disagreement {worst:.3e} > {EQUIV_TOL:.0e} on "
                f"{GRID.key[int(p)]} at {row['ticker']} {row['event_date']} — the fast "
                "pricer is not engine.structures.price_structure and the run refuses "
                "to continue")

    # -- claim 2: the linearity of the expectation --------------------------
    sim_worst, sim_checked = 0.0, 0
    if ev["m_sel"] is not None:
        for p in sample[:4]:
            idx, qty = ev["idx"][int(p)], GRID.qty[int(p)]
            legs = [{"strike": float(ev["strikes"][int(i)]), "qty": abs(float(q)),
                     "side": "sell" if q > 0 else "buy"}
                    for i, q in zip(idx, qty) if q != 0]
            got = pnl_sim.expected_pnl(
                exit_legs=legs, spot=ev["spot_entry"],
                entry_cost=float(ev["priced"][MID][0][int(p)]),
                pre_iv30=float(row["pre_iv30"]), pred_abs_move=float(row["pred_abs_move"]),
                pred_iv_crush=float(row["pred_iv_crush_30"]), dte_exit=float(ev["dte_exit"]),
                event_date=row["event_date"], pool=pool,
                key=f"EXP-133|{row['ticker']}", draws=DRAWS)
            if got is None:
                continue
            mine = 0.5 * (ev["sim_sel"][int(p)] + ev["sim_gate"][int(p)])
            sim_worst = max(sim_worst, abs(got["exp_pnl_sim"] - mine))
            sim_checked += 1
    if sim_checked and sim_worst > 1e-9:
        raise RuntimeError(
            f"expected-P&L disagreement {sim_worst:.3e} against "
            "engine.pnl_sim.expected_pnl — the linearity shortcut is not the "
            "same estimator and the run refuses to continue")

    return {
        "ticker": row["ticker"], "event_date": str(row["event_date"]),
        "candidates_sampled": int(len(sample)),
        "price_comparisons": checked,
        "price_max_abs_diff": worst,
        "price_skipped_no_structure": skipped,
        "sim_comparisons": sim_checked,
        "sim_max_abs_diff": sim_worst,
        "tolerance": EQUIV_TOL,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--years", type=int, nargs="*", default=None)
    args = ap.parse_args()
    build_all(force=args.force, limit_years=args.years)
