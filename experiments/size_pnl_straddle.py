"""EXP-188 S7: long-straddle buy rules R_L and R_L2, scored on the ladder chunks from S3.

Pure functions over numpy/pandas: no I/O, no repository, no ledger. A straddle is a long ATM call
plus a long ATM put at the first post-event expiry. Decisions use only forecast edges and entry
facts (never labels or exit quotes); marks come only from ``engine.fills.FillModel`` (section 6 of
the design). Progress logging: none needed (pure, per-quarter, called by the S9 driver).

Not wired to a production caller yet: the S9 driver (guides/size_decision_pnl_design.md section 9)
calls ``decide`` then ``score`` per quarter chunk.
"""
import math

import numpy as np
import pandas as pd

from engine.fills import BAD_QUOTE_COST_PCT, FillModel
from engine.score import GATE_MCAP_FLOOR
from engine.pnl_sim import trailing_cutoff
from engine.portfolio import CAPITAL_PER_TRADE, CONTRACT_MULTIPLIER

RULES = ("R_L", "R_L2")
GATE_RULE = "G_PROD"  # Amendment 3 sensitivity arm; not in RULES, so no hypothesis or default path reaches it. Decided only with a gate callable.
EXIT_UNQUOTED = "exit_unquoted_worst_case"
STATUSES = ("taken", "not_taken", "undetermined")
DECISION_COLS = ["event_id", "event_date", "edge", "cutoff", "rel_spread", "status", "reason"]


def _num(x):
    """``float(x)``, or NaN when ``x`` is None, pd.NA or another null."""
    if x is None or pd.isna(x):
        return float("nan")
    return float(x)


def in_universe(mcap):
    """Production's STR-THRU universe: market cap at or above engine.score GATE_MCAP_FLOOR (Amendment 5). No spread term: production applies none."""
    return math.isfinite(_num(mcap)) and _num(mcap) >= GATE_MCAP_FLOOR


def bad_quote(chunk, state, i):
    """True when the straddle's entry cost at the mid mark exceeds BAD_QUOTE_COST_PCT percent of spot (strictly greater, as production).

    The straddle universe is the market-cap floor plus production's BAD_QUOTE ceiling, with no spread term. ``state`` is the
    ``entry_state`` result for event ``i``, whose ``legs`` are set when the entry is ok."""
    c, p = state["legs"]
    cost_pct = entry_cost(chunk, c, p, 0.5) / float(chunk["spot_entry"][i]) * 100.0
    return cost_pct > BAD_QUOTE_COST_PCT


def _edge(pred, implied):
    """``pred - implied`` when both are finite, else NaN.

    Identical to ``forecast_edge`` in the v2 chooser features (``engine.v2.scoring.native_chooser_features``).
    It is kept local because this legacy experiment must not import from ``engine.v2``
    (root ARCHITECTURE.md section 3, the legacy-to-v2 layer boundary).
    """
    if math.isfinite(pred) and math.isfinite(implied):
        return pred - implied
    return float("nan")


def atm_legs(chunk, i):
    """(call_idx, put_idx) of the ATM straddle for event ``i``, or None.

    As in the registered ``straddle_through``: the ATM strike is the CALL strike nearest
    ``spot_entry``, and the put must exist at that same strike. No shifting to another common strike.
    Ties go to the lower strike by this module's own rule: the explicit ``(distance, strike)`` key
    below. This is not ``StrikeSelector``'s atm, which takes the first ``idxmin`` row and so equals
    the lower strike only on an ascending ladder.
    """
    spot = float(chunk["spot_entry"][i])
    if not math.isfinite(spot):
        return None
    lo, hi = int(chunk["offsets"][i]), int(chunk["offsets"][i + 1])
    right, strike = chunk["right"][lo:hi], chunk["strike"][lo:hi]
    calls = sorted(set(strike[right == 1]))
    if not calls:
        return None
    best = min(calls, key=lambda s: (abs(s - spot), s))  # ties go to the lower strike
    puts = np.flatnonzero((right == 0) & (strike == best))
    if puts.size == 0:
        return None
    call = lo + int(np.flatnonzero((right == 1) & (strike == best))[0])
    return call, lo + int(puts[0])


def entry_state(chunk, i, *, mcap):
    """Entry status: ``ok`` with the mean relative spread, else ``undetermined`` with a reason."""
    out = {"legs": atm_legs(chunk, i), "rel_spread": float("nan"), "status": "undetermined", "reason": None}
    if out["legs"] is None:
        out["reason"] = "no_atm_pair"
        return out
    spreads = []
    for leg in out["legs"]:
        b, a = float(chunk["bid_e"][leg]), float(chunk["ask_e"][leg])
        # The entry rule is stricter than the exit rule: zero bid, locked and crossed all refuse.
        if not (math.isfinite(b) and math.isfinite(a)) or b <= 0 or a <= 0 or b >= a:
            out["reason"] = "entry_quote_unusable"
            return out
        spreads.append((a - b) / ((a + b) / 2.0))
    if not math.isfinite(_num(mcap)):
        out["reason"] = "missing_mcap"
        return out
    out.update(status="ok", rel_spread=float(np.mean(spreads)))
    return out


def exit_value(chunk, c, p, alpha, mid_marks=False):
    """(value_per_share, rung_label) for the straddle at exit. A rung-3 leg marks the whole
    structure at worst-case payoff 0.0; rung-2 legs fill at alpha 0 (the bid), or at ``alpha`` when ``mid_marks``
    (the Amendment 1 sensitivity arm)."""
    FillModel(alpha)  # validate alpha up front, whatever the rungs
    rungs =(int(chunk["exit_rung"][c]), int(chunk["exit_rung"][p]))
    if 3 in rungs:
        return 0.0, EXIT_UNQUOTED
    total = 0.0
    for leg, rung in ((c, rungs[0]), (p, rungs[1])):
        model = FillModel(alpha if (mid_marks or rung == 1) else 0.0)
        total += float(model.sell(chunk["bid_x"][leg], chunk["ask_x"][leg]))
    return total, "rung1" if rungs == (1, 1) else "rung2"


def entry_cost(chunk, c, p, alpha):
    """Per-share cost of buying the call and the put at the alpha mark."""
    model = FillModel(alpha)  # validates alpha
    return float(sum(model.buy(chunk["bid_e"][leg], chunk["ask_e"][leg]) for leg in (c, p)))


def lots(cost_per_share):
    """Whole lots as in ``engine/portfolio.py``: round down, floor of one; bad cost -> 1. Returns a float (whole lots; the representable dtype for any positive debit)."""
    cost = float(cost_per_share)
    if not math.isfinite(cost) or cost <= 0:
        return 1.0
    return float(max(1, math.floor(CAPITAL_PER_TRADE / (cost * CONTRACT_MULTIPLIER))))


def _verdict(clears, why):
    """(status, reason) for a threshold test: taken with no reason, else not_taken with ``why``."""
    return ("taken", None) if clears else ("not_taken", why)


def decide(events, history, rule, chunk, gate=None):
    """One decision row per event. ``history`` holds the model's own ``edge`` for all its events;
    ``chunk`` is the S3 ladder chunk that carries each event's entry quotes at the true ATM.
    ``gate`` (G_PROD only) maps an event record to (status, reason) and is called only after the universe filters pass."""
    if rule == GATE_RULE:
        if gate is None:
            raise ValueError("G_PROD needs a gate callable")
    elif rule not in RULES or gate is not None:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
    frame = history[["event_date", "edge"]].rename(columns={"edge": "exp_pnl_sim"})
    cache = {}  # trailing cutoff depends only on the month, so compute it once per month
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    rows = []
    for ev in events.to_dict("records"):
        edge = _edge(_num(ev["pred_abs_move"]), _num(ev["implied_move"]))
        row = {"event_id": ev["event_id"], "event_date": ev["event_date"], "edge": edge,
               "cutoff": float("nan"), "rel_spread": float("nan"), "status": "undetermined", "reason": None}
        if not math.isfinite(edge):
            row["reason"] = "missing_forecast"
        elif str(ev["event_id"]) not in pos:
            row["reason"] = "no_ladder_chunk"
        else:
            mcap = _num(ev["mcap_usd"])  # read only once the forecast is known to be present
            state = entry_state(chunk, pos[str(ev["event_id"])], mcap=mcap)
            if state["status"] != "ok":
                row["reason"] = state["reason"]
            else:
                rel = state["rel_spread"]
                row["rel_spread"] = rel  # recorded for every ok entry; the spread is not a universe term (Amendment 5)
                if not in_universe(mcap):
                    row.update(status="not_taken", reason="universe_filter")
                elif bad_quote(chunk, state, pos[str(ev["event_id"])]):
                    row.update(status="not_taken", reason="bad_quote")
                elif rule == GATE_RULE:
                    row["status"], row["reason"] = gate(ev)
                elif rule == "R_L2":
                    row["status"], row["reason"] = _verdict(edge >= 0, "below_zero")
                else:
                    date = pd.Timestamp(ev["event_date"])
                    key = date.to_period("M")
                    if key not in cache:
                        cache[key] = trailing_cutoff(frame, date)
                    cut = cache[key]
                    if cut is None:
                        row["reason"] = "thin_history"
                    else:
                        row["cutoff"] = cut
                        row["status"], row["reason"] = _verdict(edge >= cut, "below_cutoff")
        rows.append(row)
    return pd.DataFrame(rows, columns=DECISION_COLS)


def history_rows(decisions, events):
    """The (event_date, edge) rows that feed the R_L trailing cutoff (Amendment 5).

    The history universe is the market-cap floor and not BAD_QUOTE: an event whose ``decisions`` reason is ``bad_quote``
    leaves the history as well as the straddle book. An event whose entry quote is undetermined stays in the history
    when its mcap qualifies, since its BAD_QUOTE status is unknown.
    ``decisions`` rows are in the same order as ``events`` (``decide`` emits one row per event, in order)."""
    in_mcap = events["mcap_usd"].map(in_universe).to_numpy(dtype=bool)
    not_bad = decisions["reason"].to_numpy() != "bad_quote"
    mask = in_mcap & not_bad
    return decisions.loc[mask, ["event_date", "edge"]]


def score(chunk, events, decisions, alpha, *, drop_unquoted=False, mid_marks=False):
    """Marked P&L for TAKEN events only. ``drop_unquoted`` removes the worst-case rung-3 rows. ``mid_marks`` values
    rung-2 legs at ``alpha`` (Amendment 1 sensitivity); the taken set and rung labels are unchanged."""
    FillModel(alpha)  # validate alpha up front, even when nothing is taken
    known = set(events["event_id"].astype(str))
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    rows = []
    for eid in decisions.loc[decisions["status"] == "taken", "event_id"].astype(str):
        if eid not in known or eid not in pos:
            raise ValueError(f"taken event {eid!r} is missing from the chunk or events")
        legs = atm_legs(chunk, pos[eid])
        if legs is None:
            raise ValueError(f"taken event {eid!r} has no ATM straddle")
        c, p = legs
        cost = entry_cost(chunk, c, p, alpha)
        value, rung = exit_value(chunk, c, p, alpha, mid_marks)
        if drop_unquoted and rung == EXIT_UNQUOTED:
            continue
        n, per_share = lots(cost), value - cost
        rows.append({"event_id": eid, "entry_cost": cost, "exit_value": value, "rung": rung,
                     "lots": n, "pnl_per_share": per_share,
                     "pnl_dollars": per_share * CONTRACT_MULTIPLIER * n})
    return pd.DataFrame(rows, columns=["event_id", "entry_cost", "exit_value", "rung", "lots",
                                       "pnl_per_share", "pnl_dollars"])


def rung_shares(scored):
    """Share of scored rows per rung label; empty -> {}."""
    if scored.empty:
        return {}
    return scored["rung"].value_counts(normalize=True).to_dict()


def trade_counts(decisions):
    """Counts per status; undetermined is reported on its own, never folded into not_taken."""
    return {s: int((decisions["status"] == s).sum()) for s in STATUSES}
