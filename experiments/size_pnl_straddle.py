"""EXP-188 S7: the long straddle scored under the production gate G_PROD only (R_L and R_L2 removed, Amendment 6), on the ladder chunks from S3.

Pure functions over numpy/pandas: no I/O, no repository, no ledger. A straddle is production's STR-THRU structure (a long ATM call
plus a long ATM put at the first post-event expiry), resolved and priced by ``engine.structures.price_structure``. Decisions use only
forecast edges and entry facts (never labels or exit quotes); marks come only from ``engine.fills.FillModel`` (section 6 of
the design). Progress logging: none needed (pure, per-quarter, called by the S9 driver).

Not wired to a production caller yet: the S9 driver (guides/size_decision_pnl_design.md section 9)
calls ``decide`` then ``score`` per quarter chunk.
"""
import math

import numpy as np
import pandas as pd

from engine.fills import BAD_QUOTE_COST_PCT, FillModel
from engine.portfolio import CAPITAL_PER_TRADE, CONTRACT_MULTIPLIER
from engine.score import GATE_MCAP_FLOOR, _mean_relative_spread
from engine.structures import CALL, PUT, STRUCTURES, ChainSnapshot, StructureError, price_structure

GATE_RULE = "G_PROD"  # the only straddle rule now (Amendment 6); decided only with a gate callable.
EXIT_UNQUOTED = "exit_unquoted_worst_case"
STATUSES = ("taken", "not_taken", "undetermined")
DECISION_COLS = ["event_id", "event_date", "edge", "rel_spread", "status", "reason"]


def _num(x):
    """``float(x)``, or NaN when ``x`` is None, pd.NA or another null."""
    if x is None or pd.isna(x):
        return float("nan")
    return float(x)


def in_universe(mcap):
    """Production's STR-THRU universe: market cap at or above engine.score GATE_MCAP_FLOOR (Amendment 5). No spread term: production applies none."""
    return math.isfinite(_num(mcap)) and _num(mcap) >= GATE_MCAP_FLOOR


def _edge(pred, implied):
    """``pred - implied`` when both are finite, else NaN.

    Identical to ``forecast_edge`` in the v2 chooser features (``engine.v2.scoring.native_chooser_features``).
    It is kept local because this legacy experiment must not import from ``engine.v2``
    (root ARCHITECTURE.md section 3, the legacy-to-v2 layer boundary).
    """
    if math.isfinite(pred) and math.isfinite(implied):
        return pred - implied
    return float("nan")


def chain_snapshot(chunk, i, *, puts_only=False):
    """Event i's chain at entry quotes as price_structure reads it (one expiry, the ladder's first_post_event).

    The ladder already chose the single expiry, so the snapshot carries only it, and event_date = expiry makes the
    selector's inclusive rule return it. ``puts_only`` keeps only the put rows (the short-vol view)."""
    lo, hi = int(chunk["offsets"][i]), int(chunk["offsets"][i + 1])
    slots = np.arange(lo, hi)
    if puts_only:
        slots = slots[chunk["right"][lo:hi] == 0]
    expiry = pd.Timestamp(chunk["expiry"][i])
    n = slots.size
    rows = pd.DataFrame({"expiry": [expiry] * n, "dte": [chunk["dte_entry"][i]] * n,
                         "strike": chunk["strike"][slots], "right": np.where(chunk["right"][slots] == 0, PUT, CALL),
                         "bid": chunk["bid_e"][slots], "ask": chunk["ask_e"][slots]})
    return ChainSnapshot(ticker="", obs_date=expiry, event_date=expiry, rows=rows, spot=float(chunk["spot_entry"][i]))


def resolve(chunk, i, alpha=0.5):
    """(priced, call_idx, put_idx, reason): the ATM straddle for event i through production's price_structure (audit rows 3 and 16).

    priced is production's StructurePrice at the alpha fill; call_idx/put_idx are the flat chunk indices of the legs it resolved
    (matched by right and np.isclose strike). On a refusal the first three are None and reason is no_atm_pair (StructureError: no
    call, or no put at the call's strike) or entry_quote_unusable (ValueError from FillModel: crossed, negative or NaN bid/ask).
    A zero-bid or locked entry quote is priced normally, as production does (it is no longer undetermined)."""
    fill = FillModel(alpha)  # outside the try: a bad alpha propagates
    if not math.isfinite(float(chunk["spot_entry"][i])):
        return None, None, None, "no_atm_pair"
    try:
        priced = price_structure(STRUCTURES["STR-THRU"](), chain_snapshot(chunk, i), fill)
    except StructureError:
        return None, None, None, "no_atm_pair"
    except ValueError:  # FillModel refuses a crossed, negative or NaN quote
        return None, None, None, "entry_quote_unusable"
    lo, hi = int(chunk["offsets"][i]), int(chunk["offsets"][i + 1])
    slots = np.arange(lo, hi)
    found = {}
    for leg in priced.legs:
        code = 1 if leg.right == CALL else 0
        hit = slots[(chunk["right"][lo:hi] == code) & np.isclose(chunk["strike"][lo:hi], leg.strike)]
        found[leg.right] = int(hit[0])
    return priced, found[CALL], found[PUT], None


def entry_state(chunk, i, *, mcap):
    """Entry status: ``ok`` with the priced straddle and its mean relative spread, else ``undetermined`` with a reason."""
    priced, c, p, reason = resolve(chunk, i)
    if reason is None and not math.isfinite(_num(mcap)):
        reason = "missing_mcap"
    if reason is not None:
        return {"legs": None, "rel_spread": float("nan"), "status": "undetermined", "reason": reason, "priced": None}
    rel = _mean_relative_spread(priced)
    return {"legs": (c, p), "rel_spread": float("nan") if rel is None else rel, "status": "ok", "reason": None,
            "priced": priced}


def bad_quote(priced, spot):
    """True when the straddle's entry cost exceeds BAD_QUOTE_COST_PCT percent of spot (strictly greater, as production).

    ``priced`` is resolved at alpha 0.5, so its cost is the mid mark."""
    return priced.cost / spot * 100.0 > BAD_QUOTE_COST_PCT


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


def lots(cost_per_share):
    """Whole lots as in ``engine/portfolio.py``: round down, floor of one; bad cost -> 1. Returns a float (as build_book's contracts are float64); exact only below 2**53 lots."""
    cost = float(cost_per_share)
    if not math.isfinite(cost) or cost <= 0:
        return 1.0
    return float(max(1, math.floor(CAPITAL_PER_TRADE / (cost * CONTRACT_MULTIPLIER))))


def decide(events, chunk, gate):
    """One decision row per event. ``chunk`` is the S3 ladder chunk that carries each event's entry quotes at the true ATM.
    ``gate`` maps an event record to (status, reason) and is called only after the universe filter and BAD_QUOTE pass."""
    if gate is None:
        raise ValueError("G_PROD needs a gate callable")
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    rows = []
    for ev in events.to_dict("records"):
        edge = _edge(_num(ev["pred_abs_move"]), _num(ev["implied_move"]))
        row = {"event_id": ev["event_id"], "event_date": ev["event_date"], "edge": edge,
               "rel_spread": float("nan"), "status": "undetermined", "reason": None}
        if not math.isfinite(edge):
            row["reason"] = "missing_forecast"
        elif str(ev["event_id"]) not in pos:
            row["reason"] = "no_ladder_chunk"
        else:
            i = pos[str(ev["event_id"])]
            mcap = _num(ev["mcap_usd"])  # read only once the forecast is known to be present
            state = entry_state(chunk, i, mcap=mcap)
            if state["status"] != "ok":
                row["reason"] = state["reason"]
            else:
                row["rel_spread"] = state["rel_spread"]  # recorded for every ok entry; the spread is not a universe term (Amendment 5)
                if not in_universe(mcap):
                    row.update(status="not_taken", reason="universe_filter")
                elif bad_quote(state["priced"], float(chunk["spot_entry"][i])):
                    row.update(status="not_taken", reason="bad_quote")
                else:
                    row["status"], row["reason"] = gate(ev)
        rows.append(row)
    return pd.DataFrame(rows, columns=DECISION_COLS)


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
        priced, c, p, reason = resolve(chunk, pos[eid], alpha)
        if reason is not None:
            raise ValueError(f"taken event {eid!r} has no ATM straddle")
        cost = priced.cost
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
