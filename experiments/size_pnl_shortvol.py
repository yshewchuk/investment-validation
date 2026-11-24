"""EXP-188 S5: short-vol structure scoring on the S3 ladder chunks.

S5 of guides/size_decision_pnl_design.md. The seven forecast-sized put-only families (``NAMES``) are
resolved per event from the ladder chunk, priced at entry through ``engine.fills.FillModel`` and marked
at exit by the rung rules. Pure functions over numpy/pandas: no I/O, no ledger. The S6/S9 drivers WILL
call ``simulate``, ``score_events``, ``decide`` and ``score`` per quarter (future); no production caller
yet. Progress logging: none needed (pure per-quarter functions).
"""
import dataclasses
import math

import numpy as np
import pandas as pd

from engine.entry_rules import MAX_REL_SPREAD, MCAP_FLOOR
from engine.fills import FillModel
from engine.forecast_sizing import FORECAST_SIZED, forecast_params
from engine.portfolio import CONTRACT_MULTIPLIER
from engine.pnl_sim import DRAWS, trailing_cutoff
from engine.score import _mean_relative_spread
from engine.structures import (BUY, PUT, SELL, STRUCTURES, ChainSnapshot, LadderTooCoarse, StructureError,
                               price_structure)
from experiments.size_pnl_account import secured_per_contract
from experiments.size_pnl_ladder import SELECTOR
from experiments.size_pnl_pool import put_means, structure_pnl
from experiments.size_pnl_straddle import EXIT_UNQUOTED, lots

NAMES = tuple(sorted(FORECAST_SIZED))
SCORE_COLS = ["event_id", "event_date", "structure", "width", "entry_cost", "rel_spread", "mcap_usd",
              "exp_pnl_sim", "status", "reason", "alpha"]
_OUT_COLS = ["event_id", "structure", "entry_cost", "exit_value", "rung", "lots", "pnl_per_share",
             "pnl_dollars"]
_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
_CLOSE = {BUY: SELL, SELL: BUY}
# A net premium within 1e-12 of the gross premium traded is float cancellation residue of an exact zero (a double carries
# ~1e-16 relative), not a debit; real cents-level quotes give a net of at least a cent against a gross of at most hundreds
# of dollars (ratio >= 1e-5), so no real debit is lost. Not a market rule.
ZERO_RESIDUE = 1e-12


class Refusal(ValueError):
    """A refusal with a stable ``code`` and ``details["reason"]``, kept local so this module adds no engine.v2 edge."""
    def __init__(self, code, message, details):
        super().__init__(message)
        self.code, self.details = code, details


def _num(x):
    """``float(x)``, or NaN when ``x`` is None, pd.NA or another null."""
    if x is None or pd.isna(x):
        return float("nan")
    return float(x)


def pinned(name):
    """The factory default of ``name``, refused unless it is the pinned put-only short-vol shape."""
    spec = STRUCTURES[name]()
    checks = {"entry_offset": spec.entry_offset == 0, "exit_offset": spec.exit_offset == 1,
              "decision_offset": spec.decision_offset is None,
              "anchor_offset": spec.params.get("anchor_offset", 0) == 0}
    for field, ok in checks.items():
        if not ok:
            raise Refusal(_CONFLICT, f"{name}: factory default {field} is not pinned",
                          {"reason": "factory_default_mismatch", "structure": name, "field": field})
    for leg in spec.legs:
        if leg.right != PUT:
            raise Refusal(_CONFLICT, f"{name}: leg {leg.name} is a call",
                          {"reason": "call_leg_refused", "structure": name, "field": leg.name})
        # engine.structures and engine.v2.research._pricing each define an ExpirySelector, so compare the fields, not the objects.
        if dataclasses.asdict(leg.expiry) != dataclasses.asdict(SELECTOR):
            raise Refusal(_CONFLICT, f"{name}: leg {leg.name} expiry selector differs from the ladder's",
                          {"reason": "factory_default_mismatch", "structure": name, "field": leg.name})
    return spec


def _width_params(name, pred):
    """Sizing parameters for ``name`` at production's rule (k=1), or None when the forecast declines."""
    return forecast_params(name, _num(pred))


def _put_slots(chunk, i):
    """Flat indices of event ``i``'s put rows."""
    lo, hi = int(chunk["offsets"][i]), int(chunk["offsets"][i + 1])
    return np.arange(lo, hi)[chunk["right"][lo:hi] == 0]


def _sign(side):
    """+1 for a buy, -1 for a sell."""
    return 1.0 if side == BUY else -1.0


def chain_snapshot(chunk, i):
    """Event ``i``'s put chain at entry quotes, as ``price_structure`` reads it.

    The ladder already chose the single expiry (first_post_event), so the snapshot carries only it, and
    event_date = expiry makes the selector's inclusive rule return it.
    """
    puts = _put_slots(chunk, i)
    expiry = pd.Timestamp(chunk["expiry"][i])
    rows = pd.DataFrame({"expiry": [expiry] * puts.size, "dte": [chunk["dte_entry"][i]] * puts.size,
                         "strike": chunk["strike"][puts], "right": [PUT] * puts.size,
                         "bid": chunk["bid_e"][puts], "ask": chunk["ask_e"][puts]})
    return ChainSnapshot(ticker="", obs_date=expiry, event_date=expiry, rows=rows,
                         spot=float(chunk["spot_entry"][i]))


def resolve_priced(chunk, i, name, pred, fill):
    """(priced, legs, reason) for event ``i``: production's ``StructurePrice`` and every leg as a dict.

    Both resolution and pricing run in ``engine.structures.price_structure``; the dicts carry the listed strikes.
    On a refusal the result is (None, None, reason).
    """
    params = _width_params(name, pred)
    pinned(name)
    if params is None:
        return None, None, "width_declined"
    spec = STRUCTURES[name](**params)
    try:
        priced = price_structure(spec, chain_snapshot(chunk, i), fill)
    except LadderTooCoarse:  # before StructureError: it is a subclass
        return None, None, "ladder_too_coarse"
    except StructureError:
        return None, None, "structure_unresolved"
    except ValueError:  # FillModel refuses a crossed, negative or NaN quote
        return None, None, "entry_quote_unusable"
    puts = _put_slots(chunk, i)
    legs = []
    for leg in priced.legs:
        hit = puts[np.isclose(chunk["strike"][puts], leg.strike)]
        if hit.size == 0:
            return None, None, "structure_unresolved"
        # The LISTED strike, not the arithmetic one: a mirrored leg can be 93.99999999999999 for a listed 94.0,
        # and structure_value needs exact grid equality.
        legs.append({"name": leg.name, "strike": float(chunk["strike"][hit[0]]), "side": leg.side,
                     "qty": float(leg.qty), "idx": int(hit[0])})
    return priced, legs, None


def resolve(chunk, i, name, pred):
    """(legs, reason) for event ``i``: every leg as a dict (traded and reference), or (None, reason).

    Resolution does not depend on alpha, so this resolves at alpha 0.5 and keeps only the legs.
    """
    _priced, legs, reason = resolve_priced(chunk, i, name, pred, FillModel(0.5))
    return legs, reason


def simulate(chunk, events, pool, *, draws=DRAWS):
    """{event_id: (grid, means) or None}: one simulation per event, shared by all seven structures."""
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    out = {}
    for ev in events.to_dict("records"):
        eid = str(ev["event_id"])
        if eid not in pos:
            continue
        i = pos[eid]
        strikes = chunk["strike"][_put_slots(chunk, i)]
        grid = np.unique(strikes[np.isfinite(strikes)])
        if grid.size == 0 or np.any(np.diff(grid) <= 0):
            out[eid] = None
            continue
        means = put_means(pool, key=eid, event_date=ev["event_date"], spot=float(chunk["spot_entry"][i]),
                          pre_iv30=_num(ev["pre_iv30"]), pred_move=_num(ev["pred_move"]),
                          pred_crush=_num(ev["pred_crush"]), dte_exit=_num(ev["dte_exit"]),
                          strikes=grid, draws=draws)
        out[eid] = None if means is None else (grid, means)
    return out


def entry(priced):
    """(entry cost per share, mean relative spread), or None when the entry cost is not finite.

    The cost is production's ``StructurePrice.cost``. A cost within ``ZERO_RESIDUE`` of the gross premium traded
    (sum of ``|cash_flow|`` over the legs) is float cancellation residue of an exact zero, so it is returned as 0.0.
    The spread is production's ``_mean_relative_spread`` over all resolved legs, reference legs included, as the
    board computes it.
    """
    cost = priced.cost
    if not math.isfinite(cost):
        return None
    gross = sum(abs(leg.cash_flow) for leg in priced.legs)
    if abs(cost) <= ZERO_RESIDUE * gross:
        cost = 0.0
    rel = _mean_relative_spread(priced)
    return cost, (math.nan if rel is None else rel)


def _row(ev, name, **kw):
    """A score row for ``ev`` with NaN measures and an undetermined status, overridden by ``kw``."""
    row = {"event_id": str(ev["event_id"]), "event_date": ev["event_date"], "structure": name,
           "width": math.nan, "entry_cost": math.nan, "rel_spread": math.nan,
           "mcap_usd": _num(ev["mcap_usd"]), "exp_pnl_sim": math.nan, "status": "undetermined", "reason": None,
           "alpha": math.nan}
    row.update(kw)
    return row


def score_events(chunk, events, sims, name, *, alpha=0.5):
    """One row per event: ``exp_pnl_sim`` of ``name`` at its entry cost, or the reason it is undetermined."""
    pinned(name)
    fill = FillModel(alpha)
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    rows = []
    for ev in events.to_dict("records"):
        eid = str(ev["event_id"])
        params = _width_params(name, ev["pred_move"])
        row = _row(ev, name, width=params["width_moneyness"] if params else math.nan, alpha=alpha)
        sim = sims.get(eid)
        if eid not in pos or sim is None:
            row["reason"] = "unsimulable"
        else:
            priced, legs, reason = resolve_priced(chunk, pos[eid], name, ev["pred_move"], fill)
            entered = None if legs is None else entry(priced)
            if legs is None:
                row["reason"] = reason
            elif entered is None:
                row["reason"] = "entry_quote_unusable"
            else:
                cost, rel = entered
                row.update(entry_cost=cost, rel_spread=rel)
                if not cost > 0:
                    row["reason"] = "entry_not_debit"
                else:
                    exits = [{"strike": leg["strike"], "side": _CLOSE[leg["side"]], "qty": leg["qty"]}
                             for leg in legs if leg["qty"] > 0]
                    row.update(exp_pnl_sim=structure_pnl(sim[1], sim[0], exits, cost),
                               status="scored", reason=None)
        rows.append(row)
    return pd.DataFrame(rows, columns=SCORE_COLS)


def decide(scored):
    """Taken / not_taken / undetermined per row, following ``EntryRule.evaluate`` (undetermined wins over not_taken),
    with the trailing cutoff of ``_short_vol_rule``.

    History is every scored row of the same structure with a finite ``exp_pnl_sim``, whatever its universe
    filter, so a filtered-out event still counts toward the bar. Rows already undetermined pass through.
    """
    out = scored.copy()
    value = pd.to_numeric(out["exp_pnl_sim"], errors="coerce").to_numpy(dtype=float)
    usable = (out["status"] == "scored").to_numpy() & np.isfinite(value)
    history = {name: grp[["event_date", "_v"]].rename(columns={"_v": "exp_pnl_sim"})
               for name, grp in out[usable].assign(_v=value[usable]).groupby("structure")}
    cache, cut = {}, np.full(len(out), np.nan)
    status = out["status"].to_numpy(dtype=object).copy()
    reason = out["reason"].to_numpy(dtype=object).copy()
    for k, row in enumerate(out.to_dict("records")):
        if row["status"] != "scored":
            continue
        name, month = row["structure"], pd.Timestamp(row["event_date"]).to_period("M")
        if (name, month) not in cache:  # the cutoff depends only on the month
            cache[(name, month)] = trailing_cutoff(history.get(name), row["event_date"])
        bar = cache[(name, month)]
        if bar is not None:
            cut[k] = bar
        mcap, rel = _num(row["mcap_usd"]), _num(row["rel_spread"])
        pnl_ok = None if bar is None else bool(value[k] >= bar)
        spread_ok = None if math.isnan(rel) else rel <= MAX_REL_SPREAD
        mcap_ok = None if math.isnan(mcap) else mcap >= MCAP_FLOOR
        if pnl_ok is None:
            status[k], reason[k] = "undetermined", "thin_history"
        elif spread_ok is None:
            status[k], reason[k] = "undetermined", "missing_spread"
        elif mcap_ok is None:
            status[k], reason[k] = "undetermined", "missing_mcap"
        elif not pnl_ok:
            status[k], reason[k] = "not_taken", "below_cutoff"
        elif not (spread_ok and mcap_ok):
            status[k], reason[k] = "not_taken", "universe_filter"
        else:
            status[k], reason[k] = "taken", None
    out["cutoff"], out["status"], out["reason"] = cut, status, reason
    return out


def _exit_value(chunk, traded, alpha, mid=False):
    """Value of the traded legs at exit. A BUY leg closes by selling (``FillModel.sell``, at the bid for alpha 0);
    a SELL leg closes by buying (``FillModel.buy``, at the ask for alpha 0). Rung-2 legs use alpha 0, or ``alpha``
    when ``mid`` (the Amendment 1 sensitivity arm)."""
    total = 0.0
    for leg in traded:
        k = leg["idx"]
        model = FillModel(alpha if (mid or int(chunk["exit_rung"][k]) == 1) else 0.0)
        bid, ask = float(chunk["bid_x"][k]), float(chunk["ask_x"][k])
        price = model.sell(bid, ask) if leg["side"] == BUY else model.buy(bid, ask)
        total += _sign(leg["side"]) * leg["qty"] * float(price)
    return total


def score(chunk, events, decisions, name, alpha, *, drop_unquoted=False, mid_marks=False, repriced=False):
    """Marked P&L of the TAKEN events of ``name`` only; ``drop_unquoted`` removes the worst-case rung-3 rows.

    ``mid_marks`` values rung-2 legs at ``alpha``, as rung-1 legs are (the Amendment 1 sensitivity arm); the taken set,
    rung-3 rows and rung labels are unchanged. A decision priced at a different alpha than this call is refused,
    not silently re-marked. ``repriced`` (the breakeven endpoints only) enters the trade at ``alpha`` itself, so the
    priced-at-this-alpha and the not-a-debit check are skipped; everything else is unchanged.
    """
    FillModel(alpha)  # validate up front, even when nothing is taken
    pinned(name)
    if "structure" not in decisions.columns:
        raise ValueError("decisions must carry a structure column")
    if "entry_cost" not in decisions.columns:
        raise ValueError("decisions must carry an entry_cost column")
    if "alpha" not in decisions.columns:
        raise ValueError("decisions must carry an alpha column")
    pred = dict(zip(events["event_id"].astype(str), events["pred_move"]))
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    mine = decisions[(decisions["status"] == "taken") & (decisions["structure"] == name)]
    rows = []
    for dec in mine.to_dict("records"):
        eid = str(dec["event_id"])
        if eid not in pred or eid not in pos:
            raise ValueError(f"taken event {eid!r} is missing from the chunk or events")
        priced, legs, reason = resolve_priced(chunk, pos[eid], name, pred[eid], FillModel(alpha))
        if reason == "entry_quote_unusable":
            raise ValueError(f"taken event {eid!r} has an unusable entry quote")
        if legs is None:
            raise ValueError(f"taken event {eid!r} has no resolvable {name} structure ({reason})")
        entered = entry(priced)
        if entered is None:
            raise ValueError(f"taken event {eid!r} has an unusable entry quote")
        cost = entered[0]
        bound = _num(dec["entry_cost"])
        if not repriced and not (math.isfinite(bound) and abs(bound - cost) <= 1e-9 and _num(dec["alpha"]) == float(alpha)):
            raise ValueError("decision was not priced at this alpha")
        if not repriced and not cost > 0:
            raise ValueError(f"taken event {eid!r} has an entry cost that is not a debit ({cost!r})")
        traded = [leg for leg in legs if leg["qty"] > 0]
        rungs = [int(chunk["exit_rung"][leg["idx"]]) for leg in traded]
        if 3 in rungs:  # defined-risk floor: the whole structure is marked at zero
            value, rung = 0.0, EXIT_UNQUOTED
        else:
            value = _exit_value(chunk, traded, alpha, mid=mid_marks)
            rung = "rung1" if all(r == 1 for r in rungs) else "rung2"
        if drop_unquoted and rung == EXIT_UNQUOTED:
            continue
        n, per_share = lots(cost), value - cost
        rows.append({"event_id": eid, "structure": name, "entry_cost": cost, "exit_value": value,
                     "rung": rung, "lots": n, "pnl_per_share": per_share,
                     "pnl_dollars": per_share * CONTRACT_MULTIPLIER * n})
    return pd.DataFrame(rows, columns=_OUT_COLS)


def secured(chunk, events, decisions, name):
    """[event_id, secured_per_contract] of the TAKEN rows of ``name``: the cash one contract ties up (Amendment 2).
    Legs are resolved exactly as ``score`` resolves them, so the secured cash is the one the traded legs imply."""
    pinned(name)
    pred = dict(zip(events["event_id"].astype(str), events["pred_move"]))
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    mine = decisions[(decisions["status"] == "taken") & (decisions["structure"] == name)]
    rows = []
    for dec in mine.to_dict("records"):
        eid = str(dec["event_id"])
        if eid not in pred or eid not in pos:
            raise ValueError(f"taken event {eid!r} is missing from the chunk or events")
        legs, reason = resolve(chunk, pos[eid], name, pred[eid])
        if legs is None:
            raise ValueError(f"taken event {eid!r} has no resolvable {name} structure ({reason})")
        rows.append({"event_id": eid, "secured_per_contract": secured_per_contract(legs)})
    return pd.DataFrame(rows, columns=["event_id", "secured_per_contract"]).astype({"secured_per_contract": float})


def rung_shares(scored):
    """Share of rows per rung label; {} when nothing was scored."""
    if scored.empty:
        return {}
    return scored["rung"].value_counts(normalize=True).to_dict()
