"""EXP-188 S5: short-vol structure scoring on the S3 ladder chunks.

S5 of guides/size_decision_pnl_design.md. The seven forecast-sized put-only families (``NAMES``) are
resolved per event from the ladder chunk, priced at entry through ``engine.fills.FillModel`` and marked
at exit by the rung rules. Pure functions over numpy/pandas: no I/O, no ledger. The S6/S9 drivers WILL
call ``simulate``, ``score_events``, ``decide`` and ``score`` per quarter (future); no production caller
yet. Progress logging: none needed (pure per-quarter functions).
"""
import itertools
import math

import numpy as np
import pandas as pd

from engine.entry_rules import MAX_REL_SPREAD, MCAP_FLOOR
from engine.fills import FillModel
from engine.forecast_sizing import FORECAST_SIZED, forecast_params
from engine.portfolio import CONTRACT_MULTIPLIER
from engine.pnl_sim import DRAWS, trailing_cutoff
from engine.structures import BUY, PUT, SELL, STRUCTURES, StructureError
from engine.v2.ops.errors import fail
from experiments.size_pnl_pool import MIN_WINDOW, put_means, structure_pnl
from experiments.size_pnl_straddle import EXIT_UNQUOTED, lots

NAMES = tuple(sorted(FORECAST_SIZED))
SCORE_COLS = ["event_id", "event_date", "structure", "width", "entry_cost", "rel_spread", "mcap_usd",
              "exp_pnl_sim", "status", "reason"]
_OUT_COLS = ["event_id", "structure", "entry_cost", "exit_value", "rung", "lots", "pnl_per_share",
             "pnl_dollars"]
_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
_CLOSE = {BUY: SELL, SELL: BUY}
# A numerical guard against float-noise debits, not a market rule: the design and _short_vol_rule
# set no minimum debit.
MIN_DEBIT = 1e-6


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
            raise fail(_CONFLICT, f"{name}: factory default {field} is not pinned",
                       details={"reason": "factory_default_mismatch", "structure": name, "field": field})
    for leg in spec.legs:
        if leg.right != PUT:
            raise fail(_CONFLICT, f"{name}: leg {leg.name} is a call",
                       details={"reason": "call_leg_refused", "structure": name, "field": leg.name})
    return spec


def _check_c(c):
    """``c`` as a float; it divides the forecast, so it must be finite and positive."""
    value = float(c)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"c must be finite and > 0, got {c!r}")
    return value


def _width_params(name, pred, c):
    """Sizing parameters for ``name`` at ``pred / c``, or None when the forecast declines."""
    return forecast_params(name, _num(pred) / _check_c(c))


def _put_slots(chunk, i):
    """Flat indices of event ``i``'s put rows."""
    lo, hi = int(chunk["offsets"][i]), int(chunk["offsets"][i + 1])
    return np.arange(lo, hi)[chunk["right"][lo:hi] == 0]


def _sign(side):
    """+1 for a buy, -1 for a sell."""
    return 1.0 if side == BUY else -1.0


def _floor_ok(legs):
    """EXP-133's zero floor in dollars: the traded payoff is >= 0 at every traded strike."""
    traded = [leg for leg in legs if leg["qty"] > 0]
    for at in [leg["strike"] for leg in traded]:
        payoff = sum(_sign(leg["side"]) * leg["qty"] * max(leg["strike"] - at, 0.0) for leg in traded)
        if payoff < -1e-9:
            return False
    return True


def resolve(chunk, i, name, pred, c=1.0):
    """(legs, reason) for event ``i``: every leg as a dict (traded and reference), or (None, reason)."""
    params = _width_params(name, pred, c)
    pinned(name)
    if params is None:
        return None, "width_declined"
    spec = STRUCTURES[name](**params)
    puts = _put_slots(chunk, i)
    rows = pd.DataFrame({"strike": chunk["strike"][puts], "right": ["P"] * puts.size})
    resolved = {}
    try:
        for leg in spec.legs:
            resolved[leg.name] = leg.strike.select(rows, float(chunk["spot_entry"][i]), resolved, right="P")
    except StructureError:
        return None, "structure_unresolved"
    if any(np.isclose(a, b) for a, b in itertools.combinations(resolved.values(), 2)):
        return None, "ladder_too_coarse"
    legs = []
    for leg in spec.legs:
        hit = puts[np.isclose(chunk["strike"][puts], resolved[leg.name])]
        if hit.size == 0:
            return None, "structure_unresolved"
        legs.append({"name": leg.name, "strike": float(resolved[leg.name]), "side": leg.side,
                     "qty": float(leg.qty), "idx": int(hit[0])})
    if not _floor_ok(legs):
        return None, "negative_floor"
    return legs, None


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


def entry_legs(chunk, legs, fill):
    """(entry cost per share, mean relative spread), or None when a traded leg has no finite entry quote.

    Zero-bid and locked entry quotes are priced normally, as ``price_structure`` does. This is
    deliberately NOT the straddle's stricter entry rule. The spread averages every resolved leg
    with a finite quote and a positive mid, reference legs included, exactly as the board does: `engine.score._mean_relative_spread` iterates `priced.legs`
    (engine/score.py:2476, 5333), which holds the zero-quantity reference legs too. EXP-133's `build.py` averages
    only used legs; the design (section 5) pins the board's rule.
    """
    cost, spreads = 0.0, []
    for leg in legs:
        k = leg["idx"]
        bid, ask = float(chunk["bid_e"][k]), float(chunk["ask_e"][k])
        finite = math.isfinite(bid) and math.isfinite(ask)
        if finite and bid + ask > 0:
            spreads.append((ask - bid) / ((bid + ask) / 2.0))
        if leg["qty"] > 0:
            if not finite:
                return None
            price = fill.buy(bid, ask) if leg["side"] == BUY else fill.sell(bid, ask)
            cost += _sign(leg["side"]) * leg["qty"] * float(price)
    return cost, (float(np.mean(spreads)) if spreads else math.nan)


def _row(ev, name, **kw):
    """A score row for ``ev`` with NaN measures and an undetermined status, overridden by ``kw``."""
    row = {"event_id": str(ev["event_id"]), "event_date": ev["event_date"], "structure": name,
           "width": math.nan, "entry_cost": math.nan, "rel_spread": math.nan,
           "mcap_usd": _num(ev["mcap_usd"]), "exp_pnl_sim": math.nan, "status": "undetermined", "reason": None}
    row.update(kw)
    return row


def score_events(chunk, events, sims, name, *, alpha=0.5, c=1.0):
    """One row per event: ``exp_pnl_sim`` of ``name`` at its entry cost, or the reason it is undetermined."""
    pinned(name)
    fill = FillModel(alpha)
    _check_c(c)
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    rows = []
    for ev in events.to_dict("records"):
        eid = str(ev["event_id"])
        params = _width_params(name, ev["pred_move"], c)
        row = _row(ev, name, width=params["width_moneyness"] if params else math.nan)
        sim = sims.get(eid)
        if eid not in pos or sim is None:
            row["reason"] = "unsimulable"
        else:
            legs, reason = resolve(chunk, pos[eid], name, ev["pred_move"], c)
            entered = None if legs is None else entry_legs(chunk, legs, fill)
            if legs is None:
                row["reason"] = reason
            elif entered is None:
                row["reason"] = "entry_quote_unusable"
            else:
                cost, rel = entered
                row.update(entry_cost=cost, rel_spread=rel)
                if not (math.isfinite(cost) and cost >= MIN_DEBIT):
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
            cache[(name, month)] = trailing_cutoff(history.get(name), row["event_date"], min_window=MIN_WINDOW)
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


def _exit_value(chunk, traded, alpha):
    """Value of the traded legs at exit. A BUY leg closes by selling (``FillModel.sell``, at the bid for alpha 0);
    a SELL leg closes by buying (``FillModel.buy``, at the ask for alpha 0). Rung-2 legs use alpha 0."""
    total = 0.0
    for leg in traded:
        k = leg["idx"]
        model = FillModel(alpha if int(chunk["exit_rung"][k]) == 1 else 0.0)
        bid, ask = float(chunk["bid_x"][k]), float(chunk["ask_x"][k])
        price = model.sell(bid, ask) if leg["side"] == BUY else model.buy(bid, ask)
        total += _sign(leg["side"]) * leg["qty"] * float(price)
    return total


def score(chunk, events, decisions, name, alpha, *, c=1.0, drop_unquoted=False):
    """Marked P&L of the TAKEN events of ``name`` only; ``drop_unquoted`` removes the worst-case rung-3 rows.

    A decision priced at a different alpha or c than this call is refused, not silently re-marked.
    """
    FillModel(alpha)  # validate up front, even when nothing is taken
    pinned(name)
    _check_c(c)
    if "structure" not in decisions.columns:
        raise ValueError("decisions must carry a structure column")
    if "entry_cost" not in decisions.columns:
        raise ValueError("decisions must carry an entry_cost column")
    pred = dict(zip(events["event_id"].astype(str), events["pred_move"]))
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    mine = decisions[(decisions["status"] == "taken") & (decisions["structure"] == name)]
    rows = []
    for dec in mine.to_dict("records"):
        eid = str(dec["event_id"])
        if eid not in pred or eid not in pos:
            raise ValueError(f"taken event {eid!r} is missing from the chunk or events")
        legs, reason = resolve(chunk, pos[eid], name, pred[eid], c)
        if legs is None:
            raise ValueError(f"taken event {eid!r} has no resolvable {name} structure ({reason})")
        entered = entry_legs(chunk, legs, FillModel(alpha))
        if entered is None:
            raise ValueError(f"taken event {eid!r} has an unusable entry quote")
        cost = entered[0]
        bound = _num(dec["entry_cost"])
        if not (math.isfinite(bound) and abs(bound - cost) <= 1e-9):
            raise ValueError("decision was not priced at this alpha and c")
        if not (math.isfinite(cost) and cost >= MIN_DEBIT):
            raise ValueError(f"taken event {eid!r} has an entry debit below {MIN_DEBIT} a share ({cost!r})")
        traded = [leg for leg in legs if leg["qty"] > 0]
        rungs = [int(chunk["exit_rung"][leg["idx"]]) for leg in traded]
        if 3 in rungs:  # defined-risk floor: the whole structure is marked at zero
            value, rung = 0.0, EXIT_UNQUOTED
        else:
            value = _exit_value(chunk, traded, alpha)
            rung = "rung1" if all(r == 1 for r in rungs) else "rung2"
        if drop_unquoted and rung == EXIT_UNQUOTED:
            continue
        n, per_share = lots(cost), value - cost
        rows.append({"event_id": eid, "structure": name, "entry_cost": cost, "exit_value": value,
                     "rung": rung, "lots": n, "pnl_per_share": per_share,
                     "pnl_dollars": per_share * CONTRACT_MULTIPLIER * n})
    return pd.DataFrame(rows, columns=_OUT_COLS)


def rung_shares(scored):
    """Share of rows per rung label; {} when nothing was scored."""
    if scored.empty:
        return {}
    return scored["rung"].value_counts(normalize=True).to_dict()
