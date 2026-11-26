"""EXP-189 Amendment 1, slice W7b1: F-primary taken rows with mid, harsh and sensitivity columns (design A1.2, A1.3, A1.10).

Selection is by injection only: ``size_width_pricing.load(fills=FFills(quotes))`` makes a copy whose ``fills`` is an
``FFills``, and ``is_f`` checks exactly that. With any other pricing (a CurrentFills copy, or ``size_pnl_shortvol``) the
sweep adds no column and calls nothing new.

An F taken row carries the sweep's ``_TAKEN`` columns, then ``F_COLS``:
- ``entry_cost``, ``exit_value``, ``pnl_dollars``: F primary (debit_F, proceeds_F, 0.0 when unquoted, and the P&L).
- ``exit_value_mid``, ``pnl_dollars_mid``: the mid view, a separate CurrentFills copy with mid marks at ALPHA.
- ``rung``, ``exit_fair_mid``, ``exit_negative``: F primary's rung and flags (``exit_fair_mid``: a usable quote with no
  fair value or rate, marked at V = mid; ``exit_negative``: proceeds kept below zero).
- ``entry_cost_mid``: the mid debit; it pairs with ``exit_value_mid`` and ``pnl_dollars_mid``.
- ``exit_value_at_v``: the positive legs closed at V, no concession; NaN on excluded rows. A1.3's mean exit concession is
  (exit_value_at_v - proceeds_F) / debit_F over rows with neither excluded flag.
- ``proceeds_F``: F's exit proceeds, NaN on excluded rows (unlike ``exit_value``, 0.0 there); ``debit_F``: F's entry debit.
- ``excluded_rung3``: the ladder rung is 3 (``exit_unquoted_worst_case`` in the mid view).
- ``excluded_unpriceable``: F primary is unquoted at exit but the ladder did not rung 3.
- ``rung_ladder``: the ladder rung of the mid and harsh views (taken from the mid frame).
- ``entry_cost_harsh``, ``exit_value_harsh``, ``pnl_dollars_harsh``: the harsh view (buy at ask, sell at bid at entry,
  the reverse at exit), a CurrentFills copy at alpha 0.
- ``entry_cost_v``, ``exit_value_v``, ``pnl_dollars_v`` for each v in ``VARIANTS`` (unknown0, unknown1, midfloor): F's
  sensitivity debit and proceeds (kept when negative).

Each view's P&L pairs with that view's own debit: mid with ``entry_cost_mid``, harsh with ``entry_cost_harsh``, each
sensitivity with its ``entry_cost_v``, never with ``entry_cost`` (F). Lots follow the same rule on each view's own debit.
The taken set is F's and is shared by all views, which decide nothing. ``rung`` and the flags belong to F primary;
``rung_ladder`` belongs to mid and harsh. A rung-3 leg or an F-unquoted exit (``exit_unquoted_worst_case``) puts primary
and every sensitivity exit at 0.0; mid and harsh keep the ladder's rung and value. With other fills no column is added.

Accounts, objective, fallback, T8 and T9 read F through size_width_account.deciding_mark / deciding_columns.
The ladder (ret_a*, arm E's choice ladder) is priced by ``ladder_pricing``, never by the F-loaded copy.
"""
import pandas as pd

from engine.portfolio import CONTRACT_MULTIPLIER
from experiments import size_width_pricing as seam
from experiments.fair_value_fills import VARIANTS
from experiments.size_pnl_straddle import EXIT_UNQUOTED, lots
from experiments.size_width_f_fills import FFills
from experiments.size_width_fills import Trade

ALPHA = 0.5  # the sweep's ALPHA; not imported from size_width_sweep, which imports this module
F_COLS = ["rung", "exit_fair_mid", "exit_negative", "entry_cost_mid", "rung_ladder", "entry_cost_harsh",
          "exit_value_harsh", "pnl_dollars_harsh",
          *(f"{p}_{v}" for v in VARIANTS for p in ("entry_cost", "exit_value", "pnl_dollars")),
          "exit_value_at_v", "proceeds_F", "debit_F", "excluded_rung3", "excluded_unpriceable"]


def is_f(pricing):
    return isinstance(getattr(pricing, "fills", None), FFills)


def ladder_pricing(pricing):
    return seam.PRICING if is_f(pricing) else pricing


def _pnl(exit_value, debit):
    return (exit_value - debit) * CONTRACT_MULTIPLIER * lots(debit)


def views(pricing, chunk, ev, mine, name, primary):
    """(mid, extra): ``mid`` is the CurrentFills mid score frame of the taken rows; ``extra`` is ``event_id`` plus F_COLS,
    one row per row of ``primary`` in its order (empty with those columns when ``primary`` is empty)."""
    mid = seam.PRICING.score(chunk, ev, mine, name, ALPHA, mid_marks=True, repriced=True)
    harsh = seam.PRICING.score(chunk, ev, mine, name, 0.0, mid_marks=True, repriced=True).set_index("event_id")
    mids = mid.set_index("event_id")
    pred = dict(zip(ev["event_id"].astype(str), ev["pred_move"]))
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    out = []
    for row in primary.to_dict("records"):
        eid = str(row["event_id"])
        i = pos[eid]
        legs, _ = pricing.resolve(chunk, i, name, pred[eid])
        trade = Trade(chunk, i, eid, name, tuple(legs))
        entry = pricing.fills.entry_result(trade)
        exit_res = None if row["rung"] == EXIT_UNQUOTED else pricing.fills.exit_result(trade)
        m, h = mids.loc[eid], harsh.loc[eid]
        rung3 = bool(m["rung"] == EXIT_UNQUOTED)
        got = {"event_id": eid, "rung": row["rung"], "exit_fair_mid": row["exit_fair_mid"],
               "exit_negative": row["exit_negative"], "entry_cost_mid": m["entry_cost"], "rung_ladder": m["rung"],
               "entry_cost_harsh": h["entry_cost"], "exit_value_harsh": h["exit_value"],
               "pnl_dollars_harsh": h["pnl_dollars"], "debit_F": entry.debit_F,
               "exit_value_at_v": float("nan") if exit_res is None else pricing.fills.exit_value_at_v(trade),
               "proceeds_F": float("nan") if exit_res is None else exit_res.proceeds_F,
               "excluded_rung3": rung3, "excluded_unpriceable": bool(row["rung"] == EXIT_UNQUOTED) and not rung3}
        for v in VARIANTS:
            debit = entry.sensitivity_debits[v]
            exit_v = 0.0 if exit_res is None else exit_res.sensitivity_proceeds[v]
            got.update({f"entry_cost_{v}": debit, f"exit_value_{v}": exit_v, f"pnl_dollars_{v}": _pnl(exit_v, debit)})
        out.append(got)
    return mid, pd.DataFrame(out, columns=["event_id", *F_COLS])
