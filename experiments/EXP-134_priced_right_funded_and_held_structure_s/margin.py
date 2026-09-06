#!/usr/bin/env python3
"""Cash-secured put margin: what the account can actually fund.

The user's brokerage will not grant spread margin on a put structure. Every
short put must be **cash-secured in full** — ``strike x 100`` per contract,
held from entry until the position closes — and no more than half the account
may be committed to securing puts at any moment.

That is not a refinement of position sizing. It is a different binding
constraint, and it is much tighter than anything the program has modelled:

    TWIN-P5 on an $85 stock, spacing $3: shorts are 2x $88 and 2x $82.
    Secured per contract = (2 x 88 + 2 x 82) x 100 = $34,000.
    At a $100,000 account and a 50% cap, that is ONE contract, and while it is
    open the account can fund nothing else.

Three consequences follow, and all three are the point rather than a side
effect:

* **Concurrency collapses.** The unconstrained book runs ~16 positions at once.
  A cash-secured account at this size runs one or two.
* **The universe skews cheap.** Securing four short puts on a $900 stock needs
  $360,000, so a $100k account cannot touch it at any structure width. Ticker
  price becomes an entry filter nobody wrote down.
* **Fixed-fraction sizing stops binding.** ``5% of equity / debit`` is
  irrelevant when margin allows one contract. The sizing rule the program has
  used since Phase 2 is not the rule that decides size here.

Capacity travels with the account: the cap is 50% of CURRENT marked equity, so
an account that doubles can secure twice as much. Returns are still quoted on
the debit — securing cash is a funding requirement, not capital consumed by the
trade — which is the user's registered instruction and is why ``return on
capital`` is unchanged by any of this.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["START_EQUITY", "SECURED_CAP", "CONTRACT_MULTIPLIER", "TARGET_SHARE",
           "secured_per_contract", "MarginBook", "simulate"]

#: Registered in spec.yaml. The account the constraint is evaluated against.
START_EQUITY = 100_000.0

#: Share of CURRENT marked equity that may be committed to securing short puts
#: at any one moment, across every open position.
SECURED_CAP = 0.50

#: Shares per option contract.
CONTRACT_MULTIPLIER = 100.0

#: Share of the CURRENT headroom a single position aims to consume, when
#: ``target_share`` sizing is used instead of fixed-fraction. It is an aim, not
#: a cap: a structure that needs the whole headroom for one contract may take
#: it, because one contract is indivisible and refusing it would trade nothing
#: at all. `None` keeps the fixed-fraction rule the registered run used.
TARGET_SHARE = None


def secured_per_contract(legs_blob) -> float:
    """Cash a single contract of this structure ties up, in dollars.

    Every SHORT put leg, at full strike value. Long puts secure nothing — they
    are paid for in the debit — and there is no spread offset, which is the
    whole reason this function exists rather than a margin formula.
    """
    doc = json.loads(legs_blob) if isinstance(legs_blob, str) else legs_blob
    return float(sum(
        leg["qty"] * float(leg["strike"]) * CONTRACT_MULTIPLIER
        for leg in (doc.get("entry") or []) if leg["side"] == "sell"
    ))


@dataclass
class MarginBook:
    """Open positions and the cash they are securing, in dollars."""

    equity: float = START_EQUITY
    cash: float = START_EQUITY
    secured: float = 0.0
    open_rows: list = field(default_factory=list)

    @property
    def headroom(self) -> float:
        """Secured dollars still available, against 50% of CURRENT equity."""
        return SECURED_CAP * self.equity - self.secured


def simulate(trades: pd.DataFrame, *, fraction: float = 0.05,
             start_equity: float = START_EQUITY,
             cap: float = SECURED_CAP,
             target_share: float | None = TARGET_SHARE,
             expensive_first: bool = False) -> pd.DataFrame:
    """Walk the book in date order under the cash-secured constraint.

    One row per candidate trade, in entry-date order, carrying what the account
    could actually do with it. ``contracts`` is the smaller of what
    fixed-fraction sizing asks for and what margin allows; ``funded`` is False
    when margin allowed nothing, and that trade **does not happen** — it is not
    re-selected to a cheaper structure, per the registered rule.

    Marked equity is cash plus open positions valued at cost, the same
    convention ``engine.evaluate.build_equity`` uses, so the two curves are
    read the same way. Secured cash is released when a position closes.

    ``target_share`` replaces fixed-fraction sizing with "aim to consume this
    share of the CURRENT headroom", so a position deliberately leaves room for
    the next one instead of taking everything it can afford. It is an aim and
    not a cap: contracts are indivisible, so a structure needing the whole
    headroom for one contract still takes one rather than trading nothing.

    ``expensive_first`` orders trades competing on the SAME entry date by what
    they consume, largest first. Earnings arrive in bursts; within a burst the
    expensive names are the ones that can only ever be funded from an empty
    account, so filling them first and letting cheap names use the remainder
    strictly dominates filling in date-tie order. Across dates the order is
    still chronological — this cannot see the future.
    """
    order = ["entry_date", "exit_date"]
    ascending = [True, True]
    if expensive_first:
        order = ["entry_date", "secured_per_contract"]
        ascending = [True, False]
    t = trades.sort_values(order, ascending=ascending, kind="stable").reset_index(drop=True)
    entry = pd.to_datetime(t["entry_date"]).to_numpy()
    exit_ = pd.to_datetime(t["exit_date"]).to_numpy()
    cost = t["entry_cost"].to_numpy(dtype=float) * CONTRACT_MULTIPLIER
    value = t["exit_value"].to_numpy(dtype=float) * CONTRACT_MULTIPLIER
    secure = t["secured_per_contract"].to_numpy(dtype=float)

    cash = start_equity
    open_pos: list[tuple] = []          # (exit_date, contracts, cost, value, secured)
    out = np.zeros((len(t), 6))

    for i in range(len(t)):
        # Close everything that matured before this entry, releasing its
        # secured cash and crediting its proceeds.
        still = []
        for pos in open_pos:
            if pos[0] <= entry[i]:
                cash += pos[1] * pos[3]
            else:
                still.append(pos)
        open_pos = still
        deployed = sum(p[1] * p[2] for p in open_pos)
        secured_now = sum(p[4] for p in open_pos)
        equity = cash + deployed
        headroom = cap * equity - secured_now

        if target_share is None:
            want = (fraction * equity / cost[i]) if cost[i] > 0 else 0.0
        else:
            # Aim at a share of the headroom rather than at a share of equity.
            # Below one contract the aim is overridden — see the docstring.
            want = (target_share * headroom / secure[i]) if secure[i] > 0 else np.inf
            want = max(want, 1.0)
        allowed = (headroom / secure[i]) if secure[i] > 0 else want
        contracts = max(0.0, min(want, allowed))
        # A structure needing more than the whole headroom for ONE contract is
        # not funded, and the registered rule is that the trade then simply does
        # not happen — the event is not re-selected to something affordable.
        funded = contracts >= 1.0 if secure[i] > 0 else contracts > 0
        if not funded:
            contracts = 0.0
        else:
            contracts = np.floor(contracts)          # whole contracts only
        if contracts > 0:
            cash -= contracts * cost[i]
            open_pos.append((exit_[i], contracts, cost[i], value[i],
                             contracts * secure[i]))
        out[i] = [contracts, equity, secured_now, headroom, float(funded),
                  len(open_pos)]

    for pos in open_pos:
        cash += pos[1] * pos[3]

    t["contracts"] = out[:, 0]
    t["equity_at_entry"] = out[:, 1]
    t["secured_before"] = out[:, 2]
    t["headroom_at_entry"] = out[:, 3]
    t["funded"] = out[:, 4].astype(bool)
    t["concurrency"] = out[:, 5].astype(int)
    t["pnl_usd"] = t["contracts"] * (value - cost)
    t.attrs["final_equity"] = cash
    return t
