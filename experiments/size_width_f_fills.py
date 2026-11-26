"""EXP-189 Amendment 1, slice W7a3: fill model F behind the ``size_width_fills.Fills`` seam (design A1.2, A1.4).

Public interface (W7b1 and W7d build on this; nothing else is public):

- ``FFills(quotes)``: ``quotes`` is a ``size_width_f_quotes.FQuotes``, or any object whose
  ``legs(chunk, i, resolved, *, exit=False)`` returns one F leg-table row per resolved leg, in order. Plug it in with
  ``size_width_pricing.load(fills=FFills(quotes))``; nothing selects F by itself.
- ``entry(trade, priced)`` -> ``(debit_F, rel_spread)``, ``None`` or ``Skip(reason)``. ``rel_spread`` is today's mean
  relative spread of the stored quotes (``size_pnl_shortvol.entry``), and ``None`` is that function's own refusal
  (the earlier ``entry_quote_unusable``). ``debit_F`` does not depend on the alpha ``priced`` was priced at. The skip
  reasons are ``ENTRY_REASONS``: a leg that is not ``ok`` gives the first of ``f_unpriceable``, ``f_rate_unknown``,
  ``f_no_fair_value`` present on any leg; otherwise F's guard gives ``f_not_a_debit``, ``f_fill_over_fair`` or
  ``f_spread_over_debit``, first condition that holds.
- ``exit_value(trade, traded, alpha, mid=False)`` -> ``Exit``. ``alpha`` and ``mid`` are ignored. The value is
  ``proceeds_F``, kept when negative, with flags from ``EXIT_FLAGS``: ``exit_fair_mid`` (a leg with a usable quote but no
  fair value or rate took V = mid) and ``exit_negative``. An F-``unpriceable`` leg gives ``Exit(0.0, unquoted=True)``
  with no flags: the seam labels that row's rung ``exit_unquoted_worst_case``, as it does a ladder rung 3.
- ``entry_result(trade)`` / ``exit_result(trade)`` -> F's own ``fair_value_fills.EntryResult`` / ``ExitResult`` (the
  three debits, spread sum, sensitivities, per-leg flags) for the same legs.

Only legs of positive quantity go to F (F design section 6); a reference leg of quantity 0 is never read. Entry reads
entry-date rows only (R6). Results are a function of the event and its resolved legs, so they are kept per chunk and
reused across widths, arms and draws; a new chunk object drops them. Input refusals raised by ``quotes``
(``f_quote_mismatch`` and F's R1) propagate unchanged.
"""
from experiments import fair_value_fills as fvf
from experiments import size_pnl_shortvol as sv
from experiments.size_width_fills import Exit, Skip

_STATUS_ORDER = ("unpriceable", "rate_unknown", "no_fair_value")
ENTRY_REASONS = tuple(f"f_{name}" for name in _STATUS_ORDER + ("not_a_debit", "fill_over_fair", "spread_over_debit"))
EXIT_FLAGS = ("exit_fair_mid", "exit_negative")


class FFills:
    exit_flags = EXIT_FLAGS

    def __init__(self, quotes):
        self._quotes, self._chunk, self._memo = quotes, None, {}

    def _result(self, trade, legs, exit):
        """F's structure result for the positive-quantity ``legs`` of ``trade``, computed once per chunk."""
        if trade.chunk is not self._chunk:
            self._chunk, self._memo = trade.chunk, {}
        legs = [leg for leg in legs if leg["qty"] > 0]
        key = (trade.i, exit, tuple((leg["idx"], leg["side"], leg["qty"]) for leg in legs))
        if key not in self._memo:
            table = self._quotes.legs(trade.chunk, trade.i, legs, exit=exit)
            table = table.assign(side=[leg["side"] for leg in legs], qty=[leg["qty"] for leg in legs])
            self._memo[key] = fvf.structure_exit(table) if exit else fvf.structure_entry(table)
        return self._memo[key]

    def entry_result(self, trade):
        return self._result(trade, trade.legs, False)

    def exit_result(self, trade):
        return self._result(trade, trade.legs, True)

    def entry(self, trade, priced):
        current = sv.entry(priced)
        if current is None:
            return None
        got = self.entry_result(trade)
        if got.outcome == "undetermined":
            statuses = {flag["status"] for flag in got.leg_flags}
            return Skip("f_" + next(name for name in _STATUS_ORDER if name in statuses))
        if got.outcome == "skip":
            return Skip("f_" + got.skip_reason)
        return got.debit_F, current[1]

    def exit_value(self, trade, traded, alpha, mid=False):
        got = self._result(trade, traded, True)
        if got.outcome == "unpriceable":
            return Exit(0.0, unquoted=True)
        flags = {name for name in EXIT_FLAGS if getattr(got, name)}
        return Exit(got.proceeds_F, frozenset(flags))
