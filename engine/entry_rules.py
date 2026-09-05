"""Arithmetic entry rules — the gates that are not models.

Every strategy on the board so far is gated by a fitted model with a threshold,
looked up from the registry. Some are not. TWIN-P's rule is arithmetic on the
entry close — reward greater than risk, a spread tight enough to cross sixteen
times, a name large enough to have one — and nothing in it is learned.

**Expressing such a rule as a `Gate` was tried and was wrong.** EXP-123 wrapped
it in `engine.evaluate.Gate` with a no-op ``fit`` so it would travel through the
same walk-forward harness as everything else. The harness skips years with less
than ``min_train_years`` of history, which for a gate is correct and for a rule
that learns nothing is a silent bug: 3,011 of 3,830 headline trades — 78.6% —
were reported as gated and were not filtered at all.

The fix was to stop calling it a gate. An arithmetic rule is a **universe
definition**: it applies to every row, it has no training set, and there is no
year it can be too early for. This module is that idea given a home, so the
next such rule does not have to rediscover it.

Three outcomes, not two. A rule whose facts are incomplete returns
``None`` — undetermined — rather than ``False``. A missing market cap is not
evidence against a trade, and collapsing "we could not tell" into "no" is how a
data gap turns into a silent, permanent decline that looks like a decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

__all__ = ["Term", "Verdict", "EntryRule", "ENTRY_RULES", "TWIN_P_RULE",
           "TWIN_P5_RULE", "rule_for"]


def _number(facts: Mapping, key: str) -> float | None:
    """A finite float from ``facts``, or ``None`` — never a NaN in disguise."""
    value = facts.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


@dataclass(frozen=True)
class Term:
    """One clause of a rule, named so its individual cost stays countable."""

    name: str
    describes: str
    test: Callable[[Mapping], bool | None]
    needs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Verdict:
    passed: bool | None
    terms: dict[str, bool | None] = field(default_factory=dict)
    detail: str = ""
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntryRule:
    """A set of clauses that must all hold, evaluated on entry-close arithmetic."""

    strategy: str
    terms: tuple[Term, ...]
    evidence: str

    def evaluate(self, facts: Mapping) -> Verdict:
        results = {term.name: term.test(facts) for term in self.terms}
        missing = tuple(
            sorted(
                {
                    need
                    for term in self.terms
                    if results[term.name] is None
                    for need in term.needs
                    if _number(facts, need) is None
                }
            )
        )
        if any(value is None for value in results.values()):
            undecided = [n for n, v in results.items() if v is None]
            return Verdict(
                passed=None,
                terms=results,
                detail=(
                    f"{self.strategy} entry rule undetermined: {', '.join(undecided)} "
                    + (f"(missing {', '.join(missing)})" if missing else "")
                ).strip(),
                missing=missing,
            )
        failed = [term.describes for term in self.terms if not results[term.name]]
        if failed:
            return Verdict(
                passed=False,
                terms=results,
                detail=f"{self.strategy} entry rule fails: {'; '.join(failed)}",
            )
        return Verdict(
            passed=True,
            terms=results,
            detail=f"{self.strategy} entry rule passes ({self.evidence})",
        )


# --------------------------------------------------------------------------
# TWIN-P
# --------------------------------------------------------------------------

#: Mean relative spread across the entry legs. Eight legs is sixteen crossings
#: round trip against a deliberately small debit, so a market this structure
#: cannot cross is disqualifying rather than merely expensive.
MAX_REL_SPREAD = 0.25

#: Market-cap floor. The structure needs a ladder dense enough to carry seven
#: listed strikes AND a market tight enough to cross them.
MCAP_FLOOR = 10e9


def _reward_beats_risk(facts: Mapping) -> bool | None:
    """``cost < peak / 2`` — max profit ``peak − c`` exceeds max loss ``c``.

    The single most selective term, and the one that decides what the strategy
    even is: EXP-123 measured it selecting structures TINY relative to the stock
    (median ``w`` 1.08% of spot against 4.96% unfiltered), which is why 39.6% of
    the traded events blew through a wing while only 7.3% of all resolvable ones
    did. Keeping it is a deliberate choice to trade a narrow, cheap tent rather
    than to widen until the wings stop being hit — widening re-selects the
    universe through this same term and pins the move back at ~3.14w.

    **Written against the peak, not against ``w``.** Both are the same test on
    TWIN-P, whose peak is ``2w``, and the numbers above are unchanged by this
    phrasing. They stop being the same test the moment a shape peaks somewhere
    else: TWIN-P5's wide wing peaks at ``2a`` and its tight wing at ``a``, so a
    rule hard-coded to ``cost < w`` would let the tight shape onto the board at
    twice the risk it had priced. The structure declares its own
    ``peak_multiple``; this reads it.
    """
    cost, peak = _number(facts, "cost"), _number(facts, "peak")
    if cost is None or peak is None or peak <= 0:
        return None
    return cost < peak / 2.0


def _spread_is_crossable(facts: Mapping) -> bool | None:
    spread = _number(facts, "rel_spread")
    return None if spread is None else spread <= MAX_REL_SPREAD


def _name_is_large_enough(facts: Mapping) -> bool | None:
    mcap = _number(facts, "mcap_usd")
    return None if mcap is None else mcap >= MCAP_FLOOR


TWIN_P_RULE = EntryRule(
    strategy="TWIN-P",
    terms=(
        Term("reward", "cost is not below half the peak payoff", _reward_beats_risk,
             needs=("cost", "peak")),
        Term("spread", f"mean relative spread exceeds {MAX_REL_SPREAD:.0%}",
             _spread_is_crossable, needs=("rel_spread",)),
        Term("mcap", f"market cap below ${MCAP_FLOOR/1e9:.0f}B",
             _name_is_large_enough, needs=("mcap_usd",)),
    ),
    evidence="EXP-123 spec.yaml, pre-registered",
)


def _expected_pnl_clears_the_bar(facts: Mapping) -> bool | None:
    """``exp_pnl_sim >= pnl_cutoff`` — the top fifth of the trailing six months.

    THE TERM THAT REPLACED ``cost < peak / 2`` FOR TWIN-P5, on 2026-09-05.

    The arithmetic term asked whether max profit beat max loss and answered it
    without reference to where the print was likely to land. This asks what the
    structure is expected to return, given a forecast of the move, a forecast of
    the IV crush, and the calibrated error distribution of both.

    **What changes in kind, not just in accuracy.** ``cost < peak / 2`` is an
    arithmetic GUARANTEE: it cannot be wrong about an event, whatever any model
    believes. This is a model output, so a model failure now ADMITS trades
    rather than merely mis-ranking them, and it fails silently — an
    over-optimistic exit price on a cheap wing reads as edge. That is a real
    elevation of risk and it was accepted knowingly; see
    ``guides/pnl_gate_promotion.md``.

    **The bar is relative and recomputed monthly**, over a trailing six months.
    Not because six months predicts better than twelve — no window from 6 to 36
    months is distinguishable on returns — but because it holds the admitted
    share close to a fifth: yearly SD 0.027 against 0.051 at twelve months. The
    requirement is a bounded, predictable trade count.

    Undetermined when either fact is missing, which is the case whenever the
    simulation declined: no forecast, no pre-print vol, a residual pool too
    thin, or a trailing window under 100 events. Never a rejection.
    """
    value, bar = _number(facts, "exp_pnl_sim"), _number(facts, "pnl_cutoff")
    if value is None or bar is None:
        return None
    return value >= bar


#: TWIN-P5 — the shape promoted off EXP-126, the GATE promoted off EXP-129/131.
#:
#: The reward term is gone and the two liquidity guards are untouched. That
#: split is deliberate: spread and market cap are tradeability facts, and no
#: simulation can conjure a quote that is not there, so they bind whatever the
#: forecast says. Only the term that was a PROXY for expected return has been
#: replaced by expected return itself.
#:
#: Held out on 2023-2026, with the window and quantile chosen from 2018-2022
#: alone: 287 events against the incumbent's 154, final equity 2.53x against
#: 1.68x, Sharpe 1.54 against 1.15, breakeven alpha 0.445 against 0.461, 4/4
#: years positive for both. What it did NOT clear is distinguishability — a
#: block bootstrap on the held-out CAGR difference spans [-28.99, +43.71]pp.
#: Four years cannot separate these rules and this gate is live on a decision
#: the statistics could not make.
TWIN_P5_RULE = EntryRule(
    strategy="TWIN-P5",
    terms=(
        Term("expected_pnl",
             "simulated expected return is below the trailing top-20% bar",
             _expected_pnl_clears_the_bar, needs=("exp_pnl_sim", "pnl_cutoff")),
        Term("spread", f"mean relative spread exceeds {MAX_REL_SPREAD:.0%}",
             _spread_is_crossable, needs=("rel_spread",)),
        Term("mcap", f"market cap below ${MCAP_FLOOR/1e9:.0f}B",
             _name_is_large_enough, needs=("mcap_usd",)),
    ),
    evidence="EXP-129 + EXP-131 held out on 2023-2026; promoted 2026-09-05",
)


#: Strategies gated by arithmetic rather than by a registered model.
ENTRY_RULES: dict[str, EntryRule] = {
    "TWIN-P": TWIN_P_RULE,
    "TWIN-P5": TWIN_P5_RULE,
}


def rule_for(strategy: str) -> EntryRule | None:
    return ENTRY_RULES.get(strategy)
