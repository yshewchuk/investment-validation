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

__all__ = ["Term", "Verdict", "EntryRule", "ENTRY_RULES", "TWIN_P_RULE", "rule_for"]


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
    """``cost < w`` — max profit ``2w − c`` exceeds max loss ``c``.

    The single most selective term, and the one that decides what the strategy
    even is: EXP-123 measured it selecting structures TINY relative to the stock
    (median ``w`` 1.08% of spot against 4.96% unfiltered), which is why 39.6% of
    the traded events blew through a wing while only 7.3% of all resolvable ones
    did. Keeping it is a deliberate choice to trade a narrow, cheap tent rather
    than to widen until the wings stop being hit — widening re-selects the
    universe through this same term and pins the move back at ~3.14w.
    """
    cost, width = _number(facts, "cost"), _number(facts, "w")
    if cost is None or width is None or width <= 0:
        return None
    return cost < width


def _spread_is_crossable(facts: Mapping) -> bool | None:
    spread = _number(facts, "rel_spread")
    return None if spread is None else spread <= MAX_REL_SPREAD


def _name_is_large_enough(facts: Mapping) -> bool | None:
    mcap = _number(facts, "mcap_usd")
    return None if mcap is None else mcap >= MCAP_FLOOR


TWIN_P_RULE = EntryRule(
    strategy="TWIN-P",
    terms=(
        Term("reward", "cost is not below the tent width w", _reward_beats_risk,
             needs=("cost", "w")),
        Term("spread", f"mean relative spread exceeds {MAX_REL_SPREAD:.0%}",
             _spread_is_crossable, needs=("rel_spread",)),
        Term("mcap", f"market cap below ${MCAP_FLOOR/1e9:.0f}B",
             _name_is_large_enough, needs=("mcap_usd",)),
    ),
    evidence="EXP-123 spec.yaml, pre-registered",
)


#: Strategies gated by arithmetic rather than by a registered model.
ENTRY_RULES: dict[str, EntryRule] = {"TWIN-P": TWIN_P_RULE}


def rule_for(strategy: str) -> EntryRule | None:
    return ENTRY_RULES.get(strategy)
