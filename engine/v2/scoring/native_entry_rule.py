"""Native arithmetic entry-rule gate (the gates that are not models).

Seven board strategies have no gate champion; legacy ``Scorer._score_gate``
falls through to ``Scorer._apply_entry_rule``, which evaluates
``engine/entry_rules.py`` ``ENTRY_RULES[strategy]`` on the entry-close facts.
Every one of the seven is the same three terms (``TWIN_P_RULE``,
``TWIN_P5_RULE`` and ``_short_vol_rule``):

* ``expected_pnl`` -- ``exp_pnl_sim >= pnl_cutoff``, the trailing top-20%
  bar (``engine/pnl_sim.py`` ``trailing_cutoff``);
* ``spread`` -- ``rel_spread <= MAX_REL_SPREAD``, the mean relative spread
  over the priced legs (``engine/score.py`` ``_mean_relative_spread``);
* ``mcap`` -- ``mcap_usd >= MCAP_FLOOR``.

Three outcomes, as legacy: a term whose fact is missing, NaN or infinite
(``entry_rules._number``) is undetermined (``None``); any undetermined term
makes the verdict ``None`` and the row carries ``MISSING_FEATURES``;
otherwise the verdict passes only when every term holds. The verdict is
``gate_pass`` alone: legacy writes no ``gate_score``/``gate_threshold`` for
a rule.

Sources, none of them a legacy answer: ``exp_pnl_sim`` is this row's own
simulation stage output, ``rel_spread`` is derived from the priced legs,
``mcap_usd`` is the declared market-state fact, and ``pnl_cutoff`` is read
from a frozen :class:`~engine.v2.models.trailing_cutoff_artifact.
TrailingCutoffArtifact` whose key must be the event's month (and, when
pinned, whose content hash must match). A declared cutoff that is absent or
carries another key is ``MODEL_NOT_READY``; it is never recomputed here.

The gate block, a JSON document so it can travel in a strict trace::

    {"mode": "entry_rule", "rule": "<strategy>",
     "facts": {"mcap_usd": <number | null>},
     "trailing_cutoff_key": {"history_id", "month"[, "content_hash"]},
     "trailing_cutoff": <TrailingCutoffArtifact payload | null>}
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from engine.v2.models.trailing_cutoff_artifact import (
    PNL_HISTORY_ID,
    TrailingCutoffError,
    cutoff_month,
    trailing_cutoff_from_document,
)
from engine.v2.scoring.native_chooser_features import rel_spread

__all__ = [
    "ENTRY_RULE_MODE",
    "ENTRY_RULE_STRATEGIES",
    "ENTRY_RULE_TERMS",
    "MAX_REL_SPREAD",
    "MCAP_FLOOR",
    "entry_rule_block",
    "evaluate_entry_rule",
    "execute_entry_rule",
    "number",
]

ENTRY_RULE_MODE = "entry_rule"
#: engine/entry_rules.py ``MAX_REL_SPREAD`` and ``MCAP_FLOOR`` (copied; the
#: parity test pins them equal).
MAX_REL_SPREAD = 0.25
MCAP_FLOOR = 10e9
#: engine/entry_rules.py ``ENTRY_RULES`` keys.
ENTRY_RULE_STRATEGIES = ("TWIN-P", "TWIN-P5", "CND-PS", "BFLY-P", "BFLY-P5", "RAMP7", "CTR5")
#: ``(term name, facts it needs)`` in legacy order, for every rule above.
ENTRY_RULE_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("expected_pnl", ("exp_pnl_sim", "pnl_cutoff")),
    ("spread", ("rel_spread",)),
    ("mcap", ("mcap_usd",)),
)
_KEY_FIELDS = frozenset({"history_id", "month", "content_hash"})
_BLOCK_FIELDS = frozenset({"mode", "rule", "facts", "trailing_cutoff_key", "trailing_cutoff"})


def entry_rule_block(strategy: str, *, mcap_usd: Any, cutoff: Any, month: Any = None,
                     pin: bool = True) -> dict[str, Any]:
    """The gate block declaring ``strategy``'s rule over a frozen ``cutoff``
    (a ``TrailingCutoffArtifact``; ``None`` with a ``month`` declares the
    key of one that is absent, which the stage refuses)."""
    key: dict[str, Any] = {
        "history_id": getattr(cutoff, "history_id", PNL_HISTORY_ID),
        "month": cutoff.month if cutoff is not None else cutoff_month(month)}
    if pin and cutoff is not None:
        key["content_hash"] = cutoff.content_hash
    return {"mode": ENTRY_RULE_MODE, "rule": strategy, "facts": {"mcap_usd": mcap_usd},
            "trailing_cutoff_key": key,
            "trailing_cutoff": None if cutoff is None else cutoff.payload()}


def number(value: Any) -> float | None:
    """engine/entry_rules.py ``_number``: a finite float, or ``None``."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _term(name: str, facts: Mapping[str, Any]) -> bool | None:
    if name == "expected_pnl":
        value, bar = number(facts.get("exp_pnl_sim")), number(facts.get("pnl_cutoff"))
        return None if value is None or bar is None else value >= bar
    if name == "spread":
        spread = number(facts.get("rel_spread"))
        return None if spread is None else spread <= MAX_REL_SPREAD
    mcap = number(facts.get("mcap_usd"))
    return None if mcap is None else mcap >= MCAP_FLOOR


def evaluate_entry_rule(facts: Mapping[str, Any]) -> tuple[bool | None, dict[str, bool | None]]:
    """``(verdict, per-term results)`` exactly as ``EntryRule.evaluate``."""
    terms = {name: _term(name, facts) for name, _needs in ENTRY_RULE_TERMS}
    if any(value is None for value in terms.values()):
        return None, terms
    return all(terms.values()), terms


def _cutoff(block: Mapping[str, Any], event_date: Any) -> tuple[bool, float | None]:
    """``(ready, bar)`` from the declared frozen cutoff; not ready refuses."""
    key = block.get("trailing_cutoff_key")
    document = block.get("trailing_cutoff")
    if not isinstance(key, Mapping) or set(key) - _KEY_FIELDS or not isinstance(document, Mapping):
        return False, None
    try:
        artifact = trailing_cutoff_from_document(document)
        month = cutoff_month(event_date)
    except (TrailingCutoffError, KeyError, TypeError, ValueError):
        return False, None
    if (artifact.key != (str(key.get("history_id")), str(key.get("month")))
            or artifact.month != month):
        return False, None
    pinned = key.get("content_hash")
    if pinned is not None and pinned != artifact.content_hash:
        return False, None
    return True, artifact.cutoff


def execute_entry_rule(block: Mapping[str, Any], *, strategy: str, event_date: Any,
                       legs: Sequence[Mapping[str, Any]], exp_pnl_sim: Any,
                       flags: list[str]) -> dict[str, Any]:
    """The gate stage for an arithmetic rule: ``{"gate_pass": verdict}``.

    Appends to ``flags``: ``UNSUPPORTED_GATE_RECIPE:entry_rule`` for a block
    that is malformed or names another strategy's rule, ``MODEL_NOT_READY``
    when the frozen cutoff cannot serve (no verdict then), and
    ``MISSING_FEATURES`` for an undetermined verdict.
    """
    facts_block = block.get("facts")
    if (set(block) - _BLOCK_FIELDS or block.get("rule") != strategy
            or strategy not in ENTRY_RULE_STRATEGIES or not isinstance(facts_block, Mapping)
            or set(facts_block) - {"mcap_usd"}):
        flags.append("UNSUPPORTED_GATE_RECIPE:entry_rule")
        return {}
    ready, bar = _cutoff(block, event_date)
    if not ready:
        flags.append("MODEL_NOT_READY")
        return {}
    facts = {"exp_pnl_sim": exp_pnl_sim, "pnl_cutoff": bar,
             "rel_spread": rel_spread(legs) if legs else None,
             "mcap_usd": facts_block.get("mcap_usd")}
    verdict, _terms = evaluate_entry_rule(facts)
    if verdict is None:
        flags.append("MISSING_FEATURES")
    return {"gate_pass": verdict}
