"""Deep-freeze support for ``ScoreRecord`` mapping fields.

``engine/v2/contracts`` may declare shapes only (schemas, no methods, no
imports beyond ``__future__``/``dataclasses``/``typing``/other contracts
modules) — see ``tests/test_v2_ops_contracts.py``. The recursive-immutability
behaviour for ``ScoreRecord`` therefore lives here, in the scoring package
that owns the one construction path (``with_score_id`` in ``identity.py``),
rather than on the contract itself.
"""
from __future__ import annotations

from typing import Any, Mapping

__all__ = ["FROZEN_RECORD_FIELDS", "deep_freeze", "freeze_record_fields"]

#: Every ``ScoreRecord`` field that holds nested mapping/sequence data and
#: must be made recursively immutable before a record escapes the scoring
#: package. Mirrors the historical ``ScoreRecord.__post_init__`` field list.
FROZEN_RECORD_FIELDS = (
    "canonical_request", "resolved_request", "event_ref",
    "selected_contracts", "legs", "entry_exit_plan", "quote_provenance",
    "forecasts", "uncertainty", "feature_values", "null_masks",
    "gate_terms", "chooser_candidates", "chooser_selection",
    "financial_diagnostics", "requested_payoff_views",
    "operational_envelope",
)


class _FrozenDict(dict):
    """A recursively immutable dict that remains JSON/dataclass compatible."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        source = dict(*args, **kwargs)
        dict.__init__(self, ((key, deep_freeze(value)) for key, value in source.items()))

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("ScoreRecord mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def deep_freeze(value: Any) -> Any:
    if isinstance(value, _FrozenDict):
        return value
    if isinstance(value, Mapping):
        return _FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(deep_freeze(item) for item in value)
    return value


def freeze_record_fields(record: Any) -> Any:
    """Deep-freeze the mapping fields of ``record`` in place and return it.

    ``record`` is a frozen dataclass; this bypasses its ``__setattr__`` guard
    the same way the retired ``ScoreRecord.__post_init__`` did.
    """
    for name in FROZEN_RECORD_FIELDS:
        object.__setattr__(record, name, deep_freeze(getattr(record, name)))
    return record
