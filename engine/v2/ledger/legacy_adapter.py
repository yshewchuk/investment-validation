"""The ONE ``engine.v2.ledger -> engine.*`` dependency edge (P6-3).

``checks/import_layers.py`` rule 2 confines every v2-package/legacy edge to
one named, dated, reasoned entry in ``checks/legacy_adapters.json`` and one
adapter module per package. This is that module for ``engine.v2.ledger``; no
other file in this package may import ``engine.ledger`` or
``engine.portfolio`` directly.

Every symbol reused here is a PURE function of the rows/frame it is handed.
P6-3 gave ``engine.ledger.scored_pairs`` and ``engine.portfolio.build_book``
optional ``predictions=``/``outcomes=`` row overrides (default ``None`` —
every caller before P6-3 keeps reading the jsonl ledger unchanged); fed with
:mod:`engine.v2.ledger.catalog_reader` rows instead, these compute IDENTICAL
numbers over canonical catalog decisions. ``settlement_summary`` and
``_strategy_calibration`` already took a frame with no I/O of their own. No
numerical copy: same code, different row source.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from engine.ledger import _strategy_calibration as _legacy_strategy_calibration
from engine.ledger import scored_pairs as _legacy_scored_pairs
from engine.ledger import settlement_summary as _legacy_settlement_summary
from engine.portfolio import build_book as _legacy_build_book
from engine.portfolio import summarize as _legacy_summarize

__all__ = ["build_book", "scored_pairs", "settlement_summary", "strategy_calibration", "summarize"]


def scored_pairs(*, predictions: list[dict[str, Any]],
                  outcomes: list[dict[str, Any]]) -> pd.DataFrame:
    """``engine.ledger.scored_pairs`` over caller-supplied (catalog) rows."""
    return _legacy_scored_pairs(predictions=predictions, outcomes=outcomes)


def settlement_summary(pairs: pd.DataFrame) -> dict:
    """``engine.ledger.settlement_summary`` — already a pure function of ``pairs``."""
    return _legacy_settlement_summary(pairs)


def strategy_calibration(frame: pd.DataFrame) -> dict:
    """``engine.ledger._strategy_calibration`` — already a pure function of ``frame``."""
    return _legacy_strategy_calibration(frame)


def build_book(*, predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]],
               contracts: int | None = None, capital_per_trade: float | None = None,
               include_declined: bool = False) -> pd.DataFrame:
    """``engine.portfolio.build_book`` over caller-supplied (catalog) rows.

    ``capital_per_trade=None`` omits the keyword so legacy's own default
    (``engine.portfolio.CAPITAL_PER_TRADE``) applies, rather than duplicating
    that constant here.
    """
    kwargs: dict[str, Any] = dict(contracts=contracts, include_declined=include_declined,
                                  predictions=predictions, outcomes=outcomes)
    if capital_per_trade is not None:
        kwargs["capital_per_trade"] = capital_per_trade
    return _legacy_build_book(**kwargs)


def summarize(book: pd.DataFrame) -> dict[str, Any]:
    """``engine.portfolio.summarize`` — already a pure function of ``book``."""
    return _legacy_summarize(book)
