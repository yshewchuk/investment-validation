"""Native v2 accounting: the hypothetical book, capital per trade and funding
computed over canonical catalog decisions (P6-3 ``books-funding``).

Delegates every computation to ``engine.portfolio.build_book``/``summarize``
through the one declared adapter (:mod:`engine.v2.ledger.legacy_adapter`),
fed from :mod:`engine.v2.ledger.catalog_reader` instead of the jsonl ledger,
so totals, capital-per-trade and funding numbers agree with the existing
accounting by construction — same code, different row source.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from . import catalog_reader, legacy_adapter

__all__ = ["build_book", "summarize"]


def build_book(conn, contracts: int | None = None, capital_per_trade: float | None = None,
               *, include_declined: bool = False) -> pd.DataFrame:
    """``engine.portfolio.build_book`` fed from catalog decisions/outcomes."""
    return legacy_adapter.build_book(
        predictions=catalog_reader.read_predictions(conn),
        outcomes=catalog_reader.read_outcomes(conn),
        contracts=contracts, capital_per_trade=capital_per_trade,
        include_declined=include_declined)


def summarize(book: pd.DataFrame) -> dict[str, Any]:
    """``engine.portfolio.summarize`` — already a pure function of ``book``."""
    return legacy_adapter.summarize(book)
