"""Per-source adapters. One place per API for auth, URL shape, and quirks."""
from __future__ import annotations

from typing import Callable

from engine.data.sources.base import (
    CredentialRotated,
    FetchError,
    Response,
    SourceAdapter,
    redact,
)

__all__ = [
    "Response",
    "SourceAdapter",
    "CredentialRotated",
    "FetchError",
    "redact",
    "ADAPTERS",
    "get_adapter",
]


def _orats():
    from engine.data.sources.orats import OratsAdapter

    return OratsAdapter()


def _polygon():
    from engine.data.sources.polygon import PolygonAdapter

    return PolygonAdapter()


def _oquants():
    from engine.data.sources.oquants import OquantsAdapter

    return OquantsAdapter()


def _yfinance():
    from engine.data.sources.yf import YFinanceAdapter

    return YFinanceAdapter()


#: Lazy factories — importing this package must not read credentials or pull in
#: playwright/yfinance, so a machine with neither can still run the test-suite.
ADAPTERS: dict[str, Callable[[], SourceAdapter]] = {
    "orats": _orats,
    "polygon": _polygon,
    "oquants": _oquants,
    "yfinance": _yfinance,
}


def get_adapter(source: str) -> SourceAdapter:
    try:
        return ADAPTERS[source]()
    except KeyError:
        raise KeyError(f"unknown source {source!r}; known: {sorted(ADAPTERS)}") from None
