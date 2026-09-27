"""Native provider edges for ``engine.v2.ops``.

Re-exporting the edges here lets a caller load them through the package path
``incremental_data.py`` already imports, so that module's fan-out budget does
not pay a second distinct import while every edge still loads lazily.
"""
from __future__ import annotations

import os

from engine.v2.ops.providers.nasdaq_calendar import nasdaq_calendar_fetcher
from engine.v2.ops.providers.orats_daily_market import orats_daily_market_fetcher
from engine.v2.ops.providers.yfinance_edge import (
    yfinance_earnings_fetcher,
    yfinance_history_fetcher,
)

#: Provider account -> the environment variables its fetcher needs in the
#: worker. These are the only credential names the executor's fixed launch
#: allowlist may copy; a value is never logged, printed or recorded. The
#: ``yfinance``/``nasdaq`` accounts are unmetered and keyless, so their tuples
#: are deliberately empty (budget reservation still applies).
PROVIDER_CREDENTIAL_VARIABLES = {
    "orats-daily-market": ("ORATS_API_KEY",),
    "yfinance": (),
    "nasdaq": (),
}


def provider_credentials(parameters) -> dict[str, str]:
    """The credential variables a job's provider account needs, from this env."""
    account = parameters.get("provider_account") if isinstance(parameters, dict) else None
    names = PROVIDER_CREDENTIAL_VARIABLES.get(account or "", ())
    return {name: os.environ[name] for name in names if os.environ.get(name)}


__all__ = ["PROVIDER_CREDENTIAL_VARIABLES", "nasdaq_calendar_fetcher",
           "orats_daily_market_fetcher", "provider_credentials",
           "yfinance_earnings_fetcher", "yfinance_history_fetcher"]
