"""Native provider edges for ``engine.v2.ops``.

Re-exporting the one edge here lets a caller load it through the package path
``incremental_data.py`` already imports, so that module's fan-out budget does
not pay a second distinct import while the edge still loads lazily.
"""
from __future__ import annotations

import os

from engine.v2.ops.providers.orats_daily_market import orats_daily_market_fetcher

#: Provider account -> the environment variables its fetcher needs in the
#: worker. These are the only credential names the executor's fixed launch
#: allowlist may copy; a value is never logged, printed or recorded.
PROVIDER_CREDENTIAL_VARIABLES = {"orats-daily-market": ("ORATS_API_KEY",)}


def provider_credentials(parameters) -> dict[str, str]:
    """The credential variables a job's provider account needs, from this env."""
    account = parameters.get("provider_account") if isinstance(parameters, dict) else None
    names = PROVIDER_CREDENTIAL_VARIABLES.get(account or "", ())
    return {name: os.environ[name] for name in names if os.environ.get(name)}


__all__ = ["PROVIDER_CREDENTIAL_VARIABLES", "orats_daily_market_fetcher",
           "provider_credentials"]
