"""Native provider edges for ``engine.v2.ops``.

Re-exporting the one edge here lets a caller load it through the package path
``incremental_data.py`` already imports, so that module's fan-out budget does
not pay a second distinct import while the edge still loads lazily.
"""
from engine.v2.ops.providers.orats_daily_market import orats_daily_market_fetcher

__all__ = ["orats_daily_market_fetcher"]