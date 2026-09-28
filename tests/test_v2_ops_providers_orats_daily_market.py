"""#96: the native ``daily_market`` row builder applies the legacy
``PLAUSIBLE_RANGES`` clip and the ``impliedMove <= 0`` "no quote" sentinel
mask, and its local ``PLAUSIBLE_RANGES`` mirror equals the legacy table."""
from __future__ import annotations

from engine.data.normalize.common import PLAUSIBLE_RANGES as LEGACY_RANGES
from engine.v2.ops.providers.orats_daily_market import (
    PLAUSIBLE_RANGES as NATIVE_RANGES,
    _merge_ticker_rows,
)

# Trimmed from earnings_predictions/data/raw/orats/summaries/A.json.gz (real ORATS
# hist/summaries response shape, ticker A, one row edited for this test).
SUMMARY_ROW = {
    "ticker": "AAA", "tradeDate": "2026-04-30", "stockPrice": 150.0,
    "exErnIv10d": 0.25, "exErnIv30d": 0.26, "ieeEarnEffect": 1.0,
    "impliedMove": 0.0, "iv10d": 0.24, "iv30d": 6.0, "rVol30": 0.28,
    "skewing": 0.1, "contango": 0.05, "fwd90_30": 0.27, "fexErn90_30": 0.26,
}
# Trimmed from earnings_predictions/data/raw/orats/cores/CIGI.json.gz (real ORATS
# hist/cores response shape; this test's fixture omits the row entirely for the
# "no cores data that day" case).


def _merged_row(cores=()):
    return _merge_ticker_rows(
        [SUMMARY_ROW], list(cores), expected_keys=["AAA"])[0]


def test_implied_move_zero_is_masked_to_none():
    row = _merged_row()
    assert row["implied_move"] is None


def test_out_of_range_iv30_is_masked_to_none():
    row = _merged_row()
    assert row["iv30"] is None


def test_missing_cores_row_leaves_mcap_columns_none():
    row = _merged_row()
    assert row["mcap_usd"] is None
    assert row["mcap_asof"] is None
    assert row["mcap_age_days"] is None
    assert row["src_mcap"] is None


def test_in_range_value_is_not_clipped():
    row = _merged_row()
    assert row["iv10"] == 24.0


def test_native_plausible_ranges_mirror_the_legacy_table():
    assert NATIVE_RANGES == LEGACY_RANGES
