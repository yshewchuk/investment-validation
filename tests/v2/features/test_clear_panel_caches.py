"""``engine.features.clear_panel_caches`` empties the Tier-3 lru_caches.

The nightly's fetch-only path drops its event frames and calls this before
``warm_serving`` so the warm subprocess does not overlap the parent's panel
memory (#1195). The three caches are ``lru_cache(maxsize=1)`` process globals:
this pins that the public helper clears every one of them, populating each with
a tiny CSV first so the assertion is "populated then emptied", not "already
empty".
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine import features


@pytest.fixture(autouse=True)
def _cache_starts_empty():
    features.clear_panel_caches()
    yield
    features.clear_panel_caches()


def _write_panel_csv(tmp_path) -> tuple[str, float]:
    path = tmp_path / "panel.csv"
    pd.DataFrame({
        "ticker": ["AAA"],
        "date": ["2026-10-25"],
    }).to_csv(path, index=False)
    return str(path), 0.0


def _write_forecasts_csv(tmp_path) -> tuple[str, float]:
    path = tmp_path / "forecasts.csv"
    pd.DataFrame({
        "ticker": ["AAA"],
        "event_date": ["2026-10-25"],
        **{column: [None] for column in features.FORECAST_COLUMNS},
    }).to_csv(path, index=False)
    return str(path), 0.0


def test_clear_panel_caches_empties_all_three_caches(tmp_path):
    panel_key = _write_panel_csv(tmp_path)
    forecast_key = _write_forecasts_csv(tmp_path)

    features._panel_cached(*panel_key)
    features._forecasts_cached(*forecast_key)
    features._panel_with_forecasts(panel_key, forecast_key)

    assert features._panel_cached.cache_info().currsize == 1
    assert features._forecasts_cached.cache_info().currsize == 1
    assert features._panel_with_forecasts.cache_info().currsize == 1

    features.clear_panel_caches()

    assert features._panel_cached.cache_info().currsize == 0
    assert features._forecasts_cached.cache_info().currsize == 0
    assert features._panel_with_forecasts.cache_info().currsize == 0


def test_clear_panel_caches_is_idempotent_and_exported():
    assert callable(features.clear_panel_caches)
    assert "clear_panel_caches" in features.__all__
    features.clear_panel_caches()
    features.clear_panel_caches()
    assert features._panel_cached.cache_info().currsize == 0