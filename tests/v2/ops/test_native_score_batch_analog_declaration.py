"""The nightly path declares the frozen board analog pool (#681); synthetic stand-ins only."""
import math
from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd

from engine.v2.models.analog_artifact import board_analog_pool_key
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_score_batch import _with_release_artifacts
from engine.v2.scoring.identity import score_request_key
from engine.v2.scoring.source_inputs import SourceBundle

_KEY = BoardRequest(ticker="TEST", strategy="STR-THRU",
                    event_date=pd.Timestamp("2026-01-15"), session="am")
_CALENDAR = {"event_id": "evt-1", "ticker": "TEST", "event_date": "2026-01-15",
             "entry_date": "2026-01-12", "exit_date": "2026-01-16",
             "expiry": "2026-01-16", "spot": 100.0}


@dataclass(frozen=True)
class _Pool:
    key: tuple
    content_hash: str = "sha256:abc"


def _bundle() -> SourceBundle:
    return SourceBundle(
        source_ref="t", context={"ticker": "TEST", "strategy": "STR-THRU"}, raw_quotes={},
        feature_vector={}, feature_missing_mask={}, model_identity={},
        forecast_recipes={}, model_artifact_refs={}, residual_recipe={},
        analog_recipe={}, gate_recipe={})


def _declare(pools):
    binding = SimpleNamespace(
        analog_artifacts={"STR-THRU": tuple(pools)}, payoff_artifacts={},
        driver_residual_artifacts={}, model_release="rel", frozen_inference="inf")
    return _with_release_artifacts(
        _bundle(), binding=binding, key=_KEY, strategy="STR-THRU",
        driver_identity=SimpleNamespace(role="driver", model_id="m"),
        as_of="2026-01-10", snapshot_id="snap-1", calendar_row=_CALENDAR,
        quote_max_age_sessions=None,
        panel_row={"date": "2026-01-15", "mcap_log": math.log(5e9), "im": 0.06})


def test_matching_pool_is_declared_with_the_legacy_request_key():
    pool = _Pool(board_analog_pool_key("STR-THRU", 0.5, "2026-01-10"))
    result = _declare([pool])
    assert result.analog_artifact is pool
    recipe = result.analog_artifact_recipe
    assert recipe["cutoff"] == "2026-01-10"
    assert recipe["content_hash"] == "sha256:abc"
    assert recipe["seed_snapshot"] == "snap-1"
    assert recipe["request_key"] == score_request_key(
        {**result.context, "ticker": "TEST", "strategy": "STR-THRU"})
    assert result.context["snapshot"] == "snap-1"
    # mean_prior_or_implied is absent until #651, so the ratio stays None.
    assert result.analog_query == {"mcap_bucket": "1-10B", "implied_ratio": None}


def test_pool_for_another_cutoff_leaves_the_block_undeclared():
    result = _declare([_Pool(board_analog_pool_key("STR-THRU", 0.5, "2025-01-01"))])
    assert result.analog_artifact is None
    assert result.analog_artifact_recipe == {}
    assert result.analog_query == {}


def test_release_without_analog_pools_leaves_the_block_undeclared():
    result = _declare([])
    assert result.analog_artifact is None
    assert result.analog_query == {}
