"""The nightly path declares the frozen board analog pool (#681); synthetic stand-ins only."""
import dataclasses
import math
from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.models.analog_artifact import (
    board_analog_pool_key,
    make_board_analog_pool_artifact,
)
from engine.v2.models.lineage import Lineage
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_score_batch import _with_release_artifacts
from engine.v2.scoring.identity import score_request_key
from engine.v2.scoring.source_inputs import SourceBundle
from tests.v2.ops.test_native_score_batch_driver_alias import (
    GATE,
    SIZE,
    _assemble,
    _calendar_row,
    _event_inputs,
    _stage,
)

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


def _real_pool(cutoff, strategy="STR-THRU"):
    return make_board_analog_pool_artifact(
        strategy=strategy, alpha=0.5, cutoff=cutoff, population_edges=(0.9, 1.1),
        causal_edges=(0.9, 1.1), lineage=Lineage(),
        rows=[("r1", "2025-01-01", "2025-02-01", "1-10B", "4-10", "ATM", "mid", 0.01)])


def _panel_event(**overrides):
    # panel_row and tier4_row carry DIFFERENT mcap/im/prior values, so the analog
    # query built from the wrong row gives a different bucket and ratio.
    return _event_inputs(
        panel_row={"date": "2026-01-15", "signal": 1.5, "mcap_log": math.log(5e9),
                   "im": 0.09, "mean_prior_or_implied": 0.06},
        tier4_row={"pred_abs_move": 0.05, "pred_abs_move_fold_start": "2026-01-01",
                   "mcap_log": math.log(5e10), "im": 0.04, "mean_prior_or_implied": 0.08},
        **overrides)


def test_assembly_builds_the_analog_query_from_the_panel_row(tmp_path):
    # Through the real caller (_assemble_one_event). Entry (2026-01-08) is BEFORE
    # as_of (2026-01-10), so the analog cutoff is the entry date: the pool keyed
    # at as_of must NOT be picked.
    pool = _real_pool("2026-01-08")
    decoy = _real_pool("2026-01-10")
    binding = dataclasses.replace(
        _stage(tmp_path, [SIZE, GATE]), analog_artifacts={"STR-THRU": (decoy, pool)})
    event = _panel_event(calendar_row=_calendar_row(entry_date="2026-01-08"))
    assembled, refusals = _assemble(binding, [event])
    assert refusals == ()
    block = assembled[event.key][1].analogs
    assert block["analog_artifact"] is pool
    assert block["analog_artifact_recipe"]["cutoff"] == "2026-01-08"
    assert block["query_features"]["mcap_bucket"] == "1-10B"
    assert block["query_features"]["implied_ratio"] == pytest.approx(1.5)


def test_assembly_ignores_a_pool_staged_for_another_strategy(tmp_path):
    other = _real_pool("2026-01-08", strategy="STR-OTHER")
    binding = dataclasses.replace(
        _stage(tmp_path, [SIZE, GATE]), analog_artifacts={"STR-OTHER": (other,)})
    event = _panel_event(calendar_row=_calendar_row(entry_date="2026-01-08"))
    assembled, refusals = _assemble(binding, [event])
    assert refusals == ()
    assert "analog_artifact" not in assembled[event.key][1].analogs


def test_analog_only_release_with_null_entry_date_refuses_the_row(tmp_path):
    # No driver pool and no payoff artifact: the analog cutoff is the only thing
    # that needs the entry date, so its refusal is the analog declaration's own.
    binding = dataclasses.replace(
        _stage(tmp_path, [SIZE, GATE]),
        analog_artifacts={"STR-THRU": (_real_pool("2026-01-08"),)})
    event = _panel_event(calendar_row=_calendar_row(entry_date=None))
    assembled, refusals = _assemble(binding, [event])
    assert assembled == {}
    assert [r.code for r in refusals] == ["MISSING_STAGED_INPUT"]
