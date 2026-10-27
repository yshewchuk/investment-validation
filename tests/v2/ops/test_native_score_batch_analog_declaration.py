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
from engine.v2.ops import native_score_batch
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_score_batch import _with_release_artifacts
from engine.v2.ops.native_score_batch_types import NativeScoreBatchRowRefusal
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


def _declare(pools, payoff_artifacts=()):
    binding = SimpleNamespace(
        analog_artifacts={"STR-THRU": tuple(pools)},
        payoff_artifacts={"STR-THRU": tuple(payoff_artifacts)},
        driver_residual_artifacts={}, model_release="rel", frozen_inference="inf")
    return _with_release_artifacts(
        _bundle(), binding=binding, key=_KEY, strategy="STR-THRU",
        drivers=(SimpleNamespace(
            output="driver_prediction", identity_key="driver:STR-THRU",
            identity=SimpleNamespace(role="driver", model_id="m")),),
        as_of="2026-01-10", snapshot_id="snap-1", calendar_row=_CALENDAR,
        quote_max_age_sessions=None,
        panel_row={"date": "2026-01-15", "mcap_log": math.log(5e9), "im": 0.06})


def test_matching_pool_is_seeded_with_the_native_snapshot_and_request_key():
    pool = _Pool(board_analog_pool_key("STR-THRU", 0.5, "2026-01-10"))
    result = _declare([pool])
    assert result.analog_artifact is pool
    recipe = result.analog_artifact_recipe
    assert recipe["cutoff"] == "2026-01-10"
    assert recipe["content_hash"] == "sha256:abc"
    # Native identity (user decision 2026-10-13): the NATIVE snapshot id and the
    # native request key, rendered by hand here (not with score_request_key), with
    # the resolved dates; never a legacy snapshot digest or legacy request key.
    native_key = "TEST|STR-THRU|2026-01-10|2026-01-15||2026-01-16|0.5000||||c2026-01-10|"
    assert recipe["seed_snapshot"] == "snap-1"
    assert recipe["request_key"] == native_key
    assert result.context["snapshot"] == "snap-1"
    # mean_prior_or_implied is absent until #651, so the ratio stays None.
    assert result.analog_query == {"mcap_bucket": "1-10B", "implied_ratio": None}


def test_pool_for_another_cutoff_refuses_the_row_without_fallback():
    result = _declare([_Pool(board_analog_pool_key("STR-THRU", 0.5, "2025-01-01"))])
    assert isinstance(result, NativeScoreBatchRowRefusal)
    assert result.code == "MODEL_NOT_READY"


def test_release_without_analog_pools_refuses_the_row():
    result = _declare([])
    assert isinstance(result, NativeScoreBatchRowRefusal)
    assert result.code == "MODEL_NOT_READY"


def test_payoff_family_without_one_exact_cutoff_refuses_the_row():
    pool = _Pool(board_analog_pool_key("STR-THRU", 0.5, "2026-01-10"))
    wrong_cutoff = SimpleNamespace(key=("STR-THRU", 0.5, "2025-01-01"))
    result = _declare([pool], [wrong_cutoff])
    assert isinstance(result, NativeScoreBatchRowRefusal)
    assert result.code == "MODEL_NOT_READY"

    exact = SimpleNamespace(key=("STR-THRU", 0.5, "2026-01-10"))
    duplicate = _declare([pool], [exact, exact])
    assert isinstance(duplicate, NativeScoreBatchRowRefusal)
    assert duplicate.code == "MODEL_NOT_READY"


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


def test_assembly_refuses_when_release_has_no_exact_strategy_pool(tmp_path):
    other = _real_pool("2026-01-08", strategy="STR-OTHER")
    binding = dataclasses.replace(
        _stage(tmp_path, [SIZE, GATE]), analog_artifacts={"STR-OTHER": (other,)})
    event = _panel_event(calendar_row=_calendar_row(entry_date="2026-01-08"))
    assembled, refusals = _assemble(binding, [event])
    assert assembled == {}
    assert [r.code for r in refusals] == ["MODEL_NOT_READY"]


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


def _declared_recipe(tmp_path, *, snapshot_id, ticker, event_day, expiry):
    pool = _real_pool("2026-01-08")
    binding = dataclasses.replace(
        _stage(tmp_path, [SIZE, GATE]), analog_artifacts={"STR-THRU": (pool,)})
    event = _event_inputs(
        ticker=ticker, event_date=pd.Timestamp(event_day),
        calendar_row=_calendar_row(ticker=ticker, event_date=event_day,
                                   entry_date="2026-01-08", expiry=expiry),
        panel_row={"date": event_day, "signal": 1.5, "mcap_log": math.log(5e9)})
    assembled, refusals = _assemble(binding, [event], snapshot_id=snapshot_id)
    assert refusals == ()
    return assembled[event.key][1].analogs["analog_artifact_recipe"]


def test_each_row_recipe_carries_its_own_snapshot_and_request_key(tmp_path):
    # Two rows differ in snapshot, ticker, event date and expiry: hard-coding
    # either the snapshot or the request key (to one row's value) fails one case.
    first = _declared_recipe(tmp_path, snapshot_id="snap-A", ticker="TEST",
                             event_day="2026-01-15", expiry="2026-01-16")
    second = _declared_recipe(tmp_path, snapshot_id="snap-B", ticker="OTHR",
                              event_day="2026-01-14", expiry="2026-01-17")
    assert first["seed_snapshot"] == "snap-A"
    assert first["request_key"] == (
        "TEST|STR-THRU|2026-01-10|2026-01-15||2026-01-16|0.5000||||c2026-01-10|")
    assert second["seed_snapshot"] == "snap-B"
    assert second["request_key"] == (
        "OTHR|STR-THRU|2026-01-10|2026-01-14||2026-01-17|0.5000||||c2026-01-10|")


def test_runup_strategy_also_declares_the_matching_analog_pool(monkeypatch):
    # The STR-RUNUP branch builds its model declarations with
    # runup_model_declarations (stubbed to {} here: this test is about the
    # analog declaration applied on top of it, not RUNUP's model state).
    monkeypatch.setattr(native_score_batch, "runup_model_declarations",
                        lambda *args, **kwargs: {})
    pool = _Pool(board_analog_pool_key("STR-RUNUP", 0.5, "2026-01-10"))
    binding = SimpleNamespace(
        analog_artifacts={"STR-RUNUP": (pool,)}, payoff_artifacts={},
        driver_residual_artifacts={}, model_release="rel", frozen_inference="inf")
    key = BoardRequest(ticker="TEST", strategy="STR-RUNUP",
                       event_date=pd.Timestamp("2026-01-15"), session="am")
    drivers = tuple(
        SimpleNamespace(output=name, identity_key=f"{name}:STR-RUNUP",
                        identity=SimpleNamespace(role=name, model_id="m"))
        for name in ("driver_prediction", "runup_move_prediction"))
    result = _with_release_artifacts(
        _bundle(), binding=binding, key=key, strategy="STR-RUNUP", drivers=drivers,
        as_of="2026-01-10", snapshot_id="snap-1", calendar_row=_CALENDAR,
        quote_max_age_sessions=None,
        panel_row={"date": "2026-01-15", "mcap_log": math.log(5e9), "im": 0.06})
    assert result.analog_artifact is pool
    recipe = result.analog_artifact_recipe
    assert recipe["cutoff"] == "2026-01-10"
    assert recipe["seed_snapshot"] == "snap-1"
    assert recipe["request_key"] == (
        "TEST|STR-RUNUP|2026-01-10|2026-01-15||2026-01-16|0.5000||||c2026-01-10|")


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_conflicting_pools_refuse_the_row_in_either_order(order):
    key = board_analog_pool_key("STR-THRU", 0.5, "2026-01-10")
    pools = [_Pool(key, "sha256:one"), _Pool(key, "sha256:two")]
    result = _declare([pools[i] for i in order])
    assert isinstance(result, NativeScoreBatchRowRefusal)
    assert result.code == "MODEL_NOT_READY"
