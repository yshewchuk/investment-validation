"""Capture-side gap fixed 2026-09-21: a frozen replay cannot score a role the
capture never requested.

Established on fixture 011 (CTR5, ``fixtures/tier0``): the record's release
declares ``binding_ids`` for only the "size" role even though the row's
simulation is ``planned_exit`` and every planned-exit row's forecast needs
"iv_crush" too (``engine/v2/scoring/stages.py::_required_forecast_roles``).
Two independent capture-side gaps produced this, one per branch of legacy
``Scorer._crush_forecast``:

* the SERVED branch (a forward event) called ``_phase4_record_fold``, which
  only writes ``source_inputs.frozen`` -- never ``capture_source_bundle``'s
  ``model_bindings``, the accumulator ``native_recipes.forecast
  .required_roles`` and the strict trace's ``binding_ids`` are both built
  from. Fixed in ``engine/score.py::Scorer._crush_forecast`` by adding the
  same ``capture_features``/``capture_source_bundle(model_bindings=...)``
  pair ``_size_from_forecast`` already does for "forecast_sizing".
* the STORED branch (a past event, read from the Tier-4 forecasts table) has
  no served artifact to bind at all -- legacy correctly records its value,
  row and hash under ``source_inputs.frozen.declarations``, but nothing
  folded that into the strict trace's ``forecast`` recipe. Fixed in
  ``tools/capture_tier0_corpus.py::_with_stored_crush`` (called from
  ``_captured_blocks``), which mirrors
  ``engine/v2/scoring/source_inputs.py``'s ``_stored_forecasts`` shape so
  ``engine/v2/scoring/stages.py::_execute_local_forecast`` replays the
  stored value locally rather than through a fabricated binding.

Neither strategy is CTR5-specific: ``_required_forecast_roles``'s
``_is_planned`` branch adds "iv_crush" for ANY strategy whose simulation is
planned-exit, which is every one of the seven DYN-SV menu strategies
(TWIN-P, TWIN-P5, CND-PS, BFLY-P, BFLY-P5, RAMP7, CTR5).
"""
from __future__ import annotations

from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from engine.data.features import tier4
from engine.score import Phase4TraceCollector, Scorer
from engine.v2.foundation import content_hash
from engine.v2.scoring.stages import _STRATEGY_FORECAST_ROLES, _required_forecast_roles
from tools.capture_tier0_corpus import StrictTraceCaptureError, _with_stored_crush

EVENT = "2026-09-16"
FOLD_FEATURES = ("mean_prior_abs_move", "iv30", "signed_streak", "mcap_log")
ROW = {"mean_prior_abs_move": 6.2, "iv30": 55.0, "signed_streak": 2.0, "mcap_log": 26.4}


def _collector() -> Phase4TraceCollector:
    collector = Phase4TraceCollector(retain_full_trace=False, content_hasher=content_hash)
    collector.capture_source_bundle(context={"ticker": "AAA"})
    return collector


def _source_inputs(collector: Phase4TraceCollector) -> dict:
    checkpoint = collector.diagnostic_checkpoint()
    return checkpoint["checkpoints"]["source_inputs"]["value"]


# ---------------------------------------------------------------------------
# STORED branch (fixture 011's own shape): the row-hash-verified value in
# ``source_inputs.frozen.declarations`` must reach ``forecast.required_roles``
# and ``forecast.stored`` without going through a fabricated model binding.
# ---------------------------------------------------------------------------

def test_stored_crush_declaration_earns_no_binding_before_the_fix():
    """Sanity check on the DEFECT shape: legacy's stored branch never adds a
    ``model_bindings`` entry (there is no served artifact behind a table
    lookup), so ``required_roles`` stays capture-derived from bindings alone
    unless something else folds the stored declaration in."""
    collector = _collector()
    scorer = object.__new__(Scorer)
    scorer._crush = {("AAA", pd.Timestamp(EVENT)): -17.25}
    scorer._phase4_tier4_sha = "sha256:" + "c" * 64
    request = SimpleNamespace(ticker="AAA", strategy="CTR5", decision_offset=None)
    result = SimpleNamespace(
        event_date=pd.Timestamp(EVENT), as_of=pd.Timestamp(EVENT),
        _phase4_checkpoint_collector=collector,
    )
    value = Scorer._crush_forecast(scorer, request, result, None)
    assert value == -17.25
    source = _source_inputs(collector)
    assert source.get("model_bindings") in (None, [])
    assert source["frozen"]["declarations"]["forecast:pred_iv_crush_30"]["source"] == (
        "stored_tier4"
    )


def test_with_stored_crush_adds_iv_crush_to_required_roles_and_stored_block():
    """The fix: ``_with_stored_crush`` folds the stored declaration into the
    forecast recipe the same shape ``_execute_local_forecast`` reads."""
    collector = _collector()
    scorer = object.__new__(Scorer)
    scorer._crush = {("AAA", pd.Timestamp(EVENT)): -17.25}
    scorer._phase4_tier4_sha = "sha256:" + "c" * 64
    request = SimpleNamespace(ticker="AAA", strategy="CTR5", decision_offset=None)
    result = SimpleNamespace(
        event_date=pd.Timestamp(EVENT), as_of=pd.Timestamp(EVENT),
        _phase4_checkpoint_collector=collector,
    )
    Scorer._crush_forecast(scorer, request, result, None)
    source = _source_inputs(collector)
    # This is the state fixture 011 was captured with: a size-only recipe,
    # as if no other role had ever touched the row.
    forecast_recipe = {"required_roles": ["size"]}

    fixed = _with_stored_crush(dict(forecast_recipe), source)

    assert fixed["required_roles"] == ["size", "iv_crush"]
    stored = fixed["stored"]["pred_iv_crush_30"]
    assert stored["value"] == -17.25
    assert stored["row"] == {
        "table": "tier4_forecasts", "table_sha256": "sha256:" + "c" * 64,
        "ticker": "AAA", "event_date": EVENT,
    }
    # Original recipe is untouched -- callers pass their own copy.
    assert forecast_recipe == {"required_roles": ["size"]}


def test_with_stored_crush_is_a_noop_without_a_stored_declaration():
    collector = _collector()
    source = _source_inputs(collector)
    unchanged = _with_stored_crush({"required_roles": ["size"]}, source)
    assert unchanged == {"required_roles": ["size"]}


def test_with_stored_crush_refuses_a_tampered_row_hash():
    collector = _collector()
    scorer = object.__new__(Scorer)
    scorer._crush = {("AAA", pd.Timestamp(EVENT)): -17.25}
    scorer._phase4_tier4_sha = "sha256:" + "c" * 64
    request = SimpleNamespace(ticker="AAA", strategy="CTR5", decision_offset=None)
    result = SimpleNamespace(
        event_date=pd.Timestamp(EVENT), as_of=pd.Timestamp(EVENT),
        _phase4_checkpoint_collector=collector,
    )
    Scorer._crush_forecast(scorer, request, result, None)
    source = _source_inputs(collector)
    source = dict(source)
    frozen = dict(source["frozen"])
    declarations = dict(frozen["declarations"])
    tampered = dict(declarations["forecast:pred_iv_crush_30"])
    # Legacy's own capture never sets "row_hash" (there is nothing upstream
    # to check it against); this simulates a checkpoint that carries one
    # anyway and disagrees with its own row+value, which the defensive
    # recompute in ``_with_stored_crush`` must still refuse.
    tampered["row_hash"] = "sha256:" + "0" * 64
    declarations["forecast:pred_iv_crush_30"] = tampered
    frozen["declarations"] = declarations
    source["frozen"] = frozen
    with pytest.raises(StrictTraceCaptureError, match="row_hash"):
        _with_stored_crush({"required_roles": ["size"]}, source)


# ---------------------------------------------------------------------------
# SERVED branch (a forward event): the fold's binding must reach
# ``model_bindings`` and its own feature vector, exactly like
# ``forecast_sizing`` already does.
# ---------------------------------------------------------------------------

@pytest.fixture
def crush_fold(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(tier4, "SERVING_DIR", root)
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    estimator = HistGradientBoostingRegressor(max_iter=3, random_state=0)
    X = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])
    estimator.fit(X, y)
    fold_start = pd.Timestamp("2026-09-01")
    path = tier4._serving_path("crush_synthetic", fold_start, "abc123")
    joblib.dump({
        "estimator": estimator, "model_id": "crush_synthetic",
        "fold_start": "2026-09-01", "tier3_snapshot": "abc123",
        "features": list(FOLD_FEATURES), "pool_pred": np.array([1.0, 2.0]),
        "pool_res": np.array([0.1, -0.1]),
    }, path)
    stored = joblib.load(path)
    served = tier4.ServingModel(
        estimator=stored["estimator"], model_id=stored["model_id"],
        fold_start=pd.Timestamp(stored["fold_start"]),
        tier3_snapshot=stored["tier3_snapshot"], features=tuple(stored["features"]),
        interval_floor=0.0, pool_pred=stored["pool_pred"], pool_res=stored["pool_res"],
    )
    return served


def test_crush_forecast_served_branch_registers_iv_crush_binding_before_fix_would_not(
        crush_fold):
    """FIX A regression: the served branch must add a "role": "iv_crush"
    ``model_bindings`` entry and its own ``features`` role, mirroring
    ``_size_from_forecast``'s "forecast_sizing" registration -- without this,
    ``native_recipes.forecast.required_roles`` never gains "iv_crush" for a
    forward (not-yet-printed) event."""
    collector = _collector()
    scorer = object.__new__(Scorer)
    scorer._crush = {}  # nothing stored: forces the served branch
    scorer._serving = lambda fold, produces="pred_abs_move": crush_fold
    request = SimpleNamespace(ticker="AAA", strategy="CTR5", decision_offset=None)
    result = SimpleNamespace(
        event_date=pd.Timestamp(EVENT), as_of=pd.Timestamp(EVENT),
        _phase4_checkpoint_collector=collector,
    )
    features = pd.DataFrame([ROW])

    value = Scorer._crush_forecast(scorer, request, result, features)

    assert value is not None
    source = _source_inputs(collector)
    bindings = source.get("model_bindings") or []
    roles = [b.get("role") for b in bindings]
    assert "iv_crush" in roles, f"model_bindings roles were {roles}"
    crush_binding = next(b for b in bindings if b["role"] == "iv_crush")
    assert crush_binding["output_names"] == ["pred_iv_crush_30"]
    assert crush_binding["adapter"] == "tier4-serving-fold.v1"
    required_roles = source["native_recipes"]["forecast"]["required_roles"]
    assert "iv_crush" in required_roles


# ---------------------------------------------------------------------------
# Population: this is not CTR5-specific. Derived programmatically from the
# strategy table itself, not a hard-coded strategy list.
# ---------------------------------------------------------------------------

def test_every_planned_exit_strategy_requires_iv_crush():
    """For EVERY strategy in ``_STRATEGY_FORECAST_ROLES`` (the table the
    replay validator reads), a planned-exit row's required roles include
    "iv_crush" -- ``_required_forecast_roles``'s ``_is_planned`` branch is
    strategy-agnostic. Enumerated from the table itself so a new DYN-SV
    strategy is covered automatically instead of needing a new hard-coded
    entry here."""
    planned_simulation = {"mode": "planned_exit"}
    for strategy in _STRATEGY_FORECAST_ROLES:
        fake_inputs = SimpleNamespace(
            context={}, forecast={}, simulation=planned_simulation, geometry=None,
        )
        roles = _required_forecast_roles(fake_inputs, strategy)
        assert "iv_crush" in roles, f"{strategy}: planned-exit roles were {roles}"
