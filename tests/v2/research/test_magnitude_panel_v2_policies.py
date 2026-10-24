"""Independent closed policy oracles; declarations never qualify or read sources."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.v2.features import regime, runup_math
from engine.v2.foundation import content_hash
from engine.v2.research import magnitude_panel_v2_policies as owner
from tests.v2.research.magnitude_panel_policy_support import (
    EXPECTED_AVAILABILITY,
    EXPECTED_MEAN,
    EXPECTED_SOURCES,
    EXPECTED_SPECIAL,
    forbid_policy_io,
)

_RUNUP = ["signed_streak", "dist_high", "dist_ema", "ret5", "ret10", "ret20"]
_REGIME = ["spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol5", "spy_vol20", "spy_vol60",
           "spy_vol252", "spy_vol20_rel252"]
_QUOTE = {"name": "has_implied_quote", "source": "daily_market.quote_status", "true_status": "present",
          "false_statuses": ["zero", "absent", "unavailable"], "position": "after_numeric_dictionary"}


def test_complete_policy_objects_match_independent_literals():
    result = owner.policy_definitions()
    assert set(result) == {"source_policies", "derivation_policies", "quote_indicator", "missing_indicators"}
    assert result["source_policies"] == EXPECTED_SOURCES
    expected = copy.deepcopy(EXPECTED_SPECIAL)
    for function, outputs in (("engine.v2.features.runup_math.add_runup_features", _RUNUP),
                              ("engine.v2.features.regime.add_regime_features", _REGIME)):
        for output in outputs:
            expected[output] = {"schema_version": "magnitude_existing_math.v1", "function": function, "output": output}
    assert result["derivation_policies"] == expected
    assert len(result["derivation_policies"]) == 20
    assert result["quote_indicator"] == _QUOTE
    assert result["missing_indicators"] == "family_recipe_train_only_selected_columns"


def test_aliases_sources_and_missing_policies_are_not_interchangeable():
    result = owner.policy_definitions()
    sources, derived = result["source_policies"], result["derivation_policies"]
    assert sources["market_context"]["instrument"] == "^GSPC"
    assert sources["market_context"]["value_field"] == "close"
    assert sources["runup_prices"]["value_field"] == "close_adj"
    assert {name for name in derived if name.startswith("spy_")} == set(_REGIME)
    assert derived["mean_prior_or_implied"]["zero_mean"] == "valid"
    assert derived["mean_prior_or_implied"]["zero_contributions"] == "null"
    assert derived["implied_move"]["zero"] == "null_keep_source_status"
    assert sources["daily_market"]["normalized_null"] == "never_infer_zero"
    assert sources["market_context"]["availability"]["missing_provenance"] == "refuse_before_values"
    assert sources["market_context"]["availability"]["known_late"] == "null"


def test_detachment_within_result_and_across_calls():
    first = owner.policy_definitions()
    original = copy.deepcopy(first)
    first["source_policies"]["market_context"]["availability"]["known_late"] = "poison"
    assert first["source_policies"]["runup_prices"]["availability"] == EXPECTED_AVAILABILITY
    assert first["source_policies"]["daily_market"]["availability"] == EXPECTED_AVAILABILITY
    first["derivation_policies"]["mean_prior_or_implied"]["eligibility"]["embargo_days"] = 0
    first["derivation_policies"]["ema12r_abs"]["condition"]["value"] = 0
    first["quote_indicator"]["false_statuses"].clear()
    assert owner.policy_definitions() == original


def test_declarations_only_and_exact_shared_eligibility_call(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("metadata factory attempted I/O or feature computation")

    calls = []
    original = owner.eligibility_policy

    def declare(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(owner, "eligibility_policy", declare)
        forbid_policy_io(guarded)
        guarded.setattr(runup_math, "add_runup_features", forbidden)
        guarded.setattr(regime, "add_regime_features", forbidden)
        with pytest.raises(AssertionError, match="file/database I/O"):
            Path("synthetic-unopened-file").read_text()
        with pytest.raises(AssertionError, match="feature computation"):
            runup_math.add_runup_features(None, None, None)
        result = owner.policy_definitions()
    assert calls == [{"version": "grouped_ticker_embargo.v1", "embargo_days": 30}]
    assert result["derivation_policies"]["mean_prior_or_implied"] == EXPECTED_MEAN
    assert owner.__all__ == ["policy_definitions"]
    assert not hasattr(owner, "CONTRACT") and not hasattr(owner, "recipe")


def test_shared_declaration_exception_propagates_unchanged(monkeypatch):
    error = RuntimeError("synthetic shared declaration failure")

    def refused(**kwargs):
        raise error

    monkeypatch.setattr(owner, "eligibility_policy", refused)
    with pytest.raises(RuntimeError) as caught:
        owner.policy_definitions()
    assert caught.value is error


@pytest.mark.parametrize("argument", [None, {}, "HMO"])
def test_no_caller_policy_injection(argument):
    with pytest.raises(TypeError, match="takes 0 positional arguments"):
        owner.policy_definitions(argument)


def test_policy_hashes_are_canonical_content_not_source_attestation():
    source, mean = EXPECTED_SOURCES["market_context"], EXPECTED_MEAN
    result = owner.policy_definitions()
    for document, actual, expected in (
        (source, result["source_policies"]["market_context"],
         "sha256:7c34967ce46ef061a3edca503e359e5450bc9cb811fd1cf98f66d43b6ada3f0a"),
        (mean, result["derivation_policies"]["mean_prior_or_implied"],
         "sha256:f867553f03fd532fd68fe380530f7a0ca5ac216f08c87bbfa7b83ba1d0b8a965"),
    ):
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        assert "sha256:" + hashlib.sha256(canonical).hexdigest() == expected
        assert content_hash(actual) == expected
    changed_source = result["source_policies"]["market_context"]
    changed_source["instrument"] = "SPY"
    changed_mean = result["derivation_policies"]["mean_prior_or_implied"]
    changed_mean["warmup"] = 0
    assert content_hash(changed_source) != content_hash(source)
    assert content_hash(changed_mean) != content_hash(mean)


def test_declared_existing_math_outputs_are_real_synthetic_outputs():
    dates = pd.date_range("2020-01-01", periods=310)
    closes = 100.0 + np.arange(len(dates))
    prices = pd.DataFrame({"date": dates, "close_adj": closes})
    events = pd.DataFrame({"ticker": ["SYN", "SYN"], "date": dates[-2:], "move": [1.0, -1.0],
        "n_prior": [12, 13], "ema12_prior_abs_move": [2.0, 3.0], "mean_prior_abs_move": [4.0, 5.0]})
    runup = runup_math.add_runup_features(events, {"SYN": prices}, "date")
    market = prices.rename(columns={"close_adj": "close"})
    regime_values = regime.add_regime_features(events, market)
    declared = owner.policy_definitions()["derivation_policies"]
    for output, frame in [(name, runup) for name in _RUNUP] + [(name, regime_values) for name in _REGIME]:
        assert declared[output]["output"] == output
        assert np.isfinite(frame[output].to_numpy()).all()
