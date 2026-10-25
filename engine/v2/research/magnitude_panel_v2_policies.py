"""Closed v2 panel policy metadata, not computations, sources or admission."""
import json

from engine.v2.research.eligibility_policy import eligibility_policy

__all__ = ["policy_definitions"]

_AVAILABILITY = {"event_bound": "strictly_before_event",
    "forecast_bound": "on_or_before_source_session_ceiling", "known_late": "null",
    "missing_publication": "reconstructed_only", "missing_provenance": "refuse_before_values"}
_SOURCES = {
    "stored_history": {"schema_version": "magnitude_stored_history_source.v1",
        "recipe_schema": "stored_magnitude_history.v1",
        "compatibility": "exact_existing_context_and_eligibility", "rebuild": "forbidden"},
    "prior_outcomes": {"schema_version": "magnitude_prior_outcomes_source.v1",
        "population": "existing_qualified_prior_history", "protected_values": "excluded",
        "cutoffs": "existing_history_strict_cutoffs", "knowledge_mode": "reconstructed"},
    "runup_prices": {"schema_version": "magnitude_runup_price_source.v1",
        "instrument": "event_ticker", "value_field": "close_adj",
        "value_definition": "native_price_history_adjusted_close", "availability": _AVAILABILITY},
    "market_context": {"schema_version": "magnitude_market_context_source.v1",
        "instrument": "^GSPC", "value_field": "close",
        "value_definition": "daily_index_closing_level", "availability": _AVAILABILITY},
    "daily_market": {"schema_version": "magnitude_daily_market_source.v1",
        "projection": "existing_DAILY_STATE_FIELDS", "contract_binding": "required_resolved_pinned_contract",
        "preserved_zero": "only_when_stored_source_preserves_it", "normalized_null": "never_infer_zero",
        "availability": _AVAILABILITY},
    "event_metadata": {"schema_version": "magnitude_prefix_metadata_source.v1",
        "population": "canonical_confirmed_events", "prefix_start": "first_known_event",
        "history_start": "does_not_truncate_metadata_prefix", "ordinals": "original_zero_based_no_renumbering",
        "protected_events": "metadata_only_no_quote_values"},
}
_DERIVATIONS = {
    "ema12r_abs": {"schema_version": "magnitude_ema12r_abs.v1",
        "condition": {"feature": "n_prior", "operator": ">=", "value": 12},
        "if_true": "ema12_prior_abs_move", "if_false": "mean_prior_abs_move",
        "missing": "propagate_selected_value"},
    "abs_dist_ema": {"schema_version": "magnitude_abs_dist_ema.v1", "input": "dist_ema",
        "operation": "abs", "anchors": "same_as_input", "missing": "propagate_input_null_and_reason",
        "rounding": "none", "additional_reads": "none"},
    "implied_move": {"schema_version": "magnitude_current_quote.v1",
        "source_validity": "enforce_resolved_contract", "positive": "retain",
        "zero": "null_keep_source_status", "null": "null_keep_source_status", "observable_invalid": "refuse"},
    "mcap_log": {"schema_version": "magnitude_market_cap.v1", "input": "stored_mcap_log",
        "capture_missing_or_known_late": "null", "later_capture_backfill": "forbidden"},
    "mean_prior_or_implied": {"schema_version": "magnitude_protected_implied_mean.v1",
        "metadata_prefix": "complete_canonical_confirmed_from_first_known_event",
        "target_bound": "strictly_before_target_event_and_source_session_ceiling",
        "ordinal": "original_zero_based", "warmup": 4,
        "history_start": "bounds_numeric_contributions_not_ordinals",
        "excluded_metadata": "may_count_ordinals_never_quote_values",
        "contributor": "ordinal_ge_4_on_or_after_history_start_nonprotected_retained_development",
        "quote_selection": "last_src_iv_strictly_before_historical_event", "nonmissing_fallback": "forbidden",
        "source_ceiling": "historical_preprint_and_target_forecast_ceiling",
        "known_late": "null", "missing_publication": "reconstructed_only", "realized_completion_filter": "forbidden",
        "arithmetic": "strict_prior_mean_population_admitted", "noncontributors": "mask_quote_to_null_without_reading_values",
        "target_append": "append_null_quote_take_target_mean_and_count",
        "finite_preserved_zero": "include_only_if_pinned_source_preserves_zero",
        "normalized_null": "skip_never_reconstruct_zero", "denominator": "actual_nonnull_contribution_count",
        "zero_contributions": "null", "zero_mean": "valid", "current_quote_mask": "not_applied_to_history_or_mean",
        "observable_source_violation": "refuse_not_winsorize", "missing_required_provenance": "refuse_before_values",
        "aggregate_reuse": "no_production_computed_moves_or_resumed_aggregate"},
}
_MATH = {
    "engine.v2.features.runup_math.add_runup_features": (
        "signed_streak", "dist_high", "dist_ema", "ret5", "ret10", "ret20"),
    "engine.v2.features.regime.add_regime_features": (
        "spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol5", "spy_vol20",
        "spy_vol60", "spy_vol252", "spy_vol20_rel252"),
}
_QUOTE = {"name": "has_implied_quote", "source": "daily_market.quote_status", "true_status": "present",
          "false_statuses": ["zero", "absent", "unavailable"], "position": "after_numeric_dictionary"}


def policy_definitions():
    """Return only detached requirements; no dictionary, source resolution or I/O."""
    derivations = {**_DERIVATIONS, **{name: {"schema_version": "magnitude_existing_math.v1",
        "function": function, "output": name} for function, names in _MATH.items() for name in names}}
    derivations["mean_prior_or_implied"] = {**derivations["mean_prior_or_implied"],
        "eligibility": eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30)}
    return json.loads(json.dumps({"source_policies": _SOURCES, "derivation_policies": derivations,
        "quote_indicator": _QUOTE, "missing_indicators": "family_recipe_train_only_selected_columns"}))
