"""Compatibility re-export; the ops module owns the policy."""

from engine.v2.ops.native_registration import (
    LATEST as LATEST,
    allows_hgb_axes as allows_hgb_axes,
    allows_lead_selector as allows_lead_selector,
    allows_neural as allows_neural,
    allows_size_family as allows_size_family,
    allows_workers as allows_workers,
    qualified as qualified,
    request_version as request_version,
    uses_extended_neural_profiles as uses_extended_neural_profiles,
    uses_v3_compiler as uses_v3_compiler,
)
