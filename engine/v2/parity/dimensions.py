"""The native-vs-legacy per-dimension comparison policy (spec_ns_c part c).

This module owns the numeric field groups and the one per-dimension
comparator that ``checks/phase4_real.py`` and the nightly parity report
(``engine.v2.ops.native_parity_report``) both run.  The logic moved here from
``checks/phase4_real.py`` byte-for-byte so production code no longer imports
the Phase 4 checker: the checker imports this module instead, and the two
consumers can never drift apart.  The comparison itself is still
``compare_records`` under ``SCORE_RECORD_V1`` -- the comparator core that
moved here from ``engine/v2/diagnosis`` (which re-exports it) -- and no
tolerance rule is re-implemented here.

``compare_dimension`` takes the comparator as an optional keyword so the
Phase 4 checker can keep resolving it through its own module global (its
negative controls plant a blind comparator with
``monkeypatch.setattr(phase4_real, "compare_records", ...)``); every other
caller uses the declared ``engine.v2.diagnosis`` comparator.
"""
from __future__ import annotations

from engine.v2.foundation import content_hash
from engine.v2.parity.receipt import AGREE
from engine.v2.parity.record_comparator import compare_records
from engine.v2.parity.tolerance import SCORE_RECORD_V1

__all__ = [
    "ANALOG_FIELDS",
    "FINANCIAL_FIELDS",
    "FORECAST_FIELDS",
    "GATE_FIELDS",
    "NEVER_RAN_DIMENSIONS",
    "SIMULATION_FIELDS",
    "compare_dimension",
]

FORECAST_FIELDS = (
    "driver_prediction", "driver_p10", "driver_p90",
    "forecast_abs_move", "forecast_p10", "forecast_p90", "forecast_sd",
    "runup_move_prediction", "runup_move_p10", "runup_move_p90",
    "runup_move_scale", "chooser_score",
)
SIMULATION_FIELDS = (
    "exp_pnl_sim", "exp_pnl_model", "exp_pnl_analog",
    "win_sim", "win_model", "win_model_raw", "win_analog",
)
FINANCIAL_FIELDS = (
    "entry_cost_pct", "model_vs_market", "fair_premium_pct",
    "premium_vs_fair", "cost_over_width",
)
#: gate_pass alone used to stand in for the whole gate: two runs could agree
#: on the boolean while disagreeing on the score and threshold that produced
#: it. Compare all three explicitly.
GATE_FIELDS = ("gate_score", "gate_threshold", "gate_pass")
#: ci_low/ci_high/n_analogs previously appeared in no comparison tuple at
#: all. Both sides carry these as a genuine concept (the legacy record's own
#: analog columns; the native side's resolved_request, populated by
#: _execute_analogs in stages.py) — present as a key with a possibly-None
#: value, never structurally absent, so an ordinary field comparison applies
#: with no incomparability marker needed.
ANALOG_FIELDS = ("ci_low", "ci_high", "n_analogs")

#: Dimensions the "never ran" rule (USER DECISION, 2026-09-23) applies to.
#: Forecasts are explicitly excluded -- the forecast band is being built
#: separately and this rule must not touch ``FORECAST_FIELDS`` handling.
#: ``financial_diagnostics`` is also out of scope: both sides COMPUTE it
#: from other fields rather than carrying it, so it has no notion of
#: "stage never ran" independent of the fields this rule already covers.
NEVER_RAN_DIMENSIONS = {
    "simulation": SIMULATION_FIELDS,
    "verdicts": GATE_FIELDS,
    "analogs": ANALOG_FIELDS,
}


def compare_dimension(expected: dict, actual: dict, dimension: str, *,
                      compare_records=compare_records) -> dict:
    """Compare one dimension's expected/actual views under the exact policy.

    The body is the Phase 4 checker's ``_compare_dimension`` moved here
    unchanged; ``compare_records`` is injectable only so the checker's own
    negative controls keep rebinding it through ``checks.phase4_real``.
    """
    comparison = compare_records(
        expected, actual,
        comparison_kind=f"phase4_{dimension}_parity",
        left_ref="frozen_legacy_record",
        right_ref="native_score_record",
        tolerance_policy=SCORE_RECORD_V1,
    )
    return {
        "agree": comparison.verdict == AGREE,
        "finding_fields": sorted(finding.field_path for finding in comparison.findings),
        "receipt": content_hash({
            "dimension": dimension,
            "verdict": comparison.verdict,
            "findings": [finding.field_path for finding in comparison.findings],
        }),
    }
