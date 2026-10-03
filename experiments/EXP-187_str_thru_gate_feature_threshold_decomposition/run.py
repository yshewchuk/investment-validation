#!/usr/bin/env python3
"""EXP-187 — STR-THRU gate promotion: decomposing EXP-184's feature-set and
threshold-rule effects into two separate pre-registered arms.

Run:  python3 experiments/EXP-187_str_thru_gate_feature_threshold_decomposition/run.py

EXP-184 changed the gate's feature set (the champion's registered features ->
plus forecast/analog columns) and its threshold rule (registered threshold ->
per-fold top-20%) in the same comparison, so its measured gain cannot be
attributed to either change alone. This reuses EXP-147's runner (the same
module-load pattern EXP-184 itself uses) with two different
parameterisations instead of copying it:

  * Arm A: the champion's registered features (engine/models/training/gate.py
    FEATURES; no forecast or analog columns), trained per fold with the SAME
    top-20% rule as EXP-184's candidate. Written to this experiment's own
    run_dir (REPORT.md at the experiment root).
  * Arm B: the actually-served champion gate_midfill_str_thru_forecast_analog,
    its own registered threshold, refit per fold -- written to this
    experiment's champion/ subdirectory, never overwriting arm A's report.

See spec.yaml for the full hypothesis, the comparison baselines (EXP-184's
candidate and champion-baseline metrics files), and the threshold
look-ahead caveat on arm B.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "EXP-147_str_thru_gate_promotion_confirmatory_val" / "run.py"
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402

# Evidence (REPORT.md, results/, figures/) follows the configured root, not
# wherever this checkout happens to sit -- paths.ROOT honours
# INVESTING_PLAN_ROOT. SOURCE above stays checkout-relative on purpose: it
# locates EXP-147's sibling script to load, a code location, not evidence.
HERE = paths.ROOT / "experiments" / "EXP-187_str_thru_gate_feature_threshold_decomposition"

spec = importlib.util.spec_from_file_location("exp187_runner", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.HERE = HERE

from engine.models.training import gate as gate_mod  # noqa: E402

ARM_A_FEATURES = list(gate_mod.FEATURES)
ARM_A_GATE_NAME = "STR-THRU_base_features_candidate"
ARM_A_FEATURE_LINE = (
    "engine.models.training.gate (base features): the champion's "
    "registered features (engine/models/training/gate.py FEATURES), no "
    "forecast or analog columns -- isolates the feature-set effect from "
    "EXP-184's threshold-rule change."
)
ARM_A_PROVENANCE_LINE = (
    "ga.build_dataset still joins the forecast and analog columns (the "
    "shared dataset is identical for both arms); this arm simply does not "
    "use those columns for fitting or scoring -- its feature list is the "
    "base set only. EXP-184's candidate (per-fold top-20% on the full "
    "forecast+analog feature set) and this arm (per-fold top-20% on the "
    "base feature set) are evaluated on the identical v2 snapshot, trades, "
    "repricer, walk-forward and MC settings -- see spec.yaml's "
    "comparison_baselines."
)
ARM_B_GATE_ID = "gate_midfill_str_thru_forecast_analog"

if __name__ == "__main__":
    module.main(
        candidate_features=ARM_A_FEATURES,
        candidate_gate_name=ARM_A_GATE_NAME,
        candidate_feature_line=ARM_A_FEATURE_LINE,
        candidate_provenance_line=ARM_A_PROVENANCE_LINE,
        champion_gate_id=ARM_B_GATE_ID,
    )
