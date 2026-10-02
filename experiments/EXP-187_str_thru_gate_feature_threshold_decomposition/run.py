#!/usr/bin/env python3
"""EXP-187 — STR-THRU gate promotion: decomposing EXP-184's feature-set and
threshold-rule effects into two separate pre-registered arms.

Run:  python3 experiments/EXP-187_str_thru_gate_feature_threshold_decomposition/run.py
Check (after both this experiment's arms and EXP-184 have run):
      python3 experiments/EXP-187_str_thru_gate_feature_threshold_decomposition/run.py --check-interaction

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
look-ahead caveat on arm B. ``--check-interaction`` computes spec.yaml's
interaction_check from the four arms' metrics files already on disk -- it
never runs anything, so it is also the way to check this before the heavy
run happens (it will just report what is missing).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments" / "EXP-147_str_thru_gate_promotion_confirmatory_val" / "run.py"
sys.path.insert(0, str(ROOT))

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
    "No forecast or analog columns are read for this arm. EXP-184's "
    "candidate (per-fold top-20% on the full forecast+analog feature set) "
    "and this arm (per-fold top-20% on the base feature set) are evaluated "
    "on the identical v2 snapshot, trades, repricer, walk-forward and MC "
    "settings -- see spec.yaml's comparison_baselines."
)
ARM_B_GATE_ID = "gate_midfill_str_thru_forecast_analog"

EXP184_DIR = ROOT / "experiments" / "EXP-184_str_thru_gate_promotion_confirmatory_val_registered"
CANDIDATE_METRICS = EXP184_DIR / "results" / "metrics_b8c322b24feb.json"
CHAMPION_BASELINE_METRICS = EXP184_DIR / "champion" / "results" / "metrics_3835b6283725.json"

# Interaction consistency check (judgement call, not a fixed magnitude cutoff
# meant to carry strategy meaning -- see spec.yaml's interaction_check.criterion).
RATIO_TOLERANCE = 5.0  # neither same-axis estimate may be more than this many times the other
NEGLIGIBLE = 1e-6  # guards only a sign flip from float noise when both estimates are ~0


def _latest_metrics(results_dir: Path) -> Path | None:
    files = sorted(results_dir.glob("metrics_*.json")) if results_dir.exists() else []
    return files[-1] if files else None


def _headline(path: Path) -> dict:
    return json.loads(Path(path).read_text())["headline"]


def _agrees(e1: float, e2: float) -> bool:
    """Additive (non-interacting) if both estimates are negligible, or they
    share a sign and neither's magnitude is more than RATIO_TOLERANCE times
    the other's -- the same effect measured twice, not two phenomena that
    happen to share a sign."""
    if abs(e1) <= NEGLIGIBLE and abs(e2) <= NEGLIGIBLE:
        return True
    if (e1 > 0) != (e2 > 0):
        return False
    lo, hi = sorted((abs(e1), abs(e2)))
    return hi == 0 or (hi / lo if lo else float("inf")) <= RATIO_TOLERANCE


def check_interaction() -> int:
    """Reads the four arms' metrics files already on disk and reports
    spec.yaml's interaction_check verdict. Never runs anything; if either of
    this experiment's own arms has not been run yet, says so and exits
    non-zero rather than guessing."""
    arm_a_metrics = _latest_metrics(HERE / "results")
    arm_b_metrics = _latest_metrics(HERE / "champion" / "results")
    named = [
        ("EXP-184 candidate", CANDIDATE_METRICS),
        ("EXP-184 champion_baseline", CHAMPION_BASELINE_METRICS),
        ("this experiment's arm A", arm_a_metrics),
        ("this experiment's arm B", arm_b_metrics),
    ]
    missing = [name for name, p in named if p is None or not Path(p).exists()]
    if missing:
        print("interaction_check: cannot run yet -- missing: " + ", ".join(missing))
        return 1

    candidate, champion_baseline, arm_a, arm_b = (
        _headline(p) for _, p in named
    )
    ok = True
    for metric in ("cagr", "sharpe_trade"):
        c, cb, a, b = (candidate.get(metric), champion_baseline.get(metric),
                       arm_a.get(metric), arm_b.get(metric))
        if None in (c, cb, a, b):
            print(f"interaction_check[{metric}]: a required headline value is missing")
            ok = False
            continue
        pairs = [
            ("feature effect", c - a, b - cb),
            ("threshold effect", c - b, a - cb),
        ]
        for label, e1, e2 in pairs:
            additive = _agrees(e1, e2)
            ok = ok and additive
            print(f"interaction_check[{metric}][{label}]: {e1:+.4f} vs {e2:+.4f} "
                  f"-> {'ADDITIVE' if additive else 'INTERACTION'}")
    return 0 if ok else 2


if __name__ == "__main__":
    if "--check-interaction" in sys.argv:
        raise SystemExit(check_interaction())
    module.main(
        candidate_features=ARM_A_FEATURES,
        candidate_gate_name=ARM_A_GATE_NAME,
        candidate_feature_line=ARM_A_FEATURE_LINE,
        candidate_provenance_line=ARM_A_PROVENANCE_LINE,
        champion_gate_id=ARM_B_GATE_ID,
    )
