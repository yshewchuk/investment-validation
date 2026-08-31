#!/usr/bin/env python3
"""EXP-108 — Absolute-valued features for the size model.

    python3 experiments/EXP-108_absolute_valued_features_for_the_size_mo/run.py
    python3 .../run.py --stage 1        # model accuracy only (minutes)

Stage 1 asks whether the candidate feature sets predict |move| better than the
incumbent, walk-forward, out of sample. Stage 2 — which only runs if stage 1
clears the pre-registered thresholds — asks the question that actually matters:
whether the strategy built on the better model makes more money.

The two are separate on purpose. A driver model that predicts |move| more
accurately does not automatically produce a better trade: the prediction is
pushed through a payoff map and compared against a premium, and a model can win
on MAE while moving no decision. Promoting on stage 1 alone would be exactly
the mistake the champion/challenger protocol exists to prevent.

Every arm gets its own ledger row. The headline is the primary spec only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.features import load_panel  # noqa: E402
from engine.models.training import size_model  # noqa: E402
from engine.models.training.common import walk_forward  # noqa: E402
from experiments import lib  # noqa: E402

HERE = Path(__file__).resolve().parent


def derive(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Add the pre-registered absolute-valued columns.

    Defined in spec.yaml rather than here, so the definition was registered
    before any result was seen.
    """
    out = frame
    for name, rule in (spec.get("derived_features") or {}).items():
        source = rule[len("abs("):-1] if rule.startswith("abs(") else None
        if source is None:
            raise ValueError(f"unsupported derivation {rule!r} for {name}")
        if source not in out.columns:
            raise KeyError(f"{name} needs {source}, which the panel does not carry")
        out = out.assign(**{name: pd.to_numeric(out[source], errors="coerce").abs()})
    return out


def arm_features(base: tuple[str, ...], arm: dict) -> list[str]:
    features = [f for f in base if f not in (arm.get("features_removed") or [])]
    for name in arm.get("features_added") or []:
        if name not in features:
            features.append(name)
    return features


def common_rows(frame: pd.DataFrame, feature_sets, target: str) -> pd.Index:
    """Rows usable by EVERY arm, so the arms are compared on one sample.

    ``walk_forward`` drops rows with any missing feature. An arm carrying an
    extra input would therefore be scored on a different — and possibly
    easier — subset than the incumbent, and the difference between them would
    partly be the sample rather than the model.
    """
    needed = sorted({f for fs in feature_sets for f in fs} | {target})
    values = frame[needed].apply(pd.to_numeric, errors="coerce")
    return frame.index[np.isfinite(values.to_numpy(dtype=float)).all(axis=1)]


def score_arm(data: pd.DataFrame, features, target: str, *, first_test_year: int) -> dict:
    started = time.time()
    result = walk_forward(
        data, features, target, size_model.fit, first_test_year=first_test_year
    )
    metrics = dict(result.metrics)
    metrics["elapsed_s"] = round(time.time() - started, 1)
    metrics["n_features"] = len(features)
    return metrics


def verdict(candidate: dict, incumbent: dict, target: dict) -> dict:
    """Apply the pre-registered thresholds. No judgement calls at run time."""
    r_gain = float(candidate.get("r", np.nan)) - float(incumbent.get("r", np.nan))
    mae_gain = float(incumbent.get("mae", np.nan)) - float(candidate.get("mae", np.nan))
    passes = (
        np.isfinite(r_gain)
        and np.isfinite(mae_gain)
        and r_gain >= target["stage_1_min_r_gain"]
        and mae_gain >= target["stage_1_min_mae_gain_pp"]
        and mae_gain >= -target["stage_1_max_mae_regression"]
    )
    return {
        "r_gain": round(r_gain, 5),
        "mae_gain_pp": round(mae_gain, 5),
        "clears_stage_1": bool(passes),
        "reason": (
            "clears the pre-registered thresholds"
            if passes
            else f"r gain {r_gain:+.5f} (needs >= {target['stage_1_min_r_gain']}), "
                 f"MAE gain {mae_gain:+.5f}pp (needs >= {target['stage_1_min_mae_gain_pp']})"
        ),
    }


def run_stage_1(spec: dict) -> dict:
    panel = derive(load_panel(), spec)
    data = size_model.prepare(panel)
    base = tuple(size_model.FEATURES)
    target = spec["target"]
    first_test_year = int(spec["evaluation"]["stage_1"]["first_test_year"])

    arms = [dict(spec["primary_spec"], primary=True)]
    arms += [dict(a, primary=False) for a in spec["grid"]["arms"]]
    feature_sets = [tuple(base)] + [tuple(arm_features(base, a)) for a in arms]

    keep = common_rows(data, feature_sets, target)
    data = data.loc[keep]
    print(f"[EXP-108] {len(data):,} rows usable by every arm", flush=True)

    out = {"rows": int(len(data)), "arms": {}}
    incumbent = score_arm(data, list(base), target, first_test_year=first_test_year)
    out["incumbent"] = incumbent
    print(f"  incumbent          r={incumbent['r']:.4f} mae={incumbent['mae']:.4f}", flush=True)

    for arm in arms:
        features = arm_features(base, arm)
        metrics = score_arm(data, features, target, first_test_year=first_test_year)
        call = verdict(metrics, incumbent, spec["promotion_target"])
        out["arms"][arm["arm"]] = {
            "primary": arm["primary"], "features": features,
            "metrics": metrics, "verdict": call,
        }
        flag = "PRIMARY" if arm["primary"] else "       "
        print(f"  {flag} {arm['arm']:32s} r={metrics['r']:.4f} "
              f"({call['r_gain']:+.4f})  mae={metrics['mae']:.4f} "
              f"({call['mae_gain_pp']:+.4f})  {'PASS' if call['clears_stage_1'] else 'no'}",
              flush=True)
    return out


def run_control(spec: dict) -> dict:
    """The falsification test. A GBM represents a V natively, so this arm should
    be neutral; if it is not, the linearity mechanism is wrong."""
    from engine.models.training import implied_t1
    from engine.models.training.train_all import _events_with_session

    control = spec["control"]
    panel = derive(load_panel(), spec)
    data = implied_t1.build_dataset(_events_with_session(), panel=panel)
    base = tuple(implied_t1.FEATURES)
    features = arm_features(base, control)
    target = implied_t1.TARGET

    keep = common_rows(data, [base, tuple(features)], target)
    data = data.loc[keep]
    incumbent = score_arm(data, list(base), target, first_test_year=2015)
    candidate = score_arm(data, features, target, first_test_year=2015)
    r_gain = candidate["r"] - incumbent["r"]
    print(f"  CONTROL {control['incumbent']:26s} r gain {r_gain:+.4f} "
          f"(expected {control['expected']})", flush=True)
    return {
        "incumbent": incumbent, "candidate": candidate,
        "r_gain": round(float(r_gain), 5),
        "falsifies_mechanism": bool(r_gain >= spec["promotion_target"]["stage_1_min_r_gain"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2))
    ap.add_argument("--no-control", action="store_true")
    args = ap.parse_args()

    spec = lib.load_spec(HERE / "spec.yaml")
    results = {"spec_hash": lib.spec_hash(spec), "stage_1": run_stage_1(spec)}
    if not args.no_control:
        results["control"] = run_control(spec)

    primary = results["stage_1"]["arms"][spec["primary_spec"]["arm"]]
    results["headline"] = {
        "arm": spec["primary_spec"]["arm"],
        "clears_stage_1": primary["verdict"]["clears_stage_1"],
        "note": (
            "Stage 2 is the promotion gate and has not run. A model that "
            "predicts |move| better has not yet been shown to make a better "
            "trade."
        ),
    }
    if args.stage == 2 and primary["verdict"]["clears_stage_1"]:
        results["stage_2"] = {
            "status": "not implemented",
            "next": (
                "Rescore STR-THRU with the candidate through engine.replay + "
                "engine.evaluate and compare oos_mean_mid, sharpe_trade and "
                "mc_p_loss_5pct against the champion."
            ),
        }

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "stage1_metrics.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"\nwrote {out / 'stage1_metrics.json'}")
    print("headline:", json.dumps(results["headline"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
