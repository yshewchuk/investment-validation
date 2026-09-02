#!/usr/bin/env python3
"""EXP-118 — gate retrain on the expanded replay universe.

    python3 experiments/EXP-118_gate_retrain_on_the_expanded_replay_univ/run.py

Dataset-only retrain of the two mid-fill gates. Everything about the model is
held to the champion's spec — same FEATURES, same HistGBM hyperparameters
(gate_mod.fit), same top-20% threshold rule — and only the training universe
changes: the trades table now carries ~19.2k STR-THRU and ~6.6k STR-RUNUP
mid-fill replay trades spanning 2018-2026 against the 11,080 / 3,041 rows
the champions were measured on.

The data path mirrors engine.models.training.train_all.train_gate exactly
(same trades filter, same build_dataset call), so a difference between this
experiment and the registry eval is the dataset and nothing else.

Arms (all pre-registered in spec.yaml before any run):

  str_thru_2020   PRIMARY     STR-THRU,  first_test_year=2020 (champion-comparable)
  str_runup_2020  secondary   STR-RUNUP, first_test_year=2020 (champion-comparable)
  str_thru_2018   secondary   STR-THRU,  first_test_year=2018 (extended window)
  str_runup_2018  secondary   STR-RUNUP, first_test_year=2018 (extended window)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.data import store  # noqa: E402
from engine.features import load_panel  # noqa: E402
from engine.models.registry import load_registry  # noqa: E402
from engine.models.training import gate as gate_mod  # noqa: E402
from engine.models.training.common import SEED  # noqa: E402
from experiments import lib  # noqa: E402

HERE = Path(__file__).resolve().parent

ARMS = [
    ("str_thru_2020", "STR-THRU", 2020),
    ("str_runup_2020", "STR-RUNUP", 2020),
    ("str_thru_2018", "STR-THRU", 2018),
    ("str_runup_2018", "STR-RUNUP", 2018),
]


def log(msg: str) -> None:
    print(f"[EXP-118 {datetime.now():%H:%M:%S}] {msg}", flush=True)


def engine_trades(strategy: str) -> pd.DataFrame:
    """Identical filter to engine.models.training.train_all._engine_trades."""
    trades = store.read_table("trades")
    rows = trades[
        (trades["strategy"] == strategy)
        & (trades["provenance"].astype(str) == "engine.replay")
    ]
    return rows.reset_index(drop=True)


def champion_baseline() -> dict:
    out = {}
    for entry in load_registry().entries:
        if entry.role == "gate" and entry.strategy in ("STR-THRU", "STR-RUNUP"):
            out[entry.strategy] = {
                "id": entry.id,
                "n": entry.eval.get("n"),
                "gate_lift": entry.eval.get("gate_lift"),
                "gated_mean_ret": entry.eval.get("gated_mean_ret"),
                "base_mean_ret": entry.eval.get("base_mean_ret"),
                "gated_win_rate": entry.eval.get("gated_win_rate"),
                "threshold": entry.eval.get("threshold"),
                "r": entry.eval.get("r"),
                "decile_spread": entry.eval.get("decile_spread"),
            }
    return out


def run_arm(name: str, strategy: str, first_test_year: int, panel) -> dict:
    started = time.time()
    log(f"arm {name}: loading trades for {strategy}")
    trades = engine_trades(strategy)
    log(f"arm {name}: {len(trades):,} replay trades; building gate dataset")
    dataset = gate_mod.build_dataset(trades, panel=panel)
    log(f"arm {name}: training, first_test_year={first_test_year}")
    model, result, threshold = gate_mod.train(
        dataset, seed=SEED, first_test_year=first_test_year
    )
    yearly = gate_mod.by_year_gate_table(result, threshold)
    m = result.metrics
    arm = {
        "strategy": strategy,
        "first_test_year": first_test_year,
        "n_trades": int(len(trades)),
        "n_dataset": int(len(dataset)),
        "n_complete": int(len(result.frame)),
        "years": list(result.years),
        "r": m.get("r"),
        "mae": m.get("mae"),
        "rmse": m.get("rmse"),
        "bias": m.get("bias"),
        "decile_spread": m.get("decile_spread"),
        "threshold": threshold,
        "base_mean_ret": m.get("base_mean_ret"),
        "gated_mean_ret": m.get("gated_mean_ret"),
        "gate_lift": m.get("gate_lift"),
        "base_win_rate": m.get("base_win_rate"),
        "gated_win_rate": m.get("gated_win_rate"),
        "n_passed": m.get("n_passed"),
        "by_year": yearly.to_dict(orient="records"),
        "elapsed_s": round(time.time() - started, 1),
    }
    log(
        f"arm {name}: lift {arm['gate_lift']:+.4f} "
        f"(base {arm['base_mean_ret']:+.4f} -> gated {arm['gated_mean_ret']:+.4f}), "
        f"win {arm['gated_win_rate']:.3f}, n={arm['n_complete']:,}, {arm['elapsed_s']}s"
    )
    return arm


def judge(spec: dict, arms: dict) -> dict:
    criteria = spec.get("success_criteria", {})
    verdicts = {}
    for arm_name, crit in criteria.items():
        arm = arms.get(arm_name)
        if arm is None:
            verdicts[arm_name] = {"verdict": "MISSING", "checks": {}}
            continue
        checks = {
            "gate_lift": (arm["gate_lift"] or -9) >= crit["gate_lift_at_least"],
            "gated_win_rate": (arm["gated_win_rate"] or -9) >= crit["gated_win_rate_at_least"],
            "n_complete": arm["n_complete"] > crit["n_complete_above"],
        }
        verdicts[arm_name] = {
            "verdict": "CLEARS" if all(checks.values()) else "does not clear",
            "checks": checks,
            "criteria": crit,
        }
    return verdicts


def write_report(spec: dict, arms: dict, baseline: dict, verdicts: dict, snapshot: str) -> Path:
    lines = [
        f"# {spec['id']} — {spec['title']}",
        "",
        f"Generated by run.py on {datetime.now(tz=timezone.utc):%Y-%m-%d %H:%M UTC}.",
        "",
        "## Hypothesis (pre-registered)",
        "",
        spec["hypothesis"],
        "",
        "## Champion baselines (registry.json)",
        "",
        "| strategy | champion | n | gate lift | gated mean | gated win |",
        "|---|---|---|---|---|---|",
    ]
    for strat, b in baseline.items():
        lines.append(
            f"| {strat} | {b['id']} | {b['n']:,} | {b['gate_lift']:+.4f} "
            f"| {b['gated_mean_ret']:+.4f} | {b['gated_win_rate']:.3f} |"
        )
    lines += ["", "## Arms", "",
              "| arm | status | n complete | r | lift | gated mean | gated win | threshold | verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
    arm_status = {
        "str_thru_2020": "primary",
        "str_runup_2020": "secondary (co-arm)",
        "str_thru_2018": "secondary (extended)",
        "str_runup_2018": "secondary (extended)",
    }
    for name, arm in arms.items():
        v = verdicts.get(name, {}).get("verdict", "not judged (secondary)")
        lines.append(
            f"| {name} | {arm_status.get(name, 'secondary')} | {arm['n_complete']:,} "
            f"| {arm['r']:.3f} | {arm['gate_lift']:+.4f} | {arm['gated_mean_ret']:+.4f} "
            f"| {arm['gated_win_rate']:.3f} | {arm['threshold']:.4f} | {v} |"
        )
    lines += ["", "## Per-year detail", ""]
    for name, arm in arms.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| year | n | base mean | n passed | gated mean | gated win |")
        lines.append("|---|---|---|---|---|---|")
        for row in arm["by_year"]:
            gm = f"{row['gated_mean']:+.4f}" if row["gated_mean"] is not None else "–"
            gw = f"{row['gated_win']:.3f}" if row["gated_win"] is not None else "–"
            lines.append(f"| {row['year']} | {row['n']:,} | {row['base_mean']:+.4f} "
                         f"| {row['n_passed']:,} | {gm} | {gw} |")
        lines.append("")
    lines += [
        "## Verdict",
        "",
    ]
    for arm_name, v in verdicts.items():
        lines.append(f"- **{arm_name}**: {v['verdict']} — checks {v['checks']}")
    lines += [
        "",
        "## Promotion path",
        "",
        spec["promotion_path"],
        "",
        f"Data snapshot: `{snapshot}`",
    ]
    path = HERE / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    spec = lib.load_spec(HERE / "spec.yaml")
    baseline = champion_baseline()
    log("loading panel once for all arms")
    panel = load_panel()

    arms = {}
    for name, strategy, fty in ARMS:
        arms[name] = run_arm(name, strategy, fty, panel)

    verdicts = judge(spec, arms)
    results = {
        "spec_hash": lib.spec_hash(spec),
        "snapshot": spec.get("data_snapshot"),
        "baseline": baseline,
        "arms": arms,
        "verdicts": verdicts,
    }
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1, default=str))
    for name, arm in arms.items():
        pd.DataFrame(arm["by_year"]).to_csv(out / f"by_year_{name}.csv", index=False)

    report = write_report(spec, arms, baseline, verdicts, str(spec.get("data_snapshot")))
    lib.record_evaluation(
        HERE,
        spec,
        {"headline": {"mean": arms["str_thru_2020"]["gated_mean_ret"]}},
    )

    print()
    print(f"{'arm':14s} {'n':>8} {'r':>7} {'lift':>8} {'gated':>8} {'win':>6}  verdict")
    for name, arm in arms.items():
        v = verdicts.get(name, {}).get("verdict", "-")
        print(f"{name:14s} {arm['n_complete']:>8,} {arm['r']:>7.3f} "
              f"{arm['gate_lift']:>+8.4f} {arm['gated_mean_ret']:>+8.4f} "
              f"{arm['gated_win_rate']:>6.3f}  {v}")
    print(f"\nreport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
