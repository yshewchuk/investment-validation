#!/usr/bin/env python3
"""Train, evaluate, and register every champion.

    python3 -m engine.models.training.train_all
    python3 -m engine.models.training.train_all --role size --role gate
    python3 -m engine.models.training.train_all --dry-run

Each model is walk-forward evaluated, refit on everything, saved to
``data/models/``, and registered with the metrics that evaluation produced —
never with a metric copied from a verdict document. The published research
numbers are recorded separately as ``reference``, so a drift between what this
program measures and what the earlier work reported is visible instead of
assumed away.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

from engine import paths
from engine.data import manifest, store
from engine.features import load_panel
from engine.models.registry import (
    ANY_STRATEGY,
    ARTIFACT_DIR,
    ModelArtifact,
    RegistryEntry,
    bucket_residuals,
    load_registry,
    register,
)
from engine.models.training import gate as gate_mod
from engine.models.training import iv_crush as crush_mod
from engine.models.training import implied_t1 as implied_mod
from engine.models.training import size_model as size_mod
from engine.models.training.common import SEED, log

#: What the prior research reported, for comparison only. Recorded in the
#: registry beside the metrics this program actually measured.
REFERENCE = {
    "size": {"oos_r": 0.459, "source": "EXP-040", "note": "OLS+NN blend on true implied"},
    "implied_t1": {"mae_pp": "3.3-4.0", "r": "0.60-0.72", "source": "EXP-043"},
    "gate": {"lift_per_trade": 0.046, "source": "EXP-049", "note": "top-20% on S3 mid fills"},
}


def _events_with_session(years=range(2017, 2027)) -> pd.DataFrame:
    events = store.read_table(
        "earnings_events", years=years, columns=["event_id", "ticker", "event_date", "session"]
    )
    return events[events["session"].notna()].reset_index(drop=True)


def _engine_trades(strategy: str) -> pd.DataFrame:
    trades = store.read_table("trades")
    rows = trades[
        (trades["strategy"] == strategy)
        & (trades["provenance"].astype(str) == "engine.replay")
    ]
    return rows.reset_index(drop=True)


# --------------------------------------------------------------------------
# per-role training
# --------------------------------------------------------------------------


def train_size(*, seed: int = SEED, dry_run: bool = False) -> dict:
    log("=== size: predicted |earnings move| ===")
    panel = load_panel()
    model, result = size_mod.train(panel, seed=seed)
    comparison = size_mod.compare_feature_sets(panel, seed=seed)
    log(
        f"servable vs legacy feature list: r {comparison['servable']['r']:.4f} "
        f"vs {comparison['legacy']['r']:.4f}"
    )

    # Residuals grouped by the decile of the prediction that produced them.
    # The pairing exists only here — `result.frame` carries `pred` beside the
    # target — and is gone by the time an artifact is loaded for scoring, which
    # is why the flat pool was the only thing the scorer could ever offer.
    # EXP-115: takes the shipped 80% interval from 72.8% coverage to 79.3%,
    # 12/13 years, p=0.00049.
    buckets = bucket_residuals(
        result.frame["pred"].to_numpy(dtype=float), result.residuals
    )
    if buckets:
        log(f"residual buckets: {len(buckets['pools'])} deciles, "
            f"{buckets['n']:,} paired residuals, thin={buckets['thin']}")
    else:
        log("residual buckets: sample too small to split; flat pool only")

    artifact = ModelArtifact(
        model=model,
        role="size",
        features=size_mod.FEATURES,
        residuals=result.residuals,
        residual_buckets=buckets,
        target=size_mod.TARGET,
        train_years=tuple(sorted(int(y) for y in panel["year"].dropna().unique())),
        metrics=result.metrics,
        params={"blend": "OLS + MLP(64,32)", "min_prior": size_mod.MIN_PRIOR},
        seed=seed,
        notes=(
            "v1.4 architecture. `or_implied` replaces the legacy `implied_move`, "
            "which cannot be sourced for an unrealized event. v1.4 adds the "
            "absolute-valued inputs promoted by EXP-109: mean_prior_abs_move, "
            "abs_dist_ema, abs_dist_high — the magnitudes behind V-shaped signed "
            "features that the blend's linear half cannot represent."
        ),
    )
    entry = RegistryEntry(
        id="size_v1_4",
        role="size",
        strategy=ANY_STRATEGY,
        artifact=str((ARTIFACT_DIR / "size_v1_4.joblib").relative_to(paths.ROOT)),
        artifact_sha256="",
        features=list(size_mod.FEATURES),
        target=size_mod.TARGET,
        train_window=f"walk-forward, OOS {min(result.years)}-{max(result.years)}",
        train_years=list(result.years),
        eval={
            **{k: v for k, v in result.metrics.items() if k != "by_year"},
            "feature_set_comparison": comparison,
            "reference": REFERENCE["size"],
        },
        champion=True,
        promoted=date.today().isoformat(),
        evidence="reports/phase1_models.md",
        seed=seed,
        notes=artifact.notes,
    )
    return _finalize(artifact, entry, result, dry_run)


def train_implied_t1(*, seed: int = SEED, dry_run: bool = False) -> dict:
    log("=== implied_t1: quoted implied move at the last pre-print close ===")
    events = _events_with_session()
    panel = load_panel()
    dataset = implied_mod.build_dataset(events, panel=panel)
    model, result = implied_mod.train(dataset, seed=seed)

    artifact = ModelArtifact(
        model=model,
        role="implied_t1",
        features=implied_mod.FEATURES,
        residuals=result.residuals,
        target=implied_mod.TARGET,
        train_years=tuple(result.years),
        metrics=result.metrics,
        params={"decision_days": list(implied_mod.DECISION_DAYS)},
        seed=seed,
        notes=(
            "Decision days pooled with days_before_print as a feature. Features "
            "are as-of the decision date only — the panel's market-state block "
            "is read at the close being predicted and would leak."
        ),
    )
    entry = RegistryEntry(
        id="opf_implied_t1_gbm",
        role="implied_t1",
        strategy=ANY_STRATEGY,
        artifact=str((ARTIFACT_DIR / "opf_implied_t1_gbm.joblib").relative_to(paths.ROOT)),
        artifact_sha256="",
        features=list(implied_mod.FEATURES),
        target=implied_mod.TARGET,
        train_window=f"walk-forward, OOS {min(result.years)}-{max(result.years)}",
        train_years=list(result.years),
        eval={
            **{k: v for k, v in result.metrics.items() if k != "by_year"},
            "reference": REFERENCE["implied_t1"],
        },
        champion=True,
        promoted=date.today().isoformat(),
        evidence="reports/phase1_models.md",
        seed=seed,
        notes=artifact.notes,
    )
    return _finalize(artifact, entry, result, dry_run)


def train_iv_crush(*, seed: int = SEED, dry_run: bool = False) -> dict:
    log("=== iv_crush: 30-day implied vol across the print ===")
    panel = load_panel()
    dataset = crush_mod.prepare(panel)
    model, result = crush_mod.train(dataset, seed=seed)

    notes = (
        "Target is SIGNED — iv30 falls at 83.2% of prints — so the Tier-4 "
        "producer declares interval_floor=None; a magnitude model's zero floor "
        "would clip most bands to [0, 0] without inverting one. EXP-128 cleared "
        "MAE, RMSE, year-consistency, decile spread and interval calibration and "
        "FAILED its registered coverage floor (71,864 scored rows against 80,000) "
        "because the arm consumed every numeric panel column; a curated feature "
        "list is the next iteration. Materialised on the user's explicit call "
        "with that shortfall on the record."
    )
    artifact = ModelArtifact(
        model=model,
        role="iv_crush",
        features=crush_mod.FEATURES,
        residuals=result.residuals,
        target=crush_mod.TARGET,
        train_years=tuple(result.years),
        metrics=result.metrics,
        params={"max_gap_days": crush_mod.MAX_GAP_DAYS},
        seed=seed,
        notes=notes,
    )
    entry = RegistryEntry(
        id="iv_crush_v1_gbm",
        role="iv_crush",
        strategy=ANY_STRATEGY,
        artifact=str((ARTIFACT_DIR / "iv_crush_v1_gbm.joblib").relative_to(paths.ROOT)),
        artifact_sha256="",
        features=list(crush_mod.FEATURES),
        target=crush_mod.TARGET,
        train_window=f"walk-forward, OOS {min(result.years)}-{max(result.years)}",
        train_years=list(result.years),
        eval={k: v for k, v in result.metrics.items() if k != "by_year"},
        champion=True,
        promoted=date.today().isoformat(),
        evidence="experiments/EXP-128_what_survives_the_print_a_walk_forward_m",
        seed=seed,
        notes=notes,
        tier="feature",
        produces="pred_iv_crush_30",
    )
    return _finalize(artifact, entry, result, dry_run)


def train_gate(strategy: str, *, seed: int = SEED, dry_run: bool = False) -> dict:
    log(f"=== gate: {strategy} mid-fill return ===")
    trades = _engine_trades(strategy)
    if trades.empty:
        log(f"no engine-replayed trades for {strategy} — skipping. Run engine.build_trades first.")
        return {"role": "gate", "strategy": strategy, "skipped": "no trades"}
    panel = load_panel()
    dataset = gate_mod.build_dataset(trades, panel=panel)
    model, result, threshold = gate_mod.train(dataset, seed=seed)
    yearly = gate_mod.by_year_gate_table(result, threshold)
    log("gate by year:\n" + yearly.to_string(index=False))

    slug = strategy.lower().replace("-", "_")
    artifact = ModelArtifact(
        model=model,
        role="gate",
        features=gate_mod.FEATURES,
        residuals=result.residuals,
        target=gate_mod.TARGET,
        train_years=tuple(result.years),
        metrics=result.metrics,
        params={"alpha": gate_mod.GATE_ALPHA, "top_fraction": gate_mod.TOP_FRACTION},
        seed=seed,
        notes=(
            f"Trained on engine.replay {strategy} trades at alpha=0.5 over an "
            "unselected event universe."
        ),
    )
    entry = RegistryEntry(
        id=f"gate_midfill_{slug}",
        role="gate",
        strategy=strategy,
        artifact=str((ARTIFACT_DIR / f"gate_midfill_{slug}.joblib").relative_to(paths.ROOT)),
        artifact_sha256="",
        features=list(gate_mod.FEATURES),
        target=gate_mod.TARGET,
        train_window=f"walk-forward, OOS {min(result.years)}-{max(result.years)}",
        train_years=list(result.years),
        eval={
            **{k: v for k, v in result.metrics.items() if k != "by_year"},
            "by_year": yearly.to_dict("records"),
            "reference": REFERENCE["gate"],
        },
        champion=True,
        promoted=date.today().isoformat(),
        evidence="reports/phase1_models.md",
        seed=seed,
        threshold=float(threshold),
        notes=artifact.notes,
    )
    return _finalize(artifact, entry, result, dry_run)


def _finalize(artifact: ModelArtifact, entry: RegistryEntry, result, dry_run: bool) -> dict:
    summary = {
        "id": entry.id,
        "role": entry.role,
        "strategy": entry.strategy,
        "metrics": {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in result.metrics.items()
            if not isinstance(v, (list, dict))
        },
        "n_residuals": int(artifact.residuals.size),
        "oos_years": list(result.years),
    }
    if dry_run:
        log(f"--dry-run: not writing {entry.id}")
        return summary

    path = ARTIFACT_DIR / Path(entry.artifact).name
    digest = artifact.save(path)
    entry.artifact_sha256 = digest
    register(entry)
    log(f"registered {entry.id} → {path.name} ({digest[:12]}…)")
    summary["artifact_sha256"] = digest
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

ROLE_ORDER = ("size", "implied_t1", "gate")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--role", action="append", choices=ROLE_ORDER, help="train only this role")
    ap.add_argument(
        "--gate-strategy", action="append", default=None,
        help="strategies to fit a gate for (default: STR-THRU, STR-RUNUP)",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true", help="train and evaluate, register nothing")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    roles = args.role or list(ROLE_ORDER)
    # CAL-P is deliberately absent: its exact spec has never been backtested
    # (Phase 2 backlog 1-2), the scorer refuses to score it, and a gate for a
    # structure nobody may trade would be evidence for a decision that cannot
    # be made.
    gate_strategies = args.gate_strategy or ["STR-THRU", "STR-RUNUP"]

    started = time.time()
    report = {"seed": args.seed, "generated_at": pd.Timestamp.now("UTC").isoformat(), "models": []}
    for role in roles:
        if role == "size":
            report["models"].append(train_size(seed=args.seed, dry_run=args.dry_run))
        elif role == "implied_t1":
            report["models"].append(train_implied_t1(seed=args.seed, dry_run=args.dry_run))
        elif role == "gate":
            for strategy in gate_strategies:
                report["models"].append(
                    train_gate(strategy, seed=args.seed, dry_run=args.dry_run)
                )

    if not args.dry_run:
        problems = load_registry().validate()
        report["registry_problems"] = problems
        if problems:
            print("\nREGISTRY PROBLEMS:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
        else:
            log("registry validates clean")
        report["snapshot"] = manifest.snapshot_hash()

    report["elapsed_s"] = round(time.time() - started, 1)
    log(f"done in {report['elapsed_s']:.0f}s")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str))
    return 1 if report.get("registry_problems") else 0


if __name__ == "__main__":
    sys.exit(main())
