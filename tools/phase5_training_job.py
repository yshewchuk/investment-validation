#!/usr/bin/env python3
"""P5-3: run one current training recipe over the real data, supervised.

The explicit training job entry point. It builds the recipe's dataset with
the SAME legacy builders the legacy trainers call (so membership is legacy's
by construction), adds the two columns the receipts need — the label
availability date and, for upstream Tier-4 inputs, the fold lineage — and
hands the frame to :func:`engine.v2.models.training.run_training_job`.

Heavy: every recipe loads the Tier-3 panel; the T-14 recipes also read
``daily_market``; ``gate:STR-THRU`` builds a legacy Scorer for its analog
columns. Run it under ``tools/bounded_run.py`` only, one at a time. Start
with ``--plan-only`` (receipts, no fitting). Writes only under ``--out``,
which may not be inside ``data/``. Prints names, counts, dates and codes —
never a prediction, target or PnL value.

Usage::

    python3 tools/phase5_training_job.py --list
    python3 tools/bounded_run.py --max-rss-gb 5.5 -- python3 -u tools/phase5_training_job.py \\
        --recipe size:*:champion --out /root/p5-3-runs/size-champion --plan-only

Calibration recipes (``*:calibration``) need the request's fill alpha and
one or more evidence cutoffs (legacy ``Scorer.payoff``/``recalibration`` key
``(strategy, alpha, cutoff)``); each cutoff is one fold::

    python3 tools/bounded_run.py --max-rss-gb 2.5 -- python3 -u tools/phase5_training_job.py \\
        --recipe payoff_line:STR-THRU:calibration --alpha 0.5 --cutoff 2026-09-18 \\
        --out /root/p5-3-runs/payoff-line-str-thru

Frozen P5-4 states (``--state``) write one frozen-state JSON under ``--out``
that ``tools/phase5_prepare_release.py --frozen-state`` accepts::

    python3 -u tools/phase5_training_job.py --state paired_residual_pool --out DIR

Dataset builders and the legacy selection each mirrors:
``tools/phase5_datasets.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine.v2.models.training import (  # noqa: E402
    RecipeKey,
    TrainingRefused,
    current_recipes,
    run_training_job,
)


def _key(text: str) -> RecipeKey:
    role, strategy, output = text.split(":")
    return RecipeKey(role, strategy, output)


def _next_session(dates) -> pd.Series:
    """Upper bound for a print's realization close: the next business day.

    Business days, not the exchange calendar, so a holiday can put this one
    day early; the recipes' 5-day allowance absorbs that and a planted label
    (months late) is still refused.
    """
    values = pd.to_datetime(pd.Series(dates)).dt.normalize().to_numpy("datetime64[D]")
    return pd.Series(np.busday_offset(values, 1, roll="forward").astype("datetime64[ns]"))


def _tier4_lineage(frame: pd.DataFrame, recipe) -> pd.DataFrame:
    deps = [d for d in recipe.upstream if d.lineage == "tier4_monthly_oos"]
    if not deps:
        return frame
    from engine.data.features import tier4

    forecasts = tier4.load_forecasts()
    if forecasts is None:
        raise SystemExit("tier4_forecasts is missing; lineage cannot be attached")
    cols = ["ticker", "event_date", *[c for d in deps for c in (d.fold_column, d.model_id_column)]]
    lineage = forecasts[cols].drop_duplicates(["ticker", "event_date"], keep="last").copy()
    lineage["event_date"] = pd.to_datetime(lineage["event_date"]).dt.normalize()
    out = frame.drop(columns=[c for c in cols[2:] if c in frame.columns])
    out["event_date"] = pd.to_datetime(out["event_date"]).dt.normalize()
    return out.merge(lineage, on=["ticker", "event_date"], how="left", validate="many_to_one")


def build_dataset(recipe, *, pairs_path=None) -> pd.DataFrame:
    role, strategy, output = recipe.key.role, recipe.key.strategy, recipe.key.output
    if role in ("payoff_line", "payoff_surface"):
        from tools import phase5_datasets

        return phase5_datasets.payoff_trades()
    if role == "recalibration_map":
        from tools import phase5_datasets

        return phase5_datasets.recalibration_pairs(pairs_path)

    from engine.features import load_panel
    from engine.models.training import train_all

    if role == "size":
        frame = load_panel()
        frame["label_available_at"] = _next_session(frame["date"]).to_numpy()
        return frame
    if role == "iv_crush":
        from engine.models.training import iv_crush

        frame = iv_crush.prepare(load_panel())
        # crush_frame drops the post-print date; MAX_GAP_DAYS bounds it.
        frame["label_available_at"] = (pd.to_datetime(frame["date"])
                                       + pd.Timedelta(days=iv_crush.MAX_GAP_DAYS)).to_numpy()
        return frame
    if role in ("implied_t1", "runup_move"):
        from engine.data.features import tier4

        panel = load_panel()
        if output == "tier4_monthly":  # tier4.{im_t1,runup_move}_feature_model().prepare
            factory = tier4.im_t1_feature_model if role == "implied_t1" else tier4.runup_move_feature_model
            return factory().prepare(panel).rename(columns={"date": "event_date"})
        # train_all.train_implied_t1 / train_runup_move, exactly (daily_market read inside).
        from engine.models.training import implied_t1, runup_move

        builder = implied_t1.build_dataset if role == "implied_t1" else runup_move.build_dataset
        return builder(train_all._events_with_session(), panel=panel)
    if role == "gate":
        trades = train_all._engine_trades(strategy)
        if "pred_abs_move" in recipe.features:
            from engine.models.training import gate_forecast_analog

            frame = gate_forecast_analog.build_dataset(trades)
        else:
            from engine.models.training import gate

            frame = gate.build_dataset(trades, panel=load_panel())
        return _tier4_lineage(frame, recipe)
    if role == "chooser":
        from engine.models.training import chooser

        frame, _target, _features = chooser.build_dataset()
        return _tier4_lineage(frame, recipe)
    raise SystemExit(f"{recipe.key.label()}: no dataset builder for this recipe")


STATES = tuple(f"driver_residual_pool:{role}" for role in ("size", "implied_t1", "runup_move")) + (
    "paired_residual_pool",)


def _guard(where: str) -> None:
    """Both no-fit switches, before any row is read (as ``run_training_job``)."""
    from engine.models.no_fit import forbid_fitting as legacy_forbid
    from engine.v2.models.no_fit import forbid_fitting as v2_forbid

    legacy_forbid(where)
    v2_forbid(where)


def build_state(member: str, *, cutoff=None, ticker_chunk: int = 1000, registry=None,
                paired_inputs=None):
    """``(artifact | None, summary)`` for one frozen-state catalog member.

    ``registry``/``paired_inputs`` are injectable for tests; by default the
    real registry and :func:`tools.phase5_datasets.paired_pool_inputs`.
    """
    from engine.v2.models.lineage import DataDependency, Lineage
    from engine.v2.models.training.residuals import (
        build_paired_residual_pool_artifact,
        freeze_stored_driver_residual_pool,
    )
    from tools import phase5_datasets as data

    _guard(f"tools.phase5_training_job.build_state:{member}")
    if member.startswith("driver_residual_pool:"):
        role = member.split(":", 1)[1]
        pool = data.champion_driver_pool(role, registry=registry)
        lineage = Lineage(data=(DataDependency(
            table=f"model_artifact:{pool['model_id']}",
            keys=(str(pool["artifact_sha256"]),)),))
        artifact = freeze_stored_driver_residual_pool(
            pool["residuals"], pool["buckets"], role=role, model_id=pool["model_id"],
            lineage=lineage)
        summary = {"model_id": pool["model_id"], "artifact_sha256": pool["artifact_sha256"],
                   "n_residuals": int(np.isfinite(pool["residuals"]).sum()),
                   "n_buckets": 0 if not pool["buckets"] else len(pool["buckets"]["pools"])}
        return artifact, summary
    if member != "paired_residual_pool":
        raise SystemExit(f"unknown state {member!r}; known: {', '.join(STATES)}")
    inputs = paired_inputs if paired_inputs is not None else data.paired_pool_inputs(
        ticker_chunk=ticker_chunk)
    bound = None if cutoff is None else str(pd.Timestamp(cutoff).date())
    lineage = Lineage(data=tuple(DataDependency(table=t, end_exclusive=bound) for t in (
        "tier4.forecasts", "tier3.panel", "tier2.earnings_events", "tier2.daily_market")))
    counts = {name: int(len(inputs[name])) for name in ("forecasts", "outcomes", "crush")}
    artifact = build_paired_residual_pool_artifact(
        inputs["forecasts"].to_dict("records"), inputs["outcomes"].to_dict("records"),
        inputs["crush"].to_dict("records"), move_model_id=inputs["move_model_id"],
        crush_model_id=inputs["crush_model_id"], cutoff=bound, lineage=lineage)
    summary = {"move_model_id": inputs["move_model_id"],
               "crush_model_id": inputs["crush_model_id"], "cutoff": bound,
               "input_rows": counts, "n_rows": len(artifact.rows)}
    return artifact, summary


def state_file_name(member: str) -> str:
    return member.replace(":", "__") + ".json"


def run_state_job(member: str, out: Path, *, plan_only: bool = False, **kwargs) -> dict:
    """Build one frozen state into ``out``.

    ``plan_only`` builds it and writes only the summary (counts, hash).
    A rerun keeps an identical existing file and refuses a different one.
    """
    from engine.v2.models.frozen_state import serialize_frozen_state
    from tools import phase5_datasets as data

    artifact, summary = build_state(member, **kwargs)
    summary = {"state": member, "plan_only": bool(plan_only), **summary}
    if member == "paired_residual_pool" and kwargs.get("paired_inputs") is None:
        from engine import paths

        summary["sources"] = {"tier4.forecasts": data.file_digest(paths.TIER4),
                              "tier3.panel": data.file_digest(paths.PANEL)}
    out.mkdir(parents=True, exist_ok=True)
    if artifact is None:
        summary["status"] = "empty"
    else:
        path = out / state_file_name(member)
        payload = serialize_frozen_state(artifact)
        summary.update(content_hash=artifact.content_hash, file=path.name, bytes=len(payload))
        if plan_only:
            summary["status"] = "planned"
        elif path.exists():
            if path.read_bytes() != payload:
                raise SystemExit(f"RESUME_MISMATCH: {path.name} exists with different content")
            summary["status"] = "resumed"
        else:
            tmp = path.with_name(path.name + ".partial")
            tmp.write_bytes(payload)
            tmp.replace(path)
            summary["status"] = "written"
    (out / (state_file_name(member)[:-5] + ".summary.json")).write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="print the recipe keys and exit")
    ap.add_argument("--recipe", help="role:strategy:output, e.g. size:*:champion")
    ap.add_argument("--state", choices=STATES, help="a frozen P5-4 state instead of a recipe")
    ap.add_argument("--out", help="output directory (never under data/)")
    ap.add_argument("--plan-only", action="store_true", help="receipts only; fit nothing")
    ap.add_argument("--alpha", type=float, help="fill alpha (calibration recipes)")
    ap.add_argument("--cutoff", action="append", default=[],
                    help="evidence cutoff YYYY-MM-DD (calibration: repeatable, one fold each; "
                         "paired_residual_pool: optional exclusive event-date bound)")
    ap.add_argument("--pairs", help="recalibration pairs parquet (default: legacy PAIRS_PATH)")
    ap.add_argument("--ticker-chunk", type=int, default=1000,
                    help="tickers per daily_market pass for the crush table")
    args = ap.parse_args(argv)

    recipes = current_recipes()
    if args.list:
        for key, recipe in recipes.items():
            print(f"{key.label():40s} {recipe.recipe_id:32s} folds={recipe.folds.kind} "
                  f"owner={'job' if recipe.fit_owner.startswith('engine') else 'P5-4'}")
        for state in STATES:
            print(f"{state:40s} (frozen state, --state)")
        return 0
    if bool(args.recipe) == bool(args.state) or not args.out:
        ap.error("exactly one of --recipe/--state, and --out, are required")
    from engine import paths

    out = Path(args.out).resolve()
    if out == paths.DATA.resolve() or paths.DATA.resolve() in out.parents:
        ap.error("--out may not be inside data/")
    if args.state:
        if len(args.cutoff) > 1 or (args.cutoff and args.state != "paired_residual_pool"):
            ap.error("--cutoff: at most one, and only for paired_residual_pool")
        summary = run_state_job(args.state, out, plan_only=args.plan_only,
                                cutoff=args.cutoff[0] if args.cutoff else None,
                                ticker_chunk=args.ticker_chunk)
        print(f"[p5-3] {args.state}: {summary['status']} "
              f"{summary.get('content_hash', '')} -> {out}", flush=True)
        return 0
    recipe = recipes[_key(args.recipe)]
    calibration = recipe.folds.kind == "request_cutoff"
    if calibration and (args.alpha is None or not args.cutoff):
        ap.error(f"{recipe.key.label()} needs --alpha and at least one --cutoff")
    if not calibration and (args.alpha is not None or args.cutoff):
        ap.error("--alpha/--cutoff apply to calibration recipes only")
    _guard(f"tools.phase5_training_job.main:{recipe.recipe_id}")
    dataset = build_dataset(recipe, pairs_path=args.pairs)
    print(f"[p5-3] {recipe.recipe_id}: dataset rows={len(dataset):,}", flush=True)
    extra = dict(cutoffs=tuple(args.cutoff), alpha=args.alpha) if calibration else {}
    try:
        result = run_training_job(recipe, dataset, out, plan_only=args.plan_only, **extra)
    except TrainingRefused as err:
        print(json.dumps([{"path": i.path, "code": i.code, "detail": i.detail} for i in err.issues],
                         indent=1), file=sys.stderr)
        return 2
    counts = {s: result.count(s)
              for s in ("fitted", "resumed", "skipped", "planned", "passthrough")}
    print(f"[p5-3] {recipe.recipe_id}: folds={len(result.outcomes)} {counts} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
