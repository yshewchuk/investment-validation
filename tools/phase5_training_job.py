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

Calibration recipes (``*:calibration``) are not built here: their fit and
frozen artifact belong to P5-4, whose inputs are ``Scorer.trades``
enrichments; this job only defines their membership/label seam.
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


def build_dataset(recipe) -> pd.DataFrame:
    from engine.features import load_panel
    from engine.models.training import train_all

    role, strategy, output = recipe.key.role, recipe.key.strategy, recipe.key.output
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
    raise SystemExit(f"{recipe.key.label()}: not built by this job (P5-4 owns calibration fits)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="print the recipe keys and exit")
    ap.add_argument("--recipe", help="role:strategy:output, e.g. size:*:champion")
    ap.add_argument("--out", help="output directory (never under data/)")
    ap.add_argument("--plan-only", action="store_true", help="receipts only; fit nothing")
    args = ap.parse_args(argv)

    recipes = current_recipes()
    if args.list:
        for key, recipe in recipes.items():
            print(f"{key.label():40s} {recipe.recipe_id:32s} folds={recipe.folds.kind} "
                  f"owner={'job' if recipe.fit_owner.startswith('engine') else 'P5-4'}")
        return 0
    if not args.recipe or not args.out:
        ap.error("--recipe and --out are required")
    from engine import paths

    out = Path(args.out).resolve()
    if out == paths.DATA.resolve() or paths.DATA.resolve() in out.parents:
        ap.error("--out may not be inside data/")
    recipe = recipes[_key(args.recipe)]
    dataset = build_dataset(recipe)
    print(f"[p5-3] {recipe.recipe_id}: dataset rows={len(dataset):,}", flush=True)
    try:
        result = run_training_job(recipe, dataset, out, plan_only=args.plan_only)
    except TrainingRefused as err:
        print(json.dumps([{"path": i.path, "code": i.code, "detail": i.detail} for i in err.issues],
                         indent=1), file=sys.stderr)
        return 2
    counts = {s: result.count(s) for s in ("fitted", "resumed", "skipped", "planned")}
    print(f"[p5-3] {recipe.recipe_id}: folds={len(result.outcomes)} {counts} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
