#!/usr/bin/env python3
"""Tier 4 acceptance checks — the ones a fixture cannot prove.

    python3 checks/tier4_checks.py               # everything
    python3 checks/tier4_checks.py --list
    python3 checks/tier4_checks.py --only agreement causality

``tests/test_tier4.py`` fixes the logic on synthetic data with a model whose
refit is bitwise reproducible. That is the right place for it and it is not
enough: the properties that actually matter hold between the REAL panel, the
REAL champion, and the table on disk, and every one of them is about two things
agreeing that a unit test can only ever construct one of.

The §9 acceptance tests from ``guides/tier4_feature_models.md``, in order.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.data import manifest, store  # noqa: E402
from engine.data.features import tier4  # noqa: E402
from engine.features import load_panel  # noqa: E402

#: How many stored rows the causality and agreement checks reproduce. Each one
#: costs a model fit for its fold, so this trades runtime against coverage; 200
#: spans well over a hundred folds, which is the axis that matters.
SAMPLE = 200

#: The float64 tolerance the agreement check allows, and why it is not zero.
#: The champion's MLP half is not associative across batch shapes, so the SAME
#: fitted estimator returns values a few ULPs apart depending on how many rows
#: it scores at once. Measured 2026-09-04: max 3.55e-15 over 400 events. A
#: bitwise assertion would fail for a reason that has nothing to do with
#: causality, which is the property this is here to protect.
TOLERANCE = 1e-9


@dataclass
class CheckOutcome:
    name: str
    passed: bool
    detail: str = ""
    elapsed_s: float = 0.0
    skipped: bool = False


REGISTRY: dict[str, dict] = {}
_STATE: dict[str, object] = {}


def check(name: str, *, description: str = ""):
    def wrap(fn):
        REGISTRY[name] = {"fn": fn, "description": description}
        return fn

    return wrap


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def panel() -> pd.DataFrame:
    if "panel" not in _STATE:
        _STATE["panel"] = load_panel()
    return _STATE["panel"]


def table() -> pd.DataFrame:
    if "table" not in _STATE:
        frame = tier4.load_forecasts()
        _require(
            frame is not None,
            f"{paths.TIER4} missing — build it with "
            "`python3 -m engine.data.features.tier4`",
        )
        _STATE["table"] = frame
    return _STATE["table"]


def _sample(n: int = SAMPLE) -> pd.DataFrame:
    scored = table().dropna(subset=["pred_abs_move"])
    rng = np.random.default_rng(20260904)
    idx = rng.choice(len(scored), size=min(n, len(scored)), replace=False)
    return scored.iloc[idx]


# --------------------------------------------------------------------------
# 1. a row never sees its own period
# --------------------------------------------------------------------------


@check("causality", description="refit on < fold_start reproduces every sampled row")
def check_causality() -> str:
    """The test ``assert_causal`` structurally cannot do.

    A leak inside a model's TRAINING SET leaves feature stamps perfectly
    ordered, so the existing audit passes on a table built from the final refit
    model. This refits each sampled row's fold from scratch on events strictly
    before it and demands the stored number back.
    """
    model = tier4.size_feature_model()
    _, trainable = tier4.training_frames(panel(), model)
    prepared = model.prepare(panel()).set_index(["ticker", "date"])

    worst, checked = 0.0, 0
    for fold, group in _sample().groupby("fold_start"):
        estimator = tier4.fit_fold(trainable, model, fold)
        keys = [(r.ticker, pd.Timestamp(r.event_date)) for r in group.itertuples(index=False)]
        keys = [k for k in keys if k in prepared.index]
        if not keys:
            continue
        rows = prepared.loc[keys]
        got = np.asarray(
            estimator.predict(rows[list(model.features)].to_numpy(dtype=float)), dtype=float
        ).ravel()
        want = (
            group.set_index(["ticker", "event_date"])
            .loc[keys, "pred_abs_move"]
            .to_numpy(dtype=float)
        )
        worst = max(worst, float(np.abs(got - want).max()))
        checked += len(keys)

    _require(checked > 0, "no sampled row could be reproduced — the sample is broken")
    _require(
        worst <= TOLERANCE,
        f"a stored forecast does not reproduce from a fit on < fold_start "
        f"(max |diff| {worst:.3e} over {checked} rows) — a Tier-4 row may have "
        "been built from a model that saw its own period",
    )
    return f"{checked} rows, max |diff| {worst:.2e}"


@check("leak_is_detectable", description="a full-sample refit fails the causality check")
def check_leak_is_detectable() -> str:
    """A leak test that cannot detect a leak passes for the wrong reason.

    Refit on everything and assert the stored values NO LONGER reproduce. If
    this ever passes, ``causality`` above has stopped being able to distinguish
    anything.
    """
    model = tier4.size_feature_model()
    _, trainable = tier4.training_frames(panel(), model)
    leaky = model.fit(
        trainable[list(model.features)].to_numpy(dtype=float),
        trainable[model.target].to_numpy(dtype=float),
        model.seed,
    )
    prepared = model.prepare(panel()).set_index(["ticker", "date"])
    rows = _sample(400)
    keys = [
        (r.ticker, pd.Timestamp(r.event_date))
        for r in rows.itertuples(index=False)
        if (r.ticker, pd.Timestamp(r.event_date)) in prepared.index
    ]
    got = np.asarray(
        leaky.predict(prepared.loc[keys][list(model.features)].to_numpy(dtype=float)),
        dtype=float,
    ).ravel()
    want = (
        rows.set_index(["ticker", "event_date"]).loc[keys, "pred_abs_move"].to_numpy(dtype=float)
    )
    worst = float(np.abs(got - want).max())
    _require(
        worst > TOLERANCE,
        f"a model fit on the FULL sample reproduces the stored forecasts to "
        f"{worst:.3e} — the causality check cannot tell a leak from a clean build",
    )
    return f"full-sample refit diverges by {worst:.2f}pp, as it must"


# --------------------------------------------------------------------------
# 2. --since equivalence
# --------------------------------------------------------------------------


@check("since_equivalence", description="an incremental rebuild equals the stored table")
def check_since_equivalence() -> str:
    """Non-negotiable: if these can disagree, every backfill corrupts silently.

    Rebuilt against the table on disk rather than against a second full build,
    which is both faster and the stronger statement — it is the stored table
    that everything downstream reads.
    """
    stored = table()
    since = pd.Timestamp(stored["event_date"].max()) - pd.offsets.MonthBegin(3)
    rebuilt = tier4.build_forecasts(
        panel(), since=since, existing=stored, log=lambda _m: None
    )
    _require(
        len(rebuilt) == len(stored),
        f"incremental produced {len(rebuilt):,} rows against {len(stored):,} stored",
    )
    merged = stored.merge(
        rebuilt, on=["ticker", "event_date"], how="outer", suffixes=("", "_new"),
        indicator=True,
    )
    _require(
        (merged["_merge"] == "both").all(),
        f"{int((merged['_merge'] != 'both').sum()):,} rows differ in KEY between the "
        "stored table and an incremental rebuild",
    )
    diff = (merged["pred_abs_move"] - merged["pred_abs_move_new"]).abs()
    worst = float(diff.max(skipna=True) or 0.0)
    both_null = merged["pred_abs_move"].isna() == merged["pred_abs_move_new"].isna()
    _require(
        bool(both_null.all()),
        f"{int((~both_null).sum()):,} rows disagree on whether a forecast EXISTS",
    )
    _require(
        worst <= TOLERANCE,
        f"incremental and stored forecasts differ by up to {worst:.3e} from "
        f"{since.date()} — a backfill would silently corrupt the table",
    )
    return f"rebuilt from {since.date()}, max |diff| {worst:.2e}"


# --------------------------------------------------------------------------
# 3. live and historical agree
# --------------------------------------------------------------------------


@check("agreement", description="the served fold model reproduces the stored value")
def check_agreement() -> str:
    """The property monthly cadence buys, asserted rather than assumed.

    The live board reaches a fold's model through ``serving_model``; the build
    reached it through ``fit_fold``. Both are the same call, and this is what
    keeps it that way — a second code path for live scoring is exactly how the
    board and the backtest come to disagree about what was recommended.
    """
    model = tier4.size_feature_model()
    prepared = model.prepare(panel()).set_index(["ticker", "date"])
    worst, checked = 0.0, 0
    for fold, group in _sample().groupby("fold_start"):
        served = tier4.serving_model(fold, panel=panel(), cache=False)
        for row in group.itertuples(index=False):
            key = (row.ticker, pd.Timestamp(row.event_date))
            if key not in prepared.index:
                continue
            _require(
                tier4.serving_fold(row.event_date, row.event_date) == pd.Timestamp(fold),
                f"{key}: the fold served for a same-month decision is not the "
                "fold the table used",
            )
            live = float(served.predict(prepared.loc[[key]])[0])
            worst = max(worst, abs(live - float(row.pred_abs_move)))
            checked += 1
    _require(checked > 0, "nothing was compared")
    _require(
        worst <= TOLERANCE,
        f"the live serving path and the stored table disagree by up to {worst:.3e} "
        "— the board would recommend a shape the backtest never measured",
    )
    return f"{checked} rows, max |diff| {worst:.2e}"


# --------------------------------------------------------------------------
# 4. Tier 3 is unchanged
# --------------------------------------------------------------------------


@check("tier3_untouched", description="Tier 4's hash is not folded into the snapshot")
def check_tier3_untouched() -> str:
    """If Tier 4 can move Tier 3's hash, the layering has failed.

    Tier 4 moves whenever a champion is promoted. Folding it into
    ``snapshot_hash`` would make every promotion invalidate the provenance of
    experiments that never read a forecast — the exact coupling this layer
    exists to avoid.
    """
    stats = manifest.collect_stats()
    before = manifest.snapshot_hash(stats)
    parts = [f"{n}:{stats[n]['content_hash']}" for n in sorted(stats)]
    _require(
        tier4.forecasts_digest() is not None,
        "no Tier-4 digest — the table has not been built",
    )
    _require(
        tier4.forecasts_digest() not in "|".join(parts),
        "the Tier-4 digest appears in the snapshot's inputs",
    )
    snap = manifest.read_snapshot() or {}
    _require(
        "tier4_sha256" in snap,
        "SNAPSHOT does not record tier4_sha256 — rerun the rebuild's manifest step",
    )
    _require(
        snap.get("snapshot") == before,
        "the stored snapshot hash disagrees with a fresh computation",
    )
    return f"snapshot {before[:12]}, tier4 {(tier4.forecasts_digest() or '')[:12]}"


# --------------------------------------------------------------------------
# 5. totality, NULLs, and the registry graph
# --------------------------------------------------------------------------


@check("totality", description="every Tier-3 event has a row; NULL is never zero")
def check_totality() -> str:
    frame, tier3 = table(), panel()
    _require(
        len(frame) == len(tier3),
        f"Tier 4 has {len(frame):,} rows for {len(tier3):,} Tier-3 events — a "
        "missing row must mean STALE, never 'no forecast'",
    )
    keys4 = set(map(tuple, frame[["ticker", "event_date"]].astype(str).to_numpy()))
    keys3 = set(map(tuple, tier3[["ticker", "date"]].astype(str).to_numpy()))
    _require(keys4 == keys3, f"{len(keys3 - keys4):,} Tier-3 events have no Tier-4 row")

    blank = frame[frame["pred_abs_move"].isna()]
    _require(
        not (frame["pred_abs_move"] == 0).any(),
        "a forecast of exactly zero is stored — NULL must not have been filled",
    )
    _require(
        bool(blank["model_id"].isna().all() and blank["fold_start"].isna().all()),
        "a row carries provenance without a forecast",
    )
    scored = frame[frame["pred_abs_move"].notna()]
    _require(
        bool(scored["model_id"].notna().all() and scored["fold_start"].notna().all()),
        "a forecast is stored without saying which model and fold produced it",
    )
    _require(
        bool((scored["fold_start"] <= scored["event_date"]).all()),
        "a row's fold starts after its own event",
    )
    legal = tier4.fold_start_of(scored["fold_start"]).to_numpy() == scored["fold_start"].to_numpy()
    _require(bool(legal.all()), f"{int((~legal).sum()):,} rows have an illegal fold_start")
    return (
        f"{len(frame):,} rows, {len(scored):,} forecast "
        f"({len(scored)/len(frame):.1%}), {scored['fold_start'].nunique()} folds"
    )


@check("registry_graph", description="one champion produces the forecast, and it is the one used")
def check_registry_graph() -> str:
    from engine.models.registry import load_registry

    registry = load_registry()
    problems = registry.validate(check_artifacts=False)
    _require(not problems, f"registry problems: {problems}")

    graph = registry.tier4_graph()
    producers = graph["pred_abs_move"]["produced_by"]
    _require(
        len(producers) == 1,
        f"pred_abs_move is produced by {producers} — exactly one champion must",
    )
    stored = sorted(table()["model_id"].dropna().unique().tolist())
    _require(
        stored == producers,
        f"the table was built by {stored} but the champion producer is {producers} — "
        "a promotion has happened and Tier 4 has not been rebuilt "
        "(guides/tier4_feature_models.md §6)",
    )
    return f"{producers[0]} → pred_abs_move, consumed by {graph['pred_abs_move']['consumed_by']}"


# --------------------------------------------------------------------------
# 6. the consumer
# --------------------------------------------------------------------------


@check("read_join", description="load_panel(with_forecasts=True) is left, total and opt-in")
def check_read_join() -> str:
    from engine.features import FORECAST_COLUMNS

    plain = load_panel()
    joined = load_panel(with_forecasts=True)
    _require(len(joined) == len(plain), "the read join changed the panel's row count")
    _require(
        not set(FORECAST_COLUMNS) & set(plain.columns),
        "forecast columns leak into load_panel() without with_forecasts=True",
    )
    _require(
        set(FORECAST_COLUMNS) <= set(joined.columns),
        "with_forecasts=True did not add the forecast columns",
    )
    have = int(joined["pred_abs_move"].notna().sum())
    _require(
        have == int(table()["pred_abs_move"].notna().sum()),
        "the join lost or duplicated forecasts",
    )
    return f"{have:,} of {len(joined):,} panel rows carry a forecast"


@check("sizing", description="the forecast sizes a structure, and declines when it cannot")
def check_sizing() -> str:
    from engine.forecast_sizing import WIDTH_MAX, WIDTH_MIN, forecast_params

    _require(forecast_params("TWIN-P", None) is None, "a NULL forecast produced a shape")
    _require(
        forecast_params("TWIN-P", float("nan")) is None, "a NaN forecast produced a shape"
    )
    _require(forecast_params("TWIN-P", 0.0) is None, "a zero forecast produced a shape")
    _require(forecast_params("TWIN-P", 1e9) is None, "an absurd forecast was clipped, not refused")
    params = forecast_params("TWIN-P", 7.5)
    _require(params is not None, "a normal forecast produced no shape")
    _require(
        WIDTH_MIN <= params["width_moneyness"] <= WIDTH_MAX,
        f"width {params['width_moneyness']} outside the registered bounds",
    )

    real = table()["pred_abs_move"].dropna()
    sized = sum(forecast_params("TWIN-P", v) is not None for v in real.sample(2000, random_state=1))
    return f"{sized}/2000 sampled real forecasts can size a tent"


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

ORDER = [
    "totality",
    "registry_graph",
    "read_join",
    "sizing",
    "tier3_untouched",
    "causality",
    "leak_is_detectable",
    "since_equivalence",
    "agreement",
]


def run(names: list[str]) -> list[CheckOutcome]:
    outcomes: list[CheckOutcome] = []
    for name in names:
        started = time.time()
        print(f"  ...   {name}", flush=True)
        try:
            detail = REGISTRY[name]["fn"]() or ""
            passed = True
        except Exception as exc:  # noqa: BLE001 - a failing check must not end the run
            detail = f"{type(exc).__name__}: {exc}"
            passed = False
        elapsed = time.time() - started
        outcomes.append(CheckOutcome(name, passed, detail, elapsed))
        print(f"  {'PASS' if passed else 'FAIL':5s} {name}  ({elapsed:.1f}s)  {detail}", flush=True)
    return outcomes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", choices=ORDER, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.list:
        for name in ORDER:
            print(f"  {name:20s}  {REGISTRY[name]['description']}")
        return 0

    names = args.only or ORDER
    print(f"Tier 4 acceptance checks ({len(names)} checks)\n", flush=True)
    started = time.time()
    outcomes = run(names)
    failed = [o for o in outcomes if not o.passed]
    print(f"\n{len(outcomes) - len(failed)} passed, {len(failed)} failed "
          f"in {time.time() - started:.0f}s")
    if args.json:
        Path(args.json).write_text(json.dumps([o.__dict__ for o in outcomes], indent=1, default=str))
    if failed:
        print("\nFAILED:", file=sys.stderr)
        for outcome in failed:
            print(f"  {outcome.name}: {outcome.detail}", file=sys.stderr)
        return 1
    print("\nTIER 4 CHECKS: GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
