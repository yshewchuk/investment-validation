"""EXP-188 Amendment 3: fold CLI for production's STR-THRU gate.

Run by the supervisor ONCE on real data. It is not part of CI.

Subcommands:
  fit     --train-through YEAR --out PATH [--receipt PATH]
          Refit gate_midfill_str_thru_forecast_analog on trades dated through
          Dec 31 of YEAR only. Writes a joblib artifact and a JSON receipt.
  inputs  --year YEAR --out PATH [--receipt PATH]
          Build the gate's feature table for events dated in YEAR. Writes a
          parquet file and a JSON receipt.

Nothing here touches data/models/, engine/models/registry.json or the
production artifact. refuse_protected() enforces that, and outputs are never
overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REGISTERED_ID = "gate_midfill_str_thru_forecast_analog"
REGISTERED_THRESHOLD = 0.0508851760203979
SEED = 20260829
MIN_TRAIN_ROWS = 500
LABEL_COLUMNS = ("ret", "exit_value", "exit_date")

_T0 = time.time()


def _log(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[gate_fold] {msg} (elapsed={time.time() - _T0:.1f}s rss={rss:.0f}MiB)", file=sys.stderr)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recipe_hash(features, params, seed, target, threshold, through):
    payload = {"features": list(features), "params": params, "seed": seed,
               "target": target, "threshold": threshold, "through": through}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def refuse_protected(out, root):
    out = Path(out).resolve()
    protected_dir = (Path(root) / "data" / "models").resolve()
    registry = (Path(root) / "engine" / "models" / "registry.json").resolve()
    if out == protected_dir or protected_dir in out.parents:
        raise SystemExit(f"refused: {out} is inside {protected_dir}")
    if out == registry:
        raise SystemExit(f"refused: {out} is the production registry")
    if out.exists():
        raise SystemExit(f"refused: {out} already exists (never overwrite)")
    return out


def cut_trades(trades, through):
    cutoff = pd.Timestamp(f"{through}-12-31")
    event_ok = pd.to_datetime(trades["event_date"]) <= cutoff
    exit_ok = pd.to_datetime(trades["exit_date"]) <= cutoff  # NaT compares False: dropped
    keep = event_ok & exit_ok
    kept = trades.loc[keep.to_numpy()].copy()
    stats = {
        "rows_in": int(len(trades)),
        "rows_kept": int(keep.sum()),
        "dropped_event_after_cutoff": int((~event_ok).sum()),
        "dropped_exit_after_cutoff": int((event_ok & ~exit_ok).sum()),
    }
    if not kept.empty:
        if pd.to_datetime(kept["event_date"]).max() > cutoff or pd.to_datetime(kept["exit_date"]).max() > cutoff:
            raise ValueError(f"kept trades still dated after {through}-12-31")
    return kept, stats


def train_fold(trades, *, through, build, fit, features, target, seed, threshold):
    cutoff = pd.Timestamp(f"{through}-12-31")
    features = list(features)
    kept, cut = cut_trades(trades, through)
    if kept.empty:
        raise ValueError(f"no trades left after cutoff {through}-12-31")
    dataset = build(kept)
    if (pd.to_datetime(dataset["event_date"]) > cutoff).any():
        raise ValueError(f"dataset has an event dated after {through}-12-31")
    vals = dataset[[*features, target]].to_numpy(dtype=float)
    complete = np.isfinite(vals).all(axis=1)
    n_complete = int(complete.sum())
    if n_complete < MIN_TRAIN_ROWS:
        raise ValueError(f"only {n_complete} complete rows; need {MIN_TRAIN_ROWS}")
    X = vals[complete, :len(features)]
    y = vals[complete, len(features)]
    model = fit(X, y, seed)
    params = model.get_params() if hasattr(model, "get_params") else {}
    receipt = {
        "kind": "gate_fold_receipt",
        "registered_id": REGISTERED_ID,
        "train_through": through,
        "training_rows": n_complete,
        "event_date_min": pd.to_datetime(kept["event_date"]).min().date().isoformat(),
        "event_date_max": pd.to_datetime(kept["event_date"]).max().date().isoformat(),
        "exit_date_max": pd.to_datetime(kept["exit_date"]).max().date().isoformat(),
        "cut": cut,
        "seed": seed,
        "threshold": threshold,
        "n_features": len(features),
        "recipe_hash": recipe_hash(features, params, seed, target, threshold, through),
        "fit_count": 1,
    }
    return model, receipt


def build_inputs(trades, *, year, build, features):
    cutoff = pd.Timestamp(f"{year}-12-31")
    features = list(features)
    kept = trades.loc[(pd.to_datetime(trades["event_date"]) <= cutoff).to_numpy()]
    dataset = build(kept)
    in_year = (pd.to_datetime(dataset["event_date"]).dt.year == int(year)).to_numpy()
    out = dataset.loc[in_year, ["event_id", *features]].drop_duplicates("event_id")
    bad = [c for c in out.columns if c in LABEL_COLUMNS]
    if bad:
        raise ValueError(f"label columns in gate inputs: {bad}")
    incomplete = ~np.isfinite(out[features].to_numpy(dtype=float)).all(axis=1)
    receipt = {
        "kind": "gate_inputs_receipt",
        "year": int(year),
        "rows": int(len(out)),
        "n_features": len(features),
        "incomplete_rows": int(incomplete.sum()),
    }
    return out, receipt


def _write_atomic(path, writer):
    tmp = f"{path}.tmp"
    writer(tmp)
    os.replace(tmp, path)


def _write_receipt(path, receipt):
    text = json.dumps(receipt, indent=2, sort_keys=True)
    _write_atomic(path, lambda p: Path(p).write_text(text + "\n"))
    print(text)


def _real_wiring(through_year):
    # through_year is accepted for symmetry with the CLI; the loaders derive their own years.
    import engine.models.training.gate_forecast_analog as ga
    from engine.features import FeatureContext
    from engine.models.training.train_all import _engine_trades

    def load_trades():
        return _engine_trades(ga.STRATEGY)

    def build(t):
        entry_years = pd.to_datetime(t["entry_date"]).dt.year
        years = range(int(entry_years.min()) - 1, int(entry_years.max()) + 1)
        ctx = FeatureContext.load(sorted(t["ticker"].unique()), years=years)
        return ga.build_dataset(t, context=ctx)

    return load_trades, build, ga.fit, list(ga.FEATURES), ga.TARGET


def _run_fit(args, wiring, root):
    import joblib  # fit handling only

    out = refuse_protected(args.out, root)
    receipt_path = refuse_protected(args.receipt or f"{out}.receipt.json", root)
    if wiring is None:
        wiring = _real_wiring(args.train_through)
    load_trades, build, fit, features, target = wiring
    t0 = time.time()
    _log(f"fit start: train_through={args.train_through}")
    model, receipt = train_fold(load_trades(), through=args.train_through, build=build, fit=fit,
                                features=features, target=target, seed=SEED, threshold=REGISTERED_THRESHOLD)
    payload = {"model": model, "features": list(features), "threshold": REGISTERED_THRESHOLD,
               "seed": SEED, "train_through": args.train_through, "recipe_hash": receipt["recipe_hash"]}
    _write_atomic(out, lambda p: joblib.dump(payload, p, compress=3))
    receipt["artifact_sha256"] = file_sha256(out)
    _write_receipt(receipt_path, receipt)
    _log(f"fit end: {time.time() - t0:.1f}s")
    return 0


def _run_inputs(args, wiring, root):
    out = refuse_protected(args.out, root)
    receipt_path = refuse_protected(args.receipt or f"{out}.receipt.json", root)
    if wiring is None:
        wiring = _real_wiring(args.year)
    load_trades, build, _fit, features, _target = wiring
    t0 = time.time()
    _log(f"inputs start: year={args.year}")
    table, receipt = build_inputs(load_trades(), year=args.year, build=build, features=features)
    _write_atomic(out, table.to_parquet)
    receipt["inputs_sha256"] = file_sha256(out)
    _write_receipt(receipt_path, receipt)
    _log(f"inputs end: {time.time() - t0:.1f}s")
    return 0


def main(argv=None, *, wiring=None, root=None):
    parser = argparse.ArgumentParser(prog="size_pnl_gate_fold")
    sub = parser.add_subparsers(dest="cmd", required=True)
    fit_p = sub.add_parser("fit")
    fit_p.add_argument("--train-through", type=int, required=True)
    fit_p.add_argument("--out", required=True)
    fit_p.add_argument("--receipt")
    inputs_p = sub.add_parser("inputs")
    inputs_p.add_argument("--year", type=int, required=True)
    inputs_p.add_argument("--out", required=True)
    inputs_p.add_argument("--receipt")
    args = parser.parse_args(argv)
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    if args.cmd == "fit":
        return _run_fit(args, wiring, root)
    return _run_inputs(args, wiring, root)


if __name__ == "__main__":
    raise SystemExit(main())
