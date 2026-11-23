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
import stat
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


def file_sha256(path, max_bytes=8 << 30):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except IsADirectoryError:
        raise ValueError("not a regular file") from None
    try:
        is_regular = stat.S_ISREG(os.fstat(fd).st_mode)
    except BaseException:
        os.close(fd)
        raise
    if not is_regular:
        os.close(fd)
        raise ValueError(f"not a regular file: {path}")
    with os.fdopen(fd, "rb") as f:
        h = hashlib.sha256()
        total = 0
        for chunk in iter(lambda: f.read(1 << 20), b""):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"{path} exceeds max_bytes={max_bytes}")
            h.update(chunk)
    return h.hexdigest()


def recipe_hash(features, params, seed, target, threshold, through):
    payload = {"features": list(features), "params": params, "seed": seed,
               "target": target, "threshold": threshold, "through": through}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def refuse_protected(out, root):
    out = Path(out).resolve()
    roots = [root] if isinstance(root, (str, os.PathLike)) else list(root)
    for r in roots:
        protected_dir = (Path(r) / "data" / "models").resolve()
        registry = (Path(r) / "engine" / "models" / "registry.json").resolve()
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


def _iso_or_none(ts):
    return None if pd.isna(ts) else pd.Timestamp(ts).date().isoformat()


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
    # Dates come from the rows that were actually fitted, not from every kept trade.
    fitted = dataset.loc[complete]
    fit_events = pd.to_datetime(fitted["event_date"]).dropna()
    fit_ids = set(fitted["event_id"].tolist())
    fit_exits = pd.to_datetime(kept.loc[kept["event_id"].isin(fit_ids), "exit_date"]).dropna()
    receipt = {
        "kind": "gate_fold_receipt",
        "registered_id": REGISTERED_ID,
        "train_through": through,
        "training_rows": n_complete,
        "event_date_min": _iso_or_none(fit_events.min()),
        "event_date_max": _iso_or_none(fit_events.max()),
        "exit_date_max": _iso_or_none(fit_exits.max()),
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


def _unlink_quiet(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _publish_pair(out, receipt_path, write_artifact, receipt, sha_key):
    """Publish artifact + receipt so that a receipt-less artifact never remains.

    The artifact goes to <out>.tmp and the receipt to <receipt_path>.tmp. The artifact is
    renamed into place first, then the receipt. If the receipt step fails, the published
    artifact is removed; tmp files are removed on any failure.
    """
    out = Path(out)
    receipt_path = Path(receipt_path)
    art_tmp = f"{out}.tmp"
    rec_tmp = f"{receipt_path}.tmp"
    published = False
    try:
        write_artifact(art_tmp)
        receipt[sha_key] = file_sha256(art_tmp)
        text = json.dumps(receipt, indent=2, sort_keys=True)
        Path(rec_tmp).write_text(text + "\n")
        os.replace(art_tmp, out)
        published = True
        os.replace(rec_tmp, receipt_path)
    except BaseException:
        _unlink_quiet(art_tmp)
        _unlink_quiet(rec_tmp)
        if published:
            _unlink_quiet(out)
        raise
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
        return ga.build_dataset(t, panel=ctx.panel, daily=ctx.daily, context=ctx)

    return load_trades, build, ga.fit, list(ga.FEATURES), ga.TARGET


def _resolve_outputs(out_arg, receipt_arg, root):
    out = refuse_protected(out_arg, root)
    receipt_path = refuse_protected(receipt_arg or f"{out}.receipt.json", root)
    if receipt_path == out:
        raise SystemExit(f"refused: receipt path {receipt_path} equals out path {out}")
    # Create directories only after every refusal has passed, so a refused run writes nothing.
    out.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    return out, receipt_path


def _run_fit(args, wiring, root):
    import joblib  # fit handling only

    out, receipt_path = _resolve_outputs(args.out, args.receipt, root)
    if wiring is None:
        wiring = _real_wiring(args.train_through)
    load_trades, build, fit, features, target = wiring
    t0 = time.time()
    _log(f"fit start: train_through={args.train_through}")
    model, receipt = train_fold(load_trades(), through=args.train_through, build=build, fit=fit,
                                features=features, target=target, seed=SEED, threshold=REGISTERED_THRESHOLD)
    payload = {"model": model, "features": list(features), "threshold": REGISTERED_THRESHOLD,
               "seed": SEED, "train_through": args.train_through, "recipe_hash": receipt["recipe_hash"]}
    _publish_pair(out, receipt_path, lambda p: joblib.dump(payload, p, compress=3), receipt, "artifact_sha256")
    _log(f"fit end: {time.time() - t0:.1f}s")
    return 0


def _run_inputs(args, wiring, root):
    out, receipt_path = _resolve_outputs(args.out, args.receipt, root)
    if wiring is None:
        wiring = _real_wiring(args.year)
    load_trades, build, _fit, features, _target = wiring
    t0 = time.time()
    _log(f"inputs start: year={args.year}")
    table, receipt = build_inputs(load_trades(), year=args.year, build=build, features=features)
    _publish_pair(out, receipt_path, table.to_parquet, receipt, "inputs_sha256")
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
    if root is None:
        roots = [Path(__file__).resolve().parents[1]]
        plan_root = os.environ.get("INVESTING_PLAN_ROOT")
        if plan_root:
            roots.append(Path(plan_root))
    else:
        roots = [Path(root)]
    root = roots
    if args.cmd == "fit":
        return _run_fit(args, wiring, root)
    return _run_inputs(args, wiring, root)


if __name__ == "__main__":
    raise SystemExit(main())
