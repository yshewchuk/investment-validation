"""EXP-188 Amendment 3: fold CLI for production's STR-THRU gate.

Run by the supervisor ONCE on real data. It is not part of CI.

Subcommands:
  fit     --train-through YEAR --out PATH [--receipt PATH]
          Refit gate_midfill_str_thru_forecast_analog on trades dated through
          Dec 31 of YEAR only. Writes a joblib artifact and a JSON receipt.
  inputs  --year YEAR --admitted-events PATH --out PATH [--receipt PATH]
          Build the gate's feature table for the EXP-188 admitted events dated in
          YEAR. Non-admitted events of YEAR are dropped before the build. Writes a
          parquet file and a JSON receipt.

Nothing here touches data/models/, engine/models/registry.json or the
production artifact. refuse_protected() enforces that, and outputs are never
overwritten.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import resource
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

REGISTERED_ID = "gate_midfill_str_thru_forecast_analog"
REGISTERED_THRESHOLD = 0.0508851760203979
SEED = 20260829
MAX_TRAIN_THROUGH = 2024
MIN_TRAIN_ROWS = 500
ADMITTED_MAX_BYTES = 64 << 20
LABEL_COLUMNS = ("ret", "exit_value", "exit_date")
PROGRESS_INTERVAL_S = 30
MAX_FORECAST_BYTES = 4 << 30
MAX_PARTITION_BYTES = 4 << 30
MAX_PARTITION_ROWS = 50_000_000  # post-decode sanity cap on rows per trades partition; not a memory bound
DEFAULT_MAX_PANEL_BYTES = 2 << 30
DEFAULT_MAX_DAILY_PARTITION_BYTES = 4 << 30

_T0 = time.time()


def _rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _log(msg):
    print(f"[gate_fold] {msg} (elapsed={time.time() - _T0:.1f}s rss={_rss_mib():.0f}MiB)", file=sys.stderr)


@contextlib.contextmanager
def _heartbeat(label, interval=60):
    """Log `running elapsed=<s>s rss=<MiB>` every `interval` seconds while the body runs."""
    stop = threading.Event()
    start = time.time()

    def beat():
        while not stop.wait(interval):
            print(f"[gate_fold] {label} running elapsed={time.time() - start:.1f}s rss={_rss_mib():.0f}MiB",
                  file=sys.stderr, flush=True)

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def _check_max_bytes(max_bytes):
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be positive")


class InputTooLarge(SystemExit):
    """An input file is over its byte cap. The message starts with "refused: "."""


def check_input_file(path, max_bytes):
    """Refuse an input unless it is a regular file (no symlinks) of at most max_bytes.

    A missing .parquet is looked up as .csv.gz, as tier4.load_forecasts does. Returns the
    path that was checked; raises SystemExit("refused: ...") otherwise, and InputTooLarge
    (a SystemExit subclass) when the file is over max_bytes. max_bytes is validated
    first with _check_max_bytes, before any filesystem access.
    """
    _check_max_bytes(max_bytes)
    path = str(path)
    candidates = [path]
    if path.endswith(".parquet"):
        candidates.append(path[: -len(".parquet")] + ".csv.gz")
    for cand in candidates:
        try:
            st = os.lstat(cand)  # lstat: a symlink is seen as itself and refused below
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(st.st_mode):
            raise SystemExit(f"refused: {cand} is not a regular file")
        if st.st_size > max_bytes:
            raise InputTooLarge(f"refused: {cand} is {st.st_size} bytes, over the {max_bytes} byte limit")
        return cand
    raise SystemExit(f"refused: {path} not found (no .csv.gz fallback either)")


def check_context_inputs(panel_path, daily_files_for_year, years, max_panel_bytes, max_daily_bytes):
    """Size-check every file FeatureContext.load will read, before it is called.

    Checks the panel file against max_panel_bytes, then every daily partition file for each year in
    years against max_daily_bytes. Raises InputTooLarge on the first oversize file; a daily partition
    refusal names its year. Logs one stderr line with the number of files checked.
    """
    years = list(years)
    check_input_file(panel_path, max_panel_bytes)
    daily_checked = 0
    for year in years:
        for p in daily_files_for_year(year):
            try:
                check_input_file(p, max_daily_bytes)
            except InputTooLarge as e:
                detail = str(e.code)
                if detail.startswith("refused: "):
                    detail = detail[len("refused: "):]
                raise InputTooLarge(f"refused: daily partition for year {year}: {detail}") from None
            daily_checked += 1
    _log(f"input size check passed: panel=1 daily_partitions={daily_checked} years={len(years)}")


def load_forecasts_bounded(path, max_bytes):
    """Decode the Tier-4 forecast parquet table through one bounded, non-following descriptor.

    The type and size checks run on the same descriptor that is decoded, so the bytes checked
    are the bytes read. Only the parquet table is supported: a .csv.gz or a missing file is refused.
    """
    _check_max_bytes(max_bytes)
    path = str(path)
    if not path.endswith(".parquet"):
        raise SystemExit(f"refused: {path} is not a .parquet file; only the parquet forecast table is supported")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        raise SystemExit(f"refused: {path} not found; only the parquet forecast table is supported "
                         "(no .csv.gz fallback)") from None
    except OSError as e:
        raise SystemExit(f"refused: cannot open forecast table {path}: {e.strerror}") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SystemExit(f"refused: {path} is not a regular file")
        if st.st_size > max_bytes:
            raise SystemExit(f"refused: {path} is {st.st_size} bytes, over the {max_bytes} byte limit")
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, "rb") as fh:
        frame = pd.read_parquet(fh)
    import engine.data.features.tier4 as tier4

    return tier4.normalize(tier4._assert_current_schema(frame, Path(path)))


@contextlib.contextmanager
def _patched_forecasts(tier4, loader):
    """Temporarily replace tier4.load_forecasts with loader; restore the original even on error."""
    original = tier4.load_forecasts
    tier4.load_forecasts = loader
    try:
        yield
    finally:
        tier4.load_forecasts = original


def file_sha256(path, max_bytes=8 << 30):
    _check_max_bytes(max_bytes)
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
    exit_dt = pd.to_datetime(trades["exit_date"])
    exit_null = exit_dt.isna()
    exit_ok = exit_dt <= cutoff  # NaT compares False: dropped
    keep = event_ok & exit_ok
    kept = trades.loc[keep.to_numpy()].copy()
    stats = {
        "rows_in": int(len(trades)),
        "rows_kept": int(keep.sum()),
        "dropped_event_after_cutoff": int((~event_ok).sum()),
        "dropped_exit_null": int((event_ok & exit_null).sum()),
        "dropped_exit_after_cutoff": int((event_ok & ~exit_null & ~exit_ok).sum()),
    }
    if not kept.empty:
        if pd.to_datetime(kept["event_date"]).max() > cutoff or pd.to_datetime(kept["exit_date"]).max() > cutoff:
            raise ValueError(f"kept trades still dated after {through}-12-31")
    return kept, stats


def _iso_or_none(ts):
    return None if pd.isna(ts) else pd.Timestamp(ts).date().isoformat()


def train_fold(trades, *, through, build, fit, features, target, seed, threshold):
    if through > MAX_TRAIN_THROUGH:
        raise ValueError(f"through must be <= {MAX_TRAIN_THROUGH}")
    cutoff = pd.Timestamp(f"{through}-12-31")
    features = list(features)
    kept, cut = cut_trades(trades, through)
    if kept.empty:
        raise ValueError(f"no trades left after cutoff {through}-12-31")
    with _heartbeat("fold build"):
        t_build = time.time()
        dataset = build(kept)
    _log(f"fold build done: {time.time() - t_build:.1f}s")
    if (pd.to_datetime(dataset["event_date"]) > cutoff).any():
        raise ValueError(f"dataset has an event dated after {through}-12-31")
    vals = dataset[[*features, target]].to_numpy(dtype=float)
    complete = np.isfinite(vals).all(axis=1)
    n_complete = int(complete.sum())
    if n_complete < MIN_TRAIN_ROWS:
        raise ValueError(f"only {n_complete} complete rows; need {MIN_TRAIN_ROWS}")
    X = vals[complete, :len(features)]
    y = vals[complete, len(features)]
    with _heartbeat("fold fit"):
        t_fit = time.time()
        model = fit(X, y, seed)
    _log(f"fold fit done: {time.time() - t_fit:.1f}s")
    params = model.get_params() if hasattr(model, "get_params") else {}
    # Dates come from the rows that were actually fitted, not from every kept trade.
    fitted = dataset.loc[complete]
    fit_events = pd.to_datetime(fitted["event_date"]).dropna()
    if "trade_id" in dataset.columns and "trade_id" in kept.columns:
        # Exact rows: one event can carry several trades, so match on the fitted trade itself.
        fit_trade_ids = set(fitted["trade_id"].tolist())
        exit_rows = kept["trade_id"].isin(fit_trade_ids).to_numpy()
        exit_basis = "trade_id"
    else:
        fit_ids = set(fitted["event_id"].tolist())
        exit_rows = kept["event_id"].isin(fit_ids).to_numpy()
        exit_basis = "event_id"
    if "fill_alpha" in kept.columns:
        exit_rows = exit_rows & np.isclose(kept["fill_alpha"].to_numpy(dtype=float), 0.5)
    fit_exits = pd.to_datetime(kept.loc[exit_rows, "exit_date"]).dropna()
    receipt = {
        "kind": "gate_fold_receipt",
        "registered_id": REGISTERED_ID,
        "train_through": through,
        "training_rows": n_complete,
        "event_date_min": _iso_or_none(fit_events.min()),
        "event_date_max": _iso_or_none(fit_events.max()),
        "exit_date_max": _iso_or_none(fit_exits.max()),
        "exit_date_basis": exit_basis,
        "cut": cut,
        "seed": seed,
        "threshold": threshold,
        "n_features": len(features),
        "recipe_hash": recipe_hash(features, params, seed, target, threshold, through),
        "fit_count": 1,
    }
    return model, receipt


def build_inputs(trades, *, year, build, features, admitted_ids):
    cutoff = pd.Timestamp(f"{year}-12-31")
    features = list(features)
    # Admission happens before the build: non-admitted events of `year` never reach `build`.
    # Earlier years are not filtered; they are the analog pool.
    admitted = {str(x) for x in admitted_ids}
    event_dt = pd.to_datetime(trades["event_date"])
    in_year = (event_dt.dt.year == int(year)).to_numpy()
    year_ids = set(trades.loc[in_year, "event_id"].astype(str))
    admitted_matched = len(admitted & year_ids)
    if admitted_matched == 0:
        raise ValueError("no admitted event matches the year's trades")
    not_admitted = ~trades["event_id"].astype(str).isin(admitted).to_numpy()
    drop = in_year & not_admitted
    excluded_events = int(trades.loc[drop, "event_id"].nunique())
    keep = (event_dt <= cutoff).to_numpy() & ~drop
    kept = trades.loc[keep]
    with _heartbeat("inputs build"):
        dataset = build(kept)
    if (pd.to_datetime(dataset["event_date"]) > cutoff).any():
        raise ValueError(f"dataset has an event dated after {year}-12-31")
    in_year =(pd.to_datetime(dataset["event_date"]).dt.year == int(year)).to_numpy()
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
        "excluded_events": excluded_events,
        "admitted_events": len(admitted_ids),
        "admitted_matched": admitted_matched,
    }
    return out, receipt


def read_admitted_events(path, max_bytes=ADMITTED_MAX_BYTES):
    """Read the EXP-188 admitted event ids: a JSON list of non-empty strings.

    Opens with O_NOFOLLOW, checks S_ISREG on the open descriptor, and reads at most
    max_bytes. Returns (set of ids, sha256 of the exact bytes read); the sha256 equals
    file_sha256(path) for the same file.
    """
    _check_max_bytes(max_bytes)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as e:
        raise SystemExit(f"refused: cannot open admitted events {path}: {e.strerror}") from None
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise SystemExit(f"refused: admitted events {path} is not a regular file")
        raw = f.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise SystemExit(f"refused: admitted events {path} exceeds {max_bytes} bytes")
    try:
        ids = json.loads(raw.decode("utf-8"))
    except ValueError:
        raise SystemExit(f"refused: admitted events {path} is not valid JSON") from None
    if not isinstance(ids, list) or not all(isinstance(x, str) and x for x in ids):
        raise SystemExit(f"refused: admitted events {path} must be a JSON list of non-empty strings")
    if not ids:
        raise SystemExit(f"refused: admitted events {path} is empty")
    return set(ids), hashlib.sha256(raw).hexdigest()


def _unlink_quiet(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _fsync_path(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_pair(out, receipt_path, write_artifact, receipt, sha_key):
    """Publish artifact + receipt as a verified pair, without overwriting anything.

    Each run writes its artifact and receipt to uniquely named temp files (created with mkstemp
    in the target directories), fsyncs them, and publishes each with os.link, which fails if the
    final path already exists, so a run never overwrites another run's output. A FileExistsError
    on the artifact is re-raised as SystemExit. After both links the published artifact is
    re-hashed and compared with receipt[sha_key]; a mismatch raises RuntimeError. The finals this
    run created are removed on any failure, and this run's temp files are removed on any failure.
    The parent directory is fsynced after both links. A crash between the two links can still
    leave an artifact without a receipt, which the supervisor must treat as failed.
    """
    out = Path(out)
    receipt_path = Path(receipt_path)
    art_fd, art_tmp = tempfile.mkstemp(dir=out.parent, prefix=out.name + ".", suffix=".tmp")
    os.close(art_fd)
    rec_fd, rec_tmp = tempfile.mkstemp(dir=receipt_path.parent, prefix=receipt_path.name + ".", suffix=".tmp")
    os.close(rec_fd)
    tmps = (art_tmp, rec_tmp)
    created = []
    try:
        write_artifact(art_tmp)
        _fsync_path(art_tmp)
        receipt[sha_key] = file_sha256(art_tmp)
        text = json.dumps(receipt, indent=2, sort_keys=True)
        Path(rec_tmp).write_text(text + "\n")
        _fsync_path(rec_tmp)
        try:
            os.link(art_tmp, out)
        except FileExistsError:
            raise SystemExit(f"refused: {out} already exists") from None
        created.append(out)
        _unlink_quiet(art_tmp)
        os.link(rec_tmp, receipt_path)
        created.append(receipt_path)
        _unlink_quiet(rec_tmp)
        if file_sha256(out) != receipt[sha_key]:
            raise RuntimeError("published artifact does not match its receipt")
    except BaseException:
        for tmp in tmps:
            _unlink_quiet(tmp)
        for final in created:
            _unlink_quiet(final)
        raise
    # Fsync each distinct parent directory once (artifact and receipt may share one).
    parents = {}
    for d in (out.parent, receipt_path.parent):
        parents.setdefault(d.resolve(), d)
    # Both files are already published and complete here, so a directory fsync failure keeps them.
    for d in parents.values():
        try:
            _fsync_path(d)
        except Exception as e:
            raise RuntimeError(f"published but directory fsync failed: {d}") from e
    return text


def select_trades(frame, strategy):
    """Same selection as train_all._engine_trades(strategy) with decision_offset=None."""
    rows = frame[(frame["strategy"] == strategy) & (frame["provenance"].astype(str) == "engine.replay")]
    if "decision_date" in rows.columns:
        decision = pd.to_datetime(rows["decision_date"], errors="coerce")
        entry = pd.to_datetime(rows["entry_date"], errors="coerce")
        rows = rows[~(decision < entry)]
    return rows.reset_index(drop=True)


def load_trades_streaming(store, strategy, max_year, partition_files=None):
    """Stream trades one year partition at a time, keeping only select_trades() rows.

    store.iter_table yields (year, frame) tuples; only partitions through max_year are requested.
    When partition_files(year) -> list[Path] is given, every partition file for 2000..max_year is
    size-checked before any partition is decoded.
    """
    if partition_files is not None:
        for year in range(2000, int(max_year) + 1):
            for part_file in partition_files(year):
                check_input_file(part_file, MAX_PARTITION_BYTES)
    kept = []
    cols = None
    partitions = 0
    kept_rows = 0
    t0 = time.monotonic()
    last_progress = t0
    for year, part in store.iter_table("trades", years=range(2000, int(max_year) + 1)):
        partitions += 1
        if len(part) > MAX_PARTITION_ROWS:
            raise SystemExit(f"refused: partition {year} has {len(part)} rows, over the "
                             f"{MAX_PARTITION_ROWS} post-decode sanity cap (not a memory bound)")
        if cols is None:
            cols = list(part.columns)
        selected = select_trades(part, strategy)
        if not selected.empty:
            kept.append(selected)
            kept_rows += len(selected)
        now = time.monotonic()
        if now - last_progress >= PROGRESS_INTERVAL_S:
            last_progress = now
            _log(f"trades progress: {_progress_fields(partitions, kept_rows, now - t0)}")
    if kept:
        rows = pd.concat(kept, ignore_index=True)
    else:
        rows = pd.DataFrame(columns=cols or [])
    _log(f"trades end: {_progress_fields(partitions, kept_rows, time.monotonic() - t0)}")
    return rows


def _progress_fields(done, kept_rows, elapsed):
    # No total or ETA: the requested range is not a reliable partition count.
    return f"done={done} kept_rows={kept_rows} rate={done / max(elapsed, 1e-9):.2f}/s"


def _real_wiring(through_year, max_panel_bytes=DEFAULT_MAX_PANEL_BYTES,
                 max_daily_bytes=DEFAULT_MAX_DAILY_PARTITION_BYTES):
    # through_year is accepted for symmetry with the CLI; the loaders derive their own years.
    import engine.data.features.tier4 as tier4
    import engine.models.training.gate_forecast_analog as ga
    import engine.paths as paths
    from engine.data import store
    from engine.features import FeatureContext

    def load_trades(max_year):
        def partition_files(year):
            return store._partition_files(paths.curated_partition("trades", year))

        return load_trades_streaming(store, ga.STRATEGY, max_year, partition_files=partition_files)

    def bounded_forecasts(path=None):
        return load_forecasts_bounded(paths.TIER4, MAX_FORECAST_BYTES)

    def build(t):
        entry_years = pd.to_datetime(t["entry_date"]).dt.year
        years = range(int(entry_years.min()) - 1, int(entry_years.max()) + 1)
        check_context_inputs(paths.PANEL, lambda y: store._partition_files(paths.curated_partition("daily_market", y)),
                             years, max_panel_bytes, max_daily_bytes)
        ctx = FeatureContext.load(sorted(t["ticker"].unique()), years=years)
        # The builder's own tier4.load_forecasts() call decodes through the bounded handle.
        with _patched_forecasts(tier4, bounded_forecasts):
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

    if args.train_through > MAX_TRAIN_THROUGH:
        raise SystemExit(f"refused: --train-through must be <= {MAX_TRAIN_THROUGH}")
    out, receipt_path = _resolve_outputs(args.out, args.receipt, root)
    t0 = time.time()
    # Builder progress prints go to stderr; stdout carries only the receipt JSON.
    with contextlib.redirect_stdout(sys.stderr):
        if wiring is None:
            wiring = _real_wiring(args.train_through, max_panel_bytes=args.max_panel_bytes,
                                  max_daily_bytes=args.max_daily_partition_bytes)
        load_trades, build, fit, features, target = wiring
        _log(f"fit start: train_through={args.train_through}")
        t_load = time.time()
        trades = load_trades(args.train_through)
        _log(f"fit load done: {time.time() - t_load:.1f}s")
        model, receipt = train_fold(trades, through=args.train_through, build=build, fit=fit,
                                    features=features, target=target, seed=SEED, threshold=REGISTERED_THRESHOLD)
        _log("fit cut " + " ".join(f"{k}={v}" for k, v in receipt["cut"].items()))
        _log(f"fit training_rows={receipt['training_rows']}")
        payload = {"model": model, "features": list(features), "threshold": REGISTERED_THRESHOLD,
                   "seed": SEED, "train_through": args.train_through, "recipe_hash": receipt["recipe_hash"]}
        text = _publish_pair(out, receipt_path, lambda p: joblib.dump(payload, p, compress=3), receipt,
                             "artifact_sha256")
    print(text)
    _log(f"fit end: {time.time() - t0:.1f}s")
    return 0


def _run_inputs(args, wiring, root):
    # Admission is read first so a refused admitted-events file writes nothing.
    admitted_ids, admitted_sha = read_admitted_events(args.admitted_events)
    out, receipt_path = _resolve_outputs(args.out, args.receipt, root)
    t0 = time.time()
    # Builder progress prints go to stderr; stdout carries only the receipt JSON.
    with contextlib.redirect_stdout(sys.stderr):
        if wiring is None:
            wiring = _real_wiring(args.year, max_panel_bytes=args.max_panel_bytes,
                                  max_daily_bytes=args.max_daily_partition_bytes)
        load_trades, build, _fit, features, _target = wiring
        _log(f"inputs start: year={args.year} admitted={len(admitted_ids)}")
        t_load = time.time()
        trades = load_trades(args.year)
        _log(f"inputs load done: {time.time() - t_load:.1f}s")
        t_build = time.time()
        table, receipt = build_inputs(trades, year=args.year, build=build, features=features,
                                      admitted_ids=admitted_ids)
        _log(f"inputs build done: {time.time() - t_build:.1f}s")
        _log(f"inputs excluded_events={receipt['excluded_events']} admitted_events={receipt['admitted_events']} "
             f"admitted_matched={receipt['admitted_matched']} rows={receipt['rows']}")
        receipt["admitted_events_sha256"] = admitted_sha
        text = _publish_pair(out, receipt_path, lambda p: table.to_parquet(p, index=False), receipt,
                             "inputs_sha256")
    print(text)
    _log(f"inputs end: {time.time() - t0:.1f}s")
    return 0


def main(argv=None, *, wiring=None, root=None):
    parser = argparse.ArgumentParser(prog="size_pnl_gate_fold")
    sub = parser.add_subparsers(dest="cmd", required=True)
    fit_p = sub.add_parser("fit")
    fit_p.add_argument("--train-through", type=int, required=True)
    fit_p.add_argument("--out", required=True)
    fit_p.add_argument("--receipt")
    fit_p.add_argument("--max-panel-bytes", type=int, default=DEFAULT_MAX_PANEL_BYTES)
    fit_p.add_argument("--max-daily-partition-bytes", type=int, default=DEFAULT_MAX_DAILY_PARTITION_BYTES)
    inputs_p = sub.add_parser("inputs")
    inputs_p.add_argument("--year", type=int, required=True)
    inputs_p.add_argument("--admitted-events", required=True)
    inputs_p.add_argument("--out", required=True)
    inputs_p.add_argument("--receipt")
    inputs_p.add_argument("--max-panel-bytes", type=int, default=DEFAULT_MAX_PANEL_BYTES)
    inputs_p.add_argument("--max-daily-partition-bytes", type=int, default=DEFAULT_MAX_DAILY_PARTITION_BYTES)
    args = parser.parse_args(argv)
    for flag, value in (("--max-panel-bytes", args.max_panel_bytes),
                        ("--max-daily-partition-bytes", args.max_daily_partition_bytes)):
        try:
            _check_max_bytes(value)
        except ValueError:
            raise SystemExit(f"refused: {flag} must be a positive integer") from None
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
