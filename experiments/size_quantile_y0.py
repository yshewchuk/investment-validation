"""EXP-190 Q2c Y0 report (design 4, 7). No caller yet; Q6c's real-data entry point calls it before any fit. Writes
nothing; Q6c stores it (keys y0, sealed_binding, chain_share_min, training_years_min, years: rows of year, events,
no_window, with_chains, chain_share, fold, training_years, passes). Bad inputs or malformed events refuse first; no
Y0 refuses `y0_none` after stderr year lines; MemoryError: RESOURCE_LIMIT_EXCEEDED; refusals give no partial report."""
import numbers
from collections import defaultdict
from itertools import takewhile

import pandas as pd

from engine.v2.contracts.data import KeyPredicate
from engine.v2.data import objects
from engine.v2.data.errors import DataError
from engine.v2.ops.errors import fail
from engine.v2.research._pricing import AMC, BMO, trading_calendar_from_snapshot
from engine.v2.research._scan import read_table
from engine.v2.research.progress import Progress, decision, stage
from experiments.size_pnl_sources import MAX_OBJECT_BYTES
from experiments.size_pnl_tables import windows
from experiments.size_quantile_folds import annual_folds
from experiments.size_quantile_sources import _MALFORMED, _refuse, training_labels

LAST_YEAR = 2025


def _thresholds(spec):
    wf = spec.get("walk_forward") if isinstance(spec, dict) else None
    share, floor = (wf.get("chain_share_min"), wf.get("training_years_min")) if isinstance(wf, dict) else (None, None)
    if not (isinstance(share, numbers.Real) and not isinstance(share, bool) and 0 < share <= 1
            and isinstance(floor, int) and not isinstance(floor, bool) and floor >= 1):
        raise fail("INVALID_EXPERIMENT_SPEC", "walk_forward needs chain_share_min in (0, 1], training_years_min >= 1",
                   details={"reason": "y0_inputs_invalid"})
    return share, floor


def capped_chain_keys(repository, snapshot, keys, *, max_object_bytes):
    objects.validate_max_object_bytes(max_object_bytes)
    wanted = {(str(t), pd.Timestamp(d).normalize()) for t, d in keys}
    obs = tuple(sorted({d.strftime("%Y-%m-%d") for _, d in wanted}))
    filters = (KeyPredicate(column="ticker", operator="in", values=tuple(sorted({t for t, _ in wanted}))),
               KeyPredicate(column="obs_date", operator="in", values=obs))
    frame = read_table(repository, snapshot, "option_chains", ("ticker", "obs_date"),
                       partition_keys=sorted({str(d.year) for _, d in wanted}), key_filter=filters,
                       batch_filter=lambda batch: batch.drop_duplicates(), max_object_bytes=max_object_bytes)
    return {(str(t), pd.Timestamp(d).normalize()) for t, d in zip(frame["ticker"], frame["obs_date"])} & wanted


def _parse(sealed):
    try:
        out = []
        for e in sealed["events"]:
            day = pd.Timestamp(e["event_date"])
            if pd.isna(day):
                raise ValueError("null event_date")
            out.append((e["event_id"], str(e["ticker"]), day.normalize(), e["session"]))
        return out
    except _MALFORMED as exc:
        raise _refuse("sealed_malformed", f"a sealed event is malformed ({type(exc).__name__})") from None


def _line(r):
    share = "na" if r["chain_share"] is None else f"{r['chain_share']:.3f}"
    years = "na" if r["training_years"] is None else r["training_years"]
    return (f"year={r['year']} events={r['events']} no_window={r['no_window']} with_chains={r['with_chains']} "
            f"chain_share={share} fold={r['fold']} training_years={years} passes={'yes' if r['passes'] else 'no'}")


def _report(sealed, share_min, floor, max_object_bytes, calendar, chain_keys, labels):
    events = [ev for ev in _parse(sealed) if ev[2].year <= LAST_YEAR]
    cal = calendar(sealed["repository"], sealed["snapshot"], max_object_bytes=max_object_bytes)
    scanned = []  # (event_id, year, ((ticker, entry), (ticker, exit)), or None when the event has no window)
    for eid, ticker, day, session in events:
        sess = str(session).strip().upper()
        entry, after, reason = windows(cal, day, sess) if sess in (AMC, BMO) else (None, None, "session")
        keys = None if reason else (
            (ticker, pd.Timestamp(entry).normalize()), (ticker, pd.Timestamp(after).normalize()))
        scanned.append((eid, day.year, keys))
    # One chain_keys call per (year, month) of each key's date, in sorted order; the union is the present set.
    months = defaultdict(set)
    for key in (k for _, _, ks in scanned if ks for k in ks):
        months[(key[1].year, key[1].month)].add(key)
    present, progress = set(), Progress("quantile y0 chains", len(months))
    for ym in sorted(months):
        present.update(chain_keys(sealed["repository"], sealed["snapshot"], sorted(months[ym]),
                                  max_object_bytes=max_object_bytes))
        progress.tick()
    years = [str(y) for y in range(min((ev[2].year for ev in events), default=LAST_YEAR + 1), LAST_YEAR + 1)]
    counts = {y: {"events": 0, "no_window": 0, "with_chains": 0} for y in years}
    for _, year, keys in scanned:
        c = counts[str(year)]
        c["events"] += 1
        if keys is None:
            c["no_window"] += 1
        elif keys[0] in present and keys[1] in present:
            c["with_chains"] += 1
    folds = {}
    for y in years:
        try:
            folds[y] = annual_folds(sealed, [y])[y]
        except DataError as exc:
            if exc.code != "POPULATION_COLLAPSED":
                raise
            folds[y] = None
    ok = {y: f for y, f in folds.items() if f is not None}
    year_of = {eid: year for eid, year, _ in scanned}
    training = {}
    if ok:
        ids = sorted({i for f in ok.values() for i in f["training_event_ids"]})
        cutoff = max(f["training_labels_before"] for f in ok.values())
        try:
            frame = labels(sealed, ids, cutoff)
            pairs = list(zip(frame["event_id"], frame["target_available_on"]))
        except DataError as exc:
            if exc.code != "POPULATION_COLLAPSED":
                raise
            pairs = []
        # Exact: qualification is monotone, so one max-cutoff call serves each fold's cutoff comparison.
        for y, f in ok.items():
            before = f["training_labels_before"]
            qualified = {i for i, avail in pairs if avail < before}
            training[y] = len({year_of[i] for i in f["training_event_ids"] if i in qualified and year_of[i] < int(y)})
    rows = []
    for y in years:
        c = counts[y]
        share = c["with_chains"] / c["events"] if c["events"] else None
        ty = training.get(y)
        passes = bool(c["events"]) and share >= share_min and ty is not None and ty >= floor
        fold = "ok" if folds[y] is not None else "population_collapsed"
        rows.append({"year": y, **c, "chain_share": share, "fold": fold, "training_years": ty, "passes": passes})
    tail = list(takewhile(lambda r: r["passes"], reversed(rows)))
    y0 = tail[-1]["year"] if tail else None
    for row in rows:
        decision("quantile y0", _line(row))
    decision("quantile y0", f"y0={y0 or 'none'}")
    if y0 is None:
        raise _refuse("y0_none", "no Y0: the latest reported year (2025) does not pass, or no year is reported")
    return {"y0": y0, "sealed_binding": sealed["binding"], "chain_share_min": share_min,
            "training_years_min": floor, "years": rows}


def y0_report(sealed, spec, *, max_object_bytes=MAX_OBJECT_BYTES, calendar=trading_calendar_from_snapshot,
              chain_keys=capped_chain_keys, labels=training_labels):
    with stage("quantile y0"):
        share_min, floor = _thresholds(spec)
        objects.validate_max_object_bytes(max_object_bytes)
        try:
            return _report(sealed, share_min, floor, max_object_bytes, calendar, chain_keys, labels)
        except MemoryError:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "the Y0 report ran out of memory") from None
