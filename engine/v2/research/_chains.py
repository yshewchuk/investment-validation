"""Chain access — rewritten over ``Repository``/``read_table``.

Split out of ``engine.v2.research.replay`` (review blocker: module fan-out).
The bodies are unchanged from what replay.py carried; the store-reaching edge
is ``engine.v2.research._snapshot.read_table`` against one resolved
``SnapshotRef``, and the module-level availability caches are gone (a pinned
snapshot never changes under a run, so there is nothing to invalidate and no
reason to hold global state).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from engine.v2.contracts.data import KeyPredicate
from engine.v2.research._plan import ReplayPlan
from engine.v2.research._snapshot import read_table

__all__ = [
    "ChainIndex",
    "filter_plan_by_availability",
    "load_chain_index",
    "read_chain_keys",
    "read_chain_keys_for",
    "read_chains_for_years",
]

#: The ``option_chains`` projection every chain read uses.
_CHAIN_COLUMNS = (
    "ticker", "obs_date", "expiry", "dte", "strike", "right",
    "bid", "ask", "delta", "spot", "quote_repaired",
)


def _log(message: str) -> None:
    print(f"  [replay] {message}", flush=True)


class ChainIndex:
    """``(ticker, obs_date)`` → chain rows, loaded once for a whole replay."""

    def __init__(self, groups: Mapping[tuple[str, pd.Timestamp], pd.DataFrame]):
        self._groups = dict(groups)

    def __len__(self) -> int:
        return len(self._groups)

    def __contains__(self, key) -> bool:
        return (str(key[0]), pd.Timestamp(key[1]).normalize()) in self._groups

    def get(self, ticker: str, obs_date) -> pd.DataFrame | None:
        return self._groups.get((str(ticker), pd.Timestamp(obs_date).normalize()))

    @property
    def keys(self):
        return self._groups.keys()


def read_chain_keys(repository, snapshot_ref) -> set[tuple[str, pd.Timestamp]]:
    """Every (ticker, obs_date) the snapshot's option_chains table holds.

    Rewrite of ``engine/replay.py``'s ``available_chain_keys`` — no
    module-level cache (that cache is a legacy hot-reload guard tied to a
    mutable store; a pinned snapshot never changes under a run, so there is
    nothing to invalidate and no reason to hold global state).
    """
    frame = read_table(repository, snapshot_ref, "option_chains", ("ticker", "obs_date"))
    return set(zip(frame["ticker"].astype(str), pd.to_datetime(frame["obs_date"])))


def read_chain_keys_for(repository, snapshot_ref, keys) -> set[tuple[str, pd.Timestamp]]:
    """Every ``(ticker, obs_date)`` key from ``keys`` that is actually present.

    Like :func:`read_chain_keys`, but narrowed to ``keys``'s own years,
    tickers and dates (the same key-pushdown :func:`load_chain_index` uses)
    instead of a whole-table scan across every year — and projected to only
    the two key columns, not the full chain row. A caller that already
    knows which keys it might want (:attr:`ReplayPlan.chain_keys`) uses this
    to filter BEFORE calling :func:`load_chain_index`, so that call only
    ever requests the keys that survive filtering — matching
    ``read_chain_keys`` plus a pre-filtered ``load_chain_index`` call's
    exact behavior, at a fraction of the scan cost (narrowed, two columns).
    An empty ``keys`` returns an empty set without reading anything.
    """
    wanted = {(str(t), pd.Timestamp(d).normalize()) for t, d in keys}
    if not wanted:
        return set()
    years = sorted({d.year for _, d in wanted})
    tickers = {t for t, _ in wanted}
    key_filter = _chain_key_filter(tickers, wanted)
    frame = read_table(repository, snapshot_ref, "option_chains", ("ticker", "obs_date"),
                       partition_keys=[str(y) for y in years], key_filter=key_filter)
    if frame.empty:
        return set()
    present = pd.DataFrame({"ticker": frame["ticker"].astype(str),
                            "obs_date": pd.to_datetime(frame["obs_date"]).dt.normalize()})
    pairs = pd.MultiIndex.from_frame(present).unique()
    return {(t, d) for t, d in pairs if (t, d) in wanted}


def read_chains_for_years(repository, snapshot_ref, years, batch_filter=None,
                          key_filter=()) -> pd.DataFrame:
    """option_chains rows for ``years``, projected to ``_CHAIN_COLUMNS``.

    ``batch_filter`` is an optional per-batch predicate forwarded to the read
    adapters (default ``None`` — omitted and explicit ``None`` behave
    identically to the pre-existing whole-year read). ``key_filter`` is an
    optional tuple of ``KeyPredicate`` values forwarded unchanged to the read
    adapter (default ``()`` — omitted and ``()`` behave identically to the
    pre-existing whole-year read).
    """
    return read_table(repository, snapshot_ref, "option_chains", _CHAIN_COLUMNS,
                      partition_keys=[str(y) for y in years], batch_filter=batch_filter,
                      key_filter=key_filter)


def _chain_batch_filter(tickers, wanted):
    """A per-batch exact ``(ticker, obs_date)`` pair filter for ``wanted``.

    The membership semantics are the ones :func:`load_chain_index` has always
    applied to the assembled frame — requested dates normalized, stored dates
    only ``pd.to_datetime``-converted (never normalized, so a non-midnight
    observation cannot broaden a match), exact pair membership — moved to run
    on each batch before anything accumulates. The coarse ticker membership
    only narrows work within the batch; the exact pair check is what decides
    membership.
    """
    def filter_batch(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        frame = frame[frame["ticker"].isin(tickers)]
        if frame.empty:
            return frame
        pairs = pd.MultiIndex.from_arrays([frame["ticker"],
                                           pd.to_datetime(frame["obs_date"])])
        return frame[pairs.isin(wanted)]
    return filter_batch


def _chain_key_filter(tickers, wanted) -> tuple[KeyPredicate, ...]:
    """The ``option_chains`` key filter covering ``wanted``'s ticker and date sets.

    Pushed into every scan's predicate set so the scan's explicit result limit is
    ``Repository.scan_population_bound`` for this exact pinned table, contract ref,
    these predicates and time interval -- the recorded rows of the fragments they
    survive, not the whole partition. It is the cartesian superset of ``wanted``
    (every row whose ticker AND obs_date are each individually wanted), never the
    exact pairs, which ``KeyPredicate`` cannot express (the same limitation
    ``fill_quality._chain_key_filter`` documents for this table). ``_chain_batch_filter``
    still narrows this superset down to the exact pairs after decoding, so results are
    unchanged. An empty ``tickers``/``wanted`` returns ``()`` -- ``load_chain_index``
    already returns before reading anything for empty ``wanted``.
    """
    if not tickers or not wanted:
        return ()
    obs_dates = {pd.Timestamp(d).strftime("%Y-%m-%d") for _, d in wanted}
    columns = (("ticker", tickers), ("obs_date", obs_dates))
    return tuple(KeyPredicate(column=column, operator="in", values=tuple(sorted(values)))
                 for column, values in columns if values)


def load_chain_index(repository, snapshot_ref, keys) -> ChainIndex:
    """Load exactly the chains a plan needs, one requested year at a time.

    Rewrite of ``engine/replay.py``'s ``load_chain_index``. ``keys`` is
    REQUIRED (no store-wide default); years are derived from ``keys``,
    matching the legacy function's own
    ``years = sorted({d.year for _, d in wanted})``.

    Each year is read as its own bounded scan and every Arrow batch is
    filtered to the exact requested ``(ticker, obs_date)`` pairs
    (:func:`_chain_batch_filter`) before it accumulates, so unmatched rows
    live no longer than the batch carrying them; the required groups stay
    resident (no cache, no resumability). The wanted ticker and obs_date sets
    are also pushed down as a ``key_filter`` (:func:`_chain_key_filter`), so each
    selected year is scanned once with a narrower candidate population; exact-pair
    membership still comes from ``batch_filter``. Years are visited in the manifest's
    first-seen partition order — the order ``_scan.read_table`` traverses in
    the previous whole read, where ``partition_keys`` only filtered membership
    — and each already-filtered frame is kept, then they are concatenated once
    and grouped once, exactly like that previous whole-read version: a group
    that (abnormally) occurs in more than one year partition keeps every row,
    and row order within each group, duplicates, dtypes, reset indexes,
    absent-key behavior and downstream replay pricing are unchanged.
    """
    wanted = {(str(t), pd.Timestamp(d).normalize()) for t, d in keys}
    if not wanted:
        return ChainIndex({})
    years = sorted({d.year for _, d in wanted})
    wanted_years = {str(year) for year in years}
    manifest_years = list(dict.fromkeys(
        int(record.partition_key)
        for record in repository.fragment_records(snapshot_ref, "option_chains")
        if record.partition_key in wanted_years))
    years = manifest_years + [year for year in years if year not in manifest_years]
    tickers = {t for t, _ in wanted}
    batch_filter = _chain_batch_filter(tickers, wanted)
    key_filter = _chain_key_filter(tickers, wanted)
    frames: list[pd.DataFrame] = []
    for year in years:
        frame = read_chains_for_years(repository, snapshot_ref, (year,),
                                      batch_filter=batch_filter, key_filter=key_filter)
        frame = frame[frame["ticker"].isin(tickers)]
        frame["obs_date"] = pd.to_datetime(frame["obs_date"])
        key_index = pd.MultiIndex.from_arrays([frame["ticker"], frame["obs_date"]])
        frame = frame[key_index.isin(wanted)]
        if not frame.empty:  # an absent year's fallback frame is all-object
            frames.append(frame)
        _log(f"chain index: year {year}: {len(frame):,} rows")
    if not frames:
        _log(f"chain index: 0 of {len(wanted):,} requested keys present")
        return ChainIndex({})
    frame = pd.concat(frames, ignore_index=True)
    groups = {(str(k[0]), pd.Timestamp(k[1])): g.reset_index(drop=True)
              for k, g in frame.groupby(["ticker", "obs_date"], sort=False)}
    _log(f"chain index: {len(groups):,} of {len(wanted):,} requested keys present")
    return ChainIndex(groups)


def filter_plan_by_availability(plan: ReplayPlan, available: set) -> ReplayPlan:
    """Drop planned events whose decision, entry or exit chain is not present.

    Rewrite of ``engine/replay.py``'s ``filter_plan_by_availability`` with
    ``available`` required: the legacy default was the store-reaching
    ``available_chain_keys()`` call this slice removes. A v2 caller resolves
    it once via :func:`read_chain_keys` and passes it explicitly.
    """
    if plan.frame.empty:
        return plan
    keys = available
    frame = plan.frame
    has_entry = np.array(
        [(t, d) in keys for t, d in zip(frame["ticker"], frame["entry_date"])]
    )
    has_exit = np.array(
        [(t, d) in keys for t, d in zip(frame["ticker"], frame["exit_date"])]
    )
    if "decision_date" in frame.columns:
        has_decision = np.array(
            [(t, d) in keys for t, d in zip(frame["ticker"], frame["decision_date"])]
        )
    else:
        has_decision = np.ones(len(frame), dtype=bool)
    skipped = dict(plan.skipped)
    skipped["no_entry_chain"] = skipped.get("no_entry_chain", 0) + int((~has_entry).sum())
    skipped["no_exit_chain"] = skipped.get("no_exit_chain", 0) + int(
        (has_entry & ~has_exit).sum()
    )
    skipped["no_decision_chain"] = skipped.get("no_decision_chain", 0) + int(
        (has_entry & has_exit & ~has_decision).sum()
    )
    keep = has_entry & has_exit & has_decision
    return ReplayPlan(frame=frame[keep].reset_index(drop=True), skipped=skipped)
