"""Incremental parse paths for the Tier-2 rebuild (opt-in, legacy nightly only).

Two tables are parse-bound and cached per raw input
(:class:`~engine.data.rebuild_cache.InputCache`):

``option_chains``
    One cache entry per source: a legacy strike file, or one fetch-store
    ``hist/strikes`` body. The entry holds exactly what the full rebuild
    computes for that source alone -- the validated, schema-coerced frame, its
    validation report (including any quarantine calls, which are replayed) and
    its chain kind -- or the fact that the body was a recognized empty answer.
    Unreadable and unrecognized sources are never cached, so they are re-read,
    re-counted and re-flagged every run exactly as today. Cross-source
    deduplication (``PartitionedWriter.finalize``) always runs over every
    source.

``daily_market``
    One cache entry per per-ticker raw file: the legacy summaries/cores file
    and the fetch store's ``?ticker=X`` history body. The entry holds that
    file's rows as a frame restricted to the columns the normalizer reads
    (:data:`~engine.data.normalize.n_daily.SOURCE_COLUMNS`). The market-wide
    ``tradeDate`` responses change every night and are never cached; the
    ticker's combined series, and everything computed from it (dedupe,
    market-cap as-of join, validation), is recomputed every run. A ticker
    whose parts were all parsed fresh this run goes through the unchanged
    full-rebuild path (``pd.DataFrame`` over the concatenated rows), so a
    cold-cache run is the full rebuild by construction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from engine import paths
from engine.data.fetch import iter_cached
from engine.data.normalize import fetch_store, n_chains, n_daily
from engine.data.normalize.common import read_gz_json
from engine.data.rebuild_cache import InputCache

__all__ = ["chain_sources", "normalize_ticker_cached", "EMPTY_PAYLOAD"]

#: Cached value for a fetch body that is a recognized zero-row answer.
EMPTY_PAYLOAD = "empty"


# --------------------------------------------------------------------------
# option_chains
# --------------------------------------------------------------------------


def chain_sources(cache: InputCache, fetch_stats: dict) -> Iterator[tuple[str, tuple]]:
    """The full rebuild's ``(label, (kind, handle))`` stream, cache-aware.

    ``kind`` is ``legacy`` / ``fetch`` for a source to parse (``handle`` is
    ``(path_or_source, signature, cache_path)``) or ``cached`` for a reused
    per-source result. Fetch accounting in ``fetch_stats`` matches
    :func:`~engine.data.normalize.fetch_store.iter_orats_rows`.
    """
    for path in n_chains.iter_chain_files():
        hit, value, sig = cache.lookup(path, path.name)
        yield path.name, (("cached", value) if hit else ("legacy", (path, sig, path)))
    for key in ("scanned", "payloads", "empty", "unrecognized", "unreadable"):
        fetch_stats.setdefault(key, 0)
    for entry in iter_cached("orats", "hist/strikes"):
        yield from _fetch_source(cache, entry, fetch_stats)


def _fetch_source(cache: InputCache, entry, fetch_stats: dict) -> Iterator[tuple[str, tuple]]:
    hit, value, sig = cache.lookup(entry.path, entry.source_id)
    if hit and value == EMPTY_PAYLOAD:
        fetch_store.count_outcome(fetch_stats, "empty")
        return
    if hit:
        fetch_store.count_outcome(fetch_stats, "payload")
        yield entry.source_id, ("cached", value)
        return
    outcome, source = fetch_store.classify_entry(entry, fetch_stats)
    if outcome == "empty":
        cache.store(entry.path, sig, entry.source_id, EMPTY_PAYLOAD)
    elif outcome == "payload":
        yield source.source_id, ("fetch", (source, sig, entry.path))
    else:
        cache.note_uncached()


# --------------------------------------------------------------------------
# daily_market
# --------------------------------------------------------------------------


def _project(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    wanted = [c for c in n_daily.SOURCE_COLUMNS[kind] if c in frame.columns]
    return frame[wanted].copy()


class _Part:
    """One raw input's contribution to a ticker's series."""

    def __init__(self, rows: list | None = None, frame: pd.DataFrame | None = None,
                 n: int | None = None):
        self.rows = rows
        self.frame = frame
        self.n = len(rows) if rows is not None else int(n or 0)

    @property
    def fresh(self) -> bool:
        return self.rows is not None


def _all_dicts(rows: list) -> bool:
    return all(isinstance(row, dict) for row in rows)


def _file_part(cache: InputCache, path: Path, source_id: str, kind: str, reader) -> _Part | None:
    """A cached or freshly parsed part; ``None`` when the file does not exist."""
    if not path.exists():
        return None
    hit, value, sig = cache.lookup(path, source_id)
    if hit:
        return _Part(frame=value, n=len(value))
    rows = reader(path)
    if rows is None:  # an unreadable history body: contributes nothing, as today
        cache.note_uncached()
        return _Part(rows=[])
    if _all_dicts(rows):
        cache.store(path, sig, source_id, _project(pd.DataFrame(rows), kind))
    else:
        cache.note_uncached()
    return _Part(rows=rows)


def _legacy_reader(path: Path) -> list:
    return list(read_gz_json(path) or [])


def _parts(cache: InputCache, ticker: str, kind: str) -> list[_Part]:
    directory = paths.RAW_ORATS_SUMMARIES if kind == "summaries" else paths.RAW_ORATS_CORES
    parts = [
        _file_part(cache, directory / f"{ticker}.json.gz", f"daily:{kind}:{ticker}:legacy",
                   kind, _legacy_reader),
        _Part(rows=list(n_daily.fetch_daily_index().get(ticker, {}).get(kind, []))),
        _file_part(cache, n_daily.history_body_path(ticker, kind),
                   f"daily:{kind}:{ticker}:history", kind, n_daily.read_history_body),
    ]
    return [part for part in parts if part is not None]


def _frame_from_parts(ticker: str, parts: list[_Part], kind: str) -> pd.DataFrame | None:
    """The ticker's raw frame for one kind, or ``None`` when it has no rows.

    All-fresh parts take the full rebuild's own path. Otherwise the per-part
    frames are concatenated; every column the normalizer reads is taken
    element by element (``to_datetime`` / ``to_numeric``), so the result is
    the same as one ``pd.DataFrame`` over the concatenated rows. A fresh part
    holding non-dict rows cannot be framed on its own, so that kind is re-read
    whole through the full rebuild's reader.
    """
    if sum(part.n for part in parts) == 0:
        return None
    if all(part.fresh for part in parts):
        return pd.DataFrame([row for part in parts for row in part.rows])
    if any(part.fresh and not _all_dicts(part.rows) for part in parts):
        directory = paths.RAW_ORATS_SUMMARIES if kind == "summaries" else paths.RAW_ORATS_CORES
        return pd.DataFrame(n_daily._rows_for(ticker, kind, directory))
    frames = [
        part.frame if not part.fresh else _project(pd.DataFrame(part.rows), kind)
        for part in parts if part.n
    ]
    return pd.concat(frames, ignore_index=True)


def normalize_ticker_cached(ticker: str, cache: InputCache) -> tuple[pd.DataFrame, dict]:
    """:func:`~engine.data.normalize.n_daily.normalize_ticker`, with cached parses."""
    src = _frame_from_parts(ticker, _parts(cache, ticker, "summaries"), "summaries")
    if src is None:
        return pd.DataFrame(), {"ticker": ticker, "reason": "no summaries rows"}
    return n_daily.normalize_frames(
        ticker, src, lambda: _frame_from_parts(ticker, _parts(cache, ticker, "cores"), "cores")
    )
