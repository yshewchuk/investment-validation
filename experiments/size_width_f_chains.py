"""EXP-189 Amendment 1 W7a1b (design A1.3): the `option_chains` reader for `FQuotes`.

CLI value: `--f-quotes experiments.size_width_f_chains:build_f_quotes`. It reads only the snapshot that `--sources-config` pins (`snapshot_id`); `size_width_f_select.open_run` then refuses `snapshot_binding` if that differs from the sources' snapshot. Failure order: the cap is validated first, then the sources config, the catalog open, the pinned snapshot bounds and resolve, and the F columns (`required_column`). The cap bounds each fragment's file bytes and uncompressed row-group bytes, not the size of the returned frame (`FQuotes` caps rows with `MAX_QUOTE_ROWS`).
"""
import logging
import sqlite3
import time

import pandas as pd

from engine.v2.data import objects
from engine.v2.data.repository import Repository
from engine.v2.ops.errors import fail
from engine.v2.research import _chains, _scan
from experiments import fair_value_inputs as fvi
from experiments import size_pnl_sources as src
from experiments.size_pnl_ladder import _rss_mib

MAX_OBJECT_BYTES = src.MAX_OBJECT_BYTES  # default cap per fragment; `--f-max-object-bytes` overrides it
_LOG = logging.getLogger("size_width_f_chains")


def _triples(frame):
    """MultiIndex of (ticker, obs_date, expiry): the exact keys a row is kept by."""
    return pd.MultiIndex.from_arrays([frame["ticker"].astype(str), pd.to_datetime(frame["obs_date"]),
                                      pd.to_datetime(frame["expiry"])])


def option_chain_reader(repository, snapshot, *, max_object_bytes):
    """The `read(expiries)` closure over one pinned snapshot. Refuses the cap or missing F columns before any read."""
    objects.validate_max_object_bytes(max_object_bytes)
    names = [column.name for column in repository.table_contract(snapshot, "option_chains").columns]
    missing = [c for c in fvi.QUOTE_COLUMNS if c not in names]
    if missing:
        raise fvi.FairValueRefusal("required_column", "the snapshot's option_chains contract lacks F columns", columns=missing)

    def read(expiries):
        wanted = {(str(t), pd.Timestamp(d).normalize()) for t, d in zip(expiries["ticker"], expiries["obs_date"])}
        tickers = {t for t, _ in wanted}
        years = sorted({d.year for _, d in wanted})
        triples = _triples(expiries)

        def batch_filter(frame):
            return frame if frame.empty else frame[_triples(frame).isin(triples)]

        _LOG.info("option_chains read: expiries=%d tickers=%s years=%s cap=%d", len(expiries), sorted(tickers), years,
                  max_object_bytes)
        started = time.perf_counter()
        # `_scan.read_table` visits partitions in manifest first-seen order, as `load_chain_index` does.
        frame = _scan.read_table(repository, snapshot, "option_chains", fvi.QUOTE_COLUMNS,
                                 partition_keys=[str(y) for y in years],
                                 key_filter=_chains._chain_key_filter(tickers, wanted), batch_filter=batch_filter,
                                 max_object_bytes=max_object_bytes)
        _LOG.info("option_chains read: rows=%d elapsed=%.1fs rss=%.0fMiB", len(frame), time.perf_counter() - started,
                  _rss_mib())
        return frame

    return read


def build_f_quotes(args):
    """The `--f-quotes` factory: `(read, snapshot_id)`. The connection stays open for `read`; it closes on any refusal."""
    cap = getattr(args, "f_max_object_bytes", None)
    cap = MAX_OBJECT_BYTES if cap is None else cap  # 0 is a refused cap, never the default
    objects.validate_max_object_bytes(cap)
    cfg = src._config(getattr(args, "sources_config", None))
    conn, store = src._open(cfg)
    try:
        src._bound_snapshot(conn, cfg["snapshot_id"])
        repository = Repository(conn, store, pin_snapshots=True)
        snapshot = src._snapshot(repository, cfg["snapshot_id"])
        read = option_chain_reader(repository, snapshot, max_object_bytes=cap)
    except sqlite3.Error:
        conn.close()
        raise fail("RESOURCE_UNAVAILABLE", "catalog unavailable", details={"reason": "catalog_unavailable"}) from None
    except BaseException:
        conn.close()
        raise
    _LOG.info("option_chains reader: snapshot=%s cap=%d", cfg["snapshot_id"], cap)
    return read, cfg["snapshot_id"]
