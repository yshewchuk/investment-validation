"""F quote adapter: every listed row of each requested expiry (entry and exit dates) from the pinned snapshot."""

import logging
import time
from collections import Counter

import numpy as np
import pandas as pd

from experiments import fair_value_fills as fvf
from experiments import fair_value_inputs as fvi
from experiments import fair_value_run as fvr
from experiments import size_width_f_report_inputs as ri
from experiments.size_pnl_ladder import _rss_mib

_LOG = logging.getLogger("size_width_f_quotes")
_EXPIRY = ["ticker", "obs_date", "expiry"]
_KEY = _EXPIRY + ["right", "strike"]
_LEG_COLS = fvf.OUT_COLUMNS + fvf.FILL_COLUMNS + ["bid", "ask"]
RAW_COLS = ["open_interest", "bid_size", "ask_size", "chain_kind", "assetType", "spot"]  # quote and carry fields kept for the report inputs
MAX_QUOTE_ROWS = 5_000_000  # per chunk side; read at call time


def ladder_rows(rows, *, entry):
    """(kept rows in original order, dropped count): the ladder's stable rank, then one row per normalised key."""
    bid, ask = (pd.to_numeric(rows[c], errors="coerce").astype(float).to_numpy() for c in ("bid", "ask"))
    if entry:
        valid = ~np.isnan(bid) & ~np.isnan(ask) & (bid <= ask) & (ask > 0) & (bid >= 0)
        rank = np.where(valid & (bid > 0) & (bid < ask), 0, np.where(valid, 1, 2))
    else:
        rank = np.where(np.isfinite(bid) & np.isfinite(ask) & (bid >= 0) & (ask >= bid), 0, 1)
    order = np.argsort(rank, kind="mergesort")
    ordered = rows.iloc[order]
    key = ordered[_KEY].assign(obs_date=pd.to_datetime(ordered["obs_date"], errors="coerce"),
                               expiry=pd.to_datetime(ordered["expiry"], errors="coerce"),
                               strike=pd.to_numeric(ordered["strike"], errors="coerce"))
    drop = (key.duplicated(keep="first") & ordered[_KEY].notna().all(axis=1) & key.notna().all(axis=1)).to_numpy()
    return rows.iloc[np.sort(order[~drop])].reset_index(drop=True), int(drop.sum())


def admit_inputs(pins, read_quotes):
    """Refuse a non-callable reader (f_quotes_invalid) or malformed pins (pins_invalid) before any quote is read or stored."""
    if not callable(read_quotes):
        raise fvi.FairValueRefusal("f_quotes_invalid", "read_quotes is not callable")
    if not (isinstance(pins, dict) and isinstance(pins.get("manifest_sha256"), str) and isinstance(pins.get("files"), dict) and pins["files"]
            and all(isinstance(k, str) and isinstance(v, str) for k, v in pins["files"].items())):
        raise fvi.FairValueRefusal("pins_invalid", "pins need a manifest_sha256 string and a non-empty str-to-str files dict")


def _admit(plan):
    """Refuse a malformed plan (plan_invalid) before anything is stored or read."""
    cols = ["event_id", "ticker", "entry_date", "exit_date"]
    if not isinstance(plan, pd.DataFrame) or any(c not in plan.columns for c in cols):
        raise fvi.FairValueRefusal("plan_invalid", "plan must be a DataFrame with event_id, ticker, entry_date, exit_date")
    entry, exit_ = (pd.to_datetime(plan[c], errors="coerce").dt.normalize() for c in ("entry_date", "exit_date"))
    if (plan[cols].isna().any().any() or not plan["event_id"].astype(str).is_unique or not (entry < exit_).all()
            or not all(isinstance(t, str) and t.strip() for t in plan["ticker"])):
        raise fvi.FairValueRefusal("plan_invalid", "plan: null, duplicate event_id, bad ticker, bad date, or misordered dates")


def _price(rows, vols, side, price=None):
    """batches start about every fvr.BATCH_ROWS rows and never split an expiry (a batch may exceed it); input order kept."""
    price = price or fvf.leg_fills
    if not len(rows):
        return price(rows, vols)
    grouped = rows.groupby(_EXPIRY)
    starts = grouped.size().cumsum().shift(fill_value=0).to_numpy() // fvr.BATCH_ROWS
    batches, parts, done = rows.groupby(starts[grouped.ngroup().to_numpy()]), [], 0
    started, total = time.monotonic(), len(rows)
    for _, part in batches:
        parts.append(price(part, vols))
        done, elapsed = done + len(part), time.monotonic() - started
        _LOG.info("F quotes %s batch %d/%d rows %d/%d rows/s=%.0f eta=%.0fs rss=%.0fMiB", side, len(parts), batches.ngroups,
                  done, total, done / max(elapsed, 1e-9), elapsed / done * (total - done), _rss_mib())
    return pd.concat(parts).sort_index()


class FQuotes:
    """read_quotes(expiries) must return every listed row (fvi.QUOTE_COLUMNS) of each requested (ticker, obs_date, expiry)."""

    def __init__(self, plan, read_quotes, *, tables_root, pins, snapshot_id, expected_snapshot_id, vol_scale, rate_scale,
                 rate_rules=None, model=None):
        if not (isinstance(snapshot_id, str) and snapshot_id and snapshot_id == expected_snapshot_id):
            raise fvi.FairValueRefusal("snapshot_binding", "snapshot ids are missing or differ")
        admit_inputs(pins, read_quotes)
        _admit(plan)
        self._plan = plan.set_axis(plan["event_id"].astype(str).to_numpy(), axis=0)
        self._read, self._root, self._pins, self._rules = read_quotes, tables_root, pins, dict(rate_rules or {})
        self._model = model  # None is model F; a model A (size_width_a_fills.ModelA) prices the fills and binds the leg table
        self._ids, self._scales = (snapshot_id, expected_snapshot_id), (vol_scale, rate_scale)
        self.counts, self.table_hashes, self._chunk, self._tables = Counter(), {}, None, {}
        self._built = {}
        self.coverage = []

    def table(self, chunk, *, exit=False):
        """Built once per (quarter chunk, side) and kept for the run (A1.7 R2)."""
        if chunk is not self._chunk:
            self._chunk = chunk
            key = (tuple(str(e) for e in chunk["event_id"]), tuple(str(x) for x in chunk["expiry"]))
            self._tables = self._built.setdefault(key, {})
        side = "exit" if exit else "entry"
        if side not in self._tables:
            self._tables[side] = self._build(chunk, side)
        return self._tables[side]

    def _build(self, chunk, side):
        started, events = time.monotonic(), chunk["event_id"]
        if absent := [str(e) for e in events if str(e) not in self._plan.index]:
            raise fvi.FairValueRefusal("plan_invalid", "chunk events are absent from the plan", event_ids=absent)
        plan_rows = self._plan.loc[[str(e) for e in events]]
        dates = pd.to_datetime(plan_rows["exit_date" if side == "exit" else "entry_date"]).dt.normalize()
        expiries = pd.DataFrame({"ticker": plan_rows["ticker"].astype(str).to_numpy(), "obs_date": dates.to_numpy(),
                                 "expiry": chunk["expiry"]}).drop_duplicates().reset_index(drop=True)
        _LOG.info("F quotes %s start: events=%d expiries=%d", side, len(events), len(expiries))
        if len(raw := self._read(expiries)) > MAX_QUOTE_ROWS:
            raise fvi.FairValueRefusal("input_too_large", "quote rows exceed MAX_QUOTE_ROWS", code="RESOURCE_LIMIT_EXCEEDED",
                                       side=side, rows=len(raw), max_rows=MAX_QUOTE_ROWS)
        if lacking := [c for c in fvi.QUOTE_COLUMNS if c not in raw.columns]: raise fvi.FairValueRefusal("required_column", "quote columns are absent", columns=lacking)
        rows, dropped = ladder_rows(raw, entry=side == "entry")
        self.counts[f"{side}_duplicates"] += dropped
        rows = fvi.bind_quotes(rows, snapshot_id=self._ids[0], expected_snapshot_id=self._ids[1]).reset_index(drop=True)
        rows[fvi.VOL_KEYS] = fvi._norm(rows, fvi.VOL_KEYS)
        rows = rows.merge(expiries, on=_EXPIRY, how="inner").reset_index(drop=True)
        obs = expiries["obs_date"]
        lagged = pd.concat([obs, obs - pd.Timedelta(days=fvi.RATE_LOOKBACK_DAYS)])
        rels = sorted({f"quotes/year={d.year}/month={d.month:02d}.parquet" for d in obs}
                      | {f"rates/year={d.year}/month={d.month:02d}.parquet" for d in lagged})
        if missing := [r for r in rels if r not in self._pins["files"]]:
            raise fvi.FairValueRefusal("table_absent", "no pin for a needed table", paths=missing)
        tables = fvi.load_tables(self._root, manifest_sha256=self._pins["manifest_sha256"], vol_scale=self._scales[0],
                                 rate_scale=self._scales[1], file_sha256={r: self._pins["files"][r] for r in rels}, **self._rules)
        self.table_hashes.update(tables.hashes)
        inputs = rows.join(fvi.lookup_inputs(rows, tables.rates))
        priced = _price(inputs, tables.vols, side, *(() if self._model is None else (self._model.leg_fills,)))
        self.coverage.append(ri.leg_counts(priced, inputs, point=side))  # each priced row once: a table is built once per (chunk key, side)
        priced[RAW_COLS], priced["smvVol"] = inputs[RAW_COLS], fvi.attach_vol(rows, tables.vols)
        priced["bid"], priced["ask"] = (pd.to_numeric(rows[c], errors="coerce") for c in ("bid", "ask"))
        priced.index = pd.MultiIndex.from_arrays([priced[c].to_numpy() for c in _KEY])
        _LOG.info("F quotes %s end: events=%d expiries=%d without_rows=%d rows=%d duplicates_dropped=%d elapsed=%.1fs "
                  "rss=%.0fMiB", side, len(events), len(expiries), len(expiries) - len(rows[_EXPIRY].drop_duplicates()),
                  len(priced), dropped, time.monotonic() - started, _rss_mib())
        return priced

    def leg_key(self, chunk, i, idx, *, exit=False):
        plan_row = self._plan.loc[str(chunk["event_id"][i])]
        return (str(plan_row["ticker"]), pd.Timestamp(plan_row["exit_date" if exit else "entry_date"]).normalize(),
                pd.Timestamp(chunk["expiry"][i]), ("P" if chunk["right"][idx] == 0 else "C"), float(chunk["strike"][idx]))

    def legs(self, chunk, i, resolved, *, exit=False):
        side, table, event = ("exit" if exit else "entry"), self.table(chunk, exit=exit), str(chunk["event_id"][i])
        names, pieces = (("bid_x", "ask_x") if exit else ("bid_e", "ask_e")), []
        cols = _LEG_COLS if self._model is None else _LEG_COLS + list(self._model.columns())
        bind = {} if self._model is None else self._model.columns()
        for leg in resolved:
            idx = leg["idx"]
            key = self.leg_key(chunk, i, idx, exit=exit)
            ours = np.array([chunk[names[0]][idx], chunk[names[1]][idx]], float)
            listed = key in table.index
            if listed:
                pos = table.index.get_loc(key)
                clean = all(np.isnan(c) or c == s for c, s in zip(ours, table[["bid", "ask"]].iloc[pos].to_numpy(float)))
            else:
                clean = exit and bool(np.isnan(ours).all())
            if not clean:
                raise fvi.FairValueRefusal("f_quote_mismatch", "a resolved leg's stored quote differs from the chunk's",
                                           event_id=event, side=side, right=key[3], strike=key[4])
            if listed:
                pieces.append(table.iloc[[pos]])
            else:  # F's unlisted exit quote: unpriceable, unknown liquidity, thin terms at F's values
                vals = dict.fromkeys(cols, np.nan) | dict(zip(_KEY, key)) | {
                    "status": "unpriceable", "repaired": False, "clamp_side": None, "source": None, "carry_fallback": False,
                    "dividend_fallback": False, "dividend_applied": False, "unknown_bid_size": True, "unknown_ask_size": True,
                    "unknown_oi": True, "thin_size_buy": fvf.UNKNOWN_THIN, "thin_size_sell": fvf.UNKNOWN_THIN, "thin_oi": fvf.UNKNOWN_THIN} | bind
                pieces.append(pd.DataFrame([vals], columns=cols))
        return pd.concat(pieces or [table.iloc[:0]], ignore_index=True)[cols]
