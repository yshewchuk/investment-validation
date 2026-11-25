"""Input layer for fair-value fill model F (guides/fair_value_fill_design.md, sections 3, 10, 13).

Binds quote rows to the consumer's pinned snapshot id, loads the smoothed-vol and rate tables under
caller-supplied SHA-256 pins, applies the unit and rate rules, and reports entry input coverage.
The consumer reads quotes from its pinned snapshot; the manifest pins do not prove that files and
manifest come from one run.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import numbers
import re
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

QUOTE_KEYS = ["ticker", "obs_date", "expiry", "strike", "right"]
QUOTE_COLUMNS = QUOTE_KEYS + ["bid", "ask", "iv", "spot", "quote_repaired", "open_interest", "bid_size", "ask_size", "chain_kind"]
VOL_KEYS = ["ticker", "obs_date", "expiry", "strike"]
RATE_KEYS = ["ticker", "obs_date"]
RATE_COLUMNS = ["iRate5wk", "iRateLt"]
RATE_BOUNDS = (-0.05, 0.20)
VOL_MEDIAN_BOUNDS = (0.05, 5.0)
RATE_LOOKBACK_DAYS = 7
SHORT_DAYS = 35
LONG_DAYS = 365
MAX_MANIFEST_BYTES = 1 << 20
MAX_TABLE_BYTES = 1 << 30
_TABLE_RE = re.compile(r"(quotes|rates)/year=[0-9]{4}/month=(0[1-9]|1[0-2])\.parquet")
_COUNTED = ["usable_quote", "smvvol_positive", "rate_found", "open_interest_known", "bid_size_known", "ask_size_known"]
_LOG = logging.getLogger("fair_value_inputs")


class FairValueRefusal(ValueError):
    """`code` names the class; `details["reason"]` names the rule that fired."""

    def __init__(self, reason, message, code="EXPERIMENT_IDENTITY_CONFLICT", **context):
        self.code, self.details = code, {"reason": reason, **context}
        super().__init__(f"{code} {reason}: {message}")


@dataclass(frozen=True)
class InputTables:
    vols: pd.DataFrame
    rates: pd.DataFrame
    hashes: dict


def _floats(values):
    """Float array; None, pandas NA and non-numeric values become NaN."""
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if series.dtype == object:
        series = series.mask(series.isna(), np.nan)
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _norm(frame, columns):
    """Str tickers, datetime64[ns] dates and float numbers, with a fresh RangeIndex."""
    out = {}
    for col in columns:
        raw = pd.Series(frame[col].to_numpy())
        if col == "ticker":
            out[col] = raw.astype(str).astype(object).to_numpy()
        elif col in ("obs_date", "expiry"):
            out[col] = pd.to_datetime(raw).astype("datetime64[ns]").to_numpy()
        else:
            out[col] = _floats(frame[col])
    return pd.DataFrame(out)


def _unit_scale(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
        raise FairValueRefusal("unit_scale", f"{name} must be a finite number > 0", name=name)
    return float(value)


def _read_capped(path, cap, name):
    with path.open("rb") as stream:
        blob = stream.read(cap + 1)
    if len(blob) > cap:
        raise FairValueRefusal("input_too_large", f"{name} over the read cap", code="RESOURCE_LIMIT_EXCEEDED", path=name)
    return blob


def _check_manifest(blob):
    try:
        manifest = json.loads(blob)
    except ValueError as err:
        raise FairValueRefusal("manifest_invalid", "manifest.json is not JSON") from err
    tables = manifest.get("tables") if isinstance(manifest, dict) else None
    if not isinstance(tables, dict):
        raise FairValueRefusal("manifest_invalid", "manifest.json has no tables object")
    for kind in ("quotes", "rates"):
        entry = tables.get(kind) if isinstance(tables.get(kind), dict) else {}
        for key in ("unreadable", "unrecognized"):
            if type(entry.get(key)) is not int or entry[key] != 0:
                raise FairValueRefusal("manifest_unread_payloads", f"{kind}.{key} must be integer 0", table=kind, key=key)


def _parse(blob, kind, rel):
    """Parse bytes already hashed; on failure, re-read without columns= to tell required_column from unreadable."""
    needed = VOL_KEYS + ["smvVol"] if kind == "quotes" else RATE_KEYS + RATE_COLUMNS
    try:
        return _norm(pd.read_parquet(io.BytesIO(blob), columns=needed), needed)
    except (ValueError, KeyError, OSError):
        pass
    try:
        full = pd.read_parquet(io.BytesIO(blob))
    except (ValueError, KeyError, OSError) as err:
        raise FairValueRefusal("table_unreadable", "table is not readable parquet", path=rel) from err
    absent = [c for c in needed if c not in full.columns]
    if absent:
        raise FairValueRefusal("required_column", "table lacks required columns", path=rel, columns=absent)
    raise FairValueRefusal("table_unreadable", "table is not readable parquet", path=rel)


def bind_quotes(frame, *, snapshot_id, expected_snapshot_id):
    """Check the snapshot binding and quote schema; return the QUOTE_COLUMNS rows as a copy."""
    ids_ok = all(isinstance(s, str) and s for s in (snapshot_id, expected_snapshot_id))
    if not ids_ok or snapshot_id != expected_snapshot_id:
        raise FairValueRefusal("snapshot_binding", "snapshot ids are missing or differ")
    missing = [c for c in QUOTE_COLUMNS if c not in frame.columns]
    if missing:
        raise FairValueRefusal("required_column", "quote columns are absent", columns=missing)
    if frame.duplicated(QUOTE_KEYS).any():
        raise FairValueRefusal("duplicate_key", "duplicate quote key", table="option_chains")
    return frame[QUOTE_COLUMNS].copy()


def load_tables(root, *, manifest_sha256, file_sha256, vol_scale, rate_scale):
    """Load, pin-check and convert the vol and rate tables; every refusal raises FairValueRefusal."""
    started = time.monotonic()
    vol_k, rate_k = _unit_scale(vol_scale, "vol_scale"), _unit_scale(rate_scale, "rate_scale")
    root, rels = Path(root), sorted(file_sha256)
    for rel in rels:
        if not (isinstance(rel, str) and _TABLE_RE.fullmatch(rel)):
            raise FairValueRefusal("table_path", "table path is not in the tool layout", path=str(rel))
    if not {"quotes", "rates"} <= {rel.split("/")[0] for rel in rels}:
        raise FairValueRefusal("table_absent", "a quotes and a rates file are both required")
    _LOG.info("load start: %d table files under %s", len(rels), root)
    if not (root / "manifest.json").is_file():
        raise FairValueRefusal("manifest_absent", "manifest.json is missing")
    blob = _read_capped(root / "manifest.json", MAX_MANIFEST_BYTES, "manifest.json")
    hashes = {"manifest.json": hashlib.sha256(blob).hexdigest()}
    if hashes["manifest.json"] != manifest_sha256:
        raise FairValueRefusal("manifest_hash", "manifest.json hash differs from the pin")
    _check_manifest(blob)

    frames, last_log = {"quotes": [], "rates": []}, started
    for done, rel in enumerate(rels, 1):
        if not (root / rel).is_file():
            raise FairValueRefusal("table_absent", "a listed table file is missing", path=rel)
        blob = _read_capped(root / rel, MAX_TABLE_BYTES, rel)
        hashes[rel] = hashlib.sha256(blob).hexdigest()
        if hashes[rel] != file_sha256[rel]:
            raise FairValueRefusal("table_hash", "table file hash differs from the pin", path=rel)
        kind = rel.split("/")[0]
        frames[kind].append(_parse(blob, kind, rel))
        now = time.monotonic()
        if now - last_log >= 30.0:
            last_log, rate = now, done / (now - started)
            _LOG.info("reading tables: %d/%d files, %.2f files/s, eta %.0fs", done, len(rels), rate, (len(rels) - done) / rate)

    quotes = pd.concat(frames["quotes"], ignore_index=True)
    rates = pd.concat(frames["rates"], ignore_index=True)
    if quotes.duplicated(VOL_KEYS).any():
        raise FairValueRefusal("duplicate_key", "duplicate vol key across quote files", table="quotes")
    if rates.duplicated(RATE_KEYS).any():
        raise FairValueRefusal("duplicate_key", "duplicate rate key across rate files", table="rates")
    vol = quotes["smvVol"] * vol_k
    quotes["smvVol"] = vol.where(np.isfinite(vol) & (vol > 0))
    median = quotes["smvVol"].median()
    if not VOL_MEDIAN_BOUNDS[0] < median < VOL_MEDIAN_BOUNDS[1]:
        raise FairValueRefusal("vol_unit", "median smvVol is outside the bounds or no vol is positive", median=float(median))
    for col in RATE_COLUMNS:
        value = rates[col] * rate_k
        rates[col] = value.where(np.isfinite(value) & (value >= RATE_BOUNDS[0]) & (value <= RATE_BOUNDS[1]))
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    _LOG.info("load end: %.1f s, rss %.0f MB", time.monotonic() - started, rss_mb)
    return InputTables(quotes[VOL_KEYS + ["smvVol"]], rates[RATE_KEYS + RATE_COLUMNS], hashes)


def attach_vol(quotes, vols):
    """smvVol per quote row by VOL_KEYS (both rights share it); NaN when unmatched or a key is null."""
    left = _norm(quotes, VOL_KEYS)
    merged = left.merge(vols[VOL_KEYS + ["smvVol"]], on=VOL_KEYS, how="left", validate="many_to_one")
    values = merged["smvVol"].to_numpy(dtype=float, copy=True)
    values[left.isna().any(axis=1).to_numpy()] = np.nan
    return pd.Series(values, index=quotes.index, name="smvVol")


def lookup_rate(rows, rates):
    """Rate per row: the ticker's row at obs_date, else its latest earlier one within 7 days; then a blend."""
    valid = (rows["ticker"].notna() & rows["obs_date"].notna()).to_numpy()
    keep = (rates["ticker"].notna() & rates["obs_date"].notna()).to_numpy()
    out = np.full(len(rows), np.nan)
    left = _norm(rows[valid], ["ticker", "obs_date", "expiry"])
    right = _norm(rates[keep], RATE_KEYS + RATE_COLUMNS)
    if left.empty or right.empty:
        return pd.Series(out, index=rows.index, name="rate")
    left["_pos"] = np.flatnonzero(valid)
    m = pd.merge_asof(left.sort_values("obs_date", kind="mergesort"), right.sort_values("obs_date", kind="mergesort"),
                      on="obs_date", by="ticker", direction="backward", tolerance=pd.Timedelta(days=RATE_LOOKBACK_DAYS))
    days = np.maximum((m["expiry"] - m["obs_date"]).dt.days.to_numpy(dtype=float), 0.0)
    r5, rl = m["iRate5wk"].to_numpy(dtype=float), m["iRateLt"].to_numpy(dtype=float)
    weight = np.clip((days - SHORT_DAYS) / (LONG_DAYS - SHORT_DAYS), 0.0, 1.0)
    blend = np.where(np.isfinite(r5) & np.isfinite(rl), r5 + weight * (rl - r5), np.where(np.isfinite(r5), r5, rl))
    out[m["_pos"].to_numpy()] = np.where(np.isfinite(days), blend, np.nan)
    return pd.Series(out, index=rows.index, name="rate")


def usable_quote(bid, ask):
    """Design section 7's unpriceable, negated: both finite, bid >= 0, ask > 0, bid <= ask."""
    b, a = _floats(bid), _floats(ask)
    return np.isfinite(b) & np.isfinite(a) & (b >= 0) & (a > 0) & (b <= a)


def entry_coverage(entry_rows, tables, *, emit=print):
    """Input coverage of ENTRY rows by obs_date year and chain_kind. Reads no outcome column."""
    n = len(entry_rows)
    flags = pd.DataFrame({
        "year": pd.to_datetime(entry_rows["obs_date"]).dt.year.to_numpy() if n else np.zeros(0, dtype=int),
        "chain_kind": entry_rows["chain_kind"].map(lambda v: "null" if pd.isna(v) else str(v)).to_numpy(dtype=object),
        "usable_quote": usable_quote(entry_rows["bid"], entry_rows["ask"]),
        "smvvol_positive": attach_vol(entry_rows, tables.vols).notna().to_numpy() if n else np.zeros(0, dtype=bool),
        "rate_found": lookup_rate(entry_rows, tables.rates).notna().to_numpy() if n else np.zeros(0, dtype=bool),
        "open_interest_known": np.isfinite(_floats(entry_rows["open_interest"])),
        "bid_size_known": np.isfinite(_floats(entry_rows["bid_size"])),
        "ask_size_known": np.isfinite(_floats(entry_rows["ask_size"])),
    })
    columns = ["year", "chain_kind", "rows"] + [c for name in _COUNTED for c in (name, name + "_share")]
    out = pd.DataFrame(columns=columns)
    if n:
        grouped = flags.groupby(["year", "chain_kind"], sort=True)
        out = grouped[_COUNTED].sum()
        out.insert(0, "rows", grouped.size())
        out = out.reset_index()
        for name in _COUNTED:
            out[name + "_share"] = out[name] / out["rows"]
        out = out[columns]
    emit("\n".join(["[fvf] entry input coverage", out.to_string(index=False)]))
    return out
