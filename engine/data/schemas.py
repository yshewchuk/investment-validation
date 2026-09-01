"""Tier-2 table schemas, and the gate that enforces them.

One coherent cross-source schema replaces four per-source formats. Everything
downstream reads these five tables and nothing else:

``securities``       symbology, listing range, era-normalized market cap
``earnings_events``  one row per (ticker, event date) with BMO/AMC session
``daily_market``     ticker × date: spot, IV terms, implied move, rvol, skew
``option_chains``    ticker × obs_date × expiry × strike × right: bid/ask/IV
``trades``           every simulated, paper, and live trade in one shape

Two rules make this worth the indirection:

* **Every audited field carries a source column.** When two sources disagree
  the row records which one won, so a later question about provenance is
  answerable from the data rather than from memory.
* **Every unit and convention trap is fixed here, once.** Not in every
  consumer. The traps are enumerated in :data:`CONVENTIONS` and implemented in
  ``engine/data/normalize/``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

__all__ = [
    "Column",
    "TableSchema",
    "SCHEMAS",
    "assert_schema",
    "coerce",
    "empty_frame",
    "SchemaError",
    "CONVENTIONS",
    "SOURCE_PRIORITY",
]


class SchemaError(AssertionError):
    """A frame does not match its declared Tier-2 schema."""


#: The conventions normalizers are responsible for, quoted where they came from.
CONVENTIONS = {
    "orats_mktcap_units": (
        "ORATS `mktCap` changes units twice: BILLIONS before 2017-06-28, "
        "MILLIONS from 2017-06-28 to 2026-03-10, THOUSANDS from 2026-03-11. "
        "Normalized to true USD in `securities.mcap_usd`. The legacy master "
        "panel applied x1e6 to everything before 2026-03-11, so its "
        "`or_mcap_log` is understated by log(1000) for events before "
        "2017-06-28; see the Phase 0 report."
    ),
    "realized_move": (
        "BMO = close(t-1) -> close(t); AMC = close(t) -> close(t+1) "
        "(validated, EXP-000)."
    ),
    "oquants_implied_pre_2022": (
        "oquants implied moves before 2022 are reconstructions, not live "
        "quotes; flagged `implied_reconstructed=True`."
    ),
    "orats_flt_max_sentinel": (
        "ORATS uses FLT_MAX (~3.4e38) as a missing-value sentinel in numeric "
        "fields (0.097% of cells). Masked to NaN at normalization."
    ),
    "orats_strikes_layout": (
        "Chain files are {entry_date, tickers, rows:[...]} with `_b{N}` batch "
        "naming and `.done*` markers; kinds are S2 entry, post-print exit, and "
        "T-14 (`_t14_`), plus the `_c2_` calendar pull."
    ),
    "polygon_options_window": (
        "Polygon options on this plan start 2024-08-19 and cover daily "
        "aggregates only (plus reference and live snapshots). Tick trades, "
        "NBBO quotes and intraday bars are not entitled (probed 2026-08-30), "
        "so option_daily exists only from that date and carries no bid/ask — "
        "real traded prices, not quotes."
    ),
    "polygon_equity_window": (
        "Polygon EQUITY aggregates on this plan carry only the trailing ~2 "
        "years (probed 2026-09-01, EXP-117): a historical window returns 403 "
        "NOT_AUTHORIZED 'plan doesn't include this data timeframe', while a "
        "wide range silently returns the last 500 bars. Any historical "
        "underlying-price validation must treat Polygon as a recent-era "
        "source only; plan upgrades change this."
    ),
}

#: Which source wins where two disagree. Written down because it is a decision,
#: not a fact, and every consumer must make the same one.
SOURCE_PRIORITY = {
    "option_chains": "orats",
    "option_daily": "polygon (real traded bars; 2024-08-19+ only on this plan)",
    "realized_moves": "oquants (OHLCV-validated panel)",
    "calendar": (
        "orats (anncTod, 99.52% agreement — EXP-038) for events that have "
        "happened. ORATS /hist/earnings carries NO forward dates, so the "
        "upcoming calendar comes from nasdaq (one call per date, whole market) "
        "and yfinance (per ticker); session priority is orats > yfinance > "
        "nasdaq — see engine.calendar.SESSION_PRIORITY for the measured "
        "agreement behind that order. Every row names its session source in "
        "`session_src`."
    ),
    "spot": "orats, cross-checked against yfinance close (1.3% tolerance)",
}


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    nullable: bool = True
    doc: str = ""


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...]
    partition_by: str | None = "year"
    doc: str = ""

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def column(self, name: str) -> Column:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"{self.name} has no column {name!r}")

    @property
    def dtypes(self) -> dict[str, str]:
        return {c.name: c.dtype for c in self.columns}


def _cols(*specs: tuple) -> tuple[Column, ...]:
    return tuple(Column(*spec) for spec in specs)


SECURITIES = TableSchema(
    name="securities",
    doc="One row per (ticker, as-of year): symbology and era-normalized size.",
    primary_key=("ticker", "year"),
    columns=_cols(
        ("ticker", "string", False, "Symbol as ORATS/oquants spell it"),
        ("year", "int64", False, "Partition year of the observation"),
        ("first_date", "datetime64[ns]", True, "First date with any observation"),
        ("last_date", "datetime64[ns]", True, "Last date with any observation"),
        ("mcap_usd", "float64", True, "Market cap in TRUE USD (3-era conversion)"),
        ("mcap_log", "float64", True, "log(mcap_usd)"),
        ("mcap_raw", "float64", True, "ORATS mktCap as delivered, pre-conversion"),
        ("mcap_unit_era", "string", True, "billions | millions | thousands"),
        (
            "mcap_quantized",
            "bool",
            True,
            "Pre-2017-06-28: mktCap is an INTEGER count of billions, so small "
            "caps are quantized to ~$0.5B and sub-$0.5B names round to zero",
        ),
        ("n_obs", "int64", True, "Daily observations backing this row"),
        ("src", "string", True, "Source of the size figure"),
    ),
)

EARNINGS_EVENTS = TableSchema(
    name="earnings_events",
    doc=(
        "The canonical calendar. ORATS is authoritative and oquants the "
        "cross-check, but both are history-only: Nasdaq and yfinance are what "
        "carry the FORWARD dates the monitoring board scores."
    ),
    primary_key=("event_id",),
    columns=_cols(
        ("event_id", "string", False, "{ticker}_{YYYY-MM-DD}"),
        ("ticker", "string", False, ""),
        ("event_date", "datetime64[ns]", False, ""),
        ("year", "int64", False, ""),
        ("session", "string", True, "BMO | AMC, by engine.calendar.SESSION_PRIORITY"),
        ("session_src", "string", True, "Which source supplied `session`"),
        ("annc_tod", "string", True, "Announcement time HHMM, from the session's source"),
        ("src_orats", "bool", False, "Present in the ORATS calendar (history only)"),
        ("src_oquants", "bool", False, "Present in the oquants panel (history only)"),
        ("src_nasdaq", "bool", False, "Present in the Nasdaq calendar (forward)"),
        ("src_yfinance", "bool", False, "Present in the yfinance calendar (forward + history)"),
        ("date_agree", "bool", False, "Both HISTORICAL sources carry this date"),
        ("date_conflict", "bool", False,
         "Forward sources disagree about this print's date (both rows kept)"),
        ("updated_at", "string", True, "Last-update stamp from the contributing source"),
    ),
)

DAILY_MARKET = TableSchema(
    name="daily_market",
    doc="Daily per-ticker market state. One row per (ticker, date).",
    primary_key=("ticker", "date"),
    columns=_cols(
        ("ticker", "string", False, ""),
        ("date", "datetime64[ns]", False, "Trade date (EOD)"),
        ("year", "int64", False, ""),
        ("spot", "float64", True, "ORATS stockPrice"),
        ("iv10", "float64", True, "iv10d, in vol points (%)"),
        ("iv30", "float64", True, "iv30d, in vol points (%)"),
        ("exern_iv10", "float64", True, "exErnIv10d (%)"),
        ("exern_iv30", "float64", True, "exErnIv30d (%)"),
        ("implied_move", "float64", True, "Quoted pre-earnings implied move (%)"),
        ("implied_reconstructed", "bool", True, "True where the figure is a reconstruction"),
        ("rvol30", "float64", True, "Realized vol, 30d (%)"),
        ("skew", "float64", True, "ORATS skewing"),
        ("contango", "float64", True, "ORATS contango"),
        ("fwd90_30", "float64", True, "Forward vol 90/30 (%)"),
        ("fexern90_30", "float64", True, "Ex-earnings forward vol 90/30 (%)"),
        ("iee", "float64", True, "ieeEarnEffect"),
        ("mcap_usd", "float64", True, "Market cap in TRUE USD"),
        ("mcap_log", "float64", True, "log(mcap_usd)"),
        ("mcap_asof", "datetime64[ns]", True, "Date the market cap was observed"),
        (
            "mcap_age_days",
            "float64",
            True,
            "Calendar days between `date` and `mcap_asof`. The backward join "
            "carries the last known cap forward, so a large value means a "
            "stale figure (a name whose cores series stopped), not a fresh one",
        ),
        ("src_spot", "string", True, ""),
        ("src_iv", "string", True, ""),
        ("src_mcap", "string", True, ""),
    ),
)

OPTION_CHAINS = TableSchema(
    name="option_chains",
    doc="Real EOD chains. The only sanctioned P&L price source.",
    primary_key=("ticker", "obs_date", "expiry", "strike", "right"),
    columns=_cols(
        ("ticker", "string", False, ""),
        ("obs_date", "datetime64[ns]", False, "Chain observation (trade) date"),
        ("year", "int64", False, ""),
        ("expiry", "datetime64[ns]", False, ""),
        ("dte", "int64", False, "expiry - obs_date, calendar days"),
        ("strike", "float64", False, ""),
        ("right", "string", False, "C | P"),
        ("bid", "float64", True, ""),
        ("ask", "float64", True, ""),
        ("mid", "float64", True, "(bid + ask) / 2"),
        ("iv", "float64", True, "Side-specific mid IV"),
        ("delta", "float64", True, "Call delta as ORATS reports it"),
        ("spot", "float64", True, "Underlying price on obs_date"),
        ("src", "string", True, "Source system"),
        ("src_file", "string", True, "Raw file this row was parsed from"),
        ("chain_kind", "string", True, "entry | exit | t14 | c2 — why it was pulled"),
        # Liquidity, present only on chains pulled from 2026-09 onward: the
        # earlier 19,061-file cache did not request these fields. Nullable on
        # purpose — "we never asked" and "there was no size" must not read
        # alike, and a zero would say the second.
        ("volume", "float64", True, "Contracts traded on obs_date, this side"),
        ("open_interest", "float64", True, "Open interest, this side"),
        ("bid_size", "float64", True, "Contracts resting at the bid"),
        ("ask_size", "float64", True, "Contracts resting at the ask"),
        (
            "quote_repaired",
            "bool",
            True,
            "The raw quote was crossed (bid > ask) and was collapsed to "
            "min(bid, ask). Excluding these rows instead drops whole trades, "
            "and they concentrate in the biggest movers — see validate.py",
        ),
    ),
)

OPTION_DAILY = TableSchema(
    name="option_daily",
    doc=(
        "Real traded bars per contract per day — Polygon daily aggregates. "
        "Close/VWAP/volume/trade-count are actual fills, the evidence the "
        "option_chains quotes and the fill model are calibrated against. "
        "Exists only from 2024-08-19 (the start of Polygon options on this "
        "plan); no bid/ask here by construction — see the "
        "polygon_options_window convention."
    ),
    primary_key=("contract_ticker", "obs_date"),
    columns=_cols(
        ("contract_ticker", "string", False, "OCC id, e.g. O:TSLA240906C00210000"),
        ("ticker", "string", False, "Underlying symbol"),
        ("obs_date", "datetime64[ns]", False, "Trade date of the bar (UTC date)"),
        ("year", "int64", False, "Partition year of obs_date"),
        ("expiry", "datetime64[ns]", False, "Parsed from the OCC id"),
        ("strike", "float64", False, ""),
        ("right", "string", False, "C | P"),
        ("open", "float64", True, ""),
        ("high", "float64", True, ""),
        ("low", "float64", True, ""),
        ("close", "float64", True, "Last traded price of the day"),
        ("vwap", "float64", True, "Volume-weighted average of real fills"),
        ("volume", "float64", True, "Contracts traded on the day"),
        ("n_trades", "int64", True, "Number of fills that built the bar"),
        ("src", "string", True, "Source system"),
        ("src_file", "string", True, "Tier-1 payload this row was parsed from"),
    ),
)

TRADES = TableSchema(
    name="trades",
    doc="Simulated, paper, and live trades in one schema.",
    primary_key=("trade_id",),
    columns=_cols(
        ("trade_id", "string", False, ""),
        ("kind", "string", False, "sim | paper | live"),
        ("strategy", "string", False, "CAL-P | STR-THRU | STR-RUNUP | ..."),
        ("variant", "string", True, "Parameterization within a strategy"),
        ("ticker", "string", False, ""),
        ("event_id", "string", True, "FK to earnings_events"),
        ("event_date", "datetime64[ns]", True, ""),
        ("year", "int64", False, ""),
        ("legs", "string", True, "JSON leg list, as resolved"),
        ("entry_date", "datetime64[ns]", True, ""),
        ("exit_date", "datetime64[ns]", True, ""),
        ("strike", "float64", True, "Primary strike (ATM leg)"),
        ("expiry", "datetime64[ns]", True, "Primary expiry"),
        ("fill_alpha", "float64", True, "FillModel alpha this row was priced at"),
        ("entry_cost", "float64", True, "Net debit"),
        ("exit_value", "float64", True, "Net proceeds on close"),
        ("ret", "float64", True, "(exit_value - entry_cost) / entry_cost"),
        ("provenance", "string", True, "Where this row came from"),
    ),
)

SCHEMAS: dict[str, TableSchema] = {
    s.name: s
    for s in (SECURITIES, EARNINGS_EVENTS, DAILY_MARKET, OPTION_CHAINS, OPTION_DAILY, TRADES)
}


# --------------------------------------------------------------------------
# enforcement
# --------------------------------------------------------------------------

_PANDAS_DTYPE = {
    "string": "string",
    "float64": "float64",
    "int64": "Int64",  # nullable integer: a missing DTE must not silently become 0
    "bool": "boolean",
    "datetime64[ns]": "datetime64[ns]",
}


def empty_frame(name: str) -> pd.DataFrame:
    """An empty frame with the right columns and dtypes."""
    schema = SCHEMAS[name]
    return pd.DataFrame(
        {c.name: pd.Series(dtype=_PANDAS_DTYPE[c.dtype]) for c in schema.columns}
    )


def coerce(
    df: pd.DataFrame,
    name: str,
    *,
    allow_extra: bool = False,
    only: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Reorder and cast ``df`` to the declared schema.

    Missing nullable columns are filled with nulls, so a normalizer that cannot
    source an optional field does not have to fabricate one. Missing
    non-nullable columns are an error — that is a normalizer bug, not a data gap.

    ``only`` restricts the operation to a subset of columns, for a projected
    read: the dtypes are still enforced, but a column the caller did not ask for
    is not conjured into existence.
    """
    schema = SCHEMAS[name]
    out = df.copy()
    if only is not None:
        wanted = [c for c in schema.columns if c.name in set(only)]
        unknown = sorted(set(only) - set(schema.names))
        if unknown:
            raise SchemaError(f"{name}: unknown column(s) {unknown}")
    else:
        wanted = list(schema.columns)
        missing_required = [
            c.name for c in schema.columns if not c.nullable and c.name not in out.columns
        ]
        if missing_required:
            raise SchemaError(f"{name}: missing required columns {missing_required}")

    for col in wanted:
        if col.name not in out.columns:
            out[col.name] = pd.Series([pd.NA] * len(out), index=out.index)
        target = _PANDAS_DTYPE[col.dtype]
        try:
            if col.dtype == "datetime64[ns]":
                # pandas 3 parses to microsecond resolution and Parquet may
                # round-trip to yet another unit; pin nanoseconds so a dtype
                # comparison is a real check rather than a platform quirk.
                out[col.name] = pd.to_datetime(out[col.name], errors="coerce").astype(
                    "datetime64[ns]"
                )
            elif col.dtype == "int64":
                out[col.name] = pd.to_numeric(out[col.name], errors="coerce").astype(target)
            elif col.dtype == "float64":
                out[col.name] = pd.to_numeric(out[col.name], errors="coerce").astype("float64")
            else:
                out[col.name] = out[col.name].astype(target)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"{name}.{col.name}: cannot cast to {col.dtype} ({exc})") from exc

    ordered = [c.name for c in wanted]
    extra = [c for c in out.columns if c not in schema.names]
    if extra and not allow_extra:
        out = out.drop(columns=extra)
        extra = []
    return out[ordered + extra]


def assert_schema(df: pd.DataFrame, name: str, *, check_keys: bool = True) -> None:
    """Raise :class:`SchemaError` unless ``df`` satisfies its table contract.

    Checks columns, dtypes, non-null constraints, and primary-key uniqueness.
    Called by every normalizer before a frame is allowed into ``curated/``, so a
    schema drift shows up at write time rather than three phases downstream.
    """
    if name not in SCHEMAS:
        raise SchemaError(f"unknown table {name!r}; known: {sorted(SCHEMAS)}")
    schema = SCHEMAS[name]

    missing = [c for c in schema.names if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing}")

    for col in schema.columns:
        actual = str(df[col.name].dtype)
        expected = _PANDAS_DTYPE[col.dtype]
        if actual != expected:
            raise SchemaError(
                f"{name}.{col.name}: dtype {actual!r}, expected {expected!r}"
            )
        if not col.nullable and df[col.name].isna().any():
            n = int(df[col.name].isna().sum())
            raise SchemaError(f"{name}.{col.name}: {n} null(s) in a non-nullable column")

    if check_keys and schema.primary_key and len(df):
        dupes = df.duplicated(subset=list(schema.primary_key)).sum()
        if dupes:
            raise SchemaError(
                f"{name}: {int(dupes)} duplicate row(s) on primary key "
                f"{schema.primary_key}"
            )
