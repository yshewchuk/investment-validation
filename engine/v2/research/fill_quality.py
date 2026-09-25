"""Measured fill quality: real Polygon trades vs ORATS quoted bid/ask.

The mid-fill assumption is the program's verdict flipper, and until this table
existed there was no evidence for it — only the assumption. This tool joins
``option_chains`` (ORATS EOD quotes) with ``option_daily`` (Polygon real traded
bars) on (contract, date) and measures, wherever both exist:

* how far the last real trade and the day's VWAP sit from the quoted mid,
  expressed as an implied fill alpha in [0, 1] (0 = filled at the touch against
  you, 0.5 = mid, 1 = filled at the touch in your favour, outside = through it);
* how that gap behaves as a function of liquidity — trade count, volume, and
  the quoted relative spread — which is the slicing any fill model must
  condition on.

Phase 6 slice 6 moved this from ``tools/fill_quality.py``. The join arithmetic
is unchanged; both table reads are now bounded ``Repository.scan`` calls
against one pinned snapshot, and :func:`run` writes the snapshot id into the
output::

    python3 tools/v2_fill_quality.py --catalog <cat.sqlite> --store-root <objects>

Read-only on the snapshot; writes the joined table and a summary to
``reports/``.

Caveat, stated in the output: the Polygon close is the day's LAST trade and the
VWAP is over the whole day, while the ORATS quote is the EOD quote; on thin
names the two did not necessarily happen at the same instant. That is why the
liquidity slices are printed next to the averages — an alpha measured on one
trade a day is not an alpha you can execute.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from engine.v2.contracts.data import KeyPredicate
from engine.v2.data import errors
from engine.v2.research._scan import DEFAULT_SCOPE, read_table, resolve_snapshot

__all__ = [
    "OPTION_CHAIN_COLUMNS",
    "OPTION_CHAIN_TABLE",
    "OPTION_DAILY_COLUMNS",
    "OPTION_DAILY_TABLE",
    "contract_ticker_column",
    "join_from_snapshot",
    "join_quotes_and_trades",
    "read_option_chains",
    "read_option_daily",
    "run",
    "summarize",
    "summary_text",
    "write_report",
]

OPTION_DAILY_TABLE = "option_daily"
OPTION_CHAIN_TABLE = "option_chains"
#: Exactly the ``option_daily`` columns the join and summary read.
OPTION_DAILY_COLUMNS = ("contract_ticker", "obs_date", "close", "vwap", "n_trades", "volume")
#: Exactly the ``option_chains`` columns the join key and measured gaps read.
OPTION_CHAIN_COLUMNS = ("ticker", "obs_date", "expiry", "strike", "right", "bid", "ask",
                        "mid", "spot", "chain_kind")


def contract_ticker_column(chains: pd.DataFrame) -> pd.Series:
    """The OCC id for each chain row, built the same way the pull builds jobs."""
    strike_int = (chains["strike"] * 1000).round().astype("int64")
    return (
        "O:"
        + chains["ticker"].astype(str)
        + chains["expiry"].dt.strftime("%y%m%d")
        + chains["right"].astype(str)
        + strike_int.astype(str).str.zfill(8)
    )


def read_option_daily(repository, snapshot_ref) -> pd.DataFrame:
    """Polygon real traded bars, read through the pinned snapshot."""
    return read_table(repository, snapshot_ref, OPTION_DAILY_TABLE, OPTION_DAILY_COLUMNS)


#: The OCC contract-id shape :func:`contract_ticker_column` writes (and the
#: pull's legacy ``option_ticker`` writes): parsed back here so the chain read
#: can be pinned to the traded contracts without importing the legacy pull.
_CONTRACT_TICKER_RE = re.compile(
    r"^O:(?P<ticker>.+)(?P<yymmdd>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")


def _contract_components(contract_ticker: str) -> tuple[str, str, str]:
    """``(ticker, expiry, right)`` parsed from one OCC contract id."""
    match = _CONTRACT_TICKER_RE.match(contract_ticker)
    if match is None:
        raise errors.fail("CONTRACT_MISMATCH",
                          "option_daily carries a contract_ticker that is not OCC-shaped")
    yymmdd = match["yymmdd"]
    return (match["ticker"], f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:]}", match["right"])


def _chain_key_filter(trades: pd.DataFrame) -> tuple[KeyPredicate, ...]:
    """The ``option_chains`` key filter covering the traded contract-days.

    ``strike`` is a float64 column and ``KeyPredicate.values`` excludes floats
    (component contracts §2.1), so the filter pins every other primary-key
    component; the join on ``(contract_ticker, obs_date)`` still enforces
    exact contract identity on whatever the scan returns.
    """
    if trades.empty:
        return ()
    tickers, expiries, rights = set(), set(), set()
    for contract in trades["contract_ticker"].dropna():
        ticker, expiry, right = _contract_components(str(contract))
        tickers.add(ticker)
        expiries.add(expiry)
        rights.add(right)
    obs_dates = {pd.Timestamp(value).strftime("%Y-%m-%d")
                 for value in trades["obs_date"].dropna()}
    columns = (("ticker", tickers), ("obs_date", obs_dates),
               ("expiry", expiries), ("right", rights))
    return tuple(KeyPredicate(column=column, operator="in", values=tuple(sorted(values)))
                 for column, values in columns if values)


def read_option_chains(repository, snapshot_ref, *, years=None,
                       key_filter=()) -> pd.DataFrame:
    """ORATS EOD quotes, read through the pinned snapshot (optionally by year)."""
    return read_table(repository, snapshot_ref, OPTION_CHAIN_TABLE, OPTION_CHAIN_COLUMNS,
                      partition_keys=years, key_filter=key_filter)


def join_quotes_and_trades(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Inner join of ORATS quotes and Polygon traded bars on (contract, date)."""
    if trades.empty:
        return trades
    if quotes.empty:
        return pd.DataFrame()

    quotes = quotes.copy()
    quotes["contract_ticker"] = contract_ticker_column(quotes)
    out = trades.merge(
        quotes[["contract_ticker", "obs_date", "bid", "ask", "mid", "spot", "chain_kind"]],
        on=["contract_ticker", "obs_date"],
        how="inner",
        validate="one_to_one",
    )
    out["rel_spread"] = np.where(
        out["mid"] > 0, (out["ask"] - out["bid"]) / out["mid"], np.nan
    )
    width = out["ask"] - out["bid"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["alpha_close"] = np.where(
            width > 0, (out["ask"] - out["close"]) / width, np.nan
        )
        out["alpha_vwap"] = np.where(
            width > 0, (out["ask"] - out["vwap"]) / width, np.nan
        )
    return out


def join_from_snapshot(repository, snapshot_ref) -> pd.DataFrame:
    """The legacy flow, split at its two store reads: read both tables from the
    pinned snapshot, then run the pure join on the resulting frames.

    The chain read carries a key filter for the traded contract-days, so a
    year-sized chain table is not read in full to discard almost all of it.
    """
    trades = read_option_daily(repository, snapshot_ref)
    if trades.empty:
        return trades
    years = sorted({int(y) for y in trades["obs_date"].dt.year})
    quotes = read_option_chains(repository, snapshot_ref, years=years,
                                key_filter=_chain_key_filter(trades))
    return join_quotes_and_trades(trades, quotes)


def _bucket(n: pd.Series) -> pd.Series:
    return pd.cut(
        n.fillna(0),
        bins=[-1, 1, 5, 50, np.inf],
        labels=["1", "2-5", "6-50", "50+"],
    )


def summarize(joined: pd.DataFrame) -> pd.DataFrame:
    """Per-liquidity-bucket stats of the measured fill gaps."""
    df = joined.copy()
    df["trades_bucket"] = _bucket(df["n_trades"])
    rows = []
    for bucket, chunk in df.groupby("trades_bucket", observed=True):
        rows.append(
            {
                "n_trades_bucket": str(bucket),
                "days": len(chunk),
                "contracts": int(chunk["contract_ticker"].nunique()),
                "median_alpha_close": round(float(chunk["alpha_close"].median()), 3),
                "median_alpha_vwap": round(float(chunk["alpha_vwap"].median()), 3),
                "pct_close_beyond_worst": round(
                    float(
                        (
                            (chunk["alpha_close"] < 0) | (chunk["alpha_close"] > 1)
                        ).mean()
                    ),
                    3,
                ),
                "median_rel_spread": round(float(chunk["rel_spread"].median()), 3),
                "median_volume": float(chunk["volume"].median()),
            }
        )
    return pd.DataFrame(rows)


def summary_text(table: pd.DataFrame, snapshot_id: str) -> str:
    """The printed summary as markdown, with the pinned snapshot named."""
    return "\n".join([
        "# Fill quality — measured",
        "",
        f"Pinned snapshot: `{snapshot_id}`",
        "",
        "_no rows_" if table.empty else table.to_string(index=False),
        "",
        "alpha = 0 is a fill at the touch against you, 0.5 is mid, 1 is the "
        "touch in your favour; beyond [0,1] the trade happened THROUGH the "
        "quoted spread. Median over days, bucketed by how many real trades "
        "the contract saw that day.",
        "",
    ])


def write_report(joined: pd.DataFrame, table: pd.DataFrame, *, snapshot_id: str,
                 out_dir: Path, stamp: str, csv: str | None = None) -> dict[str, str]:
    """Write the joined rows and summary, both carrying ``snapshot_id``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"fill_quality_{stamp}.parquet"
    joined.assign(snapshot_id=snapshot_id).to_parquet(parquet_path, index=False)
    md_path = out_dir / f"fill_quality_{stamp}.md"
    md_path.write_text(summary_text(table, snapshot_id))
    paths = {"parquet": str(parquet_path), "md": str(md_path)}
    if csv:
        joined.assign(snapshot_id=snapshot_id).to_csv(csv, index=False)
        paths["csv"] = str(csv)
    return paths


def run(repository, *, reports_dir: Path = Path("reports"), scope: str = DEFAULT_SCOPE,
        snapshot_id: str | None = None, since: str | None = None,
        csv: str | None = None, stamp: str | None = None) -> dict:
    """Read one pinned snapshot, measure the gaps, and write the report pair."""
    snapshot = resolve_snapshot(repository, scope=scope, snapshot_id=snapshot_id)
    joined = join_from_snapshot(repository, snapshot)
    if since:
        joined = joined[joined["obs_date"] >= pd.Timestamp(since)]
    if joined.empty:
        raise errors.fail("POPULATION_COLLAPSED", "no overlap rows in this snapshot")
    table = summarize(joined)
    stamp = stamp or pd.Timestamp.now().strftime("%Y-%m-%d")
    paths = write_report(joined, table, snapshot_id=snapshot.snapshot_id,
                         out_dir=Path(reports_dir), stamp=stamp, csv=csv)
    return {"snapshot_id": snapshot.snapshot_id, "rows": int(len(joined)),
            "contracts": int(joined["contract_ticker"].nunique()),
            "summary": table, "paths": paths}
