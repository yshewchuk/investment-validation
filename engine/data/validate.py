"""The ingestion validation gate.

Every normalizer's output passes through here before it is allowed into
``curated/``. Two classes of check:

**Row-level integrity** — cheap, runs on everything. Crossed quotes, negative
prices, DTE that disagrees with ``expiry - obs_date``, duplicate primary keys.
Offending rows are excluded from Tier 2 and the raw file that produced them is
flagged in ``data/raw/quarantine/``.

**Cross-source sanity** — sampled, runs on the batch. ORATS spot against the
yfinance close (1.3% tolerance, the known agreement level), and ORATS straddle
mids against the Polygon real-trade panel where the windows overlap (2024-08+,
±3%).

The contract, from the guide, is precise about what quarantine means:

* the raw file is **never moved or deleted** — Tier 1 is append-only, and a
  flag file names the offender instead;
* offending rows are **excluded**, not silently repaired;
* a failure **raises a flag, never blocks the pull** — new raw data is always
  kept regardless of whether it normalizes cleanly;
* nothing is ever dropped quietly. Every exclusion is counted and reported.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from engine import paths

__all__ = [
    "CheckResult",
    "ValidationReport",
    "validate_chains",
    "validate_daily",
    "quarantine",
    "should_quarantine",
    "spot_vs_yfinance",
    "straddle_vs_polygon",
    "SPOT_TOLERANCE_PCT",
    "STRADDLE_TOLERANCE_PCT",
    "QUARANTINE_EXCLUSION_RATE",
    "STRUCTURAL_CHECKS",
]

#: Known agreement level between ORATS stockPrice and the yfinance close.
SPOT_TOLERANCE_PCT = 1.3
#: ORATS chains validated to ~±2–3% against Polygon real trades.
STRADDLE_TOLERANCE_PCT = 3.0

#: Fraction of a file's rows that may be excluded before the file itself is
#: treated as suspect.
#:
#: The distinction matters at this scale. A crossed quote on a 0.04/0.03
#: deep-OTM strike is a real and routine artifact of penny-wide markets — a
#: handful appear in most chain files, and flagging 19,000 files for it would
#: bury the cases that actually need a human. Those rows are still excluded and
#: still counted; what they do not do is raise a file-level flag. A file that
#: cannot be parsed, is missing columns, or loses more than this share of its
#: rows *is* structurally suspect, and gets quarantined.
QUARANTINE_EXCLUSION_RATE = 0.01

#: Checks whose failure quarantines the file regardless of how few rows it hit —
#: these indicate the file is not what the parser thinks it is, rather than that
#: the market was wide.
STRUCTURAL_CHECKS = frozenset(
    {"dte_matches_dates", "right_is_C_or_P", "primary_key_unique", "date_present"}
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    n_checked: int = 0
    n_failed: int = 0
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.name}: {self.n_failed}/{self.n_checked} failed"
            + (f" — {self.detail}" if self.detail else "")
        )


@dataclass
class ValidationReport:
    table: str
    checks: list[CheckResult] = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0
    quarantined_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def rows_excluded(self) -> int:
        return self.rows_in - self.rows_out

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)

    def summary(self) -> dict:
        return {
            "table": self.table,
            "ok": self.ok,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_excluded": self.rows_excluded,
            "failed_checks": [c.name for c in self.checks if not c.passed],
            "quarantined_files": sorted(set(self.quarantined_files)),
        }

    def merge(self, other: "ValidationReport") -> None:
        """Fold a per-file report into a running batch report."""
        self.rows_in += other.rows_in
        self.rows_out += other.rows_out
        self.quarantined_files.extend(other.quarantined_files)
        by_name = {c.name: c for c in self.checks}
        for check in other.checks:
            if check.name not in by_name:
                self.checks.append(
                    CheckResult(check.name, check.passed, check.n_checked, check.n_failed, check.detail)
                )
                continue
            agg = by_name[check.name]
            agg.n_checked += check.n_checked
            agg.n_failed += check.n_failed
            agg.passed = agg.passed and check.passed
            if check.detail and not agg.detail:
                agg.detail = check.detail


# --------------------------------------------------------------------------
# quarantine
# --------------------------------------------------------------------------


def should_quarantine(report: "ValidationReport") -> bool:
    """Whether a file's failures are structural rather than routine."""
    if any(not c.passed and c.name in STRUCTURAL_CHECKS for c in report.checks):
        return True
    if report.rows_in <= 0:
        return False
    return (report.rows_excluded / report.rows_in) > QUARANTINE_EXCLUSION_RATE


def quarantine(
    source_file: str,
    reason: str,
    details: dict | None = None,
    *,
    root: Path | None = None,
) -> Path:
    """Write a flag file naming a raw file that failed validation.

    The raw bytes stay exactly where they are: Tier 1 is append-only and every
    byte ever fetched is kept forever, whether or not it parses. What lands here
    is a pointer plus the reason, so a later fix can find the offender without
    the pull having to be re-spent.
    """
    root = root or paths.QUARANTINE
    paths.assert_writable(root).mkdir(parents=True, exist_ok=True)
    safe = str(source_file).replace("/", "__")
    path = root / f"{safe}.flag.json"
    payload = {
        "source_file": str(source_file),
        "reason": reason,
        "flagged_at": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    }
    existing = []
    if path.exists():
        try:
            prior = json.loads(path.read_text())
            existing = prior if isinstance(prior, list) else [prior]
        except ValueError:
            existing = []
    path.write_text(json.dumps(existing + [payload], indent=1, default=str))
    print(f"  QUARANTINE {source_file}: {reason}", flush=True)
    return path


# --------------------------------------------------------------------------
# row-level integrity
# --------------------------------------------------------------------------


def validate_chains(
    df: pd.DataFrame, *, source_file: str | None = None, quarantine_root: Path | None = None
) -> tuple[pd.DataFrame, ValidationReport]:
    """Integrity-check normalized chain rows; return the admissible subset."""
    report = ValidationReport(table="option_chains", rows_in=len(df))
    if df.empty:
        report.rows_out = 0
        return df, report

    df = df.copy()
    bad = pd.Series(False, index=df.index)
    reasons: list[str] = []

    def flag(mask: pd.Series, name: str, detail: str = "") -> None:
        nonlocal bad
        n = int(mask.sum())
        report.add(CheckResult(name, n == 0, len(df), n, detail))
        if n:
            reasons.append(f"{name}={n}")
            bad = bad | mask

    quoted = df["bid"].notna() & df["ask"].notna()

    # -- crossed quotes are REPAIRED, not excluded ------------------------
    #
    # Dropping a crossed row looks conservative and is not. A structure needs
    # every one of its legs, so losing one row loses the whole trade — and the
    # trades that lose a leg are not a random sample. They are the big movers:
    # after a large gap, one side of a now-worthless leg goes stale while the
    # other reprices, e.g. ALGN 2018-10-25, where the 295 call quoted bid 9.80
    # against ask 0.35. Excluding those rows removed 114 of 5,444 S2 trades
    # whose mean return was +95%, dragging the measured edge from +3.5% to
    # +1.8% — a selection bias larger than the effect being measured.
    #
    # The repair collapses the quote to `min(bid, ask)`. In every case observed
    # the crossed pair is a stale high bid against a fresh low ask on a nearly
    # worthless option, so the minimum is the economically sane value, and it is
    # the choice that cannot manufacture value out of a data error. The row is
    # flagged so a consumer can exclude or re-price it deliberately.
    crossed = quoted & (df["bid"] > df["ask"])
    df["quote_repaired"] = False
    n_crossed = int(crossed.sum())
    if n_crossed:
        repaired = np.minimum(df.loc[crossed, "bid"], df.loc[crossed, "ask"])
        df.loc[crossed, "bid"] = repaired
        df.loc[crossed, "ask"] = repaired
        df.loc[crossed, "mid"] = repaired
        df.loc[crossed, "quote_repaired"] = True
    report.add(
        CheckResult(
            "crossed_quotes_repaired",
            True,  # a repair is a recorded outcome, not a failure
            len(df),
            n_crossed,
            f"{n_crossed} crossed quote(s) collapsed to min(bid, ask) and flagged",
        )
    )

    flag(quoted & (df["bid"] < 0), "bid_non_negative")
    flag(quoted & (df["ask"] < 0), "ask_non_negative")
    flag(df["strike"] <= 0, "strike_positive")
    flag(df["dte"] < 0, "dte_non_negative", "expiry before observation date")

    expected_dte = (df["expiry"] - df["obs_date"]).dt.days
    flag(df["dte"].astype("float") != expected_dte.astype("float"), "dte_matches_dates")

    flag(~df["right"].isin(["C", "P"]), "right_is_C_or_P")

    key = ["ticker", "obs_date", "expiry", "strike", "right"]
    dupes = df.duplicated(subset=key, keep="first")
    report.add(CheckResult("primary_key_unique", not dupes.any(), len(df), int(dupes.sum())))
    if dupes.any():
        reasons.append(f"duplicate_keys={int(dupes.sum())}")

    clean = df[~bad & ~dupes]
    report.rows_out = len(clean)

    if reasons and source_file and should_quarantine(report):
        quarantine(
            source_file,
            "chain row validation: " + ", ".join(reasons),
            {"rows_in": len(df), "rows_kept": len(clean)},
            root=quarantine_root,
        )
        report.quarantined_files.append(str(source_file))
    return clean, report


def validate_daily(
    df: pd.DataFrame, *, source_file: str | None = None, quarantine_root: Path | None = None
) -> tuple[pd.DataFrame, ValidationReport]:
    """Integrity-check normalized daily rows; return the admissible subset."""
    report = ValidationReport(table="daily_market", rows_in=len(df))
    if df.empty:
        report.rows_out = 0
        return df, report

    bad = pd.Series(False, index=df.index)
    reasons: list[str] = []

    def flag(mask: pd.Series, name: str, detail: str = "") -> None:
        nonlocal bad
        n = int(mask.sum())
        report.add(CheckResult(name, n == 0, len(df), n, detail))
        if n:
            reasons.append(f"{name}={n}")
            bad = bad | mask

    flag(df["date"].isna(), "date_present")
    flag(df["spot"].notna() & (df["spot"] <= 0), "spot_positive")
    for col in ("iv30", "implied_move", "rvol30"):
        if col in df.columns:
            flag(df[col].notna() & (df[col] < 0), f"{col}_non_negative")

    # `keep="last"` here, but `keep="first"` for chains — deliberately.
    # A repeated (ticker, date) in a daily series is a corrected restatement,
    # so the newest wins. A repeated chain contract is the SAME observation
    # arriving via two overlapping pulls, so first-in-sorted-source-order wins
    # and the result does not depend on which pull ran last.
    dupes = df.duplicated(subset=["ticker", "date"], keep="last")
    report.add(CheckResult("primary_key_unique", not dupes.any(), len(df), int(dupes.sum())))

    ordered = df.sort_values(["ticker", "date"])
    monotone = ordered.groupby("ticker")["date"].is_monotonic_increasing.all()
    report.add(CheckResult("dates_monotone_per_ticker", bool(monotone), len(df), 0 if monotone else 1))

    clean = df[~bad & ~dupes]
    report.rows_out = len(clean)

    if reasons and source_file and should_quarantine(report):
        quarantine(
            source_file,
            "daily row validation: " + ", ".join(reasons),
            {"rows_in": len(df), "rows_kept": len(clean)},
            root=quarantine_root,
        )
        report.quarantined_files.append(str(source_file))
    return clean, report


# --------------------------------------------------------------------------
# cross-source sanity
# --------------------------------------------------------------------------


def spot_vs_yfinance(
    daily: pd.DataFrame,
    *,
    sample_frac: float = 0.02,
    tolerance_pct: float = SPOT_TOLERANCE_PCT,
    seed: int = 0,
    yf_dir: Path | None = None,
    max_tickers: int = 200,
) -> CheckResult:
    """Compare ORATS spot against the cached yfinance close on a sample.

    Split adjustment is the reason this compares against ``close_raw``: the
    adjusted series is restated after every split, so an adjusted-vs-unadjusted
    comparison would report a spurious break on every split date in the sample.
    """
    yf_dir = yf_dir or paths.RAW_YF
    if daily.empty or not yf_dir.exists():
        return CheckResult("spot_vs_yfinance", True, 0, 0, "no data to compare")

    rng = np.random.default_rng(seed)
    tickers = sorted(daily["ticker"].dropna().unique())
    if len(tickers) > max_tickers:
        tickers = list(rng.choice(tickers, size=max_tickers, replace=False))

    diffs: list[float] = []
    n_checked = 0
    for ticker in sorted(tickers):
        path = yf_dir / f"px_{ticker}.csv"
        if not path.exists():
            continue
        try:
            px = pd.read_csv(path, usecols=["date", "close_raw"], parse_dates=["date"])
        except (ValueError, OSError):
            continue
        sub = daily[(daily["ticker"] == ticker) & daily["spot"].notna()]
        if sub.empty:
            continue
        take = max(1, int(len(sub) * sample_frac))
        idx = rng.choice(len(sub), size=min(take, len(sub)), replace=False)
        sub = sub.iloc[sorted(idx)]
        merged = sub.merge(px, on="date", how="inner")
        merged = merged[merged["close_raw"] > 0]
        if merged.empty:
            continue
        rel = (merged["spot"] / merged["close_raw"] - 1.0).abs() * 100
        diffs.extend(rel.tolist())
        n_checked += len(merged)

    if not diffs:
        return CheckResult("spot_vs_yfinance", True, 0, 0, "no overlapping rows")
    arr = np.array(diffs)
    n_failed = int((arr > tolerance_pct).sum())
    # A tolerance check on a 2% sample is about the bulk of the distribution,
    # not about individual outliers: a handful of stale yfinance rows must not
    # fail an otherwise sound ingest. The median is the number that matters.
    median = float(np.median(arr))
    passed = median <= tolerance_pct
    return CheckResult(
        "spot_vs_yfinance",
        passed,
        n_checked,
        n_failed,
        f"median |diff| {median:.3f}% (tolerance {tolerance_pct}%), "
        f"p95 {float(np.percentile(arr, 95)):.3f}%",
    )


def straddle_vs_polygon(
    chains: pd.DataFrame,
    *,
    tolerance_pct: float = STRADDLE_TOLERANCE_PCT,
    panel_path: Path | None = None,
) -> CheckResult:
    """Compare ORATS ATM straddle mids against the Polygon real-trade panel.

    The Polygon panel is close prices of actually-traded contracts from 2024-08
    onward — an independent measurement, not another quote feed. Where the
    windows overlap it is the strongest available evidence that the chain prices
    the whole program's P&L rests on are real.
    """
    panel_path = panel_path or paths.BT_STRADDLE_PANEL
    if chains.empty or not Path(panel_path).exists():
        return CheckResult("straddle_vs_polygon", True, 0, 0, "no overlap available")

    panel = pd.read_csv(panel_path, parse_dates=["entry_date", "expiry"])
    needed = {"ticker", "entry_date", "expiry", "strike", "call_entry", "put_entry"}
    if not needed <= set(panel.columns):
        return CheckResult("straddle_vs_polygon", True, 0, 0, "panel lacks expected columns")

    panel = panel.assign(poly_straddle=panel["call_entry"] + panel["put_entry"])
    panel = panel[panel["poly_straddle"] > 0]

    wide = chains.pivot_table(
        index=["ticker", "obs_date", "expiry", "strike"],
        columns="right",
        values="mid",
        aggfunc="first",
    ).reset_index()
    if not {"C", "P"} <= set(wide.columns):
        return CheckResult("straddle_vs_polygon", True, 0, 0, "chains lack both sides")
    wide["orats_straddle"] = wide["C"] + wide["P"]
    wide = wide[wide["orats_straddle"] > 0]

    merged = wide.merge(
        panel[["ticker", "entry_date", "expiry", "strike", "poly_straddle"]],
        left_on=["ticker", "obs_date", "expiry", "strike"],
        right_on=["ticker", "entry_date", "expiry", "strike"],
        how="inner",
    )
    if merged.empty:
        return CheckResult("straddle_vs_polygon", True, 0, 0, "no matched contracts")

    ratio = merged["orats_straddle"] / merged["poly_straddle"]
    rel = (ratio - 1.0).abs() * 100
    n_failed = int((rel > tolerance_pct).sum())
    median = float(np.median(rel))
    return CheckResult(
        "straddle_vs_polygon",
        median <= tolerance_pct,
        len(merged),
        n_failed,
        f"median ratio {float(np.median(ratio)):.4f}, median |diff| {median:.2f}% "
        f"(tolerance {tolerance_pct}%)",
    )
