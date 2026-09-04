"""Logical name → physical path registry.

Nothing in ``engine/`` (or in any phase built on top of it) hardcodes a
filesystem path after this module exists. Two rules make that worth enforcing:

* **Grandfathered directories are read-only.** The existing research trees
  under ``earnings_predictions/`` and ``polygon_cache/`` hold ~57k files that
  cost real quota to acquire and that the current verdicts rest on. They are
  registered here and never moved, renamed, or written to. :func:`assert_writable`
  turns that convention into a runtime check.
* **New data lives under one root.** Tiers 1/2/3 all hang off ``data/`` at the
  repo root so a single ``.gitignore`` line keeps them out of git and a single
  path move would relocate the whole store.

The root can be overridden with the ``INVESTING_PLAN_ROOT`` environment
variable, which is what the test-suite uses to build throwaway trees.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# roots
# --------------------------------------------------------------------------

#: Repo root. ``engine/paths.py`` → ``engine/`` → repo root.
ROOT = Path(os.environ.get("INVESTING_PLAN_ROOT", Path(__file__).resolve().parents[1]))

ENGINE = ROOT / "engine"
CHECKS = ROOT / "checks"
TESTS = ROOT / "tests"
GUIDES = ROOT / "guides"
TOOLS = ROOT / "tools"

ENV_FILE = ROOT / ".env"

# --------------------------------------------------------------------------
# grandfathered research trees — REGISTERED, NEVER WRITTEN
# --------------------------------------------------------------------------

EP = ROOT / "earnings_predictions"
EP_DATA = EP / "data"
EP_RAW = EP_DATA / "raw"
EP_PROCESSED = EP_DATA / "processed"
EP_SRC = EP / "src"
EP_STRATEGIES = EP / "strategies"
EP_OPF = EP / "opf"

RAW_ORATS = EP_RAW / "orats"
RAW_ORATS_SUMMARIES = RAW_ORATS / "summaries"
RAW_ORATS_CORES = RAW_ORATS / "cores"
RAW_ORATS_EARNINGS = RAW_ORATS / "earnings"
RAW_ORATS_STRIKES = RAW_ORATS / "strikes"
RAW_ORATS_QUOTA_LOG = RAW_ORATS / "quota_log.csv"
RAW_ORATS_TICKERS = RAW_ORATS / "tickers.json"

RAW_OQUANTS = EP_RAW / "oquants"
RAW_OQUANTS_MOVES = RAW_OQUANTS / "moves"
RAW_OQUANTS_RETURNS = RAW_OQUANTS / "returns"  # DEPRECATED for P&L (fitted marks)
RAW_OQUANTS_FFTS = RAW_OQUANTS / "ffts"
RAW_OQUANTS_SKEW = RAW_OQUANTS / "skew"

RAW_YF = EP_RAW / "yfinance"
RAW_POLYGON = EP_RAW / "polygon"
RAW_POLYGON_LEGACY = ROOT / "polygon_cache"

GSPC_DAILY = RAW_POLYGON / "gspc_daily.csv"

#: The panel every current verdict rests on. The migration test reconciles the
#: rebuilt Tier-3 panel against this file; it is never regenerated in place.
LEGACY_PANEL = EP_PROCESSED / "events_with_orats_sum.csv"
LEGACY_EVENTS = EP_PROCESSED / "events.csv"
LEGACY_TRUE_IMPLIED = EP_PROCESSED / "true_implied.csv"

BT = ROOT / "bt"
BT_STRADDLE_PANEL = BT / "straddle" / "straddle_panel.csv"

#: Simulated trade sets produced by the pre-engine research, normalized into
#: the Tier-2 ``trades`` table. ``(strategy, path, entry_col, exit_col)``.
LEGACY_TRADE_SETS = {
    "S1_calendar": (EP_STRATEGIES / "s1_vrp_calendar_straddle" / "data" / "trades_real.csv", "entry_date", "exit_date"),
    "S2_short_dte": (EP_STRATEGIES / "s2_underpriced_vol" / "data" / "trades_real.csv", "entry_date", "exit_date"),
    "S3_runup": (EP_STRATEGIES / "s3_pre_earnings_long_vol" / "data" / "trades_real_t14.csv", "t10_date", "exit_date"),
}

#: Everything above this line is read-only to engine code.
GRANDFATHERED = (EP, BT, RAW_POLYGON_LEGACY)

# --------------------------------------------------------------------------
# new three-tier store
# --------------------------------------------------------------------------

DATA = ROOT / "data"

# Tier 1 — raw immutable cache (append-only, every byte ever fetched)
RAW = DATA / "raw"
RAW_FETCH = RAW / "fetch"
FETCH_LOG = RAW_FETCH / "fetch_log.csv"
QUOTA_LOG = RAW_FETCH / "quota_log.csv"
POLYGON_LOCK = RAW_FETCH / ".polygon.lock"
QUARANTINE = RAW / "quarantine"

#: Synthesized oquants-format moves files for tickers the oquants panel does
#: not carry (EXP-117 universe extension): dates/sessions from the ORATS
#: calendar, moves computed session-aware from yfinance closes (validated
#: exact against Polygon in EXP-117), implied moves from daily_market. The
#: target provenance of these rows is COMPUTED, not vendor-supplied.
COMPUTED_MOVES = RAW / "computed_moves"

# Tier 2 — normalized store (Parquet, partitioned by year)
CURATED = DATA / "curated"

# Tier 3 — feature / serving layer (rebuildable, snapshot-hashed)
FEATURES = DATA / "features"
SNAPSHOT_FILE = FEATURES / "SNAPSHOT"
PANEL = FEATURES / "panel.parquet"

#: Tier 4 — the feature-model forecasts, keyed on (ticker, event_date). Narrow
#: by design: keys, model outputs and provenance, never a copy of Tier 3. It
#: carries its own hash rather than joining the Tier-3 snapshot, so a rebuild
#: here cannot invalidate an experiment that never read a forecast.
TIER4 = FEATURES / "tier4_forecasts.parquet"

MANIFEST = DATA / "MANIFEST.md"

REPORTS = ROOT / "reports"
LEDGER = ROOT / "ledger"

#: Directories engine code creates on demand.
WRITABLE_ROOTS = (DATA, REPORTS, LEDGER)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def curated_table(name: str) -> Path:
    """Directory holding one Tier-2 table (``year=YYYY/`` partitions inside)."""
    return CURATED / name


def curated_partition(name: str, year: int | str) -> Path:
    return curated_table(name) / f"year={year}"


def raw_fetch_path(source: str, key: str) -> Path:
    """Tier-1 cache location for one fetch, sharded two hex chars deep.

    Sharding keeps any single directory well under the tens-of-thousands of
    entries that make ``os.listdir`` on this filesystem painful — the existing
    ``strikes/`` directory with 19k files is already at that edge.
    """
    return RAW_FETCH / source / key[:2] / key


def is_grandfathered(path: Path | str) -> bool:
    """True if ``path`` lies inside a read-only research tree."""
    p = Path(path).resolve()
    for root in GRANDFATHERED:
        try:
            p.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def assert_writable(path: Path | str) -> Path:
    """Raise if ``path`` is inside a grandfathered tree; else return it.

    Every engine write path funnels through this. The grandfathered trees are
    irreplaceable at anything short of a full quota re-spend, so an accidental
    write there is a category of bug worth making impossible rather than
    unlikely.
    """
    p = Path(path)
    if is_grandfathered(p):
        raise PermissionError(
            f"refusing to write inside a grandfathered read-only tree: {p}"
        )
    return p


def ensure_dirs() -> None:
    """Create the new-tier directory skeleton. Never touches grandfathered."""
    for d in (
        RAW_FETCH,
        QUARANTINE,
        CURATED,
        FEATURES,
        REPORTS,
        LEDGER,
    ):
        assert_writable(d).mkdir(parents=True, exist_ok=True)


def describe() -> dict[str, str]:
    """Flat ``{logical name: path}`` map, for manifests and report provenance."""
    out: dict[str, str] = {}
    for name, value in sorted(globals().items()):
        if name.isupper() and isinstance(value, Path):
            out[name] = str(value)
    return out
