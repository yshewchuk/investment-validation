"""The Polygon real-trade pull's universe read path, on a pinned snapshot.

Moved from ``engine/data/pulls/polygon_fills.py`` by Phase 6 slice 6:
:func:`collect_contracts` verbatim and the ``trades`` read rewritten as a
bounded ``Repository.scan`` against one pinned snapshot. The network half of
that pull — ``build_plan``/``execute`` with their ``Fetcher`` pacing and raw
cache — stays legacy and out of scope here; this module plans the universe and
writes it with the snapshot id::

    python3 tools/v2_polygon_fills.py --catalog <cat.sqlite> --store-root <objects>

``option_ticker`` is a verbatim copy of
``engine.data.sources.polygon.option_ticker``: a v2 package may not import
legacy code without a declared adapter, and the function is a pure four-line
identity.

Output: a plan summary (contract count, contract-days, ordered contracts) as
JSON under ``reports/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from engine.v2.research._scan import DEFAULT_SCOPE, read_table, resolve_snapshot

__all__ = [
    "POLYGON_OPTIONS_START",
    "TRADES_COLUMNS",
    "collect_contracts",
    "option_ticker",
    "plan_summary",
    "read_trades",
    "run",
]

#: First day of Polygon options data on this plan (established by the
#: bt/straddle pulls; daily aggs before this date do not exist here).
POLYGON_OPTIONS_START = "2024-08-19"

#: The ``trades`` projection ``collect_contracts`` reads.
TRADES_COLUMNS = ("ticker", "legs", "entry_date", "exit_date")


def option_ticker(symbol: str, expiry: str, right: str, strike: float) -> str:
    """OCC contract id: ``O:{SYM}{YYMMDD}{C|P}{strike*1000:08d}``."""
    yymmdd = "".join(expiry.split("-"))[2:]
    r = right.upper()[0]
    if r not in ("C", "P"):
        raise ValueError(f"right must be C or P, got {right!r}")
    return f"O:{symbol.upper()}{yymmdd}{r}{round(strike * 1000):08d}"


def read_trades(repository, snapshot_ref) -> pd.DataFrame:
    """The simulated-trade universe, read through the pinned snapshot."""
    return read_table(repository, snapshot_ref, "trades", TRADES_COLUMNS)


def collect_contracts(trades: pd.DataFrame,
                      min_date: str = POLYGON_OPTIONS_START) -> dict[str, dict]:
    """Every contract the simulated trades hold, with its observed date range.

    Returns ``{contract_ticker: {ticker, first_obs, last_obs, expiry,
    n_obs_dates}}``. Only observation dates on/after ``min_date`` count — that
    is where Polygon has data to answer with — but a contract observed only
    before it can still be relevant to no one, so it simply does not appear.

    The legacy ``trades=None`` branch (read the store) moved to
    :func:`read_trades`; the frame is now always supplied.
    """
    info: dict[str, dict] = {}
    for row in trades.itertuples(index=False):
        legs = getattr(row, "legs", None)
        if not isinstance(legs, str):
            continue
        try:
            doc = json.loads(legs)
        except ValueError:
            continue
        for phase, obs in (("entry", row.entry_date), ("exit", row.exit_date)):
            date = pd.Timestamp(obs)
            if pd.isna(date) or date.strftime("%Y-%m-%d") < min_date:
                continue
            for leg in doc.get(phase) or []:
                try:
                    expiry = pd.Timestamp(leg["expiry"])
                    contract = option_ticker(
                        str(row.ticker),
                        expiry.strftime("%Y-%m-%d"),
                        str(leg["right"]),
                        float(leg["strike"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue  # a leg shape we do not understand adds no job
                rec = info.setdefault(
                    contract,
                    {
                        "ticker": str(row.ticker),
                        "first_obs": date,
                        "last_obs": date,
                        "expiry": expiry,
                        "dates": set(),
                    },
                )
                rec["first_obs"] = min(rec["first_obs"], date)
                rec["last_obs"] = max(rec["last_obs"], date)
                rec["expiry"] = max(rec["expiry"], expiry)
                rec["dates"].add(date.strftime("%Y-%m-%d"))
    return info


def plan_summary(info: dict[str, dict], *, snapshot_id: str,
                 min_date: str = POLYGON_OPTIONS_START) -> dict:
    """The universe plan as a JSON-safe document, naming its snapshot."""
    return {
        "schema_version": "polygon_fills_plan.v1",
        "snapshot_id": snapshot_id,
        "min_date": min_date,
        "contracts_in_trades": len(info),
        "contract_days": sum(len(rec["dates"]) for rec in info.values()),
        "contracts": [
            {
                "contract_ticker": contract,
                "ticker": rec["ticker"],
                "first_obs": rec["first_obs"].strftime("%Y-%m-%d"),
                "last_obs": rec["last_obs"].strftime("%Y-%m-%d"),
                "expiry": rec["expiry"].strftime("%Y-%m-%d"),
                "n_obs_dates": len(rec["dates"]),
            }
            for contract, rec in sorted(info.items())
        ],
    }


def run(repository, *, out_dir: Path = Path("reports"), scope: str = DEFAULT_SCOPE,
        snapshot_id: str | None = None, min_date: str = POLYGON_OPTIONS_START,
        stamp: str | None = None) -> dict:
    """Plan the pull universe from one pinned snapshot and write the plan."""
    snapshot = resolve_snapshot(repository, scope=scope, snapshot_id=snapshot_id)
    info = collect_contracts(read_trades(repository, snapshot), min_date=min_date)
    plan = plan_summary(info, snapshot_id=snapshot.snapshot_id, min_date=min_date)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or pd.Timestamp.now().strftime("%Y-%m-%d")
    path = out_dir / f"polygon_fills_plan_{stamp}.json"
    path.write_text(json.dumps(plan, indent=1, sort_keys=True))
    plan["path"] = str(path)
    return plan
