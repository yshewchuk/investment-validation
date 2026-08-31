"""The prediction ledger: frozen predictions, scored later, never edited.

    python3 -m engine.ledger snapshot --as-of 2026-09-02 --horizon 21
    python3 -m engine.ledger score --through 2026-11-30
    python3 -m engine.ledger calibrate
    python3 -m engine.ledger status

Every backtest in this program shares a code path with the thing it is trying
to validate. The ledger does not: predictions are written **before** outcomes
exist, in a file nobody may rewrite, and scored afterwards by re-pricing the
same structure through :mod:`engine.replay`. It is the out-of-time,
out-of-code-path test that catches what the backtests cannot.

Layout::

    ledger/predictions/YYYY-MM-DD.jsonl   one row per (event, strategy, structure)
    ledger/outcomes/YYYY-MM-DD.jsonl      written once the event resolves
    ledger/calibration/REPORT.md          regenerated every 50 newly scored rows
    ledger/health.json                    what the Phase 3 model-health view reads

**Append-only, enforced here rather than by convention.** A date file is
created with ``"x"`` and appended to thereafter; a row whose ``row_id`` already
exists is refused. There is no delete path. Corrections are new rows carrying
``supersedes`` and a reason, and readers resolve to the last non-superseded row
per id — so the record of what was actually predicted survives the correction.

**The file date follows ``as_of``, never the wall clock.** The nightly job that
starts at 23:58 and finishes at 00:03 must not split one board across two
files, and a prediction's identity is the decision it was made on.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine import paths
from engine.jsonio import json_safe

#: Bumped when a prediction row's shape changes. Rows carry it so a reader
#: five schema versions later can still tell what it is holding.
SCHEMA_VERSION = 1

#: Newly scored outcomes that trigger a calibration recompute (plan §P4.2).
CALIBRATION_TRIGGER = 50


class LedgerError(RuntimeError):
    """Any attempt to rewrite history, or to write a malformed row."""


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


def predictions_dir() -> Path:
    return paths.LEDGER / "predictions"


def outcomes_dir() -> Path:
    return paths.LEDGER / "outcomes"


def calibration_dir() -> Path:
    return paths.LEDGER / "calibration"


def health_path() -> Path:
    return paths.LEDGER / "health.json"


def _date_file(directory: Path, as_of) -> Path:
    return directory / f"{pd.Timestamp(as_of).date()}.jsonl"


def row_id(as_of, ticker: str, strategy: str, strike: Any, expiry: Any) -> str:
    """Stable identity of one prediction.

    Keyed on the decision date rather than the event date: re-scoring the same
    event on a later day is a NEW prediction (the evidence moved), not an edit
    of the old one.
    """
    def _d(value) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "na"
        try:
            return str(pd.Timestamp(value).date())
        except (ValueError, TypeError):
            return str(value)

    strike_key = "atm" if strike is None or not np.isfinite(float(strike)) else f"{float(strike):.4f}"
    return f"{_d(as_of)}|{ticker}|{strategy}|{strike_key}|{_d(expiry)}"


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_predictions(*, as_of=None, resolve_supersedes: bool = True) -> list[dict]:
    """Every prediction row, or one date's.

    With ``resolve_supersedes`` (the default) a row that a later row supersedes
    is dropped from the returned view — but never from the file.
    """
    directory = predictions_dir()
    files = [_date_file(directory, as_of)] if as_of is not None else sorted(directory.glob("*.jsonl"))
    rows: list[dict] = []
    for path in files:
        rows.extend(_read_jsonl(path))
    if not resolve_supersedes:
        return rows
    superseded = {r["supersedes"] for r in rows if r.get("supersedes")}
    return [r for r in rows if r["row_id"] not in superseded]


def read_outcomes() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(outcomes_dir().glob("*.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


def existing_row_ids() -> set[str]:
    return {r["row_id"] for r in read_predictions(resolve_supersedes=False)}


# --------------------------------------------------------------------------
# writing — the append-only contract
# --------------------------------------------------------------------------


REQUIRED_PREDICTION_FIELDS = ("row_id", "as_of", "decision_ts", "ticker", "strategy",
                              "event_date", "score", "snapshot_hash")


def write_predictions(rows: Sequence[Mapping[str, Any]], *, as_of=None) -> Path:
    """Append prediction rows to their as-of date file.

    Refuses any ``row_id`` already on file — including one written minutes ago
    by a re-run of the same job. A re-score with new evidence is a supersede
    (:func:`supersede`), never an overwrite.
    """
    if not rows:
        raise LedgerError("no rows to write")
    as_of = pd.Timestamp(as_of if as_of is not None else rows[0]["as_of"]).normalize()
    for row in rows:
        missing = [f for f in REQUIRED_PREDICTION_FIELDS if f not in row]
        if missing:
            raise LedgerError(f"prediction row missing fields: {missing}")
        if pd.Timestamp(row["as_of"]).normalize() != as_of:
            raise LedgerError(
                f"row {row['row_id']} has as_of {row['as_of']}, file date is {as_of.date()} "
                "— the file date follows as_of, so a batch cannot span two dates")

    directory = paths.assert_writable(predictions_dir())
    directory.mkdir(parents=True, exist_ok=True)
    path = _date_file(directory, as_of)

    on_file = existing_row_ids()
    clashes = [r["row_id"] for r in rows if r["row_id"] in on_file]
    if clashes:
        raise LedgerError(
            f"{len(clashes)} row_id(s) already in the ledger, e.g. {clashes[0]!r}. "
            "The ledger is append-only: correct a prediction with supersede(), "
            "never by rewriting it.")

    mode = "a" if path.exists() else "x"
    with open(path, mode) as fh:
        for row in rows:
            try:
                line = json.dumps(row, default=str, allow_nan=False)
            except ValueError as exc:  # NaN/Infinity survived the sanitizer
                raise LedgerError(
                    f"row {row['row_id']} is not strict JSON ({exc}); the ledger "
                    "cannot be corrected after the fact, so it is not written"
                ) from exc
            fh.write(line + "\n")
    return path


def supersede(old_row_id: str, new_row: Mapping[str, Any], reason: str) -> Path:
    """Correct a prediction by appending a replacement that names it.

    The original stays on file and stays readable. Nothing in this module can
    remove it, which is the property that makes the ledger evidence.
    """
    if not reason:
        raise LedgerError("a supersede needs a reason")
    if old_row_id not in existing_row_ids():
        raise LedgerError(f"cannot supersede unknown row_id {old_row_id!r}")
    row = json_safe(dict(new_row))
    row["supersedes"] = old_row_id
    row["supersede_reason"] = reason
    return write_predictions([row])


# --------------------------------------------------------------------------
# snapshot — writing predictions before outcomes exist
# --------------------------------------------------------------------------


def _event_ids(events_wanted: pd.DataFrame) -> pd.DataFrame:
    """Attach the store's ``event_id`` to (ticker, event_date) pairs."""
    from engine.data import store

    events = store.read_table("earnings_events",
                              columns=["event_id", "ticker", "event_date", "session"])
    events["event_date"] = pd.to_datetime(events["event_date"])
    return events_wanted.merge(events, on=["ticker", "event_date"], how="left")


def build_prediction_rows(scores: pd.DataFrame, *, as_of, decision_ts=None,
                          audit_receipts: Mapping[str, Any] | None = None) -> list[dict]:
    """Turn a scored board into ledger rows.

    The whole ``ScoreResult`` is embedded rather than a hand-picked subset:
    scoring one prediction correctly a month later needs the model versions,
    the analog buckets, the flags and the evidence cutoff, and a schema that
    picks favourites now is one that loses the field it turns out to need.
    """
    as_of = pd.Timestamp(as_of).normalize()
    decision_ts = pd.Timestamp(decision_ts or datetime.now(tz=timezone.utc))
    if decision_ts.tzinfo is None:
        decision_ts = decision_ts.tz_localize("UTC")

    wanted = scores[["ticker", "event_date"]].drop_duplicates().copy()
    wanted["event_date"] = pd.to_datetime(wanted["event_date"])
    ids = _event_ids(wanted).set_index(["ticker", "event_date"])["event_id"].to_dict()

    written_at = datetime.now(tz=timezone.utc).isoformat()
    rows: list[dict] = []
    for record in scores.to_dict(orient="records"):
        event_date = pd.to_datetime(record.get("event_date"))
        rid = row_id(as_of, record.get("ticker"), record.get("strategy"),
                     record.get("strike"), record.get("expiry"))
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "row_id": rid,
            "written_at": written_at,
            "as_of": str(as_of.date()),
            "decision_ts": decision_ts.isoformat(),
            "ticker": record.get("ticker"),
            "event_id": ids.get((record.get("ticker"), event_date)),
            "event_date": str(event_date.date()) if pd.notna(event_date) else None,
            "session": record.get("session"),
            "strategy": record.get("strategy"),
            "structure": {"strike": record.get("strike"), "expiry": record.get("expiry"),
                          "entry_date": record.get("entry_date"),
                          "exit_date": record.get("exit_date"),
                          "dte_entry": record.get("dte_entry")},
            "intended_prices": {"alpha": record.get("fill"),
                                "entry_cost": record.get("entry_cost"),
                                "spot": record.get("spot")},
            "score": record,
            "model_versions": record.get("model_versions") or {},
            "snapshot_hash": record.get("snapshot_hash") or "",
            "audit_receipt": (audit_receipts or {}).get(rid),
            "supersedes": None,
            "supersede_reason": None,
        })
    return [json_safe(row) for row in rows]


def snapshot(as_of=None, *, horizon_days: int = 21,
             strategies: Sequence[str] | None = None,
             tickers: Iterable[str] | None = None,
             scores: pd.DataFrame | None = None) -> dict:
    """Score the upcoming calendar and freeze it into the ledger.

    Phase 3's nightly job calls this; until Phase 3 exists the CLI does, so the
    ledger accrues rows before the Q3 season rather than starting empty on the
    day it matters.
    """
    from engine.audit import audit_receipt_for_snapshot
    from engine.score import score_calendar

    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    started = time.time()
    if scores is None:
        scores = score_calendar(as_of, horizon_days=horizon_days,
                                strategies=strategies, tickers=tickers)
    if not len(scores):
        return {"as_of": str(as_of.date()), "rows": 0, "path": None,
                "note": "no confirmed events in the horizon"}

    rows = build_prediction_rows(scores, as_of=as_of)
    receipt = audit_receipt_for_snapshot(scores, decision_ts=rows[0]["decision_ts"])
    for row in rows:
        row["audit_receipt"] = receipt.as_dict()

    on_file = existing_row_ids()
    fresh = [r for r in rows if r["row_id"] not in on_file]
    skipped = len(rows) - len(fresh)
    path = write_predictions(fresh, as_of=as_of) if fresh else None
    return {"as_of": str(as_of.date()), "rows": len(fresh), "skipped_existing": skipped,
            "path": str(path) if path else None,
            "elapsed_s": round(time.time() - started, 1)}


# --------------------------------------------------------------------------
# outcome scoring
# --------------------------------------------------------------------------


def _unresolved(through: pd.Timestamp) -> list[dict]:
    scored = {o["row_id"] for o in read_outcomes()}
    out = []
    for row in read_predictions():
        if row["row_id"] in scored or not row.get("event_date"):
            continue
        if pd.Timestamp(row["event_date"]) <= through:
            out.append(row)
    return out


def _write_outcomes(rows: Sequence[Mapping[str, Any]]) -> Path | None:
    """Append outcome rows, keyed by resolution date. Idempotent by row_id."""
    if not rows:
        return None
    directory = paths.assert_writable(outcomes_dir())
    directory.mkdir(parents=True, exist_ok=True)
    already = {o["row_id"] for o in read_outcomes()}
    fresh = [r for r in rows if r["row_id"] not in already]
    if not fresh:
        return None
    resolved_on = pd.Timestamp(fresh[0]["resolved_at"]).normalize()
    path = _date_file(directory, resolved_on)
    mode = "a" if path.exists() else "x"
    with open(path, mode) as fh:
        for row in fresh:
            # Same rule as the prediction side: an outcome file that is not
            # strict JSON is a corrupt record nobody may rewrite.
            fh.write(json.dumps(json_safe(row), default=str, allow_nan=False) + "\n")
    return path


def score_outcomes(through=None, *, resolved_at=None) -> dict:
    """Join realized P&L onto every prediction whose event has passed.

    The realized side is priced through :mod:`engine.replay` — the same path
    the backtests use — so a ledger-vs-backtest gap can never be a pricing
    difference. Predictions that cannot be resolved are written as outcomes
    with ``status="unresolvable"`` and a reason, never dropped: silently
    omitting the ones whose chains never arrived would bias the ledger toward
    liquid names, which is the one bias it exists to be free of.
    """
    from engine import replay as replay_mod

    through = pd.Timestamp(through).normalize() if through is not None else pd.Timestamp.today().normalize()
    resolved_at = pd.Timestamp(resolved_at or datetime.now(tz=timezone.utc))
    pending = _unresolved(through)
    if not pending:
        return {"resolved": 0, "unresolvable": 0, "path": None}

    frame = pd.DataFrame([{
        "row_id": r["row_id"], "event_id": r.get("event_id"), "ticker": r["ticker"],
        "event_date": pd.Timestamp(r["event_date"]), "session": r.get("session"),
        "strategy": r["strategy"],
        "alpha": (r.get("intended_prices") or {}).get("alpha", 0.5),
        "entry_cost": (r.get("intended_prices") or {}).get("entry_cost"),
        "event_date_at_prediction": r["event_date"],
    } for r in pending])

    priced: dict[tuple[str, str], dict] = {}
    for strategy, group in frame.groupby("strategy"):
        events = group[["event_id", "ticker", "event_date", "session"]].drop_duplicates()
        events = events[events["event_id"].notna()]
        if not len(events):
            continue
        try:
            result = replay_mod.replay(strategy, events.reset_index(drop=True),
                                       progress_every=0)
        except (KeyError, ValueError) as exc:      # unknown strategy, empty plan
            print(f"  [ledger] {strategy}: replay unavailable — {exc}", flush=True)
            continue
        trades = result.trades
        if not len(trades):
            continue
        for record in trades.to_dict(orient="records"):
            priced[(str(record["event_id"]), f"{float(record['fill_alpha']):.2f}")] = record

    rows: list[dict] = []
    for row, record in zip(pending, frame.to_dict(orient="records")):
        key = (str(record["event_id"]), f"{float(record['alpha'] or 0.5):.2f}")
        trade = priced.get(key)
        base = {
            "schema_version": SCHEMA_VERSION,
            "row_id": row["row_id"],
            "resolved_at": resolved_at.isoformat(),
            "ticker": row["ticker"],
            "strategy": row["strategy"],
            "event_date": row["event_date"],
            "predicted_win": (row.get("score") or {}).get("win_model"),
            "predicted_pnl": (row.get("score") or {}).get("exp_pnl_model"),
            "predicted_win_analog": (row.get("score") or {}).get("win_analog"),
            "gate_pass": (row.get("score") or {}).get("gate_pass"),
        }
        if trade is None:
            rows.append(base | {
                "status": "unresolvable",
                "reason": ("no priced replay for this event at the intended alpha — "
                           "the exit chain never arrived, or the event moved"),
                "realized_pnl": None, "realized_win": None,
            })
            continue
        realized = float(trade.get("ret", np.nan))
        rows.append(base | {
            "status": "resolved",
            "reason": None,
            "fill_alpha_used": float(trade.get("fill_alpha", np.nan)),
            "realized_pnl": realized,
            "realized_win": bool(realized > 0),
            "realized_entry_cost": float(trade.get("entry_cost", np.nan)),
            "realized_exit_value": float(trade.get("exit_value", np.nan)),
            "exit_source": trade.get("exit_mode") or "chain",
            "event_date_changed": str(pd.Timestamp(trade["event_date"]).date())
                                  != row["event_date"],
        })

    path = _write_outcomes(rows)
    resolved = sum(1 for r in rows if r["status"] == "resolved")
    return {"resolved": resolved, "unresolvable": len(rows) - resolved,
            "path": str(path) if path else None}


# --------------------------------------------------------------------------
# calibration + health
# --------------------------------------------------------------------------


def scored_pairs() -> pd.DataFrame:
    """Predicted vs realized, one row per resolved outcome."""
    outcomes = [o for o in read_outcomes() if o.get("status") == "resolved"]
    if not outcomes:
        return pd.DataFrame(columns=["row_id", "strategy", "predicted_win",
                                     "realized_win", "predicted_pnl", "realized_pnl"])
    return pd.DataFrame(outcomes)[
        ["row_id", "strategy", "predicted_win", "realized_win",
         "predicted_pnl", "realized_pnl", "event_date"]]


def _strategy_calibration(frame: pd.DataFrame) -> dict:
    from engine.calibrate import brier, brier_skill, decile_table, monotonicity, reliability_table

    ok = frame["predicted_win"].notna() & frame["realized_win"].notna()
    frame = frame[ok]
    n = int(len(frame))
    if not n:
        return {"available": False, "reason": "no scored predictions with a win probability"}
    p = frame["predicted_win"].to_numpy(dtype=float)
    y = frame["realized_win"].to_numpy(dtype=float)
    table = reliability_table(p, y)
    out = {
        "available": True,
        "n": n,
        "base_rate": float(y.mean()),
        "brier": float(brier(p, y)),
        "brier_base_rate": float(y.mean() * (1 - y.mean())),
        "brier_skill": float(brier_skill(p, y)),
        "reliability_monotonicity": float(monotonicity(table)),
        "deciles": [{"predicted": float(r["predicted"]), "realized": float(r["realized"]),
                     "n": int(r["n"])} for _i, r in table.iterrows()],
    }
    pnl = frame[frame["predicted_pnl"].notna() & frame["realized_pnl"].notna()]
    if len(pnl):
        out["predicted_mean_pnl"] = float(pnl["predicted_pnl"].mean())
        out["realized_mean_pnl"] = float(pnl["realized_pnl"].mean())
        out["pnl_deciles"] = decile_table(pnl["predicted_pnl"].to_numpy(dtype=float),
                                          pnl["realized_pnl"].to_numpy(dtype=float)
                                          ).to_dict(orient="records")
    return out


def write_health(per_strategy: Mapping[str, Any], *, n_scored: int) -> Path:
    """The frozen contract the Phase 3 model-health view reads.

    Frozen now, before the dashboard exists, so Phase 3 builds against a shape
    that will not move under it.
    """
    from engine.data import store  # noqa: F401  (import proves the store is reachable)

    snapshot_hash = ""
    if paths.SNAPSHOT_FILE.exists():
        try:
            snapshot_hash = json.loads(paths.SNAPSHOT_FILE.read_text()).get("snapshot", "")
        except (ValueError, OSError):
            snapshot_hash = ""
    quota_state = None
    if paths.QUOTA_LOG.exists():
        lines = paths.QUOTA_LOG.read_text().splitlines()
        quota_state = lines[-1] if len(lines) > 1 else None

    predictions = read_predictions()
    latest = max((p["as_of"] for p in predictions), default=None)
    champions: dict[str, Any] = {}
    for row in predictions[-50:]:
        champions.update(row.get("model_versions") or {})

    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_scored": int(n_scored),
        "n_predictions": len(predictions),
        "latest_prediction_as_of": latest,
        "per_strategy": dict(per_strategy),
        "champion_versions": champions,
        "snapshot_hash": snapshot_hash,
        "data_freshness": {"latest_prediction_as_of": latest},
        "quota_state": quota_state,
    }
    path = paths.assert_writable(health_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str))
    return path


def _calibration_state_path() -> Path:
    return calibration_dir() / "state.json"


def calibration_due(*, trigger: int = CALIBRATION_TRIGGER) -> tuple[bool, int, int]:
    """(due, n_scored_now, n_at_last_report) — the ≥50-new-rows trigger."""
    n_now = int(len(scored_pairs()))
    state_path = _calibration_state_path()
    last = 0
    if state_path.exists():
        try:
            last = int(json.loads(state_path.read_text()).get("n_scored_at_last_report", 0))
        except (ValueError, OSError):
            last = 0
    return (n_now - last) >= trigger, n_now, last


def calibrate(*, force: bool = False, trigger: int = CALIBRATION_TRIGGER) -> dict:
    """Regenerate the ledger calibration report and health.json.

    Emitted through :class:`engine.report.Report` like every other result in
    the program — the ledger does not get its own bespoke format.
    """
    from engine.report import Report, build_provenance

    due, n_now, last = calibration_due(trigger=trigger)
    if not (due or force):
        return {"regenerated": False, "n_scored": n_now, "n_at_last_report": last,
                "note": f"{n_now - last} new scored row(s); trigger is {trigger}"}

    pairs = scored_pairs()
    per_strategy = {str(s): _strategy_calibration(g) for s, g in pairs.groupby("strategy")} \
        if len(pairs) else {}
    overall = _strategy_calibration(pairs) if len(pairs) else {"available": False,
                                                              "reason": "no scored outcomes yet"}

    out_dir = paths.assert_writable(calibration_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "kind": "calibration",
        "spec": {"id": "LEDGER", "title": "Prediction-ledger calibration",
                 "type": "descriptive",
                 "hypothesis": "The shipped win probabilities and expected P&L match "
                               "what the frozen predictions actually realized."},
        "results": {"headline": {}, "stress": {}, "mc": {},
                    "calibration": overall, "per_strategy": per_strategy,
                    "n_scored": n_now},
        "headline": {},
        "backtest": {},
        "checklist": [],
        "provenance": build_provenance(spec_hash=None, seeds={},
                                       input_files=sorted(outcomes_dir().glob("*.jsonl"))),
        "survivorship_note": "",
        "calibration": overall if overall.get("available") else None,
        "calibration_raw": overall,
        "funnel": [{"stage": "predictions written", "events": len(read_predictions()),
                    "note": "frozen before the event, append-only"},
                   {"stage": "outcomes resolved", "events": n_now,
                    "note": "priced through engine.replay after the event",
                    "headline": True}],
        "extra_sections": _per_strategy_sections(per_strategy),
    }
    report_path = Report(context).write(out_dir)
    write_health(per_strategy, n_scored=n_now)
    _calibration_state_path().write_text(json.dumps(
        {"n_scored_at_last_report": n_now,
         "generated_at": datetime.now(tz=timezone.utc).isoformat()}, indent=1))
    return {"regenerated": True, "n_scored": n_now, "report": str(report_path),
            "health": str(health_path())}


def _per_strategy_sections(per_strategy: Mapping[str, Any]) -> list[dict]:
    if not per_strategy:
        return []
    rows = []
    for strategy, block in sorted(per_strategy.items()):
        if not block.get("available"):
            rows.append([strategy, "n/a", "n/a", "n/a", block.get("reason", "")])
            continue
        rows.append([strategy, f"{block['n']:,}", f"{block['brier_skill']:.3f}",
                     f"{block['base_rate']:.1%}",
                     f"predicted {block.get('predicted_mean_pnl', float('nan')):+.2%} vs "
                     f"realized {block.get('realized_mean_pnl', float('nan')):+.2%}"
                     if "predicted_mean_pnl" in block else ""])
    return [{
        "title": "Per-strategy calibration",
        "note": "Each strategy's frozen predictions against what they realized.",
        "columns": ["strategy", "n scored", "Brier skill", "base rate", "mean P&L"],
        "align": ["---", "---:", "---:", "---:", "---"],
        "rows": rows,
        "falsifies": "a strategy's Brier skill staying below −0.05 as its sample grows — "
                     "then its shipped win rate is a ranking, not a probability.",
    }]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def status() -> dict:
    predictions = read_predictions()
    outcomes = read_outcomes()
    resolved = [o for o in outcomes if o.get("status") == "resolved"]
    due, n_now, last = calibration_due()
    return {
        "predictions": len(predictions),
        "prediction_files": len(list(predictions_dir().glob("*.jsonl"))),
        "outcomes": len(outcomes),
        "resolved": len(resolved),
        "unresolvable": len(outcomes) - len(resolved),
        "calibration_due": due,
        "n_scored": n_now,
        "n_at_last_report": last,
        "health": str(health_path()) if health_path().exists() else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="score the upcoming calendar and freeze it")
    snap.add_argument("--as-of", default=None)
    snap.add_argument("--horizon", type=int, default=21)
    snap.add_argument("--strategy", action="append", default=None)

    sc = sub.add_parser("score", help="resolve predictions whose events have passed")
    sc.add_argument("--through", default=None)

    cal = sub.add_parser("calibrate", help="regenerate the calibration report + health.json")
    cal.add_argument("--force", action="store_true")

    sub.add_parser("status", help="what the ledger holds")

    args = parser.parse_args(argv)
    if args.command == "snapshot":
        out = snapshot(args.as_of, horizon_days=args.horizon, strategies=args.strategy)
    elif args.command == "score":
        out = score_outcomes(args.through)
        if out["resolved"]:
            out["calibration"] = calibrate()
    elif args.command == "calibrate":
        out = calibrate(force=args.force)
    else:
        out = status()
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
