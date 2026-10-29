"""Paired research-book comparisons; conditional policy deltas, not a rerun.

Calendar and ticker resampling keep whole events together. They do not refit,
reallocate capital, authorize holdout reads or establish executable capacity.
"""
from collections import defaultdict
from datetime import date

import numpy as np

from engine.v2.ops.errors import fail
from engine.v2.research.progress import Progress, stage
from experiments.distribution_book import validate_book_config

__all__ = ["paired_report"]


def _require(condition):
    if not condition:
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution paired report inputs or arithmetic")


def _day(value):
    try:
        day = date.fromisoformat(value)
        _require(day.isoformat() == value)
        return day
    except (TypeError, ValueError):
        _require(False)


def _interval(rows, field, *, seed, resamples, confidence):
    groups = defaultdict(lambda: np.zeros(2))
    for row in rows:
        groups[row[field]] += (row["delta"], 1)
    if len(groups) < 2:
        return {"status": "unavailable_fewer_than_two_clusters", "clusters": len(groups)}
    values = np.array([groups[key] for key in sorted(groups)])
    rng, samples = np.random.default_rng(seed), []
    progress = Progress("distribution paired uncertainty", resamples)
    with stage("distribution paired uncertainty"), np.errstate(over="raise", invalid="raise"):
        for _ in range(resamples):
            total = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
            samples.append(total[0] / total[1])
            progress.tick()
        interval = np.quantile(samples, [(1 - confidence) / 2, (1 + confidence) / 2]).tolist()
    _require(np.isfinite(interval).all())
    return {"status": "available", "clusters": len(groups), "interval": interval}


def _inputs(opportunities, books):
    """Validate supplied document shapes and status/value semantics, not provenance."""
    summary_fields = {"opportunities", "funded", "unresolved", "funded_coverage", "settled_net_profit",
        "cash", "settlement_equity", "capital_days", "settlement_drawdown", "net_profit_per_opportunity"}
    nontrades = ("no_trade", "abstained", "missing_entry", "trade_cap", "ticker_cap", "concurrent_cap", "cash")
    try:
        _require(type(opportunities) is list and type(books) is dict and all(type(arm) is str for arm in books))
        for row in opportunities:
            _require(type(row) is dict and {"event_id", "ticker", "decision_date"} <= row.keys()
                     and type(row["event_id"]) is str)
        for book in books.values():
            if book is None:
                continue
            _require(type(book) is dict and {"schema_version", "config", "report_end", "per_event", "summary"} <= book.keys())
            _require(type(book["per_event"]) is list and type(book["summary"]) is dict)
            validate_book_config(book["config"])
            _require(set(book["summary"]) == summary_fields and type(book["report_end"]) is str)
            _day(book["report_end"])
            for key, value in book["summary"].items():
                _require((value is None and key in ("funded_coverage", "net_profit_per_opportunity"))
                         or (type(value) in (int, float) and np.isfinite(value)))
            for row in book["per_event"]:
                _require(type(row) is dict and {"event_id", "ticker", "decision_at", "event_date", "status", "net_profit"} <= row.keys()
                         and type(row["event_id"]) is str)
                _require(all(type(row[key]) is str for key in ("ticker", "event_date", "status")))
                _day(row["event_date"])
                _require(row["decision_at"] is None or (type(row["decision_at"]) is str and bool(row["decision_at"])))
                status, value = row["status"], row["net_profit"]
                _require((status == "unresolved" and value is None) or (type(value) in (int, float)
                    and np.isfinite(value) and (status == "settled" or (status in nontrades and value == 0))))
    except (KeyError, TypeError, ValueError, OverflowError):
        _require(False)


def paired_report(opportunities, books, *, seed, resamples, block_days, block_origin, confidence):
    """Compare histogram to same_mean (primary), matched_point and optional refs.

    No-trades/refusals are zero; unresolved funded events are NOT zero. A missing
    outcome makes the full-opportunity contrast unavailable; scoreable-only
    diagnostics/intervals remain labelled conditional, with exclusions explicit.
    Fixed occupied calendar blocks and ticker clusters use ratio-of-sums means;
    these intervals condition on realized allocation, not refitted book policies.
    O(events log events + resamples * clusters), O(events + resamples) workspace per contrast.
    """
    _require(type(seed) is int and seed >= 0 and type(resamples) is int and resamples > 0)
    _require(type(block_days) is int and block_days > 0 and type(confidence) in (int, float) and 0 < confidence < 1)
    origin = _day(block_origin)
    _inputs(opportunities, books)
    metadata = {row["event_id"]: row for row in opportunities}
    _require(len(metadata) == len(opportunities))
    for row in opportunities:
        _require(all(type(row[k]) is str and row[k] for k in ("event_id", "ticker")))
        _day(row["decision_date"])
    indexed, summaries, reference = {}, {}, None
    for arm, book in sorted(books.items()):
        if book is None:
            continue
        _require(book["schema_version"] == "distribution_book.v1")
        rows = {row["event_id"]: row for row in book["per_event"]}
        _require(len(rows) == len(book["per_event"]) and set(rows) == set(metadata))
        identity = (book["config"], book["report_end"],
                    {event: row["event_date"] for event, row in rows.items()})
        _require(reference is None or identity == reference)
        reference = identity
        for event, row in rows.items():
            _require((row["ticker"], row["decision_at"] or row["event_date"]) ==
                     (metadata[event]["ticker"], metadata[event]["decision_date"]))
        indexed[arm], summaries[arm] = rows, dict(book["summary"])
    contrasts = {}
    for arm in ("same_mean", "matched_point", "incumbent", "gate"):
        if "histogram" not in indexed or arm not in indexed:
            contrasts[arm] = {"status": "unavailable_arm"}
            continue
        rows, excluded = [], []
        for event, meta in sorted(metadata.items()):
            a, b = indexed["histogram"][event]["net_profit"], indexed[arm][event]["net_profit"]
            if a is None or b is None:
                excluded.append(event)
            else:
                _require(np.isfinite(a - b))
                rows.append({"event_id": event, "ticker": meta["ticker"], "delta": a - b,
                    "calendar": (_day(meta["decision_date"]) - origin).days // block_days})
        delta = float(np.mean([row["delta"] for row in rows])) if rows else None
        _require(delta is None or np.isfinite(delta))
        try:
            intervals = {field: _interval(rows, field, seed=seed, resamples=resamples, confidence=confidence)
                for field in ("calendar", "ticker")}
        except (FloatingPointError, OverflowError, ValueError):
            _require(False)
        contrasts[arm] = {"status": "complete" if rows and not excluded else "incomplete",
            "net_delta_per_opportunity": delta if rows and not excluded else None,
            "scoreable_delta": delta, "scoreable_events": len(rows), "unresolved_events": excluded,
            "conditional_intervals": intervals}
    return {"schema_version": "distribution_paired_report.v1", "opportunities": len(metadata),
        "primary": "histogram_minus_same_mean", "arms": summaries, "contrasts": contrasts,
        "uncertainty": {"seed": seed, "resamples": resamples, "block_days": block_days,
            "block_origin": block_origin, "confidence": confidence, "calendar_blocks": "occupied_fixed_nonoverlapping",
            "interpretation": "conditional_on_realized_policy_not_portfolio_replay"}, "final_read_authority": "none"}
