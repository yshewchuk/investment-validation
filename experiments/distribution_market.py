"""Native pinned STR-THRU prices and explicit USD costs, never a survivor filter."""
import json
from datetime import date
from math import isfinite

import pandas as pd

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import from_document
from engine.v2.models.payoff_artifact import PayoffArtifactLoader, PayoffLineArtifact
from engine.v2.research._chains import load_chain_index
from engine.v2.research._plan import plan_events
from engine.v2.research._pricing import (
    STRUCTURES,
    ChainSnapshot,
    FillModel,
    StructureError,
    price_structure,
    trading_calendar_from_snapshot,
)
from engine.v2.research.progress import Progress
from engine.v2.research.replay import clean_chain_rows
from experiments.distribution_run_request import number, require
from experiments.magnitude_search_shards import iter_section

STRUCTURE = STRUCTURES["STR-THRU"](entry_offset=0, exit_offset=1, decision_offset=-1)


def read_bridge(store, entry, fold):
    """Existing frozen artifact plus declared strict window; not a new source proof."""
    require(entry["outer_year"] == fold["year"])
    ref = from_document(ArtifactRef, entry["ref"])
    require(0 < ref.byte_size <= 16 * 1024 * 1024)
    artifact = PayoffArtifactLoader(store.root).load_bytes(store.read_verified(ref), ref.content_hash)
    require(isinstance(artifact, PayoffLineArtifact) and artifact.strategy == "STR-THRU"
            and artifact.driver == "abs_move" and artifact.alpha == entry["alpha"]
            and artifact.cutoff is not None and artifact.cutoff <= fold["training_labels_before"]
            and artifact.window_start is not None and artifact.window_end is not None
            and artifact.window_start <= artifact.window_end < artifact.cutoff and bool(artifact.residuals))
    require(all(date.fromisoformat(day).isoformat() == day for day in (artifact.window_start, artifact.window_end, artifact.cutoff))
            and artifact.n > 0 and all(isfinite(value) for value in (artifact.intercept, artifact.slope, *artifact.residuals)))
    return artifact


def read_benchmark(store, entry, fold):
    """Import existing fold-tagged forecasts; hashes are identity, not causal proof."""
    if entry is None:
        return {}
    require(entry["outer_year"] == fold["year"])
    ref = from_document(ArtifactRef, entry["ref"])
    require(0 < ref.byte_size <= 16 * 1024 * 1024)
    document = json.loads(store.read_verified(ref))
    require(set(document) == {"schema_version", "outer_year", "training_labels_before", "rows", "forecast_horizon", "decision_policy"}
            and document["schema_version"] == "distribution_benchmark.v1"
            and type(document["forecast_horizon"]) is int and document["forecast_horizon"] == 1
            and document["decision_policy"] == "native_h1_source_session_ceiling"
            and document["outer_year"] == fold["year"]
            and document["training_labels_before"] <= fold["training_labels_before"])
    rows = list(iter_section(store, "distribution_benchmark", document["rows"], lambda row: (row["event_id"],)))
    require(date.fromisoformat(document["training_labels_before"]).isoformat() == document["training_labels_before"])
    require(len(rows) == len(fold["evaluation_event_ids"])
            and {row["event_id"] for row in rows} == set(fold["evaluation_event_ids"]))
    for row in rows:
        require(set(row) == {"event_id", "incumbent_magnitudes", "gate_selected"}
                and (row["gate_selected"] is None or type(row["gate_selected"]) is bool))
        if row["incumbent_magnitudes"] is not None:
            require(type(row["incumbent_magnitudes"]) is list and bool(row["incumbent_magnitudes"]))
            for sample in row["incumbent_magnitudes"]:
                number(sample)
    return {row["event_id"]: row for row in rows}


def side_cost(economics, contracts):
    """USD per two-long-leg side; alpha already prices spread, no spread surcharge."""
    require(type(contracts) is int and contracts > 0)
    require(economics["currency"] == "USD" and type(economics["multiplier"]) is int
            and economics["multiplier"] == 100)
    for key in ("commission", "exchange_fee", "minimum_side_fee", "slippage_per_share_per_leg"):
        number(economics[key])
    return number(max(economics["minimum_side_fee"], 2 * contracts *
                (economics["commission"] + economics["exchange_fee"]))
            + 2 * contracts * economics["multiplier"] * economics["slippage_per_share_per_leg"])


def _price(index, row, when, alpha, economics, *, pin=None, closing=False):
    rows = index.get(row["ticker"], row[when])
    if rows is None or rows.empty:
        return None
    rows = clean_chain_rows(rows)
    if rows.empty:
        return None
    try:
        priced = price_structure(STRUCTURE, ChainSnapshot(ticker=row["ticker"], obs_date=row[when],
            event_date=row["event_date"], rows=rows, session=row["session"]), FillModel(alpha), pin=pin, closing=closing)
        if not closing and any((leg.ask - leg.bid) / ((leg.ask + leg.bid) / 2)
                               > economics["max_relative_spread"] for leg in priced.legs):
            return None
        return priced
    except (StructureError, ValueError):
        return None


def decision_quotes(repository, snapshot, events, alphas, economics, *, calendar=None):
    """Read D-1 only; return every plan including missing decision quotes."""
    number(economics["max_relative_spread"])
    calendar = calendar if calendar is not None else trading_calendar_from_snapshot(repository, snapshot, extend_days=0)
    clocks = {event["event_id"]: next(c for c in event["forecasts"] if c["horizon_sessions"] == 1) for event in events}
    plan = plan_events(STRUCTURE, pd.DataFrame([{**event, "session": clocks[event["event_id"]]["session"]} for event in events]), calendar)
    index = load_chain_index(repository, snapshot, zip(plan.frame.ticker, plan.frame.decision_date))
    progress, result = Progress("distribution decision quotes", len(plan.frame)), {alpha: {} for alpha in alphas}
    for row in plan.frame.to_dict("records"):
        require(row["decision_date"].date().isoformat() == clocks[row["event_id"]]["source_session_ceiling"])
        for alpha in alphas:
            result[alpha][row["event_id"]] = (row, _price(index, row, "decision_date", alpha, economics))
        progress.tick()
    return result


def outcome_quotes(repository, snapshot, decisions, alpha, economics, contracts):
    """Only after intents are committed: D0/D+1 with the exact D-1 contracts."""
    fee = side_cost(economics, contracts)
    scale = contracts * economics["multiplier"]
    number(economics["max_relative_spread"])
    keys = [(row["ticker"], row[when]) for row, decision in decisions.values()
            if decision is not None for when in ("entry_date", "exit_date")]
    index = load_chain_index(repository, snapshot, keys)
    progress, result = Progress("distribution realized quotes", len(decisions)), {}
    for event, (row, decision) in decisions.items():
        if decision is not None:
            entry = _price(index, row, "entry_date", alpha, economics, pin=decision.legs)
            exit_ = _price(index, row, "exit_date", alpha, economics, pin=decision.legs, closing=True)
            result[event] = {"entry_cash": None if entry is None else entry.cost * scale + fee,
                "exit_cash": None if exit_ is None else exit_.exit_value * scale - fee,
                "exit_at": row["exit_date"].date().isoformat(), "contracts": contracts,
                "multiplier": economics["multiplier"]}
        progress.tick()
    return result
