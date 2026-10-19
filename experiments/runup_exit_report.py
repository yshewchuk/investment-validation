"""Exploratory, same-position STR-RUNUP exit comparisons; no publication."""
from __future__ import annotations

from dataclasses import asdict, replace
from math import isfinite
from pathlib import Path

from engine.v2.data.errors import DataError
from engine.v2.foundation import content_hash, to_document
from engine.v2.foundation.experiment_exits import valid_threshold_units
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.fingerprints import environment_identity, source_closure
from engine.v2.research._chains import load_chain_index
from engine.v2.research._plan import plan_events
from engine.v2.research._pricing import (
    MIN_MEANINGFUL_COST,
    STRUCTURES,
    ChainSnapshot,
    FillModel,
    StructureError,
    price_structure,
    trading_calendar_from_snapshot,
)
from engine.v2.research.experiment_exits import EnteredPosition, PositionLeg, walk_exit
from engine.v2.research.experiment_population import load_population
from engine.v2.research.replay import clean_chain_rows

UNITS = "quote_price_times_quantity"
HORIZON = 14


def _number(value, lower, upper):
    return type(value) in (int, float) and isfinite(value) and lower <= value <= upper


def validate_request(document):
    """Detach a small closed declaration before repository access; no defaults."""
    try:
        if not isinstance(document, dict) or set(document) != {
                "event_ids", "fill_alpha", "threshold_units", "arms"}:
            raise ValueError
        ids, arms = document["event_ids"], document["arms"]
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 10000
                or any(not isinstance(i, str) or not i.strip() for i in ids)
                or len(set(ids)) != len(ids)
                or not _number(document["fill_alpha"], 0, 1)
                or not valid_threshold_units(document["threshold_units"])
                or not isinstance(arms, list) or not 1 <= len(arms) <= 8):
            raise ValueError
        normalized, names = [], {"fixed_day"}
        for arm in arms:
            if (not isinstance(arm, dict) or set(arm) != {"arm_id", "target_pnl", "stop_pnl"}
                    or not isinstance(arm["arm_id"], str) or not arm["arm_id"].strip()
                    or arm["arm_id"] in names
                    or not _number(arm["target_pnl"], 0, float("inf"))
                    or not _number(arm["stop_pnl"], -float("inf"), 0)
                    or arm["target_pnl"] <= 0 or arm["stop_pnl"] >= 0):
                raise ValueError
            names.add(arm["arm_id"])
            normalized.append({"arm_id": arm["arm_id"],
                               "target_pnl": float(arm["target_pnl"]),
                               "stop_pnl": float(arm["stop_pnl"])})
        return {"event_ids": sorted(ids), "fill_alpha": float(document["fill_alpha"]),
                "threshold_units": document["threshold_units"],
                "arms": sorted(normalized, key=lambda a: a["arm_id"])}
    except (ValueError, TypeError, OverflowError):
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid RUNUP comparison request") from None


def _positions(repository, snapshot, events, fill):
    """Resolve contracts from entry chains only, never from later marks."""
    calendar = trading_calendar_from_snapshot(repository, snapshot, extend_days=0)
    structure = STRUCTURES["STR-RUNUP"]()
    plan = plan_events(structure, events, calendar=calendar)
    if (len(plan.frame) != len(events)
            or set(plan.frame["event_id"]) != set(events["event_id"])):
        raise fail("EXPERIMENT_VARIANT_FAILED", "incomplete RUNUP calendar plan",
                   details={"planned_count": len(plan.frame), "skipped": plan.skipped})
    index = load_chain_index(repository, snapshot,
                             list(zip(plan.frame["ticker"], plan.frame["entry_date"])))
    positions = []
    for row in plan.frame.sort_values("event_id").itertuples(index=False):
        try:
            rows = index.get(row.ticker, row.entry_date)
            if rows is None or rows.empty:
                raise ValueError
            chain = ChainSnapshot(row.ticker, row.entry_date, row.event_date,
                                  clean_chain_rows(rows).sort_values(["expiry", "strike", "right"]),
                                  session=row.session)
            if not isfinite(chain.spot_price) or chain.spot_price <= 0:
                raise ValueError
            entry = price_structure(structure, chain, fill)
            if not isfinite(entry.cost) or entry.cost <= MIN_MEANINGFUL_COST:
                raise ValueError
            if any((leg.expiry - row.entry_date).days != leg.dte
                   or leg.dte < structure.params["target_dte"] for leg in entry.legs):
                raise ValueError
            if calendar.shift(row.entry_date, HORIZON) != row.exit_date:
                raise ValueError
            legs = tuple(PositionLeg(leg.expiry.date().isoformat(), leg.strike,
                                     leg.right, leg.side, leg.qty) for leg in entry.legs)
            positions.append(EnteredPosition(row.event_id, row.ticker,
                                             row.entry_date.date().isoformat(), legs))
        except (ValueError, TypeError, KeyError, OverflowError, StructureError):
            raise fail("EXPERIMENT_VARIANT_FAILED", "RUNUP entry selection failed",
                       details={"trade_id": row.event_id,
                                "selected_count": len(positions)}) from None
    return tuple(positions), structure


def _binding(snapshot, events, request, positions, structure, as_of_month):
    context = {name: str(events.iloc[0][name]) for name in (
        "random_membership_version", "rolling_membership_version")}
    return {
        "schema_version": "runup_exit_comparison.v1", "snapshot": to_document(snapshot),
        "as_of_month": as_of_month, "purpose": "selection", "holdouts": context,
        "request": request, "structure": structure.to_dict(),
        "positions": [asdict(position) for position in positions],
        "baseline": {"kind": "fixed_day", "trading_days": HORIZON},
        "cost_policy": {"spread": "alpha_ladder", "commissions": "not_modeled",
                        "fees": "not_modeled", "contract_multiplier": "not_applied"},
        "coverage": "every_held_leg_every_observed_session_through_common_horizon",
        "source_closure": source_closure(Path(__file__).resolve().parents[1],
                                          ["experiments/v2_runup_exits.py"]),
        "environment": environment_identity(None),
    }


def _paired_rows(arm_id, decisions, baseline, *, fractional_arm=None):
    rows = []
    if tuple(d.trade_id for d in decisions) != tuple(d.trade_id for d in baseline):
        raise fail("EXPERIMENT_VARIANT_FAILED", "comparison population changed")
    for decision, control in zip(decisions, baseline, strict=True):
        delta = decision.pnl - control.pnl
        if decision.entry_cost != control.entry_cost or not isfinite(delta):
            raise fail("EXPERIMENT_VARIANT_FAILED", "comparison economics changed")
        rows.append({**asdict(decision), "arm_id": arm_id,
                     "fixed_day_pnl": control.pnl, "pnl_difference": delta,
                     "fixed_day_exit_date": control.exit_date, "pnl_units": UNITS})
        if fractional_arm is not None:
            rows[-1]["thresholds"] = {
                "units": UNITS, "entry_debit": decision.entry_cost,
                "target_pnl": fractional_arm["target_pnl"] * decision.entry_cost,
                "stop_pnl": fractional_arm["stop_pnl"] * decision.entry_cost}
    return rows


def compare_runup_exits(repository, snapshot, *, as_of_month, document):
    """Return the complete paired report or refuse; never compare survivors.

    Fixed-day walks every held leg/session (not only endpoints), so its complete
    success is required before any early-exit arm can run. P&L retains the
    existing quote-price-times-quantity units; fractional thresholds are ratios
    of net opening debit, never contract-dollar conversion or portfolio returns.
    """
    request = validate_request(document)
    counts = {"requested_count": len(request["event_ids"]), "admitted_count": 0,
              "selected_count": 0, "compared_count": 0, "arm_count": len(request["arms"]) + 1}
    stage = "population"
    try:
        events = load_population(repository, snapshot, as_of_month=as_of_month,
                                 purpose="selection", event_ids=request["event_ids"])
        if sorted(events["event_id"]) != request["event_ids"]:
            raise fail("EXPERIMENT_VARIANT_FAILED", "comparison population incomplete")
        counts["admitted_count"] = len(events)
        stage = "entry_selection"
        positions, structure = _positions(repository, snapshot, events, FillModel(request["fill_alpha"]))
        counts["selected_count"] = len(positions)
        stage = "common_horizon_coverage"
        baseline = walk_exit(repository, snapshot, positions, economic_params={
            "fill": request["fill_alpha"], "exit": {"kind": "fixed_day", "trading_days": HORIZON}})
        stage = "exit_comparison"
        rows = _paired_rows("fixed_day", baseline, baseline)
        for arm in request["arms"]:
            recipe = {key: value for key, value in arm.items() if key != "arm_id"}
            recipe.update(kind="target_stop", trading_days=HORIZON)
            fractional = request["threshold_units"] == "entry_debit_fraction"
            if fractional:
                recipe["threshold_units"] = request["threshold_units"]
            decisions = walk_exit(repository, snapshot, positions,
                                  economic_params={"fill": request["fill_alpha"], "exit": recipe})
            rows.extend(_paired_rows(arm["arm_id"], decisions, baseline,
                                     fractional_arm=arm if fractional else None))
        stage = "report_identity"
        binding = _binding(snapshot, events, request, positions, structure, as_of_month)
        counts["compared_count"] = len(positions)
        report = {"status": "complete", "mode": "exploratory_no_ledger",
                  "binding": binding, "comparison_id": content_hash(binding),
                  "counts": counts, "rows": rows}
        return {**report, "report_hash": content_hash(report)}
    except (DataError, OpsError) as exc:
        details = {**counts, **exc.problem.details, "stage": stage}
        raise type(exc)(replace(exc.problem, details=details)) from None
    except (ValueError, TypeError, KeyError, OverflowError):
        raise fail("EXPERIMENT_VARIANT_FAILED", "RUNUP comparison failed",
                   details={**counts, "stage": stage}) from None
