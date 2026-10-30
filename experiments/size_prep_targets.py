"""Workspace development-target derivation over a pinned bounds record. Library-only: no production caller until slice 4 (#1614 slice 3b)."""
import time

from engine.v2.foundation import canonical_json, content_hash
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts, calendar_date
from engine.v2.ops.errors import fail
from engine.v2.research import eligible_history
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.experiment_population import load_population
from experiments import size_panel, size_prep_history
from experiments.size_prep_inputs import _log

TARGETS_VERSION = "size_prep_targets.v1"


def _no_targets():
    return fail("PUBLICATION_REFUSED", "the requested window has no development targets",
                details={"reason": "no_targets"})


def derive_workspace_targets(workspace, scope, record, *, tickers, event_start, event_end, as_of_month,
                             history_start, metadata_limits, clock):
    """Derive universe ∩ retained development targets; refuses before writing; the read pin is last."""
    started = time.monotonic()
    tickers_ok = (type(tickers) is list and bool(tickers)
                  and all(type(t) is str and t and t == t.strip() for t in tickers)
                  and tickers == sorted(set(tickers)))
    if not tickers_ok:
        raise fail("INVALID_REQUEST", "sorted unique non-empty ticker list required")
    if (type(scope) is not str or type(as_of_month) is not str or type(history_start) is not str
            or type(event_start) is not str or type(event_end) is not str
            or type(metadata_limits) is not dict):
        raise fail("INVALID_REQUEST", "invalid workspace targets request")
    try:
        first, last = calendar_date(event_start), calendar_date(event_end)
    except (ValueError, TypeError, OverflowError):
        raise fail("INVALID_REQUEST", "canonical half-open event window required") from None
    if not first < last:
        raise fail("INVALID_REQUEST", "event window start must precede end")
    size_panel.validate_panel_inputs({"source_start": event_start, "metadata_limits": metadata_limits})
    try:
        ExperimentHoldouts(as_of_month)
        calendar_date(history_start)
    except (ValueError, TypeError, OverflowError):
        raise fail("INVALID_REQUEST", "canonical as-of month and history start required") from None
    _log("workspace targets start tickers=" + str(len(tickers)) + " window=" + event_start + ".." + event_end,
         started)
    repository, bounds = size_prep_history.checked_bounds(workspace, scope, record, metadata_limits)
    population = load_population(repository, bounds, as_of_month=as_of_month)
    days = population.event_date.dt.strftime("%Y-%m-%d")
    kept = population[population.ticker.isin(tickers) & (days >= event_start) & (days < event_end)]
    universe = sorted(kept.event_id.tolist())
    _log("workspace targets universe count=" + str(len(universe)), started)
    if not universe:
        raise _no_targets()
    limits = size_prep_history._derive_history_limits(repository, bounds, len(universe))
    policy = eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30)
    plan = eligible_history.plan_targets(repository, bounds, as_of_month=as_of_month, policy=policy,
                                         event_ids=universe, horizons=[1, 3, 5],
                                         history_start=history_start, limits=limits)
    targets = plan["retained"]
    if not targets:
        raise _no_targets()
    excluded = {e["event_id"]: e["reasons"] for e in plan["exclusions"]}
    unavailable = set(plan["unavailable"])
    before_start = set(plan["before_start"])
    retained = set(targets)
    dropped = []
    for event_id in universe:
        if event_id in retained:
            continue
        if event_id in before_start:
            dropped.append({"event_id": event_id, "reasons": ["before_history_start"]})
        elif event_id in excluded:
            dropped.append({"event_id": event_id, "reasons": excluded[event_id]})
        elif event_id in unavailable:
            dropped.append({"event_id": event_id, "reasons": ["label_bounds_unavailable"]})
        else:
            raise fail("PUBLICATION_REFUSED", "the eligibility plan does not account for a universe event",
                       details={"reason": "plan_incomplete"})
    fields = {"schema_version": TARGETS_VERSION, "scope": scope, "record_id": record["record_id"],
              "bounds_snapshot_id": bounds.snapshot_id, "event_start": event_start,
              "event_end": event_end, "as_of_month": as_of_month, "history_start": history_start,
              "tickers_sha": content_hash(tickers), "universe_count": len(universe),
              "universe_sha": content_hash(universe), "target_count": len(targets),
              "targets_sha": content_hash(targets), "dropped_count": len(dropped),
              "dropped_sha": content_hash(dropped), "eligibility_plan_id": plan["plan_id"],
              "limits": limits}
    doc = {**fields, "targets_id": content_hash(fields)}
    ref = workspace.record_read_pin("size-prep-targets:" + doc["targets_id"].split(":", 1)[1], doc, clock=clock)
    counts = {}
    for item in dropped:
        counts[item["reasons"][0]] = counts.get(item["reasons"][0], 0) + 1
    _log("workspace targets end targets=" + str(len(targets)) + " dropped=" + str(len(dropped))
         + " first_reasons=" + canonical_json(counts), started)
    return {**doc, "event_ids": targets, "dropped": dropped, "read_pin_ref": ref}
