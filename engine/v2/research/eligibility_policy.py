"""Opt-in grouped embargo planning over metadata, not a data-read capability."""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from datetime import timedelta

from engine.v2.data import errors
from engine.v2.foundation import content_hash, to_document
from engine.v2.foundation.experiment_holdouts import calendar_date
from engine.v2.research import progress
from engine.v2.research.experiment_population import canonical_events
from engine.v2.research.experiment_trades import classify_events
from engine.v2.research.snapshot_inputs import admit_snapshot


def eligibility_policy(*, version, embargo_days):
    """Declare whole calendar days explicitly; there is no global default width."""
    if version != "grouped_ticker_embargo.v1" or type(embargo_days) is not int or embargo_days < 0:
        raise errors.fail("CONTRACT_MISMATCH", "explicit supported policy and nonnegative integer days required")
    identity = {"schema_version": version, "embargo_days": embargo_days,
        "endpoints": "inclusive", "group_key": "event_id", "window_key": "ticker",
        "applies_to": "development", "evaluation": "preserve_membership"}
    return {**identity, "policy_id": content_hash(identity)}


def _day(value):
    if type(value) is not str:
        raise ValueError
    return calendar_date(value)


def _ids(values):
    if (type(values) is not list or any(type(v) is not str or not v or v != v.strip() for v in values)
            or len(set(values)) != len(values)):
        raise ValueError
    return sorted(values)


def _envelope(snapshot, as_of_month, policy, metadata):
    expected = eligibility_policy(version=policy["schema_version"], embargo_days=policy["embargo_days"])
    if type(policy) is not dict or policy != expected:
        raise ValueError
    if (type(metadata) is not dict or set(metadata) != {
            "schema_version", "source_snapshot", "holdout_as_of_month", "policy_id",
            "horizons", "groups", "metadata_id"}
            or metadata["schema_version"] != "event_group_bounds.v1"
            or metadata["source_snapshot"] != to_document(snapshot)
            or metadata["holdout_as_of_month"] != as_of_month or metadata["policy_id"] != policy["policy_id"]
            or metadata["metadata_id"] != content_hash({k: v for k, v in metadata.items() if k != "metadata_id"})):
        raise ValueError
    return expected


def _forecasts(forecasts, horizons, event):
    if (type(forecasts) is not list or any(type(f) is not dict or set(f) != {
            "horizon_sessions", "decision_date", "source_session_ceiling"} for f in forecasts)
            or any(type(f["horizon_sessions"]) is not int for f in forecasts)
            or sorted(f["horizon_sessions"] for f in forecasts) != horizons):
        raise ValueError
    for forecast in forecasts:
        if not _day(forecast["source_session_ceiling"]) <= _day(forecast["decision_date"]) <= event:
            raise ValueError
    return sorted(deepcopy(forecasts), key=lambda f: f["horizon_sessions"])


def _group(group, horizons):
    if type(group) is not dict or set(group) != {
            "event_id", "ticker", "event_date", "label_start", "label_end", "forecasts"}:
        raise ValueError
    if type(group["ticker"]) is not str or not group["ticker"] or group["ticker"] != group["ticker"].strip():
        raise ValueError
    event = _day(group["event_date"])
    if not _day(group["label_start"]) <= event <= _day(group["label_end"]):
        raise ValueError
    return {**deepcopy(group), "forecasts": _forecasts(group["forecasts"], horizons, event)}


def _metadata(snapshot, as_of_month, policy, metadata):
    """Closed producer declaration: a digest binds bytes, not truth of bounds."""
    try:
        expected = _envelope(snapshot, as_of_month, policy, metadata)
        horizons = metadata["horizons"]
        if (type(horizons) is not list or not horizons or any(type(h) is not int or h <= 0 for h in horizons)
                or horizons != sorted(set(horizons)) or type(metadata["groups"]) is not list):
            raise ValueError
        _ids([g["event_id"] for g in metadata["groups"]])
        groups = []
        meter = progress.Progress("eligibility metadata", len(metadata["groups"]))
        for group in metadata["groups"]:
            groups.append(_group(group, horizons))
            meter.tick()
        return expected, sorted(groups, key=lambda g: g["event_id"]), metadata["metadata_id"]
    except (KeyError, TypeError, ValueError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH", "invalid or stale eligibility policy/group metadata") from None


def _windows(events, labels, width, ambiguous):
    """One sorted union per ticker; no per-group scan of all protected events."""
    by_ticker = {}
    try:
        delta = timedelta(days=width)
        meter = progress.Progress("eligibility windows", len(events))
        for event in events.itertuples(index=False):
            memberships = labels.get(event.event_id, set())
            if "random" in memberships:
                if event.event_id in ambiguous:
                    raise ValueError
                day = calendar_date(event.event_date)
                by_ticker.setdefault(event.ticker, []).append((day - delta, day + delta))
            meter.tick()
        merged = {}
        merging = progress.Progress("eligibility window merge", sum(map(len, by_ticker.values())))
        for ticker, intervals in sorted(by_ticker.items()):
            union = []
            for start, end in sorted(intervals):
                if union and start <= union[-1][1]:
                    union[-1] = (union[-1][0], max(end, union[-1][1]))
                else:
                    union.append((start, end))
                merging.tick()
            merged[ticker] = union
        return merged
    except (TypeError, ValueError, OverflowError):
        raise errors.fail("HOLDOUT_ACCESS_DENIED", "random holdout window bounds cannot be established") from None


def _overlaps(intervals, starts, start, end):
    index = bisect_right(starts, end) - 1
    return index >= 0 and intervals[index][1] >= start


def _classify_groups(groups, indexed, labels, windows):
    starts = {ticker: [start for start, _ in union] for ticker, union in windows.items()}
    retained, exclusions = [], []
    meter = progress.Progress("eligibility groups", len(groups))
    for group in groups:
        key, ticker = group["event_id"], group["ticker"]
        try:
            if (key not in indexed.index or ticker != indexed.loc[key, "ticker"]
                    or group["event_date"] != calendar_date(indexed.loc[key, "event_date"]).isoformat()):
                raise ValueError
        except (TypeError, ValueError):
            raise errors.fail("HOLDOUT_ACCESS_DENIED", "group identity differs from canonical population") from None
        reasons = sorted(labels.get(key, set()))
        intervals = windows.get(ticker, [])
        bounds = [(group["event_date"], group["event_date"], "event_window"),
                  (group["label_start"], group["label_end"], "label_window")]
        bounds += [(f[field], f[field], field) for f in group["forecasts"]
                   for field in ("decision_date", "source_session_ceiling")]
        reasons += sorted({"embargo_" + reason for start, end, reason in bounds
            if _overlaps(intervals, starts.get(ticker, []), _day(start), _day(end))})
        if reasons:
            exclusions.append({"event_id": key, "reasons": reasons})
        else:
            retained.append(key)
        meter.tick()
    return retained, exclusions


def plan_eligibility(repository, snapshot, *, as_of_month, policy, metadata, evaluation_event_ids):
    """Recheck population membership; retain evaluation IDs without granting reads.

    Producer-declared label/forecast bounds must already be established against
    the pinned source. They are validated structurally, NOT proven from payloads.
    Consumers must bind/apply this plan before preprocessing, selection and fits.
    Existing consumers are unchanged; neither a hash nor this plan authorizes IO.
    """
    with progress.stage("eligibility admission", report_refusals=True):
        admit_snapshot(repository, snapshot)
        policy, groups, metadata_id = _metadata(snapshot, as_of_month, policy, metadata)
        try:
            evaluation = _ids(evaluation_event_ids)
        except (TypeError, ValueError):
            raise errors.fail("CONTRACT_MISMATCH", "explicit unique evaluation event IDs required") from None
    with progress.stage("eligibility plan", report_refusals=True):
        events = canonical_events(repository, snapshot)
        try:
            _ids(events.event_id.tolist())
        except ValueError:
            raise errors.fail("HOLDOUT_ACCESS_DENIED", "canonical event identities collide or are invalid") from None
        canonical, labels, ambiguous, context = classify_events(events, snapshot, as_of_month=as_of_month)
        indexed = events.set_index("event_id")
        if set(evaluation) - set(indexed.index):
            raise errors.fail("HOLDOUT_ACCESS_DENIED", "evaluation membership cannot be resolved")
        windows = _windows(canonical, labels, policy["embargo_days"], ambiguous)
        retained, exclusions = _classify_groups(groups, indexed, labels, windows)
        identity = {"schema_version": "group_eligibility_plan.v1", **context,
            "source_snapshot": to_document(snapshot), "policy": policy,
            "metadata_id": metadata_id, "groups": groups,
            "random_windows": [{"ticker": ticker, "start": start.isoformat(), "end": end.isoformat()}
                for ticker, union in windows.items() for start, end in union],
            "retained_event_ids": retained, "exclusions": exclusions,
            "evaluation_event_ids": evaluation,
            "coverage": {"development_groups": len(groups), "retained_groups": len(retained),
                         "excluded_groups": len(exclusions), "evaluation_groups": len(evaluation)}}
        progress.decision("eligibility plan", f"retained {len(retained)}/{len(groups)} development groups; "
                          f"preserved {len(evaluation)} evaluation IDs")
        return {**identity, "plan_id": content_hash(identity)}
