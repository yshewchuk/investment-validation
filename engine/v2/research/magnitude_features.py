"""Pinned, reconstructed history-feature inputs; never a verified PIT forecast."""
from __future__ import annotations

import math
import weakref

from engine.v2.data import errors
from engine.v2.foundation import content_hash, market_calendar
from engine.v2.foundation.experiment_holdouts import _calendar_date
from engine.v2.research._scan import read_table
from engine.v2.research.event_history import load_event_history
from engine.v2.research.experiment_population import load_population
from engine.v2.research.magnitude_inputs import forecast_cutoff, magnitude_recipe

HISTORY_FEATURE_COLUMNS = ("n_prior", "mean_prior_move", "mean_prior_abs_move",
                           *(f"ema{span}_prior_{kind}" for span in (2, 4, 8, 12) for kind in ("move", "abs_move")))


#: Pinned Repository -> {snapshot id: sorted canonical session days}. A pinned snapshot is
#: immutable, so a hit is exact; the entry dies with the repository; failures are never stored.
_OBSERVED_DAYS = weakref.WeakKeyDictionary()


def _observed_days(repository, snapshot):
    """Sorted canonical daily_market dates; read once per pinned repository and snapshot."""
    cache = None
    if getattr(repository, "pins_snapshots", False) is True:
        try:
            cache = _OBSERVED_DAYS.setdefault(repository, {})
        except TypeError:  # not weak-referenceable: read every call
            cache = None
    if cache is not None and snapshot.snapshot_id in cache:
        return cache[snapshot.snapshot_id]
    frame = read_table(repository, snapshot, "daily_market", ("date",),
                       batch_filter=lambda batch: batch.drop_duplicates())
    if frame.empty:
        raise errors.fail("CALENDAR_UNAVAILABLE", "magnitude history requires observed sessions")
    try:
        days = tuple(sorted({_calendar_date(value).isoformat() for value in frame.date}))
    except (ValueError, TypeError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH", "magnitude calendar dates must be canonical") from None
    if cache is not None:
        cache[snapshot.snapshot_id] = days
    return days


def _calendar_input(repository, snapshot):
    """The native observed date union, without normalization or projected days."""
    days = _observed_days(repository, snapshot)
    records = repository.fragment_records(snapshot, "daily_market")
    source = {"snapshot_id": snapshot.snapshot_id, "calendar_version": snapshot.calendar_version,
        "table": "daily_market", "columns": ["date"],
        "dataset_version": snapshot.table_versions["daily_market"].dataset_version_id,
        "objects": [{"fragment_id": record.fragment_id, "object_id": record.object_ref.object_id,
                     "content_hash": record.object_ref.content_hash} for record in records]}
    return market_calendar.CalendarSessions(days, days[-1]), source


def _feature_recipe(features):
    """Keep exact native history names and explicit nulls; never infer aliases."""
    if (tuple(features) != HISTORY_FEATURE_COLUMNS or any(value is not None and
            (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value))
            for value in features.values())):
        raise errors.fail("EXPERIMENT_VARIANT_FAILED", "magnitude history features are invalid")
    return {"schema_version": "magnitude_history_features.v1", "columns": list(HISTORY_FEATURE_COLUMNS),
            "arithmetic": "panel_math.history_features", "missing_values": "null", "imputation": "none"}


def load_magnitude_history(repository, snapshot, *, event_id, as_of_month,
                           horizon, history_start, purpose="selection"):
    """Bind one admitted event's calendar/cutoff to protected historical features."""
    target = load_population(repository, snapshot, as_of_month=as_of_month,
                             purpose=purpose, event_ids=[event_id]).iloc[0]
    calendar, source = _calendar_input(repository, snapshot)
    cutoff = forecast_cutoff(target.event_date, target.session, horizon, calendar)
    history = load_event_history(repository, snapshot, event_id=event_id, as_of_month=as_of_month,
        decision_session=cutoff["source_session_ceiling"], history_start=history_start, purpose=purpose)
    recipe = _feature_recipe(history["features"])
    identity = {"schema_version": "magnitude_history_inputs.v1", "snapshot_id": snapshot.snapshot_id,
        "event_id": event_id, "ticker": target.ticker, "event_date": target.event_date.date().isoformat(),
        "session": target.session, "calendar_input": source, "forecast_cutoff": cutoff,
        "history_input": history, "target_recipe": magnitude_recipe(), "feature_recipe": recipe,
        "features": dict(history["features"]), "availability": {
            "basis": "reconstructed_date_bounds", "event_schedule_as_known": "unverified",
            "source_publication": "unverified"}}
    return {**identity, "inputs_id": content_hash(identity)}
