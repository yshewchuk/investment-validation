"""Trades read for experiment consumers, over one pinned snapshot.

This is the research entrypoint for v2 experiment reads of the committed
``trades`` version published by ``tools/v2_build_trades.py``. Completed
experiment wrappers are not maintained; missing holdout context refuses.
There is no CLI of its own; loader output stays in memory except for the
optional private holdout refusal signal written for registered subprocess
orchestration.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from engine.v2.data import errors
from engine.v2.foundation import SystemClock
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.research._trades_publish import read_event_rows, read_existing_trades
from engine.v2.research._trades_revisions import PROVENANCE

__all__ = ["PROVENANCE", "load_trades"]

_HOLDOUT_REFUSAL_SIGNAL_ENV = "INVESTMENT_PLAN_HOLDOUT_REFUSAL_SIGNAL"
_HOLDOUT_REFUSAL_SIGNAL_SCHEMA = "holdout_refusal_signal.v1"
_HOLDOUT_REFUSAL_PINS = ("snapshot_id", "holdout_as_of_month",
                         "random_membership_version", "rolling_membership_version")


def load_trades(repository, snapshot, strategy: str, *, as_of_month=None,
                purpose="training", event_ids=None) -> pd.DataFrame:
    """The committed v2-replay trade rows for ``strategy``, one pinned snapshot.

    Trade columns from ``experiments.common.load_engine_trades``: every
    ``trades`` column plus a ``session`` column joined from
    ``earnings_events`` on ``event_id``. Filters to this package's own
    ``PROVENANCE`` tag (never a legacy ``engine.replay`` row the same
    snapshot might also carry). ``event_date``, ``entry_date`` and
    ``exit_date`` are returned as ``pd.Timestamp`` (via ``pd.to_datetime``),
    matching the legacy loader.

    Raises ``engine.v2.data.errors.DataError``:
    - ``CONTRACT_MISMATCH`` — the snapshot has no ``trades`` table at all
      (raised by ``read_existing_trades`` itself), its ``earnings_events``
      table has no fragments at all (the same issue #70 bare ``ValueError``,
      converted here too — trivially every row's session is then unmatched).
    - ``POPULATION_COLLAPSED`` — the snapshot's ``trades`` table has no row
      for this ``strategy``/``PROVENANCE`` pair, or no fragments at all (the
      latter is ``_snapshot.read_table``'s documented issue #70 bare
      ``ValueError``, converted here so every ``load_trades`` failure stays
      typed).

    Does not itself resolve a snapshot id and does not itself decide "missing
    snapshot" — that is ``Repository.resolve``'s ``SNAPSHOT_NOT_FOUND``,
    raised by the caller's own resolve call before this function is reached
    (see ``experiments/common_v2.py``).
    ``as_of_month`` explicitly pins shared holdout membership. Bulk reads
    exclude both sets; an explicitly requested excluded event refuses the
    entire read. No final-read purpose is available. Context columns and
    ``holdout_exclusions`` attrs contain membership evidence, not metrics.
    """
    policy, context = _holdout_context(snapshot, as_of_month, purpose, event_ids)
    try:
        trades = read_existing_trades(repository, snapshot)
    except ValueError:
        if repository.fragment_records(snapshot, "trades"):
            raise
        raise errors.fail(
            "POPULATION_COLLAPSED",
            "the trades table has no fragments in this snapshot",
            details={"snapshot_id": snapshot.snapshot_id},
        ) from None
    rows = trades[
        (trades["strategy"] == strategy) & (trades["provenance"] == PROVENANCE)
    ].reset_index(drop=True)
    if event_ids is not None:
        missing = set(event_ids) - set(rows["event_id"])
        if missing:
            details = {**context, "purpose": purpose, "holdout_exclusions": [
                {"event_id": key, "memberships": ["ambiguous"]} for key in sorted(missing)]}
            _emit_holdout_refusal_signal(details)
            raise errors.fail("HOLDOUT_ACCESS_DENIED", "requested event membership cannot be resolved",
                              details=details)
        rows = rows[rows["event_id"].isin(event_ids)].copy()
    if rows.empty:
        raise errors.fail(
            "POPULATION_COLLAPSED",
            "no trades rows for this strategy/provenance in the resolved snapshot",
            details={"snapshot_id": snapshot.snapshot_id, "strategy": strategy},
        )
    for column in ("event_date", "entry_date", "exit_date"):
        rows[column] = pd.to_datetime(rows[column], errors="coerce" if column == "event_date" else "raise")

    try:
        events = read_event_rows(repository, snapshot, columns=(
            "event_id", "ticker", "event_date", "session", "date_conflict", "event_cluster_id"))
    except ValueError:
        if repository.fragment_records(snapshot, "earnings_events"):
            raise
        raise errors.fail(
            "CONTRACT_MISMATCH",
            "a trades row has no matching earnings_events session",
            details={"snapshot_id": snapshot.snapshot_id, "strategy": strategy},
        ) from None
    return _exclude_holdouts(rows, events, policy, context, purpose, event_ids)


def _holdout_context(snapshot, as_of_month, purpose, event_ids):
    try:
        policy = ExperimentHoldouts(as_of_month)
        if as_of_month > SystemClock().now().strftime("%Y-%m"):
            raise ValueError
        if not isinstance(purpose, str) or purpose not in ("training", "selection", "sweep"):
            raise ValueError
        if event_ids is not None and (
            not isinstance(event_ids, (list, tuple, set, frozenset))
            or any(not isinstance(value, str) or not value.strip() for value in event_ids)
        ):
            raise ValueError
    except ValueError:
        raise errors.fail("HOLDOUT_ACCESS_DENIED", "explicit valid holdout context is required") from None
    return policy, {"snapshot_id": snapshot.snapshot_id, "holdout_as_of_month": as_of_month,
                    "random_membership_version": policy.random_version,
                    "rolling_membership_version": policy.rolling_version}


def _emit_holdout_refusal_signal(details):
    """Durable private typed denial pins for the enabled registered runner child.

    ``legacy_adapter.run_legacy_script`` points
    ``INVESTMENT_PLAN_HOLDOUT_REFUSAL_SIGNAL`` at a staged-run file that
    ``worker._registered_experiment_runner`` reads on child nonzero exit.
    Without that environment variable (direct or in-process loader callers)
    nothing is written and behavior is unchanged. Only the four nonblank
    string identity pins travel; ``holdout_exclusions``, event IDs, messages
    and every other denial detail never leave the typed exception. An
    enabled write or fsync failure propagates rather than being swallowed.
    """
    signal_path = os.environ.get(_HOLDOUT_REFUSAL_SIGNAL_ENV)
    if not signal_path:
        return
    pins = {name: (details or {}).get(name) for name in _HOLDOUT_REFUSAL_PINS}
    if not all(isinstance(value, str) and value.strip() for value in pins.values()):
        return
    document = {"schema_version": _HOLDOUT_REFUSAL_SIGNAL_SCHEMA,
                "failure_code": "HOLDOUT_ACCESS_DENIED", **pins}
    destination = Path(signal_path)
    temp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                       dir=destination.parent,
                                       prefix=".holdout_refusal_signal.",
                                       suffix=".tmp", delete=False)
    replaced = False
    try:
        temp.write(json.dumps(document, indent=2, sort_keys=True))
        temp.flush()
        os.fsync(temp.fileno())
        temp.close()
        os.replace(temp.name, destination)
        replaced = True
        directory_fd = os.open(destination.parent,
                               os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            try:
                os.fsync(directory_fd)
            except OSError:
                if replaced:
                    _rollback_replaced_signal(destination)
                raise
        finally:
            os.close(directory_fd)
    finally:
        temp.close()
        Path(temp.name).unlink(missing_ok=True)


def _rollback_replaced_signal(destination):
    """Undo a non-durable replacement after the parent-directory fsync failed.

    Only called once ``os.replace`` has succeeded for this invocation, so the
    destination holds this invocation's document and is not durable. Best-effort
    remove it and resync the parent, swallowing every failure so the original
    sync ``OSError`` still propagates unmasked.
    """
    try:
        Path(destination).unlink(missing_ok=True)
    except OSError:
        return
    try:
        directory_fd = os.open(Path(destination).parent,
                               os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass


def _ambiguous(row, duplicates):
    return (row.event_id in duplicates or pd.isna(row.canonical_date)
            or pd.isna(row.event_date) or row.event_date != row.canonical_date
            or not isinstance(row.canonical_ticker, str) or row.ticker != row.canonical_ticker
            or not isinstance(row.session, str) or row.session not in ("AMC", "BMO")
            or pd.isna(row.date_conflict) or bool(row.date_conflict))


def _exclude_holdouts(rows, events, policy, context, purpose, event_ids):
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    duplicates = set(events.loc[events["event_id"].duplicated(False), "event_id"])
    duplicates.update(events.loc[events["event_cluster_id"].notna()
        & events.duplicated(["ticker", "event_cluster_id"], keep=False), "event_id"])
    duplicates.update(events.loc[
        events.duplicated(["ticker", "event_date", "session"], keep=False), "event_id"])
    rows = rows.merge(
        events.rename(columns={"event_date": "canonical_date", "ticker": "canonical_ticker"})
        .drop_duplicates("event_id"),
        on="event_id", how="left",
    )
    exclusions = {}
    for row in rows.itertuples(index=False):
        labels = policy.classify(row.event_id, row.canonical_date)
        if _ambiguous(row, duplicates):
            labels = labels | {"ambiguous"}
        if labels:
            key = row.event_id if isinstance(row.event_id, str) else None
            exclusions.setdefault(key, set()).update(labels)
    labels = [{"event_id": key, "memberships": sorted(value)}
              for key, value in sorted(exclusions.items(), key=lambda item: (item[0] is not None, item[0] or ""))]
    excluded = rows["event_id"].isin(exclusions) | rows["event_id"].isna()
    if exclusions and (event_ids is not None or excluded.all()):
        details = {**context, "purpose": purpose, "holdout_exclusions": labels}
        _emit_holdout_refusal_signal(details)
        raise errors.fail("HOLDOUT_ACCESS_DENIED", "requested events are excluded from experiment reads",
                          details=details)
    rows = rows[~excluded].drop(
        columns=["canonical_date", "canonical_ticker", "date_conflict", "event_cluster_id"]
    ).reset_index(drop=True)
    for key, value in context.items():
        rows[key] = value
    rows["population_use"] = "post-release selection"
    rows.attrs["holdout_exclusions"] = labels
    return rows
