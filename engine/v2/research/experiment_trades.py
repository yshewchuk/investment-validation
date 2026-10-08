"""Trades read for experiment consumers, over one pinned snapshot.

This is the research package's entrypoint for engine/v2 consumers outside
``tools/v2_*.py``: it lets ``experiments/common_v2.py`` read the committed
``trades`` dataset version ``tools/v2_build_trades.py`` publishes, without
the caller touching this package's internal read helpers directly. No CLI
of its own; no write path.
"""
from __future__ import annotations

import pandas as pd

from engine.v2.data import errors
from engine.v2.foundation import SystemClock
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.research._trades_publish import read_event_rows, read_existing_trades
from engine.v2.research._trades_revisions import PROVENANCE

__all__ = ["PROVENANCE", "load_trades"]


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
            raise errors.fail("HOLDOUT_ACCESS_DENIED", "requested event membership cannot be resolved",
                details={**context, "purpose": purpose, "holdout_exclusions": [
                    {"event_id": key, "memberships": ["ambiguous"]} for key in sorted(missing)]})
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
        raise errors.fail("HOLDOUT_ACCESS_DENIED", "requested events are excluded from experiment reads",
                          details={**context, "purpose": purpose, "holdout_exclusions": labels})
    rows = rows[~excluded].drop(
        columns=["canonical_date", "canonical_ticker", "date_conflict", "event_cluster_id"]
    ).reset_index(drop=True)
    for key, value in context.items():
        rows[key] = value
    rows["population_use"] = "post-release selection"
    rows.attrs["holdout_exclusions"] = labels
    return rows
