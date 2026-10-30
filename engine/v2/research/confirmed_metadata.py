"""Pure complete-view reduction, not source admission or numerical eligibility."""
import numpy as np
import pandas as pd

from engine.v2.data.confirmed_event_metadata import classify_metadata
from engine.v2.data.errors import fail
from engine.v2.data.event_history import event_sequence

_COLUMNS = ("event_id", "ticker", "event_date", "session", "date_conflict",
            "event_cluster_id", "src_orats")


def _text(value, field="metadata text"):
    if type(value) is not str or not value:
        raise fail("CONTRACT_MISMATCH", f"{field} must be a nonempty string")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise fail("CONTRACT_MISMATCH", f"{field} must be valid Unicode") from None
    return value


def validate_tickers(tickers):
    """Return sorted canonical ticker names; no source access or admission."""
    if type(tickers) is not list or not tickers:
        raise fail("CONTRACT_MISMATCH", "tickers must be a nonempty plain list")
    for ticker in tickers:
        _text(ticker, "ticker")
        if ticker != ticker.strip() or "\0" in ticker:
            raise fail("CONTRACT_MISMATCH", "ticker names must be canonical")
    if len(set(tickers)) != len(tickers):
        raise fail("CONTRACT_MISMATCH", "ticker names must be distinct")
    return sorted(tickers)


def _null(value):
    return (value is None or value is pd.NA
            or isinstance(value, (float, np.floating)) and np.isnan(value))


def _source_shape(events):
    if not isinstance(events, pd.DataFrame):
        raise fail("CONTRACT_MISMATCH", "metadata requires a DataFrame")
    columns = list(events.columns)
    if (len(columns) != len(_COLUMNS) or not all(type(c) is str for c in columns)
            or set(columns) != set(_COLUMNS)):
        raise fail("CONTRACT_MISMATCH", "metadata requires exactly seven unique columns")
    if events["event_date"].dtype != np.dtype("datetime64[ns]"):
        raise fail("CONTRACT_MISMATCH", "metadata dates require naive datetime64[ns]")
    for session in events["session"]:
        if not _null(session) and type(session) is not str:
            raise fail("CONTRACT_MISMATCH", "metadata session must be scalar text or null")


def metadata_rows(events, *, tickers):
    """Validate rows/tickers and detach evidence/reasons; no source or conflict proof."""
    tickers = set(validate_tickers(tickers))
    _source_shape(events)
    reasons = classify_metadata(events)
    rows = []
    for values in events.loc[:, _COLUMNS].itertuples(index=False, name=None):
        row = dict(zip(_COLUMNS, values))
        _text(row["event_id"])
        if _text(row["ticker"]) not in tickers:
            raise fail("CONTRACT_MISMATCH", "metadata ticker is outside the request")
        cluster = row["event_cluster_id"]
        row["event_cluster_id"] = None if _null(cluster) else _text(cluster)
        if not isinstance(row["date_conflict"], (bool, np.bool_)):
            raise fail("CONTRACT_MISMATCH", "metadata date_conflict must be a bool")
        row["date_conflict"] = bool(row["date_conflict"])
        row["src_orats"] = bool(row["src_orats"])
        row["session"] = None if _null(row["session"]) else row["session"]
        row["event_date"] = row["event_date"].date().isoformat()
        rows.append(row)
    return rows, reasons


def _conflicts(rows):
    ids, days = set(), set()
    for row in rows:
        identity = row["event_id"]
        day = (row["ticker"], row["event_date"])
        if identity in ids or day in days:
            raise fail("IDENTITY_CONFLICT", "metadata view has conflicting event identities")
        ids.add(identity)
        days.add(day)


def _ambiguity(rows, reasons):
    clusters = {}
    for row in rows:
        cluster = row["event_cluster_id"]
        if cluster is not None:
            key = (row["ticker"], cluster)
            clusters[key] = clusters.get(key, 0) + 1
    counts = {"unconfirmed_date_conflict": 0, "unconfirmed_cluster_ambiguous": 0}
    updated = []
    for row, reason in zip(rows, reasons):
        cluster = row["event_cluster_id"]
        if reason is None and row["date_conflict"]:
            reason = "date_conflict"
            counts["unconfirmed_date_conflict"] += 1
        elif (reason is None and cluster is not None
                and clusters[(row["ticker"], cluster)] > 1):
            reason = "cluster_ambiguous"
            counts["unconfirmed_cluster_ambiguous"] += 1
        updated.append(reason)
    return tuple(updated), counts


def _sequence(ticker, selected, confirmed):
    confirmed.sort(key=lambda row: (row["event_date"], row["event_id"]))
    sequence = event_sequence([row["event_date"] for row in confirmed], warmup=4)
    events = [dict(row, ordinal=event.ordinal, warmup_complete=event.admitted)
              for row, event in zip(confirmed, sequence)]
    status = "confirmed_events" if events else (
        "no_confirmed_events" if selected else "absent_in_snapshot")
    return {"ticker": ticker, "status": status, "selected_rows": selected,
            "confirmed_rows": len(events),
            "first_confirmed_date": events[0]["event_date"] if events else None,
            "events": events}


def reduce_with_counts(events: pd.DataFrame, *, tickers: list[str]) -> tuple[list[dict], dict]:
    """Reduce all supplied selected rows and count date/cluster ambiguity reasons."""
    tickers = validate_tickers(tickers)
    rows, reasons = metadata_rows(events, tickers=tickers)
    _conflicts(rows)  # Duplicate identities invalidate the entire view.
    reasons, counts = _ambiguity(rows, reasons)
    selected = dict.fromkeys(tickers, 0)
    confirmed = {ticker: [] for ticker in tickers}
    for row, reason in zip(rows, reasons):
        ticker = row["ticker"]
        selected[ticker] += 1
        if reason is None:
            confirmed[ticker].append(row)
    sequences = [_sequence(ticker, selected[ticker], confirmed[ticker]) for ticker in tickers]
    return sequences, counts


def reduce_sequences(events: pd.DataFrame, *, tickers: list[str]) -> list[dict]:
    """Reduce all supplied selected rows, or refuse without returning a partial view.

    Caller owns allocation bounds and complete requested-ticker membership.
    No source identity, time cutoff, coverage or payload authority is accepted.
    """
    return reduce_with_counts(events, tickers=tickers)[0]
