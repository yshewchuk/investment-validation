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


def _tickers(tickers):
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


def _rows(events, tickers):
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
    ids, days, clusters = set(), set(), set()
    for row in rows:
        identity = row["event_id"]
        day = (row["ticker"], row["event_date"])
        cluster = (row["ticker"], row["event_cluster_id"])
        if (identity in ids or day in days or row["date_conflict"]
                or row["event_cluster_id"] is not None and cluster in clusters):
            raise fail("IDENTITY_CONFLICT", "metadata view has conflicting event identities")
        ids.add(identity)
        days.add(day)
        if row["event_cluster_id"] is not None:
            clusters.add(cluster)


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


def reduce_sequences(events: pd.DataFrame, *, tickers: list[str]) -> list[dict]:
    """Reduce all supplied selected rows, or refuse without returning a partial view.

    Caller owns allocation bounds and complete requested-ticker membership.
    No source identity, time cutoff, coverage or payload authority is accepted.
    """
    tickers = _tickers(tickers)
    rows, reasons = _rows(events, set(tickers))
    _conflicts(rows)  # Unconfirmed rows can invalidate the entire view.
    selected = dict.fromkeys(tickers, 0)
    confirmed = {ticker: [] for ticker in tickers}
    for row, reason in zip(rows, reasons):
        ticker = row["ticker"]
        selected[ticker] += 1
        if reason is None:
            confirmed[ticker].append(row)
    return [_sequence(ticker, selected[ticker], confirmed[ticker]) for ticker in tickers]
