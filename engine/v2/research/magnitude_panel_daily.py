"""Bounded reconstructed daily/SPY blocks; deliberately not fit-admissible rows."""
from __future__ import annotations

import json
import math
from bisect import bisect_right
from datetime import date, timedelta

import pandas as pd

from engine.v2.contracts.data import DataQuery, KeyPredicate, PriceQuery, TimeInterval
from engine.v2.data import errors, objects, price_history_query, price_history_table
from engine.v2.features import panel_math, regime
from engine.v2.foundation import canonical_json, content_hash, to_document
from engine.v2.research import (
    eligibility_policy,
    magnitude_panel_plan,
    magnitude_panel_table,
    progress,
    snapshot_inputs,
)

_DAILY = tuple(panel_math.DAILY_STATE_FIELDS)
_SPY = magnitude_panel_table.BLOCKS["spy"]
_COLUMNS = ("ticker", "date", "year", "src_iv", "mcap_asof", *_DAILY)
_STATUS = "reconstructed_partial_blocks_not_fit_admissible"


def _limit(value):
    if type(value) is not int or not 0 < value <= 1_000_000:
        raise errors.fail("CONTRACT_MISMATCH", "explicit positive bounded panel limits required")
    return value


def _producer(parent, producer):
    """Reuse the closed metadata validator; this binds, but does not attest, code."""
    try:
        check = {**parent, "producer": producer}
        check["context_id"] = content_hash({k: v for k, v in check.items() if k != "context_id"})
        magnitude_panel_plan.validate_context(check)
        detached = json.loads(canonical_json(producer))
        if "engine/v2/research/magnitude_panel_daily.py" not in detached["source_closure"]:
            raise ValueError
        return detached
    except (KeyError, TypeError, ValueError, RecursionError):
        raise errors.fail("CONTRACT_MISMATCH", "explicit daily block producer identity required") from None


def _day(value):
    """Native daily rows use timestamps; accept only naive midnight days."""
    if isinstance(value, (bool, int, float)):
        raise ValueError
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
        raise ValueError
    return stamp.date().isoformat()


def _optional_number(value):
    if value is None or (not isinstance(value, bool) and pd.isna(value)):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError
    return float(value)


def _group_clocks(parent, metadata):
    """Check clock declarations against the admitted parent, not their hashes alone."""
    events = parent["history_context"]["temporal_plan"]["events"]
    try:
        groups = {group["event_id"]: group for group in metadata["groups"]}
        if len(groups) != len(events) or len(groups) != len(metadata["groups"]):
            raise ValueError
        for event in events:
            group = groups[event["event_id"]]
            expected = [{"horizon_sessions": cutoff["horizon_sessions"],
                "decision_date": cutoff["source_session_ceiling"],
                "source_session_ceiling": cutoff["source_session_ceiling"]}
                for cutoff in event["forecasts"]]
            if (any(group[key] != event[key] for key in ("ticker", "event_date"))
                    or sorted(group["forecasts"], key=lambda f: f["horizon_sessions"])
                    != sorted(expected, key=lambda f: f["horizon_sessions"])):
                raise ValueError
    except (KeyError, TypeError, ValueError):
        raise errors.fail("CONTRACT_MISMATCH", "daily block eligibility clocks differ from panel plan") from None
    return events


def _admit(repository, source, history, parent, policy, metadata, source_start, max_events):
    if getattr(repository, "pins_snapshots", False) is not True:
        raise errors.fail("CONTRACT_MISMATCH", "daily blocks require a snapshot-pinned repository")
    magnitude_panel_plan.preflight_context(repository, source, history, parent)
    events = _group_clocks(parent, metadata)
    if len(events) > max_events:
        raise errors.fail("RESULT_LIMIT_EXCEEDED", "daily block event bound exceeded")
    # A new consumer opts in; it neither migrates nor changes the v1 history context.
    expected = eligibility_policy.eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30)
    if policy != expected:
        raise errors.fail("CONTRACT_MISMATCH", "daily blocks require the stage-two eligibility policy")
    plan = eligibility_policy.plan_eligibility(repository, source,
        as_of_month=parent["history_context"]["temporal_plan"]["holdout_as_of_month"],
        policy=policy, metadata=metadata, evaluation_event_ids=[])
    retained = set(plan["retained_event_ids"])
    events = [event for event in events if event["event_id"] in retained]
    if not events:
        raise errors.fail("POPULATION_COLLAPSED", "no daily block events survive eligibility")
    if source_start > min(c["source_session_ceiling"] for e in events for c in e["forecasts"]):
        raise errors.fail("CONTRACT_MISMATCH", "daily source start exceeds a forecast ceiling")
    return events, plan


def _query(repository, snapshot, ticker, year, start, end, limit):
    ref = snapshot.table_versions["daily_market"].table_contract_ref
    contract = repository.table_contract(snapshot, "daily_market")
    if (contract.contract_id, contract.semantic_version) != ("legacy.daily_market.v2", "2.0.0"):
        raise errors.fail("CONTRACT_MISMATCH", "unsupported daily block source contract")
    keys = (KeyPredicate(column="ticker", operator="eq", values=(ticker,)),
            KeyPredicate(column="year", operator="eq", values=(year,)))
    interval = TimeInterval(column="date", start_inclusive=start,
        end_exclusive=(date.fromisoformat(end) + timedelta(days=1)).isoformat())
    bound = repository.scan_population_bound(snapshot.snapshot_id, table_name="daily_market",
        table_contract_ref=ref, key_filter=keys, time_interval=interval)
    return DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=ref,
        columns=_COLUMNS, key_filter=keys, time_interval=interval,
        order_by=contract.primary_key, max_result_rows=min(limit, bound),
        max_batch_rows=min(contract.maximum_batch_rows, limit, max(1, bound)))


def _source_row(raw, ticker, year, start, end, seen):
    try:
        day = _day(raw["date"])
        if (raw["ticker"] != ticker or int(day[:4]) != year
                or not start <= day <= end or day in seen):
            raise ValueError
        seen.add(day)
        row = {**raw, "date": day}
        cap = None if raw["mcap_asof"] is None else _day(raw["mcap_asof"])
        row["mcap_asof"] = cap
        for name in _DAILY:
            row[name] = (None if name == "mcap_log" and (cap is None or cap > day)
                         else _optional_number(raw[name]))
        if row["implied_move"] is not None and row["implied_move"] < 0:
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH", "invalid daily block source row") from None
    return row


def _source_rows(repository, snapshot, ticker, start, end, limit, max_object_bytes):
    """One scan per ticker/year, bounded across years; no per-event source work."""
    rows, seen = [], set()
    meter = progress.Progress("daily block source years", int(end[:4]) - int(start[:4]) + 1)
    for year in range(int(start[:4]), int(end[:4]) + 1):
        request = _query(repository, snapshot, ticker, year, start, end, limit)
        for batch in repository.scan(request, table_name="daily_market", max_object_bytes=max_object_bytes):
            for raw in batch.to_pylist():
                rows.append(_source_row(raw, ticker, year, start, end, seen))
                if len(rows) > limit:
                    raise errors.fail("RESULT_LIMIT_EXCEEDED", "daily block source row bound exceeded")
        meter.tick()
    return sorted(rows, key=lambda row: row["date"])


def _read_regime_series(repository, snapshot, start, end, max_source_rows, max_object_bytes):
    """The pinned ``REGIME_TICKER`` (^GSPC) ``price_history`` closes on ``[start, end]``.

    Read exactly like ``features.panel_row_inputs._read_regime_prices`` -- one
    bounded ``price_history_query.get_price_series`` as of ``end``
    (``lookback_sessions=0``), whose ``close_raw`` is the regime close (the
    series carries no dividend adjustment). A missing ^GSPC partition propagates
    ``get_price_series``'s ``CONTRACT_MISMATCH`` unchanged (no ``daily_market``
    spot fallback); a non-convertible close refuses the same way the panel reader
    does, and the bounded row cap reuses ``max_source_rows``. The materialization's
    ``max_object_bytes`` is forwarded to that read, so a regime fragment is bound
    before its values decode, exactly like the daily-market source reads.
    """
    request = PriceQuery(ticker=price_history_table.REGIME_TICKER, session_date=end,
        observation_ceiling=end, lookback_sessions=0)
    series = price_history_query.get_price_series(repository, request, snapshot,
        max_object_bytes=max_object_bytes)
    rows = []
    for row in series:
        if row.date < start:
            continue
        try:
            close = float(row.close_raw)
        except (TypeError, ValueError, OverflowError):
            raise errors.fail("CONTRACT_MISMATCH", "price_history value conversion failed",
                details={"table_name": price_history_table.PRICE_HISTORY_TABLE_NAME}) from None
        rows.append({"date": row.date, "close": close})
        if len(rows) > max_source_rows:
            raise errors.fail("RESULT_LIMIT_EXCEEDED", "daily block regime row bound exceeded")
    return rows


def _daily(rows, dates, ceiling):
    index = bisect_right(dates, ceiling) - 1
    values = dict.fromkeys(_DAILY)
    if index < 0:
        return {"values": values, "anchor": None, "quote_status": "unavailable",
                "missing": dict.fromkeys(_DAILY, "unavailable")}
    row = rows[index]
    values.update({name: row[name] for name in _DAILY})
    quote = "absent" if values["implied_move"] is None else "zero" if values["implied_move"] == 0 else "present"
    if quote == "zero":
        values["implied_move"] = None
    missing = {name: "unavailable" for name, value in values.items() if value is None}
    if quote != "present":
        missing["implied_move"] = "no_quote"
    cap_date = row["mcap_asof"]
    if cap_date is None or cap_date > ceiling or cap_date > row["date"]:
        values["mcap_log"] = None
        missing["mcap_log"] = "late_capture" if cap_date is not None else "unavailable"
    return {"values": values, "quote_status": quote, "missing": missing,
        "anchor": {"ticker": row["ticker"], "source_date": row["date"],
            "mcap_asof": cap_date, "available_as_of": None, "knowledge_mode": "reconstructed"}}


def _spy_record(calculated, inputs):
    try:
        values = {name: _optional_number(float(calculated[name])) for name in _SPY}
    except ValueError:
        raise errors.fail("CONTRACT_MISMATCH", "nonfinite SPY block calculation") from None
    windows = {"spy_ret21": 21, "spy_ret63": 63, "spy_vol5": 5, "spy_vol20": 20, "spy_vol60": 60}
    return {"values": values, "source_dates": inputs,
        "ticker": price_history_table.REGIME_TICKER,
        "available_as_of": None, "knowledge_mode": "reconstructed",
        "missing": {name: "insufficient_history" if 0 < len(inputs) <= windows.get(name, 252) else "unavailable"
                    for name, value in values.items() if value is None}}


def _spy_blocks(events, rows):
    """Existing rolling arithmetic once for the full bounded event batch.

    ``rows`` are the pinned ^GSPC ``price_history`` closes (``date``/``close``)
    produced by :func:`_read_regime_series`; the nine ``spy_*`` names are
    historical and read this regime close, never a ``daily_market`` SPY spot.
    """
    forecasts = [(event, cutoff) for event in events for cutoff in event["forecasts"]]
    frame = pd.DataFrame({"date": [pd.Timestamp(e["event_date"]) for e, _ in forecasts],
        "decision": [pd.Timestamp(c["source_session_ceiling"]) for _, c in forecasts]})
    market = pd.DataFrame({"date": pd.to_datetime([row["date"] for row in rows]),
        "close": [row["close"] for row in rows]})
    try:
        regime.validated_regime_closes([row["close"] for row in rows])
    except ValueError:
        raise errors.fail("CONTRACT_MISMATCH", "regime source closes must be positive") from None
    calculated = regime.add_regime_features(frame, market, as_of_column="decision")
    dates = [row["date"] for row in rows]
    result = {}
    for index, (event, cutoff) in enumerate(forecasts):
        end = bisect_right(dates, cutoff["source_session_ceiling"])
        # All fixed windows use at most 253 observations; preserve every contributor.
        inputs = dates[max(0, end - 253):end]
        result[event["event_id"], cutoff["horizon_sessions"]] = _spy_record(calculated.iloc[index], inputs)
    return result


def build_daily_blocks(repository, source_snapshot, history_snapshot, panel_context, *,
                       price_history_snapshot, producer, policy, eligibility_metadata, source_start,
                       max_source_rows, max_events, max_object_bytes):
    """Produce detached partial blocks, not a fit/read capability or stored snapshot.

    The ``spy`` block's nine regime values read the pinned ``price_history_snapshot``'s
    ``REGIME_TICKER`` (^GSPC) ``price_history`` close -- one regime source, never a
    ``daily_market`` SPY spot. Label bounds are producer declarations, not verified
    outcome intervals. The imported daily table has no publication stamps; source
    dates only establish reconstructed prefixes. Never call this output observed-PIT
    or fit-admissible.
    """
    objects.validate_max_object_bytes(max_object_bytes)
    source_start = magnitude_panel_table.canonical_day(source_start)
    max_source_rows, max_events = _limit(max_source_rows), _limit(max_events)
    producer = _producer(panel_context, producer)
    with progress.stage("daily block admission", report_refusals=True):
        events, eligibility = _admit(repository, source_snapshot, history_snapshot,
            panel_context, policy, eligibility_metadata, source_start, max_events)
        # Admitted exactly like the other snapshot inputs, before the regime
        # reference is bound into the materialization identity below.
        price_history_snapshot = snapshot_inputs.admit_snapshot(repository, price_history_snapshot)
    ends, by_ticker = {}, {}
    for event in events:
        by_ticker.setdefault(event["ticker"], []).append(event)
        end = max(c["source_session_ceiling"] for c in event["forecasts"])
        ends[event["ticker"]] = max(ends.get(event["ticker"], ""), end)
    regime_end = max(ends.values())
    fragments = [{"fragment_id": record.fragment_id, "object_id": record.object_ref.object_id,
                  "content_hash": record.object_ref.content_hash}
        for record in repository.fragment_records(source_snapshot, "daily_market")]
    identity = {"schema_version": "magnitude_panel_daily_materialization.v1", "status": _STATUS,
        "panel_context_id": panel_context["context_id"], "source_snapshot": to_document(source_snapshot),
        "price_history_snapshot": to_document(price_history_snapshot),
        "producer": producer, "producer_identity": "caller_declared_not_installed_attestation",
        "eligibility_plan": eligibility, "label_bounds": "unverified_producer_declaration",
        "source_start": source_start, "max_source_rows": max_source_rows, "max_events": max_events,
        "max_object_bytes": max_object_bytes,
        "snapshot_daily_fragments": fragments, "daily_columns": list(_DAILY), "spy_columns": list(_SPY),
        "availability": "reconstructed_session_prefix_publication_unverified"}
    context = {**identity, "context_id": content_hash(identity)}
    with progress.stage("daily block sources", report_refusals=True):
        regime_rows = _read_regime_series(repository, price_history_snapshot, source_start,
            regime_end, max_source_rows, max_object_bytes)
        spy = _spy_blocks(events, regime_rows)
    output = []
    meter = progress.Progress("daily block events", len(events))
    with progress.stage("daily block values", report_refusals=True):
        for ticker in sorted(by_ticker):
            rows = _source_rows(
                repository, source_snapshot, ticker, source_start, ends[ticker], max_source_rows, max_object_bytes)
            surface = [row for row in rows if pd.notna(row["src_iv"])]
            dates = [row["date"] for row in surface]
            for event in by_ticker[ticker]:
                for cutoff in event["forecasts"]:
                    row = {"event_id": event["event_id"], "horizon_sessions": cutoff["horizon_sessions"],
                        "cutoff_id": cutoff["cutoff_id"], "context_id": context["context_id"], "status": _STATUS,
                        "source_session_ceiling": cutoff["source_session_ceiling"],
                        "daily_market": _daily(surface, dates, cutoff["source_session_ceiling"]),
                        "spy": spy[event["event_id"], cutoff["horizon_sessions"]]}
                    output.append({**row, "row_id": content_hash(row)})
                meter.tick()
    return {"context": context, "rows": output}
