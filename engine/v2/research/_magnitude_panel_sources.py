"""Qualified panel source composition; internal orchestration, not read authority."""
import json
from bisect import bisect_left, bisect_right
from datetime import date, timedelta
from functools import cache

import numpy as np
import pandas as pd

from engine.v2.contracts.data import DataQuery, KeyPredicate, PriceQuery, TimeInterval
from engine.v2.data import price_history, price_history_query
from engine.v2.data.computed_move_bounds import source_bounds
from engine.v2.data.errors import fail
from engine.v2.data.event_history import event_sequence, strict_prior_mean
from engine.v2.data.manifests import table_contract_hash
from engine.v2.data.price_history_table import PRICE_HISTORY_CONTRACT
from engine.v2.data.query import plan_scan_population
from engine.v2.features import runup_math
from engine.v2.foundation import canonical_json, content_hash, to_document
from engine.v2.research import eligibility_policy, event_history
from engine.v2.research import magnitude_panel_daily as daily
from engine.v2.research import magnitude_panel_qualified_table as table_contract
from engine.v2.research.magnitude_features import calendar_input
from engine.v2.research.magnitude_inputs import forecast_cutoff
from engine.v2.research.progress import Progress


def inventory(repository, snapshot, table, limits):
    """Bound physical source inventory before native readers decode any payload."""
    contract = repository.table_contract(snapshot, table)
    if ((table == "daily_market" and (contract.contract_id != "legacy.daily_market.v2"
            or table_contract_hash(contract) != table_contract._DAILY_HASH))
            or (table == "price_history" and contract != PRICE_HISTORY_CONTRACT)):
        raise fail("UNSUPPORTED_CONTRACT", "qualified panel source definition is unsupported")
    records = repository.fragment_records(snapshot, table)
    if not any(r.row_count for r in records):
        raise fail("CONTRACT_MISMATCH", "required panel source inventory is empty")
    if (len(records) > limits["max_fragments"]
            or sum(r.row_count for r in records) > limits["max_source_rows"]
            or sum(r.object_ref.byte_size for r in records) > limits["max_total_bytes"]
            or any(r.object_ref.byte_size > limits["max_object_bytes"] for r in records)):
        raise fail("RESULT_LIMIT_EXCEEDED", "panel source inventory exceeds explicit limits")
    return records


def price_dates(repository, snapshot, ticker, limits):
    """Read only version/date metadata, never future price values for label brackets."""
    contract = repository.table_contract(snapshot, "price_history")
    predicates = (KeyPredicate(column="ticker", operator="eq", values=(ticker,)),)
    ref = snapshot.table_versions["price_history"].table_contract_ref
    count = repository.scan_population_bound(snapshot.snapshot_id, table_name="price_history",
        table_contract_ref=ref, key_filter=predicates)
    if not count:
        raise fail("CONTRACT_MISMATCH", "quote group source dates unavailable")
    request = DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=ref,
        columns=("date", "retrieved_at", "deleted", "source_hash", "capture_id"), key_filter=predicates,
        order_by=contract.primary_key, max_result_rows=count,
        max_batch_rows=min(max(1, count), contract.maximum_batch_rows))
    rows = [r for b in repository.scan(request, table_name="price_history",
        max_object_bytes=limits["max_object_bytes"]) for r in b.to_pylist()]
    if not rows:
        raise fail("CONTRACT_MISMATCH", "quote group source dates unavailable")
    pool = price_history.resolve_pool(pd.DataFrame(rows), latest_snapshot=True)
    return pool.loc[~pool.deleted].sort_values("date").to_dict("records")


def quote_plan(repository, source, index, history_context, request, price_metadata):
    """Independent quote population; missing realized labels never select quotes."""
    calendar, _ = calendar_input(repository, source)
    groups = []
    meter = Progress("qualified quote group metadata", len(index["sequences"]))
    for sequence in index["sequences"]:
        sd = np.array([r["date"] for r in price_metadata[sequence["ticker"]]], dtype="datetime64[ns]")
        latest = max(c["source_session_ceiling"] for t in history_context["targets"]
                     if t["ticker"] == sequence["ticker"] for c in t["forecasts"])
        for event in sequence["events"]:
            # Full ordinals remain in the index, including pre-start/protected rows.
            if event["ordinal"] < 4 or event["event_date"] < request["history_start"]:
                continue
            if event["event_date"] >= latest:
                continue
            lo, hi = source_bounds(sd, event["event_date"], event["session"])
            if lo is None:
                raise fail("CONTRACT_MISMATCH", "historical quote label bracket unavailable")
            clocks = [forecast_cutoff(pd.Timestamp(event["event_date"]), event["session"], h, calendar)
                      for h in (1, 3, 5)]
            groups.append({"event_id": event["event_id"], "ticker": event["ticker"],
                "event_date": event["event_date"], "label_start": lo, "label_end": hi,
                "forecasts": [{"horizon_sessions": c["horizon_sessions"],
                    "decision_date": c["source_session_ceiling"],
                    "source_session_ceiling": c["source_session_ceiling"]} for c in clocks]})
        meter.tick()
    metadata = {"schema_version": "event_group_bounds.v1", "source_snapshot": to_document(source),
        "holdout_as_of_month": request["as_of_month"], "policy_id": request["policy"]["policy_id"],
        "horizons": [1, 3, 5], "groups": groups}
    metadata["metadata_id"] = content_hash(metadata)
    return eligibility_policy.plan_eligibility(repository, source, as_of_month=request["as_of_month"],
        policy=request["policy"], metadata=metadata, evaluation_event_ids=[])


def signed_moves(repository, source, admitted, population):
    """One guarded read per qualified union contributor, not signs inferred from H11."""
    meter = Progress("eligible-history identity rows",
        sum(len(target["forecasts"]) for target in admitted["context"]["targets"]))
    expected = {}
    for row in admitted["rows"]:
        history = json.loads(row["history_json"])
        for entry in history["history"]:
            expected[entry["event_id"]] = {name: entry[name] for name in
                ("fragment_id", "object_id", "content_hash")}
        meter.tick()
    candidates = population[population.event_id.isin(expected)].sort_values(["event_date", "event_id"])
    plans, reader = event_history.plan_with_reader(repository, source, candidates,
        repository.table_contract(source, "computed_moves"))
    values = {}
    bounds = {r["event_id"]: r for r in admitted["context"]["contributors"]}
    for event, query, record in plans:
        anchor = expected[event.event_id]
        if any(anchor[name] != value for name, value in (
                ("fragment_id", record.fragment_id), ("object_id", record.object_ref.object_id),
                ("content_hash", record.object_ref.content_hash))):
            raise fail("CONTRACT_MISMATCH", "signed history differs from admitted artifact")
        rows = reader(event, query)
        available = (date.fromisoformat(bounds[event.event_id]["post_close_date"]) + timedelta(days=1)).isoformat()
        outcome, reason = event_history._historical_target(rows, event, date.fromisoformat(available))
        if reason or outcome[2] != available:
            raise fail("CONTRACT_MISMATCH", "signed source availability differs from qualified history")
        values[event.event_id] = outcome[0]
    if values.keys() != expected.keys():
        raise fail("CONTRACT_MISMATCH", "signed history union incomplete")
    return values


def streak(history, moves):
    values = [moves[e["event_id"]] for e in history["history"]]
    frame = pd.DataFrame({"ticker": [history["ticker"]] * (len(values) + 1),
                          "move": [*values, 0.]})
    return float(runup_math._signed_streak(frame)[-1])


def anchor(records, ticker, day, version=None, *, daily_contract=None):
    """Bind daily rows with the native ticker/year/time membership planner."""
    if daily_contract is not None:
        keys = (KeyPredicate(column="ticker", operator="eq", values=(ticker,)),
                KeyPredicate(column="year", operator="eq", values=(int(day[:4]),)))
        interval = TimeInterval(column="date", start_inclusive=day,
            end_exclusive=(date.fromisoformat(day) + timedelta(days=1)).isoformat())
        matches = plan_scan_population(daily_contract, records,
            key_filter=keys, time_interval=interval).records
    else:
        matches = [r for r in records if r.partition_key == ticker
                   and r.primary_key_min[1] <= day <= r.primary_key_max[1]]
    if not matches:
        raise fail("CONTRACT_MISMATCH", "source row cannot be bound to a pinned fragment")
    return {"ticker": ticker, "source_date": day, "available_as_of": None,
        "knowledge_mode": "reconstructed", "row_key": [ticker, day],
        "version": version,
        "fragments": [{"fragment_id": r.fragment_id, "object_id": r.object_ref.object_id,
                       "content_hash": r.object_ref.content_hash} for r in matches]}


def anchor_lookup(records, *, daily_contract=None):
    """Index once per invocation; cache canonical anchors, return detached values."""
    partitions = {}
    for i, record in enumerate(records):
        partitions.setdefault(record.partition_key, []).append((i, record))

    @cache
    def encoded(ticker, day, version):
        candidates = dict(partitions.get(ticker, []) + partitions.get(day[:4], []))
        return canonical_json(anchor([candidates[i] for i in sorted(candidates)],
                                     ticker, day, json.loads(version), daily_contract=daily_contract))

    def lookup(ticker, day, version=None):
        return json.loads(encoded(ticker, day, canonical_json(version)))
    return lookup


def implied_history(sequence, target, ceiling, retained, rows, anchor_for, history_start):
    prefix = [e for e in sequence["events"] if e["event_date"] < min(target["event_date"], ceiling)]
    events = event_sequence([e["event_date"] for e in prefix] + [target["event_date"]], warmup=4)
    dates, values, sources, membership = [r["date"] for r in rows], [], [], []
    for event in prefix:
        eligible = (event["ordinal"] >= 4 and event["event_date"] >= history_start
                    and event["event_id"] in retained)
        value, source = None, None
        if eligible:
            index = bisect_left(dates, event["event_date"]) - 1
            if index >= 0:
                row = rows[index]
                if row["date"] > ceiling:
                    raise fail("CONTRACT_MISMATCH", "historical quote exceeds forecast ceiling")
                value = row["implied_move"]
                source = anchor_for(target["ticker"], row["date"])
                sources.append(source)
        values.append(value)
        membership.append({"event_id": event["event_id"], "ordinal": event["ordinal"],
                           "eligible": eligible, "source": source})
    mean, count = strict_prior_mean(events, [*values, None], population="admitted")[-1]
    return mean, {"sources": sources, "prefix": membership, "contribution_count": count,
                  "missing": {} if mean is not None else {"mean_prior_or_implied": "unavailable"}}


def prices_for(repository, snapshot, ticker, start, end, limits):
    series = price_history_query.get_price_series(repository,
        PriceQuery(ticker=ticker, session_date=end, observation_ceiling=end, lookback_sessions=0),
        snapshot, max_object_bytes=limits["max_object_bytes"])
    rows = [row for row in series if row.date >= start]
    if len(rows) > limits["max_source_rows"]:
        raise fail("RESULT_LIMIT_EXCEEDED", "panel price row bound exceeded")
    if any(row.close_adj is None or not np.isfinite(row.close_adj) or row.close_adj <= 0 for row in rows):
        raise fail("CONTRACT_MISMATCH", "invalid adjusted runup price")
    return rows


def runup(targets, rows):
    forecasts = [(t, c) for t in targets for c in t["forecasts"]]
    frame = pd.DataFrame({"date": pd.to_datetime([t["event_date"] for t, _ in forecasts]),
        "decision": pd.to_datetime([c["source_session_ceiling"] for _, c in forecasts])})
    prices = pd.DataFrame({"date": pd.to_datetime([r.date for r in rows]),
                           "close_adj": [r.close_adj for r in rows]})
    names = ("dist_high", "dist_ema", "ret5", "ret10", "ret20")
    # Preserve the native >=300-source-observation gate and full EMA prefix.
    calculated = (runup_math._price_features(frame, prices, frame.decision.to_numpy())
                  if len(rows) >= 300 else {name: [np.nan] * len(frame) for name in names})
    dates = [r.date for r in rows]
    result = {}
    for i, (target, cutoff) in enumerate(forecasts):
        end = min(bisect_right(dates, cutoff["source_session_ceiling"]),
                  bisect_left(dates, target["event_date"]))
        values = {name: (daily._optional_number(float(calculated[name][i])) if end >= 300 else None)
                  for name in names}
        values["abs_dist_ema"] = None if values["dist_ema"] is None else abs(values["dist_ema"])
        result[target["event_id"], cutoff["horizon_sessions"]] = (values, dates[:end])
    return result


def daily_surface(repository, source, ticker, start, end, limits):
    """Decode selected raw daily columns once per ticker/year, then select rows.

    The native scanner decodes selected columns in whole surviving fragments
    before date filtering. This is explicitly permitted raw-market-source access,
    NOT permission for excluded events to contribute or for protected outcomes.
    Quote grouping is already admitted; implied_history never looks up excluded
    events' quotes. Physical and decoded-object byte limits remain native.
    """
    rows = daily._source_rows(repository, source, ticker, start, end,
                              limits["max_source_rows"], limits["max_object_bytes"])
    _daily_ranges(rows)
    return [row for row in rows if pd.notna(row["src_iv"])]


def _daily_ranges(rows):
    for row in rows:
        for name, (lo, hi) in table_contract._DAILY_RANGES.items():
            if row[name] is not None and not lo <= row[name] <= hi:
                raise fail("CONTRACT_MISMATCH", "daily source violates pinned normalization range")
