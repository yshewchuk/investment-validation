"""Complete immutable qualified SIZE panel; private installed-wrapper backend."""
import json
from bisect import bisect_right
from collections import Counter

import numpy as np

from engine.v2.data.errors import fail
from engine.v2.features.calendar_features import calendar_features
from engine.v2.features.own_vol import compute_own_vol
from engine.v2.foundation import canonical_json, content_hash, to_document
from engine.v2.research import _magnitude_panel_sources as sources
from engine.v2.research import magnitude_confirmed_metadata, season_heat, season_population
from engine.v2.research import magnitude_panel_daily as daily
from engine.v2.research import magnitude_panel_qualified_table as table
from engine.v2.research import magnitude_panel_qualified_x as x
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.eligible_history_table import FEATURES as HISTORY
from engine.v2.research.experiment_population import load_population
from engine.v2.research.iv_dynamics import compute_iv_dynamics
from engine.v2.research.magnitude_features import calendar_input
from engine.v2.research.magnitude_panel_table import canonical_day
from engine.v2.research.progress import Progress, decision, stage
from engine.v2.research.snapshot_inputs import admit_snapshot
from engine.v2.research.tail_history import compute_tail_history


def _policy(policy):
    if policy != eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30):
        raise fail("HOLDOUT_ACCESS_DENIED", "qualified SIZE panel requires frozen 30-day policy")


def _horizons(horizons):
    if (type(horizons) is not list or not horizons
            or any(type(h) is not int or h not in (1, 3, 5) for h in horizons)
            or horizons != sorted(set(horizons))):
        raise fail("CONTRACT_MISMATCH", "ordered native horizons required")


def _daily_start(records):
    dates = [canonical_day(r.time_min[:10]) for r in records
             if r.row_count and type(r.time_min) is str and r.time_min]
    if not dates:
        raise fail("CONTRACT_MISMATCH", "required dated daily source inventory is empty")
    return min(dates)


def _price_inputs(repository, snapshot, tickers, limits):
    """Per-ticker row digests plus date arrays; each ticker's version rows are dropped after hashing."""
    digests, dates = {}, {}
    for ticker in sorted(set(tickers) | {"^GSPC"}):
        rows = sources.price_dates(repository, snapshot, ticker, limits)
        digests[ticker] = {"rows": len(rows), "digest": content_hash(rows)}
        dates[ticker] = np.array([row["date"] for row in rows], dtype="datetime64[ns]")
    return digests, dates


def _season_plan(repository, source, history, *, as_of_month, limits):
    """Roster plan at context derivation: pinned calendar identity plus the receipt and build inputs."""
    calendar, observed_source = calendar_input(repository, source,
                                               max_object_bytes=limits["max_object_bytes"])
    calendar_source = history.get("calendar_source", observed_source)
    if (type(calendar_source) is dict and type(observed_source) is dict
            and calendar_source != observed_source):
        raise fail("IDENTITY_CONFLICT", "panel calendar differs from the admitted history")
    plan = season_population.plan_roster(repository, source, as_of_month=as_of_month,
        decision_sessions=sorted({c["source_session_ceiling"] for t in history["targets"]
                                  for c in t["forecasts"]}),
        calendar=calendar, calendar_source=calendar_source, limits=limits)
    return plan["receipt"], plan["retained"], calendar.days


def _prepare(repository, source, *, history_snapshot, price_history_snapshot, as_of_month,
             policy, event_ids, horizons, history_start, source_start, limits, metadata_limits, producer, history_admitted):
    _policy(policy)
    source_start = canonical_day(source_start)
    _horizons(horizons)
    request = dict(as_of_month=as_of_month, policy=policy, event_ids=event_ids, horizons=horizons,
                   history_start=history_start, limits=limits)
    admitted = history_admitted
    history = admitted["context"]
    price_history_snapshot = admit_snapshot(repository, price_history_snapshot)
    daily_records = sources.inventory(repository, source, "daily_market", limits)
    price_records = sources.inventory(repository, price_history_snapshot, "price_history", limits)
    daily_start = _daily_start(daily_records)
    targets = history["targets"]
    if source_start > min(c["source_session_ceiling"] for t in targets for c in t["forecasts"]):
        raise fail("CONTRACT_MISMATCH", "panel source start exceeds forecast ceiling")
    tickers = sorted({t["ticker"] for t in targets})
    index = magnitude_confirmed_metadata.build_index(repository, source, tickers=tickers, limits=metadata_limits)
    price_digests, price_day_arrays = _price_inputs(repository, price_history_snapshot, tickers, limits)
    quotes, precoverage = sources.quote_plan(repository, source, index, history, request, price_day_arrays)
    receipt, roster, sessions = _season_plan(repository, source, history,
                                             as_of_month=as_of_month, limits=limits)
    groups = {g["event_id"]: g for g in history["eligibility"]["groups"]}
    event_sessions = {event["event_id"]: event["session"]
        for sequence in index["sequences"] for event in sequence["events"]}
    missing = [t["event_id"] for t in targets if t["event_id"] not in event_sessions]
    if missing:
        raise fail("CONTRACT_MISMATCH", "target event has no confirmed session")
    targets = [{**t, "session": event_sessions[t["event_id"]],
                **{k: groups[t["event_id"]][k] for k in ("label_start", "label_end")}}
               for t in targets]
    identity = {"schema_version": "magnitude_panel_context.v5", "source_snapshot": to_document(source),
        "history_snapshot": to_document(history_snapshot), "history_context": history,
        "season_heat_population": receipt,
        "price_history_snapshot": to_document(price_history_snapshot), "target_events": targets,
        "eligibility_plan": history["eligibility"],
        "quote_eligibility_plan": quotes, "confirmed_metadata": index, "producer": producer,
        "price_row_digests": price_digests,
        "feature_recipe": table.recipe("HMOSX"), "request": {**request, "source_start": source_start,
            "metadata_limits": metadata_limits},
        "quote_label_windows": "independent_pinned_price_dates_not_computed_outcome_capture",
        "daily_source_start": daily_start,
        "raw_daily_access": "selected_columns_yearly_decode_then_eligible_event_contributions_only",
        "availability": "reconstructed_prefix_publication_and_schedule_unverified",
        "source_fragments": {"daily_market": [to_document(r) for r in daily_records],
                             "price_history": [to_document(r) for r in price_records]}}
    identity.update(sources.precoverage_document(precoverage))
    context = {**identity, "context_id": content_hash(identity)}
    if len(canonical_json(context).encode()) > limits["max_total_bytes"]:
        raise fail("RESULT_LIMIT_EXCEEDED", "panel context exceeds explicit byte limit")
    # The retained roster and the pinned sessions are value-build inputs; neither is stored
    # in the context, so no per-event roster bounds the identity and no calendar is re-read.
    return context, admitted, daily_records, price_records, roster, sessions


def _block(values, anchors, **extra):
    return {"sources": anchors, "missing": {name: "insufficient_history"
            for name, value in values.items() if value is None}, **extra}


_IV_FIELDS = ("date", "implied_move", "iv10", "iv30", "exern_iv30", "rvol30")
_IV_LAGS = (5, 10, 20)


def _ivdyn_x(surface, dates, ceiling, anchor_for, ticker):
    """G2 columns from the last 21 surface rows dated <= ceiling; lineage anchors the lag reference rows."""
    end = bisect_right(dates, ceiling)
    window = surface[max(0, end - 21):end]
    values = compute_iv_dynamics([{k: row[k] for k in _IV_FIELDS} for row in window])
    anchors = [anchor_for(ticker, window[-1 - lag]["date"]) for lag in _IV_LAGS if len(window) > lag]
    return values, _block(values, anchors)


def _ownvol_x(prices, price_dates, ceiling):
    """G3 columns from the last 61 adjusted closes dated <= ceiling; lineage records window dates and row count."""
    end = bisect_right(price_dates, ceiling)
    window = prices[max(0, end - 61):end]
    values = compute_own_vol([(row.date, float(row.close_adj)) for row in window], ceiling)
    extra = {"first_date": window[0].date if window else None,
             "last_date": window[-1].date if window else None, "rows": len(window)}
    return values, _block(values, [], **extra)


def _history_x(history, moves, target):
    """G1 tail and G4 calendar columns from the H11 contributor list and the stored clock.
    Invoked by `_rows` during `experiments.size_panel.materialize`; request binding is slice 7,
    contract text slice 8."""
    entries = history["history"]
    tail = compute_tail_history([abs(moves[e["event_id"]]) for e in entries])
    prior = entries[-1]["event_date"] if entries else None
    return {**tail, **calendar_features(target["session"], target["event_date"], prior)}


def _daily_lineage(market, anchor_for, ticker):
    anchors = [] if market["anchor"] is None else [
        anchor_for(ticker, market["anchor"]["source_date"])]
    return {**market, "sources": anchors}


def _versions(repository, price_snapshot, context, ticker):
    """Re-read one ticker's version rows and verify them against the context digest."""
    rows = sources.price_dates(repository, price_snapshot, ticker, context["request"]["limits"])
    pin = context["price_row_digests"][ticker]
    if len(rows) != pin["rows"] or content_hash(rows) != pin["digest"]:
        raise fail("IDENTITY_CONFLICT", "price version rows differ from the panel context")
    return {row["date"]: row for row in rows}


def _ticker_inputs(repository, source, price_snapshot, context, ticker, sequences,
                   selected_by_ticker, request):
    sequence = sequences[ticker]
    selected = selected_by_ticker[ticker]
    end = max(c["source_session_ceiling"] for t in selected for c in t["forecasts"])
    prices = sources.prices_for(repository, price_snapshot, ticker, request["source_start"],
                                end, request["limits"])
    runup = sources.runup(selected, prices)
    surface = sources.daily_surface(repository, source, ticker, context["daily_source_start"],
                                   end, request["limits"])
    return sequence, runup, surface, [row["date"] for row in surface], prices


def _season_record(record):
    """The five stored season columns; the count is always int, the rest follow the compute result."""
    return {**{name: record[name] for name in table.SEASON},
            "season_heat_count_20": int(record["season_heat_count_20"])}


def _season_heat(repository, source, context, roster, sessions, targets):
    """One population label read and one forward heat sweep over deduplicated target clocks."""
    receipt = context["season_heat_population"]
    request = context["request"]
    labels = season_population.read_labels(repository, source, roster,
        as_of_month=request["as_of_month"], limits=request["limits"],
        latest_decision=receipt["window"]["latest_decision"])
    heat_targets = {(target["event_id"], cutoff["source_session_ceiling"]):
        {"event_id": target["event_id"], "ticker": target["ticker"], "event_date": target["event_date"],
         "decision_session": cutoff["source_session_ceiling"]}
        for target in targets for cutoff in target["forecasts"]}
    results = season_heat.compute_season_heat(labels, list(heat_targets.values()), sessions)
    return results, receipt["receipt_id"]


class _NullCounts:
    """Per-build null counts of the HMOSX columns; one stderr line, never per row."""
    def __init__(self):
        self.rows, self.counts = 0, Counter()

    def add(self, row):
        self.rows += 1
        self.counts.update(k for k in x.X_COLUMNS if row[k] is None)

    def log(self):
        decision("panel x columns", f"rows={self.rows} null counts: " + ", ".join(
            f"{k}={self.counts[k]}" for k in x.X_COLUMNS))


def _history_missing(history_values, hx):
    return {**history_values, **{k: hx[k] for k in (*x.TAIL, "cal_days_since_prior")}}


def _prior_events(history, moves, stored):
    """Signed streak and the 12-event EMA (mean prior move until twelve prior events exist)."""
    return {"signed_streak": sources.streak(history, moves),
            "ema12r_abs": stored["ema12_prior_abs_move"] if stored["n_prior"] >= 12
            else stored["mean_prior_abs_move"]}


def _market_x(inputs, ceiling, anchor_for, ticker):
    """G2 and G3 columns plus their lineage blocks for one decision ceiling."""
    surface, dates, prices, price_dates = inputs
    iv, iv_block = _ivdyn_x(surface, dates, ceiling, anchor_for, ticker)
    own, own_block = _ownvol_x(prices, price_dates, ceiling)
    return {**iv, **own}, {"ivdyn": iv_block, "ownvol": own_block}


def _rows(repository, source, price_snapshot, prepared):
    context, admitted, daily_records, price_records, roster, sessions = prepared
    daily_anchors = sources.anchor_lookup(daily_records,
        daily_contract=repository.table_contract(source, "daily_market"))
    price_anchors = sources.anchor_lookup(price_records)
    gspc_anchors = sources.anchor_lookup(price_records)  # regime dates only; survives ticker changes
    request = context["request"]
    population = load_population(repository, source, as_of_month=request["as_of_month"])
    moves = sources.signed_moves(repository, source, admitted, population)
    targets = context["target_events"]
    heat, season_receipt_id = _season_heat(repository, source, context, roster, sessions, targets)
    targets_by_id = {target["event_id"]: target for target in targets}
    cutoffs = {(target["event_id"], cutoff["horizon_sessions"]): cutoff
        for target in targets for cutoff in target["forecasts"]}
    selected_by_ticker = {}
    for target in targets:
        selected_by_ticker.setdefault(target["ticker"], []).append(target)
    sequences = {sequence["ticker"]: sequence
        for sequence in context["confirmed_metadata"]["sequences"]}
    ceiling = max(c["source_session_ceiling"] for t in targets for c in t["forecasts"])
    regime = daily._read_regime_series(repository, price_snapshot, request["source_start"], ceiling,
        request["limits"]["max_source_rows"], request["limits"]["max_object_bytes"])
    spy = daily._spy_blocks(targets, regime)
    gspc_versions = _versions(repository, price_snapshot, context, "^GSPC")
    versions = {}
    retained = set(context["quote_eligibility_plan"]["retained_event_ids"])
    meter = Progress("qualified panel events", len(targets))
    history_rows = iter(admitted["rows"])
    current_ticker = None
    nulls = _NullCounts()
    for stored in history_rows:
        key = stored["event_id"], stored["horizon_sessions"]
        target = targets_by_id[key[0]]
        cutoff = cutoffs[key]
        ticker = target["ticker"]
        if ticker != current_ticker:
            price_anchors.clear()  # anchor caches are per active ticker, not per run
            daily_anchors.clear()
            sequence, runup, surface, dates, prices = _ticker_inputs(repository, source, price_snapshot,
                context, ticker, sequences, selected_by_ticker, request)
            price_dates = [row.date for row in prices]
            inputs = (surface, dates, prices, price_dates)
            versions = _versions(repository, price_snapshot, context, ticker)
            current_ticker = ticker
        history = json.loads(stored["history_json"])
        hx = _history_x(history, moves, target)
        prior = _prior_events(history, moves, stored)
        runvalues, rundates = runup[key]
        market = daily._daily(surface, dates, cutoff["source_session_ceiling"])
        xm, xm_blocks = _market_x(inputs, cutoff["source_session_ceiling"], daily_anchors, ticker)
        implied, implied_lineage = sources.implied_history(sequence, target,
            cutoff["source_session_ceiling"], retained, surface, daily_anchors, request["history_start"])
        history_values = {name: stored[name] for name in HISTORY}
        blocks = {"history": _block(_history_missing(history_values, hx), history["history"]),
            "prior_events": _block(prior, history["history"]),
            "runup": _block(runvalues, [price_anchors(ticker, d, versions[d]) for d in rundates]),
            "spy": _block(spy[key]["values"], [gspc_anchors("^GSPC", d, gspc_versions[d])
                                              for d in spy[key]["source_dates"]]),
            "daily_market": _daily_lineage(market, daily_anchors, ticker),
            "implied_history": implied_lineage,
            "season_heat": {"receipt_id": season_receipt_id},
            **xm_blocks}
        row = {"schema_version": table.CONTRACT.contract_id, "event_id": key[0],
            "context_id": context["context_id"], "history_context_id": admitted["context"]["context_id"],
            "horizon_sessions": key[1], "event_date": target["event_date"],
            "cutoff_id": cutoff["cutoff_id"], "source_session_ceiling": cutoff["source_session_ceiling"],
            "lineage": canonical_json(blocks), "current_quote_status": market["quote_status"],
            "has_implied_quote": int(market["quote_status"] == "present"), **history_values,
            **prior, **runvalues, **spy[key]["values"], **market["values"],
            "mean_prior_or_implied": implied,
            **_season_record(heat[key[0], cutoff["source_session_ceiling"]]),
            **hx, **xm}
        nulls.add(row)
        yield {**row, "row_id": content_hash(row)}
        if key[1] == target["forecasts"][-1]["horizon_sessions"]:
            meter.tick()
    nulls.log()


def _materialize(repository, source, **request):
    from engine.v2.research import _magnitude_panel_store as store

    with stage("qualified panel admission", report_refusals=True):
        prepared = _prepare(repository, source, **request)
    existing = store.existing(repository, prepared[0])
    if existing is not None:
        admitted = prepared[1]
        total = sum(len(target["forecasts"]) for target in admitted["context"]["targets"])
        with stage("qualified panel replay history validation", report_refusals=True):
            meter = Progress("qualified panel replay history", total)
            for _ in admitted["rows"]:
                meter.tick()
        return {"context": prepared[0], "feature_snapshot": existing}
    with stage("qualified panel values", report_refusals=True):
        rows = _rows(repository, source, request["price_history_snapshot"], prepared)
        snapshot = store.publish(repository, source, prepared[0], rows)
    return {"context": prepared[0], "feature_snapshot": snapshot}


def _read(repository, source, feature_snapshot, *, columns, lineage=True, **request):
    from engine.v2.research import _magnitude_panel_store as store

    table.admit_columns(columns, season=True, x_groups=True)
    with stage("qualified stored panel admission", report_refusals=True):
        context = _prepare(repository, source, **request)[0]
        rows = store.read(repository, feature_snapshot, context, columns, lineage)
    return {"context": context, "feature_snapshot": to_document(feature_snapshot), "rows": rows}
