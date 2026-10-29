"""Complete immutable qualified SIZE panel; private installed-wrapper backend."""
import json

from engine.v2.data.errors import fail
from engine.v2.foundation import canonical_json, content_hash, to_document
from engine.v2.research import _magnitude_panel_sources as sources
from engine.v2.research import magnitude_confirmed_metadata
from engine.v2.research import magnitude_panel_daily as daily
from engine.v2.research import magnitude_panel_qualified_table as table
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.eligible_history_table import FEATURES as HISTORY
from engine.v2.research.experiment_population import load_population
from engine.v2.research.magnitude_panel_table import canonical_day
from engine.v2.research.progress import Progress, stage
from engine.v2.research.snapshot_inputs import admit_snapshot


def _policy(policy):
    if policy != eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30):
        raise fail("HOLDOUT_ACCESS_DENIED", "qualified SIZE panel requires frozen 30-day policy")


def _daily_start(records):
    dates = [canonical_day(r.time_min[:10]) for r in records
             if r.row_count and type(r.time_min) is str and r.time_min]
    if not dates:
        raise fail("CONTRACT_MISMATCH", "required dated daily source inventory is empty")
    return min(dates)


def _prepare(repository, source, *, history_snapshot, price_history_snapshot, as_of_month,
             policy, event_ids, horizons, history_start, source_start, limits, metadata_limits, producer, history_admitted):
    _policy(policy)
    source_start = canonical_day(source_start)
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
    price_metadata = {ticker: sources.price_dates(repository, price_history_snapshot, ticker, limits)
                      for ticker in sorted(set(tickers) | {"^GSPC"})}
    quotes = sources.quote_plan(repository, source, index, history, request, price_metadata)
    groups = {g["event_id"]: g for g in history["eligibility"]["groups"]}
    targets = [{**t, **{k: groups[t["event_id"]][k] for k in ("label_start", "label_end")}}
               for t in targets]
    identity = {"schema_version": "magnitude_panel_context.v3", "source_snapshot": to_document(source),
        "history_snapshot": to_document(history_snapshot), "history_context": history,
        "price_history_snapshot": to_document(price_history_snapshot), "target_events": targets,
        "eligibility_plan": history["eligibility"],
        "quote_eligibility_plan": quotes, "confirmed_metadata": index, "producer": producer,
        "price_row_metadata": price_metadata,
        "feature_recipe": table.recipe("HMO"), "request": {**request, "source_start": source_start,
            "metadata_limits": metadata_limits},
        "quote_label_windows": "independent_pinned_price_dates_not_computed_outcome_capture",
        "daily_source_start": daily_start,
        "raw_daily_access": "selected_columns_yearly_decode_then_eligible_event_contributions_only",
        "availability": "reconstructed_prefix_publication_and_schedule_unverified",
        "source_fragments": {"daily_market": [to_document(r) for r in daily_records],
                             "price_history": [to_document(r) for r in price_records]}}
    context = {**identity, "context_id": content_hash(identity)}
    if len(canonical_json(context).encode()) > limits["max_total_bytes"]:
        raise fail("RESULT_LIMIT_EXCEEDED", "panel context exceeds explicit byte limit")
    return context, admitted, daily_records, price_records


def _block(values, anchors, **extra):
    return {"sources": anchors, "missing": {name: "insufficient_history"
            for name, value in values.items() if value is None}, **extra}


def _daily_lineage(market, anchor_for, ticker):
    anchors = [] if market["anchor"] is None else [
        anchor_for(ticker, market["anchor"]["source_date"])]
    return {**market, "sources": anchors}


def _price_versions(context):
    return {(ticker, row["date"]): row for ticker, rows in context["price_row_metadata"].items() for row in rows}


def _next_history_row(rows, key):
    row = next(rows)
    if (row["event_id"], row["horizon_sessions"]) != key:
        raise fail("CONTRACT_MISMATCH", "streamed history target/cutoff order differs")
    return row


def _rows(repository, source, price_snapshot, prepared):
    context, admitted, daily_records, price_records = prepared
    daily_anchors = sources.anchor_lookup(daily_records,
        daily_contract=repository.table_contract(source, "daily_market"))
    price_anchors = sources.anchor_lookup(price_records)
    request = context["request"]
    versions = _price_versions(context)
    population = load_population(repository, source, as_of_month=request["as_of_month"])
    moves = sources.signed_moves(repository, source, admitted, population)
    history_rows = iter(admitted["rows"])
    targets = context["target_events"]
    ceiling = max(c["source_session_ceiling"] for t in targets for c in t["forecasts"])
    regime = daily._read_regime_series(repository, price_snapshot, request["source_start"], ceiling,
        request["limits"]["max_source_rows"], request["limits"]["max_object_bytes"])
    spy = daily._spy_blocks(targets, regime)
    retained = set(context["quote_eligibility_plan"]["retained_event_ids"])
    meter = Progress("qualified panel events", len(targets))
    for sequence in context["confirmed_metadata"]["sequences"]:
        ticker = sequence["ticker"]
        selected = [t for t in targets if t["ticker"] == ticker]
        end = max(c["source_session_ceiling"] for t in selected for c in t["forecasts"])
        prices = sources.prices_for(repository, price_snapshot, ticker, request["source_start"],
                                    end, request["limits"])
        runup = sources.runup(selected, prices)
        surface = sources.daily_surface(repository, source, ticker, context["daily_source_start"], end, request["limits"])
        dates = [r["date"] for r in surface]
        for target in selected:
            for cutoff in target["forecasts"]:
                key = target["event_id"], cutoff["horizon_sessions"]
                stored = _next_history_row(history_rows, key)
                history = json.loads(stored["history_json"])
                prior = {"signed_streak": sources.streak(history, moves),
                    "ema12r_abs": stored["ema12_prior_abs_move"] if stored["n_prior"] >= 12
                    else stored["mean_prior_abs_move"]}
                runvalues, rundates = runup[key]
                market = daily._daily(surface, dates, cutoff["source_session_ceiling"])
                implied, implied_lineage = sources.implied_history(sequence, target,
                    cutoff["source_session_ceiling"], retained, surface, daily_anchors, request["history_start"])
                history_values = {name: stored[name] for name in HISTORY}
                blocks = {"history": _block(history_values, history["history"]),
                    "prior_events": _block(prior, history["history"]),
                    "runup": _block(runvalues, [price_anchors(ticker, d, versions[ticker, d]) for d in rundates]),
                    "spy": _block(spy[key]["values"], [price_anchors("^GSPC", d, versions["^GSPC", d])
                                                      for d in spy[key]["source_dates"]]),
                    "daily_market": _daily_lineage(market, daily_anchors, ticker),
                    "implied_history": implied_lineage}
                row = {"schema_version": table.CONTRACT.contract_id, "event_id": key[0],
                    "context_id": context["context_id"], "history_context_id": admitted["context"]["context_id"],
                    "horizon_sessions": key[1], "event_date": target["event_date"],
                    "cutoff_id": cutoff["cutoff_id"], "source_session_ceiling": cutoff["source_session_ceiling"],
                    "lineage": canonical_json(blocks), "current_quote_status": market["quote_status"],
                    "has_implied_quote": int(market["quote_status"] == "present"), **history_values,
                    **prior, **runvalues, **spy[key]["values"], **market["values"],
                    "mean_prior_or_implied": implied}
                yield {**row, "row_id": content_hash(row)}
            meter.tick()

    try:
        next(history_rows)
    except StopIteration:
        pass
    else:
        raise fail("CONTRACT_MISMATCH", "streamed history has extra rows")


def _materialize(repository, source, **request):
    from engine.v2.research import _magnitude_panel_store as store

    with stage("qualified panel admission", report_refusals=True):
        prepared = _prepare(repository, source, **request)
    existing = store.existing(repository, prepared[0])
    if existing is not None:
        return {"context": prepared[0], "feature_snapshot": existing}
    with stage("qualified panel values", report_refusals=True):
        rows = _rows(repository, source, request["price_history_snapshot"], prepared)
        snapshot = store.publish(repository, source, prepared[0], rows)
    return {"context": prepared[0], "feature_snapshot": snapshot}


def _read(repository, source, feature_snapshot, *, columns, **request):
    from engine.v2.research import _magnitude_panel_store as store

    table.admit_columns(columns)
    with stage("qualified stored panel admission", report_refusals=True):
        context = _prepare(repository, source, **request)[0]
        rows = store.read(repository, feature_snapshot, context, columns)
    return {"context": context, "feature_snapshot": to_document(feature_snapshot), "rows": rows}
