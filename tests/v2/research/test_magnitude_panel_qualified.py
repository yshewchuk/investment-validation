"""Synthetic full qualified panel, protected quote independence and stored replay."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.data import objects
from engine.v2.data.errors import DataError
from engine.v2.data.price_history_table import PRICE_HISTORY_CONTRACT
from engine.v2.data.repository import Repository
from engine.v2.foundation import SystemClock, content_hash
from engine.v2.research import _magnitude_panel_sources as sources
from engine.v2.research import magnitude_panel_qualified_table as table
from engine.v2.research import magnitude_panel_v2_table as v2
from experiments import eligible_history, size_panel
from tests.data_scan_support import (
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.v2.research.magnitude_trial_support import calendar_sessions, poison_outcomes
from tests.v2.research.test_eligible_history import source
from tests.v2.research.test_magnitude_confirmed_metadata import LIMITS as METADATA_LIMITS


@pytest.fixture
def panel_data(tmp_path, request):
    spec = getattr(request, "param", {})
    days = spec.get("days", ["2022-06-02", "2023-01-03", "2023-03-20", "2023-04-03",
                             "2023-05-15", "2023-06-02", "2024-06-17"])
    data = source(tmp_path, null_day="2023-05-15", days=days, values=spec.get("values"))
    store = data.repository._store
    repo = data.repository = Repository(data.conn, store, pin_snapshots=True)
    daily_contract = repo.table_contract(data.snapshot, "daily_market")
    days = calendar_sessions("2022-05-20", "2024-06-20").days
    market = []
    for day in days:
        values = {name: float(i + 1) for i, name in enumerate(sources.daily._DAILY)}
        values.update(ticker="SYNTH", date=pd.Timestamp(day), year=int(day[:4]),
                      src_iv="synthetic", mcap_asof=pd.Timestamp(day), implied_move=8., spot=100.)
        if day == "2023-06-01":
            values["implied_move"] = 0.
        if day == "2024-06-14":
            values.update(implied_move=0., mcap_asof=pd.Timestamp("2030-01-01"))
        market.append(values)
    daily_records = list(repo.fragment_records(data.snapshot, "daily_market"))
    for year in sorted({r["year"] for r in market}):
        group = [r for r in market if r["year"] == year]
        daily_records.append(publish_and_inspect(store, daily_contract, contract_ref_for(daily_contract),
            sorted(group, key=lambda r: (r["ticker"], r["date"])), str(year)))
    tables = {name: list(repo.fragment_records(data.snapshot, name)) for name in data.snapshot.table_versions}
    tables["daily_market"] = sorted(daily_records, key=lambda r: (r.partition_key, r.primary_key_min))
    contracts = {name: repo.table_contract(data.snapshot, name) for name in data.snapshot.table_versions}
    hashes = {}
    for name, records in tables.items():
        hashes[name] = {partition: objects.partition_logical_hash(store,
            [r.object_ref for r in records if r.partition_key == partition], contracts[name],
            contract_ref_for(contracts[name]), partition) for partition in {r.partition_key for r in records}}
    snapshot = commit_tables(data.conn, SystemClock(), tables, contracts, store=store,
        partition_logical_hashes=hashes, scope="panel-source", receipt_id="panel-source", attempt_id="panel-source")
    price_records = []
    for ticker in ("SYNTH", "^GSPC"):
        rows = [dict(ticker=ticker, date=day, close_adj=100. + i / 10,
            close_raw=200. + i / 10, high_raw=201. + i / 10, retrieved_at="2025-01-01T00:00:00Z",
            deleted=False, source_kind="massive", source_hash="sha256:" + "ab" * 32,
            capture_id="synthetic-prices") for i, day in enumerate(days)]
        price_records.append(publish_and_inspect(store, PRICE_HISTORY_CONTRACT,
            contract_ref_for(PRICE_HISTORY_CONTRACT), rows, ticker))
    prices = commit_tables(data.conn, SystemClock(), {"price_history": price_records},
        {"price_history": PRICE_HISTORY_CONTRACT}, store=store, scope="panel-prices",
        receipt_id="panel-prices", attempt_id="panel-prices")
    limits = {**data.request["limits"], "max_object_bytes": 8 << 20, "max_total_bytes": 32 << 20}
    history_request = {**data.request, "limits": limits}
    artifact = eligible_history.materialize(repo, snapshot, **history_request)
    request = {**history_request, "history_snapshot": artifact["feature_snapshot"],
        "price_history_snapshot": prices, "source_start": days[0],
        "metadata_limits": {**METADATA_LIMITS, "max_catalog_fragments": 100, "max_fragments": 100,
                            "max_object_bytes": 8 << 20}}
    yield SimpleNamespace(**vars(data), panel_source=snapshot, prices=prices, panel_request=request)
    data.conn.close()


def read(data, artifact, **changes):
    return size_panel.read(data.repository, data.panel_source, artifact["feature_snapshot"],
        columns=[*table.FEATURES, "has_implied_quote"], **{**data.panel_request, **changes})


def test_public_full_table_and_independent_missing_realized_quote(panel_data, monkeypatch):
    data = panel_data
    poison_outcomes(monkeypatch, data.repository, data.panel_source,
                    ["2023-03-20", "2023-04-03", "2023-05-15", "2024-06-17"])
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    result = read(data, artifact)
    assert [r["horizon_sessions"] for r in result["rows"]] == [1, 3, 5]
    assert artifact["context"]["schema_version"] == "magnitude_panel_context.v3"
    plan = artifact["context"]["eligibility_plan"]
    assert plan["plan_id"] == content_hash({k: v for k, v in plan.items() if k != "plan_id"})
    assert plan["policy"]["policy_id"] == data.panel_request["policy"]["policy_id"]
    assert artifact["context"]["target_events"][0]["label_end"] == "2024-06-18"
    assert artifact["context"]["quote_label_windows"] == "independent_pinned_price_dates_not_computed_outcome_capture"
    assert "quote_precoverage_excluded" not in artifact["context"]  # no pre-coverage events: identity unchanged
    for row in result["rows"]:
        assert row["n_prior"] == 3
        assert row["mean_prior_move"] == 0.
        assert row["signed_streak"] == 1.
        assert row["ema12r_abs"] == 4.
        assert row["abs_dist_ema"] == abs(row["dist_ema"])
        assert row["mean_prior_or_implied"] == 4.  # missing realized label's 8 + preserved 0
        lineage = json.loads(row["lineage"])
        assert lineage["implied_history"]["contribution_count"] == 2
        assert [r["ordinal"] for r in lineage["implied_history"]["prefix"]] == list(range(6))
        assert lineage["implied_history"]["prefix"][4]["eligible"] is True
        assert row["spy_ret21"] is not None
    last = result["rows"][0]
    assert last["current_quote_status"] == "zero"
    assert last["implied_move"] is None and last["has_implied_quote"] == 0
    assert last["mcap_log"] is None
    assert json.loads(last["lineage"])["daily_market"]["missing"]["mcap_log"] == "late_capture"
    poison_outcomes(monkeypatch, data.repository, data.panel_source, (), all_outcomes=True)
    assert size_panel.materialize(data.repository, data.panel_source, **data.panel_request) == artifact
    assert read(data, artifact) == result


def test_versioned_recipe_preserves_existing_hashes_and_dictionary_order():
    for subset, length in (("H", 11), ("HM", 28), ("HMO", 42)):
        before = v2.recipe(subset)
        current = table.recipe(subset)
        assert current["dictionary_hash"] != before["dictionary_hash"]
        assert current["model_columns"] == before["model_columns"]
        assert len(current["dictionary"]) == length
        assert v2.recipe(subset) == before
        assert current["schema_version"] == "research_magnitude_panel.v3"


@pytest.mark.parametrize("name", ["producer", "history_producer", "history_admitted"])
def test_public_producer_override_refused_before_io(name):
    with pytest.raises(DataError, match="installed producer"):
        size_panel.materialize(None, None, **{name: {}})
    with pytest.raises(DataError, match="installed producer"):
        size_panel.read(None, None, None, columns=["n_prior"], **{name: {}})


def test_projection_and_context_mismatch(panel_data):
    data = panel_data
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    rows = size_panel.read(data.repository, data.panel_source, artifact["feature_snapshot"],
        columns=["n_prior"], **data.panel_request)["rows"]
    assert set(rows[0]) == set(table.METADATA) | {"n_prior"}
    for changes in ({"horizons": [1]}, {"source_start": "2022-05-21"},
                    {"event_ids": [data.events[-2]["event_id"]]}):
        with pytest.raises(DataError):
            read(data, artifact, **changes)
    for columns in (["realized_move_pct"], ["or_implied"], ["n_prior", "n_prior"], []):
        with pytest.raises(DataError):
            size_panel.read(data.repository, data.panel_source, artifact["feature_snapshot"],
                            columns=columns, **data.panel_request)


def test_native_streak_not_inferred_from_aggregate():
    history = {"ticker": "A", "history": [{"event_id": str(i)} for i in range(4)]}
    assert sources.streak(history, dict(zip(map(str, range(4)), [-2., 2., 3., 4.]))) == 3.
    assert sources.streak(history, dict(zip(map(str, range(4)), [4., 3., 2., -2.]))) == -1.
    assert sources.streak(history, dict(zip(map(str, range(4)), [1., 2., 3., 0.]))) == 0.
    assert sources.streak({"ticker": "A", "history": []}, {}) == 0.


def test_all_horizons_native_daily_prices_and_metadata_projection(panel_data, monkeypatch):
    data = panel_data
    observed = []
    original = data.repository.scan
    def scan(request, **kwargs):
        observed.append((kwargs["table_name"], request))
        return original(request, **kwargs)
    monkeypatch.setattr(data.repository, "scan", scan)
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    result = read(data, artifact)
    source_rows = [(name, q) for name, q in observed if q.snapshot_id == data.panel_source.snapshot_id]
    numerical_daily = [q for name, q in source_rows if name == "daily_market" and "implied_move" in q.columns]
    assert numerical_daily
    assert len(numerical_daily) == 3  # one selected ticker/year, never one per anchor
    assert {q.key_filter[1].values[0] for q in numerical_daily} == {2022, 2023, 2024}
    date_queries = [q for name, q in observed if name == "price_history" and "deleted" in q.columns
                    and "close_adj" not in q.columns]
    assert date_queries and all(set(q.columns) == {"date", "retrieved_at", "deleted", "source_hash", "capture_id"} for q in date_queries)
    days = list(calendar_sessions("2022-05-20", "2024-06-20").days)
    for row in result["rows"]:
        index = days.index(row["source_session_ceiling"])
        close = 100. + index / 10
        assert row["ret20"] == pytest.approx((close / (100. + (index - 20) / 10) - 1) * 100)
        assert row["ret5"] == pytest.approx((close / (100. + (index - 5) / 10) - 1) * 100)
        assert row["spy_ret21"] == pytest.approx(((200. + index / 10) /
            (200. + (index - 21) / 10) - 1) * 100)
        ema = pd.Series([100. + i / 10 for i in range(index + 1)]).ewm(span=252, adjust=False).mean().iloc[-1]
        assert row["dist_ema"] == pytest.approx((close / ema - 1) * 100)
        lineage = json.loads(row["lineage"])
        assert lineage["runup"]["sources"][0]["source_date"] == days[0]
        assert lineage["runup"]["sources"][-1]["source_date"] == row["source_session_ceiling"]
        assert all(a["source_date"] <= row["source_session_ceiling"]
                   for name in ("runup", "spy", "daily_market") for a in lineage[name]["sources"])


def test_bounds_fail_closed_without_relaxation(panel_data):
    data = panel_data
    for key, value in (("max_source_rows", 1), ("max_object_bytes", 1),
                       ("max_total_bytes", 1), ("max_output_rows", 2), ("max_fragments", 1)):
        request = {**data.panel_request, "limits": {**data.panel_request["limits"], key: value}}
        with pytest.raises(DataError) as refused:
            size_panel.materialize(data.repository, data.panel_source, **request)
        assert refused.value.code == "RESULT_LIMIT_EXCEEDED"
    request = {**data.panel_request,
        "metadata_limits": {**data.panel_request["metadata_limits"], "max_retained_rows": 1}}
    with pytest.raises(DataError) as refused:
        size_panel.materialize(data.repository, data.panel_source, **request)
    assert refused.value.code == "RESULT_LIMIT_EXCEEDED"


def test_frozen_policy_refused_before_source_access(monkeypatch):
    from engine.v2.research.eligibility_policy import eligibility_policy

    snapshot, request = _public_shape_request()
    request["policy"] = eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=31)
    monkeypatch.setattr(size_panel, "_fingerprint", lambda: pytest.fail("fingerprinted incompatible policy"))
    with pytest.raises(DataError, match="frozen 30-day") as refused:
        size_panel.materialize(None, snapshot, **request)
    assert refused.value.code == "HOLDOUT_ACCESS_DENIED"


def test_implied_prefix_last_null_is_not_nonmissing_fallback():
    sequence = {"events": [{"event_id": str(i), "ordinal": i, "event_date": f"2020-0{i+1}-15"}
                           for i in range(6)]}
    target = {"ticker": "A", "event_date": "2020-08-15"}
    ref = SimpleNamespace(object_id="object", content_hash="sha256:" + "a" * 64)
    records = [SimpleNamespace(partition_key="2020", time_min="2020-01-01", time_max="2020-12-31",
                              fragment_id="fragment", object_ref=ref, row_count=1,
                              primary_key_min=None, primary_key_max=None)]
    rows = [{"date": "2020-05-12", "implied_move": 999.},
            {"date": "2020-05-14", "implied_move": None},
            {"date": "2020-06-14", "implied_move": 0.}]
    value, lineage = sources.implied_history(sequence, target, "2020-08-10",
        {"4", "5"}, rows, sources.anchor_lookup(records, daily_contract=contract_for("daily_market")), "2020-05-01")
    assert value == 0 and lineage["contribution_count"] == 1
    assert lineage["prefix"][4]["source"]["source_date"] == "2020-05-14"
    assert [entry["ordinal"] for entry in lineage["prefix"]] == list(range(6))
    value, lineage = sources.implied_history(sequence, target, "2020-08-10",
        set(), rows, sources.anchor_lookup(records, daily_contract=contract_for("daily_market")), "2020-05-01")
    assert value is None and lineage["contribution_count"] == 0
    assert all(entry["source"] is None for entry in lineage["prefix"])


def test_stored_rows_detect_identity_or_feature_corruption(panel_data):
    from engine.v2.research import _magnitude_panel_store as store

    data = panel_data
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    rows = read(data, artifact)["rows"]
    for key, value in (("context_id", "sha256:" + "a" * 64), ("signed_streak", 100.),
                       ("current_quote_status", "present"), ("n_prior", float("inf"))):
        changed = [dict(r) for r in rows]
        changed[0][key] = value
        with pytest.raises(DataError) as caught:
            list(store._validate(changed, artifact["context"]))
        assert caught.value.code == "CONTRACT_MISMATCH"
    with pytest.raises(DataError) as caught:
        list(store._validate(rows[:-1], artifact["context"]))
    assert caught.value.code == "CONTRACT_MISMATCH"
    with pytest.raises(DataError) as caught:
        list(store._validate([*rows, rows[0]], artifact["context"]))
    assert caught.value.code == "CONTRACT_MISMATCH"
    with pytest.raises(DataError) as caught:
        list(store._validate([rows[1], rows[0], *rows[2:]], artifact["context"]))
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_size_panel_read_revalidates_history_coverage(panel_data, monkeypatch):
    data = panel_data
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    original_scan = data.repository.scan

    def shortened_history_scan(query, **kwargs):
        batches = original_scan(query, **kwargs)
        if kwargs.get("table_name") != "research_eligible_history":
            return batches
        batches = list(batches)
        assert batches
        batches[-1] = batches[-1].slice(0, batches[-1].num_rows - 1)
        return iter(batches)

    monkeypatch.setattr(data.repository, "scan", shortened_history_scan)
    with pytest.raises(DataError) as caught:
        read(data, artifact)
    assert caught.value.code == "CONTRACT_MISMATCH"
    with pytest.raises(DataError) as caught:
        size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_daily_nonfinite_and_negative_quote_refused():
    row = {name: 1. for name in sources.daily._DAILY}
    row.update(ticker="A", date="2020-01-01", mcap_asof="2020-01-01")
    for name, value in (("implied_move", -1.), ("rvol30", float("inf"))):
        with pytest.raises(DataError):
            sources.daily._source_row({**row, name: value}, "A", 2020, "2020-01-01", "2020-01-02", set())


@pytest.mark.parametrize("count", [250, 299, 300])
def test_runup_batch_membership_cannot_change_causal_warmup(count):
    dates = pd.date_range("2020-01-01", periods=350).strftime("%Y-%m-%d").tolist()
    prices = [SimpleNamespace(date=day, close_adj=100. + i) for i, day in enumerate(dates)]
    target = {"event_id": "early", "event_date": dates[count], "forecasts": [
        {"horizon_sessions": 1, "source_session_ceiling": dates[count - 1]}]}
    later = {"event_id": "late", "event_date": "2021-01-01", "forecasts": [
        {"horizon_sessions": 1, "source_session_ceiling": dates[-1]}]}
    alone = sources.runup([target], prices[:count])["early", 1]
    batched = sources.runup([target, later], prices)["early", 1]
    assert alone == batched
    assert (alone[0]["ret5"] is None) == (count < 300)


def test_exact_source_contract_refuses_same_name_changed_definition(panel_data, monkeypatch):
    from dataclasses import replace

    from engine.v2.data.manifests import table_contract_hash

    data = panel_data
    original = data.repository.table_contract
    def changed(snapshot, name):
        contract = original(snapshot, name)
        if name == "daily_market":
            columns = tuple(replace(c, unit="fraction") if c.name == "implied_move" else c
                            for c in contract.columns)
            contract = replace(contract, columns=columns)
            return replace(contract, definition_hash=table_contract_hash(contract))
        return contract
    monkeypatch.setattr(data.repository, "table_contract", changed)
    with pytest.raises(DataError) as refused:
        sources.inventory(data.repository, data.panel_source, "daily_market", data.panel_request["limits"])
    assert refused.value.code == "UNSUPPORTED_CONTRACT"


def test_frozen_source_ranges_match_native_normalizer_and_not_comparator_masks():
    from engine.v2.ops.providers.orats_daily_market import PLAUSIBLE_RANGES

    assert table._DAILY_RANGES == PLAUSIBLE_RANGES
    recipe = table.recipe("HMO")
    implied = next(f for f in recipe["dictionary"] if f["name"] == "implied_move")
    policy = implied["source_dependencies"][0]["policy"]
    assert policy["normalization_ranges"]["implied_move"] == [0., 100.]
    assert policy["normalization_ranges"]["rvol30"] == [0., 500.]
    assert policy["supported_contract_hash"] == table._DAILY_HASH


@pytest.mark.parametrize("name,value", [("implied_move", 101.), ("rvol30", 501.), ("rvol30", -1.)])
def test_observable_source_range_violation_refuses(name, value):
    row = {key: 1. for key in table._DAILY_RANGES}
    with pytest.raises(DataError, match="normalization range"):
        sources._daily_ranges([{**row, name: value}])


@pytest.mark.parametrize("columns", [[{}], [["n_prior"]], [True], [None]])
def test_malformed_projection_is_closed_refusal(columns):
    with pytest.raises(DataError):
        table.admit_columns(columns)


def test_excluded_decoded_quotes_cannot_change_aggregate(panel_data, monkeypatch):
    from engine.v2.research import magnitude_panel_qualified as builder

    data = panel_data
    prepared = builder._prepare(data.repository, data.panel_source,
        producer=size_panel._producer({}), history_admitted=size_panel._history(data.repository, data.panel_source, data.panel_request), **data.panel_request)
    baseline = list(builder._rows(data.repository, data.panel_source, data.prices, prepared))
    original = sources.daily._source_rows
    # Excluded protected, embargoed and warm-up quote anchors may pass through
    # raw decoding, but no contribution lookup or arithmetic may depend on them.
    poisoned = {"2022-06-01", "2023-01-02", "2023-03-17", "2023-03-31"}
    def changed(*args, **kwargs):
        rows = original(*args, **kwargs)
        return [{**r, "implied_move": 99.} if r["date"] in poisoned else r for r in rows]
    monkeypatch.setattr(sources.daily, "_source_rows", changed)
    assert list(builder._rows(data.repository, data.panel_source, data.prices, prepared)) == baseline
    def admitted_changed(*args, **kwargs):
        rows = original(*args, **kwargs)
        return [{**r, "implied_move": 10.} if r["date"] == "2023-05-12" else r for r in rows]
    monkeypatch.setattr(sources.daily, "_source_rows", admitted_changed)
    changed_rows = list(builder._rows(data.repository, data.panel_source, data.prices, prepared))
    assert all(r["mean_prior_or_implied"] == 5. for r in changed_rows)
    assert all(r["mean_prior_or_implied"] == 4. for r in baseline)


def test_daily_scan_count_does_not_scale_with_targets(panel_data, monkeypatch):
    data = panel_data
    request = dict(data.panel_request)
    history_request = {k: request[k] for k in data.request}
    history_request["event_ids"] = [data.events[5]["event_id"], data.events[-1]["event_id"]]
    artifact = eligible_history.materialize(data.repository, data.panel_source, **history_request)
    request.update(event_ids=history_request["event_ids"], history_snapshot=artifact["feature_snapshot"])
    queries = []
    original = data.repository.scan
    def scan(query, **kwargs):
        if kwargs["table_name"] == "daily_market" and "implied_move" in query.columns:
            queries.append(query)
        return original(query, **kwargs)
    monkeypatch.setattr(data.repository, "scan", scan)
    result = size_panel.materialize(data.repository, data.panel_source, **request)
    assert len(result["context"]["target_events"]) == 2
    assert len(queries) == 3


def test_price_warmup_start_cannot_truncate_historical_quote_prefix(panel_data):
    data = panel_data
    request = {**data.panel_request, "source_start": "2024-01-01"}
    artifact = size_panel.materialize(data.repository, data.panel_source, **request)
    result = read(data, artifact, source_start="2024-01-01")
    assert artifact["context"]["daily_source_start"] == "2022-05-20"
    for row in result["rows"]:
        assert row["mean_prior_or_implied"] == 4.
        assert row["ret5"] is None  # Explicit price warmup is independent of quote history.
        lineage = json.loads(row["lineage"])["implied_history"]
        assert lineage["prefix"][4]["source"]["source_date"] == "2023-05-12"
        assert lineage["contribution_count"] == 2


def test_history_start_bounds_events_not_prior_quote_dates():
    sequence = {"events": [{"event_id": str(i), "ordinal": i, "event_date": f"2020-0{i+1}-15"}
                           for i in range(5)]}
    ref = SimpleNamespace(object_id="object", content_hash="sha256:" + "a" * 64)
    records = [SimpleNamespace(partition_key="2020", time_min="2020-01-01", time_max="2020-12-31",
                              fragment_id="fragment", object_ref=ref, row_count=1,
                              primary_key_min=None, primary_key_max=None)]
    value, lineage = sources.implied_history(sequence, {"ticker": "A", "event_date": "2020-07-15"},
        "2020-07-10", {"4"}, [{"date": "2020-04-30", "implied_move": 8.}], sources.anchor_lookup(records, daily_contract=contract_for("daily_market")), "2020-05-01")
    assert value == 8. and lineage["contribution_count"] == 1
    assert lineage["prefix"][4]["source"]["source_date"] == "2020-04-30"


@pytest.mark.parametrize("panel_data", [{
    "days": ["2022-06-02", "2022-07-01", "2022-09-01", "2023-01-03", "2023-03-20",
             "2023-04-03", "2023-05-15", "2023-06-02", "2024-06-17"],
    "values": [-1., -1., -2., -4., 9999., 9999., 9999., 6., 8.],
}], indirect=True)
def test_grouped_quote_exclusion_after_warmup(panel_data, monkeypatch):
    data = panel_data
    poison_outcomes(monkeypatch, data.repository, data.panel_source,
                    ["2023-03-20", "2023-04-03", "2023-05-15", "2024-06-17"])
    original = sources.daily._source_rows
    def poisoned(*args, **kwargs):
        return [{**r, "implied_move": 99.} if r["date"] in ("2023-03-17", "2023-03-31") else r
                for r in original(*args, **kwargs)]
    monkeypatch.setattr(sources.daily, "_source_rows", poisoned)
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    for row in read(data, artifact)["rows"]:
        lineage = json.loads(row["lineage"])["implied_history"]
        assert lineage["prefix"][4]["ordinal"] == 4 and lineage["prefix"][4]["eligible"] is False
        assert lineage["prefix"][5]["ordinal"] == 5 and lineage["prefix"][5]["eligible"] is False
        assert all(lineage["prefix"][i]["source"] is None for i in (4, 5))
        assert [lineage["prefix"][i]["eligible"] for i in (6, 7)] == [True, True]
        assert row["mean_prior_or_implied"] == 4. and lineage["contribution_count"] == 2


@pytest.mark.parametrize("route", ["materialize", "read"])
def test_imported_panel_identity_drift_refuses_before_source_reads(panel_data, monkeypatch, route):
    monkeypatch.setattr(size_panel, "_fingerprint", lambda: {"source_closure": {"changed": "sha256:new"}})
    with pytest.raises(DataError) as refused:
        if route == "read":
            size_panel.read(None, panel_data.panel_source, panel_data.panel_source,
                            columns=["n_prior"], **panel_data.panel_request)
        else:
            size_panel.materialize(None, panel_data.panel_source, **panel_data.panel_request)
    assert refused.value.code == "INPUT_CHANGED"


def test_panel_uses_public_history_admission_before_backend(panel_data, monkeypatch):
    data = panel_data
    def refused(*args, **kwargs):
        from engine.v2.data.errors import fail
        raise fail("INPUT_CHANGED", "synthetic imported history drift")
    monkeypatch.setattr(eligible_history, "iter_read", refused)
    with pytest.raises(DataError, match="synthetic imported history drift"):
        size_panel.materialize(data.repository, data.panel_source, **data.panel_request)


@pytest.mark.parametrize("columns", [["realized_move_pct"], ["or_implied"], [None]])
def test_public_projection_refuses_before_history_access(monkeypatch, columns):
    monkeypatch.setattr(eligible_history, "iter_read", lambda *a, **k: pytest.fail("history was read"))
    with pytest.raises(DataError) as refused:
        size_panel.read(None, None, None, columns=columns)
    assert refused.value.code == "HOLDOUT_ACCESS_DENIED"


@pytest.mark.parametrize("budget_delta", [-1, 0])
def test_publication_enforces_native_expanded_parquet_budget(panel_data, budget_delta):
    """Compressible lineage must not publish an artifact its bounded reader refuses."""
    from copy import deepcopy

    from engine.v2.research import _magnitude_panel_store as transport

    data = panel_data
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    rows = read(data, artifact)["rows"]
    context = deepcopy(artifact["context"])
    context["context_id"] = content_hash({k: v for k, v in context.items() if k != "context_id"})
    for row in rows:
        row["context_id"] = context["context_id"]
        row["lineage"] = json.dumps({"synthetic_compressible_lineage": "x" * 200_000})
        row["row_id"] = content_hash({k: v for k, v in row.items() if k != "row_id"})
    schema = transport.pa.schema([transport.pa.field(c.name,
        transport.query.arrow_type_for(c.physical_type), nullable=c.nullable) for c in table.CONTRACT.columns])
    sink = transport.pa.BufferOutputStream()
    transport.pq.write_table(transport.pa.Table.from_pylist(rows, schema=schema), sink)
    payload = sink.getvalue()
    with transport.pq.ParquetFile(transport.pa.BufferReader(payload)) as parquet:
        expanded = sum(parquet.metadata.row_group(i).total_byte_size
                       for i in range(parquet.metadata.num_row_groups))
    assert len(payload) < expanded - 1
    context["request"]["limits"]["max_object_bytes"] = expanded + budget_delta
    context["context_id"] = content_hash({k: v for k, v in context.items() if k != "context_id"})
    for row in rows:
        row["context_id"] = context["context_id"]
        row["row_id"] = content_hash({k: v for k, v in row.items() if k != "row_id"})
    if budget_delta < 0:
        with pytest.raises(DataError, match="uncompressed"):
            transport.publish(data.repository, data.panel_source, context, rows)
        assert transport.existing(data.repository, context) is None
    else:
        snapshot = transport.publish(data.repository, data.panel_source, context, rows)
        assert transport.read(data.repository, snapshot, context, [*table.FEATURES, "has_implied_quote"]) == rows


@pytest.mark.parametrize("damage", [None, "scope", "snapshot"])
def test_panel_receipt_projection_excludes_diagnostics(panel_data, damage):
    import sqlite3

    from engine.v2.research import _magnitude_panel_store as transport

    data = panel_data
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    slot = transport._slot(artifact["context"])
    data.conn.execute("DROP TRIGGER data_import_receipts_no_update")
    data.conn.commit()
    data.conn.execute("PRAGMA foreign_keys=OFF")
    data.conn.execute("UPDATE data_import_receipts SET problem_json=? WHERE receipt_id=?",
                      ('{"message":"' + 'x' * (1 << 20) + '"}', slot))
    if damage:
        column = "scope" if damage == "scope" else "result_snapshot_id"
        value = "x" * (1 << 20) if damage == "scope" else "snap_" + "g" * 32
        data.conn.execute(f"UPDATE data_import_receipts SET {column}=? WHERE receipt_id=?", (value, slot))
    def authorizer(action, name, column, *rest):
        if action == sqlite3.SQLITE_READ and name == "data_import_receipts" and column in {"problem_json", "registered_at"}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    data.conn.set_authorizer(authorizer)
    try:
        if damage:
            with pytest.raises(DataError) as caught:
                transport.existing(data.repository, artifact["context"])
            assert caught.value.code == "IDENTITY_CONFLICT"
        else:
            assert transport.existing(data.repository, artifact["context"]) == artifact["feature_snapshot"]
    finally:
        data.conn.set_authorizer(None)


def test_anchor_lookup_indexes_once_and_caches_only_exact_invocation_keys(monkeypatch):
    class Records(list):
        visits = 0
        def __iter__(self):
            for record in super().__iter__():
                self.visits += 1
                yield record
    ref = SimpleNamespace(object_id="object", content_hash="sha256:" + "a" * 64)
    records = Records(SimpleNamespace(partition_key=f"T{i}", primary_key_min=(f"T{i}", "2020-01-01"),
        primary_key_max=(f"T{i}", "2020-12-31"), fragment_id=f"f{i}", object_ref=ref) for i in range(100))
    real, calls = sources.anchor, []
    def counted(selected, ticker, day, version=None, **kwargs):
        calls.append((len(selected), ticker, day, version))
        return real(selected, ticker, day, version, **kwargs)
    monkeypatch.setattr(sources, "anchor", counted)
    lookup = sources.anchor_lookup(records)
    version = {"capture_id": "v1"}
    first = lookup("T0", "2020-06-01", version)
    for _ in range(20):
        assert lookup("T0", "2020-06-01", {"capture_id": "v1"}) == first
    assert records.visits == 100 and len(calls) == 1 and calls[0][0] == 1
    first["fragments"].clear()
    version["capture_id"] = "v2"
    assert lookup("T0", "2020-06-01", {"capture_id": "v1"})["fragments"]
    assert lookup("T0", "2020-06-01", version)["version"] == version
    lookup("T1", "2020-06-01", version)
    lookup("T0", "2020-06-02", version)
    assert len(calls) == 4 and all(call[0] == 1 for call in calls)
    fresh = sources.anchor_lookup(records)
    fresh("T0", "2020-06-01", {"capture_id": "v1"})
    assert records.visits == 200 and len(calls) == 5
    for _ in range(2):
        with pytest.raises(DataError) as caught:
            lookup("T0", "2019-01-01")
        assert caught.value.code == "CONTRACT_MISMATCH"
    assert len(calls) == 7  # refusals are not cached


@pytest.mark.parametrize("name", ["daily_market", "price_history"])
@pytest.mark.parametrize("empty", [[], [SimpleNamespace(row_count=0)]])
def test_empty_source_inventory_refuses_before_values(panel_data, monkeypatch, name, empty):
    data = panel_data
    original = data.repository.fragment_records
    monkeypatch.setattr(data.repository, "fragment_records",
        lambda snapshot, table: empty if table == name else original(snapshot, table))
    snapshot = data.panel_source if name == "daily_market" else data.prices
    with pytest.raises(DataError) as caught:
        sources.inventory(data.repository, snapshot, name, data.panel_request["limits"])
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_missing_ticker_price_dates_refuses_before_scan(panel_data, monkeypatch):
    data = panel_data
    monkeypatch.setattr(data.repository, "scan", lambda *a, **k: pytest.fail("empty source scan"))
    with pytest.raises(DataError) as caught:
        sources.price_dates(data.repository, data.prices, "ABSENT", data.panel_request["limits"])
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_no_dated_daily_fragment_refuses_before_metadata_index(panel_data, monkeypatch):
    from engine.v2.research import magnitude_panel_qualified as panel
    data = panel_data
    original = sources.inventory
    monkeypatch.setattr(sources, "inventory", lambda repository, snapshot, name, limits:
        [SimpleNamespace(row_count=1, time_min=None)] if name == "daily_market"
        else original(repository, snapshot, name, limits))
    monkeypatch.setattr(panel.magnitude_confirmed_metadata, "build_index",
                        lambda *a, **k: pytest.fail("metadata read before daily coverage"))
    with pytest.raises(DataError) as caught:
        size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    assert caught.value.code == "CONTRACT_MISMATCH"


def _public_shape_request():
    from engine.v2.contracts.data import SnapshotRef
    from engine.v2.research.eligibility_policy import eligibility_policy
    snapshot = SnapshotRef(snapshot_id="snap_" + "a" * 32, manifest_hash="sha256:" + "a" * 64,
        table_versions={}, calendar_version="synthetic", source_priority_version="synthetic",
        finality_receipt_refs=(), knowledge_mode_by_table={})
    return snapshot, dict(history_snapshot=snapshot, price_history_snapshot=snapshot,
        as_of_month="2024-07", policy=eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30),
        event_ids=["synthetic"], horizons=[1, 3, 5], history_start="2020-01-01", source_start="2019-01-01",
        limits={name: 100 for name in ("max_source_rows", "max_fragments", "max_object_bytes",
                                     "max_total_bytes", "max_output_rows")}, metadata_limits=dict(METADATA_LIMITS))


@pytest.mark.parametrize("route", ["materialize", "read"])
@pytest.mark.parametrize("key", ["history_snapshot", "price_history_snapshot", "as_of_month", "policy",
    "event_ids", "horizons", "history_start", "source_start", "limits", "metadata_limits"])
@pytest.mark.parametrize("damage", ["missing", "null", "wrong_type"])
def test_public_request_shape_refuses_before_fingerprinting(monkeypatch, route, key, damage):
    snapshot, request = _public_shape_request()
    if damage == "missing":
        del request[key]
    else:
        request[key] = None if damage == "null" else 0
    monkeypatch.setattr(size_panel, "_fingerprint", lambda: pytest.fail("fingerprinted malformed request"))
    with pytest.raises(DataError) as caught:
        if route == "read":
            size_panel.read(None, snapshot, snapshot, columns=["n_prior"], **request)
        else:
            size_panel.materialize(None, snapshot, **request)
    expected = "SNAPSHOT_UNRESOLVED" if key.endswith("snapshot") and damage != "missing" else "CONTRACT_MISMATCH"
    assert caught.value.code == expected


@pytest.mark.parametrize("route", ["materialize", "read"])
@pytest.mark.parametrize("key,value", [
    ("unknown", 1), ("event_ids", [None]), ("event_ids", [[]]), ("event_ids", ["x", "x"]),
    ("event_ids", [""]), ("event_ids", "x"), ("horizons", [True, 3, 5]), ("horizons", [1, 3]),
    ("as_of_month", "2024-13"), ("history_start", "2024-02-30"), ("source_start", "2024-1-01"),
    ("limits", {"max_object_bytes": 1}), ("metadata_limits", {"max_object_bytes": 1}),
    ("policy", {"embargo_days": True})])
def test_public_request_semantics_refuse_before_fingerprinting(monkeypatch, route, key, value):
    snapshot, request = _public_shape_request()
    request[key] = value
    monkeypatch.setattr(size_panel, "_fingerprint", lambda: pytest.fail("fingerprinted malformed request"))
    with pytest.raises(DataError) as caught:
        if route == "read":
            size_panel.read(None, snapshot, snapshot, columns=["n_prior"], **request)
        else:
            size_panel.materialize(None, snapshot, **request)
    assert caught.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("route", ["materialize", "read"])
@pytest.mark.parametrize("key", ["limits", "metadata_limits"])
@pytest.mark.parametrize("value", [True, 0, -1, None, "1"])
def test_public_limit_values_refuse_before_fingerprinting(monkeypatch, route, key, value):
    snapshot, request = _public_shape_request()
    request[key]["max_object_bytes"] = value
    monkeypatch.setattr(size_panel, "_fingerprint", lambda: pytest.fail("fingerprinted malformed limits"))
    with pytest.raises(DataError) as caught:
        if route == "read":
            size_panel.read(None, snapshot, snapshot, columns=["n_prior"], **request)
        else:
            size_panel.materialize(None, snapshot, **request)
    assert caught.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("route", ["materialize", "read"])
@pytest.mark.parametrize("value", [None, {}, "snap_fake"])
def test_public_source_shape_refuses_before_fingerprinting(monkeypatch, route, value):
    snapshot, request = _public_shape_request()
    monkeypatch.setattr(size_panel, "_fingerprint", lambda: pytest.fail("fingerprinted malformed snapshot"))
    with pytest.raises(DataError) as caught:
        if route == "read":
            size_panel.read(None, value, snapshot, columns=["n_prior"], **request)
        else:
            size_panel.materialize(None, value, **request)
    assert caught.value.code == "SNAPSHOT_UNRESOLVED"


@pytest.mark.parametrize("value", [None, {}, "snap_fake"])
def test_public_feature_shape_refuses_before_fingerprinting(monkeypatch, value):
    snapshot, request = _public_shape_request()
    monkeypatch.setattr(size_panel, "_fingerprint", lambda: pytest.fail("fingerprinted malformed feature snapshot"))
    with pytest.raises(DataError) as caught:
        size_panel.read(None, snapshot, value, columns=["n_prior"], **request)
    assert caught.value.code == "SNAPSHOT_UNRESOLVED"


def test_missing_public_projection_is_typed_before_fingerprinting(monkeypatch):
    snapshot, request = _public_shape_request()
    monkeypatch.setattr(size_panel, "_fingerprint", lambda: pytest.fail("fingerprinted absent projection"))
    with pytest.raises(DataError) as caught:
        size_panel.read(None, snapshot, snapshot, **request)
    assert caught.value.code == "HOLDOUT_ACCESS_DENIED"


def test_daily_lineage_matches_native_mixed_ticker_membership(panel_data, monkeypatch):
    from datetime import date, timedelta

    from engine.v2.contracts.data import KeyPredicate, TimeInterval
    from engine.v2.data import query

    data = panel_data
    contract = data.repository.table_contract(data.panel_source, "daily_market")
    records = data.repository.fragment_records(data.panel_source, "daily_market")
    seen, original = set(), data.repository.scan
    def observed(request, *, table_name, **kwargs):
        if table_name == "daily_market" and any(p.column == "ticker" and p.values == ("SYNTH",)
                                               for p in request.key_filter):
            seen.update(r.object_ref.object_id for r in query.plan_scan_population(
                contract, records, key_filter=request.key_filter, time_interval=request.time_interval).records)
        yield from original(request, table_name=table_name, **kwargs)
    monkeypatch.setattr(data.repository, "scan", observed)
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    for row in read(data, artifact)["rows"]:
        lineage = json.loads(row["lineage"])
        for anchor in lineage["daily_market"]["sources"] + lineage["implied_history"]["sources"]:
            day = anchor["source_date"]
            expected = query.plan_scan_population(contract, records, key_filter=(
                KeyPredicate(column="ticker", operator="eq", values=("SYNTH",)),
                KeyPredicate(column="year", operator="eq", values=(int(day[:4]),))),
                time_interval=TimeInterval(column="date", start_inclusive=day,
                    end_exclusive=(date.fromisoformat(day) + timedelta(days=1)).isoformat())).records
            assert len(expected) == 1
            assert any(r.partition_key == day[:4] and r not in expected for r in records)
            identities = [{"fragment_id": r.fragment_id, "object_id": r.object_ref.object_id,
                           "content_hash": r.object_ref.content_hash} for r in expected]
            assert anchor["fragments"] == identities
            assert {item["object_id"] for item in identities} <= seen
        assert row["n_prior"] == 3 and row["mean_prior_or_implied"] == 4.


def test_streamed_history_panel_rows_match_materialized_history(panel_data):
    from engine.v2.research import magnitude_panel_qualified as builder

    data = panel_data
    history_request = {key: data.panel_request[key] for key in data.request}
    from engine.v2.research.eligible_history_table import FEATURES as HISTORY_FEATURES
    columns = list(HISTORY_FEATURES)
    materialized = eligible_history.read(data.repository, data.panel_source, data.panel_request["history_snapshot"],
        columns=columns, **history_request)
    streamed = eligible_history.iter_read(data.repository, data.panel_source, data.panel_request["history_snapshot"],
        columns=columns, **history_request)
    producer = size_panel._producer({})
    materialized_prepared = builder._prepare(data.repository, data.panel_source,
        producer=producer, history_admitted=materialized, **data.panel_request)
    streamed_prepared = builder._prepare(data.repository, data.panel_source,
        producer=producer, history_admitted=streamed, **data.panel_request)

    streamed_rows = list(builder._rows(data.repository, data.panel_source, data.prices, streamed_prepared))
    materialized_rows = list(builder._rows(data.repository, data.panel_source, data.prices, materialized_prepared))
    assert len(streamed_rows) == 3
    assert streamed_rows == materialized_rows


def test_panel_rows_consume_canonical_history_order_across_prefix_tickers(monkeypatch):
    from engine.v2.research import magnitude_panel_qualified as builder
    from engine.v2.research.eligible_history_table import FEATURES as HISTORY_FEATURES

    targets = []
    rows = []
    for event_id, ticker in (("M_2024-01-02", "M"), ("MA_2024-01-02", "MA")):
        forecasts = [{"horizon_sessions": horizon, "cutoff_id": f"{event_id}:{horizon}",
            "source_session_ceiling": "2024-01-01"} for horizon in (1, 3, 5)]
        targets.append({"event_id": event_id, "ticker": ticker, "event_date": "2024-01-02",
            "forecasts": forecasts})
        for cutoff in forecasts:
            rows.append({**{name: None for name in HISTORY_FEATURES},
                "event_id": event_id, "horizon_sessions": cutoff["horizon_sessions"],
                "history_json": json.dumps({"history": []}), "n_prior": 0,
                "ema12_prior_abs_move": None, "mean_prior_abs_move": None})
    rows.sort(key=lambda row: (row["event_id"], row["horizon_sessions"]))
    context = {"request": {"as_of_month": "2024-01", "source_start": "2023-01-01",
            "history_start": "2023-01-01", "limits": {"max_source_rows": 20, "max_object_bytes": 100}},
        "target_events": targets,
        "confirmed_metadata": {"sequences": [{"ticker": "M"}, {"ticker": "MA"}]},
        "price_row_metadata": {"M": [{"date": "2024-01-01"}],
                                "MA": [{"date": "2024-01-01"}],
                                "^GSPC": [{"date": "2024-01-01"}]},
        "quote_eligibility_plan": {"retained_event_ids": []}, "daily_source_start": "2023-01-01",
        "context_id": "panel-context"}
    admitted = {"context": {"context_id": "history-context"}, "rows": rows}
    prepared = (context, admitted, [], [])
    monkeypatch.setattr(builder, "load_population", lambda *args, **kwargs: object())
    monkeypatch.setattr(sources, "signed_moves", lambda *args: {})
    monkeypatch.setattr(sources, "anchor_lookup", lambda *args, **kwargs: lambda *args: {})
    price_tickers = []

    def prices_for(repository, price_snapshot, ticker, *args):
        price_tickers.append(ticker)
        return []

    monkeypatch.setattr(sources, "prices_for", prices_for)
    monkeypatch.setattr(sources, "runup", lambda selected, prices: {
        (target["event_id"], cutoff["horizon_sessions"]): ({}, [])
        for target in selected for cutoff in target["forecasts"]})
    monkeypatch.setattr(sources, "daily_surface", lambda *args: [])
    monkeypatch.setattr(sources, "streak", lambda *args: 0.)
    implied_tickers = []

    def implied_history(sequence, target, *args):
        implied_tickers.append((sequence["ticker"], target["ticker"]))
        return None, {}

    monkeypatch.setattr(sources, "implied_history", implied_history)
    monkeypatch.setattr(builder.daily, "_read_regime_series", lambda *args: [])
    monkeypatch.setattr(builder.daily, "_spy_blocks", lambda selected, regime: {
        (target["event_id"], cutoff["horizon_sessions"]): {"values": {}, "source_dates": []}
        for target in selected for cutoff in target["forecasts"]})
    monkeypatch.setattr(builder.daily, "_daily", lambda *args: {
        "quote_status": "missing", "anchor": None,
        "values": {name: None for name in builder.daily._DAILY}})
    monkeypatch.setattr(builder, "_block", lambda values, anchors: {})

    repository = SimpleNamespace(table_contract=lambda *args: object())
    emitted = list(builder._rows(repository, object(), object(), prepared))
    assert [row["event_id"] for row in emitted] == [row["event_id"] for row in rows]
    assert [row["horizon_sessions"] for row in emitted] == [row["horizon_sessions"] for row in rows]
    assert price_tickers == ["MA", "M"]
    assert implied_tickers == [(row["ticker"], row["ticker"]) for row in
        [target for target in targets for _ in target["forecasts"]][::-1]]
