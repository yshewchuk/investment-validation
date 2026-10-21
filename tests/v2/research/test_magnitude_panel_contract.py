"""Synthetic stage-2 schema and metadata-only admission refusal oracles."""
import copy
import json
from dataclasses import replace
from datetime import timedelta

import pytest

from engine.v2.contracts.data import (
    ColumnContract,
    DataQuery,
    KeyPredicate,
    TableContract,
    TimeInterval,
)
from engine.v2.data.errors import DataError
from engine.v2.data.manifests import table_contract_hash
from engine.v2.foundation import canonical_json, content_hash, to_document
from engine.v2.research import magnitude_features
from engine.v2.research import magnitude_history_store as history
from engine.v2.research import magnitude_panel_plan as plan
from engine.v2.research import magnitude_panel_table as table
from tests.v2.research.magnitude_trial_support import build_history, calendar_sessions
from tests.v2.research.test_magnitude_history_store import history_source

# Literal scientific dictionary oracle: do not derive this from implementation constants.
_EXPECTED_FIELDS = [
    ("n_prior", "history", "count"),
    ("mean_prior_move", "history", "percent"), ("mean_prior_abs_move", "history", "percent"),
    ("ema2_prior_move", "history", "percent"), ("ema2_prior_abs_move", "history", "percent"),
    ("ema4_prior_move", "history", "percent"), ("ema4_prior_abs_move", "history", "percent"),
    ("ema8_prior_move", "history", "percent"), ("ema8_prior_abs_move", "history", "percent"),
    ("ema12_prior_move", "history", "percent"), ("ema12_prior_abs_move", "history", "percent"),
    ("signed_streak", "prior_events", "count"), ("ema12r_abs", "prior_events", "percent"),
    ("dist_high", "runup", "percent"), ("dist_ema", "runup", "percent"),
    ("ret5", "runup", "percent"), ("ret10", "runup", "percent"), ("ret20", "runup", "percent"),
    ("spy_ret21", "spy", "percent"), ("spy_ret63", "spy", "percent"),
    ("spy_ret252", "spy", "percent"), ("spy_dd252", "spy", "percent"),
    ("spy_vol5", "spy", "percent"), ("spy_vol20", "spy", "percent"),
    ("spy_vol60", "spy", "percent"), ("spy_vol252", "spy", "percent"),
    ("spy_vol20_rel252", "spy", "ratio"),
    ("implied_move", "daily_market", "percent"), ("iv10", "daily_market", "percent"),
    ("iv30", "daily_market", "percent"), ("exern_iv10", "daily_market", "percent"),
    ("exern_iv30", "daily_market", "percent"), ("iee", "daily_market", "unknown"),
    ("skew", "daily_market", "unknown"), ("contango", "daily_market", "unknown"),
    ("fwd90_30", "daily_market", "percent"), ("fexern90_30", "daily_market", "percent"),
    ("rvol30", "daily_market", "percent"), ("spot", "daily_market", "usd"),
    ("mcap_log", "daily_market", "log_usd"),
]


def _expected_recipe(subset):
    fields = _EXPECTED_FIELDS[:{"H": 11, "HM": 27, "HMO": 40}[subset]]
    dictionary = [{"name": name, "block": block, "source_alias": name,
        "native_output_alias": "im" if name == "implied_move" else name,
        "physical_type": "int64" if name == "n_prior" else "float64", "unit": unit,
        "nullable": name != "n_prior", "on_missing": "zero_count" if name == "n_prior" else "null"}
        for name, block, unit in fields]
    return {"schema_version": "research_magnitude_panel.v1", "subset": subset,
        "dictionary": dictionary, "dictionary_hash": content_hash(dictionary), "imputation": "none",
        "missing_indicators": [name for name, _, _ in fields if name != "n_prior"],
        "quote_indicator": subset == "HMO"}


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    source = history_source(tmp_path_factory.mktemp("panel-contract"))
    build, output = build_history(source)
    producer = copy.deepcopy(build["context"]["producer"])
    context = plan.build_context(source.repository, source.snapshot, output,
                                 build["context"], producer=producer)
    yield source, output, build["context"], producer, context
    source.conn.close()


def _hash(document, key):
    document[key] = content_hash({name: value for name, value in document.items() if name != key})
    return document


def _context_hash(context):
    history = context["history_context"]
    temporal = history["temporal_plan"]
    for event in temporal["events"]:
        for cutoff in event["forecasts"]:
            _hash(cutoff, "cutoff_id")
    _hash(temporal, "plan_id")
    _hash(history, "context_id")
    return _hash(context, "context_id")


def _rows(context):
    result = []
    for event in context["history_context"]["temporal_plan"]["events"]:
        for cutoff in event["forecasts"]:
            row = {name: None for name in table.FEATURES}
            row.update(schema_version=table.CONTRACT.contract_id, event_id=event["event_id"],
                context_id=context["context_id"], history_context_id=context["history_context"]["context_id"],
                horizon_sessions=cutoff["horizon_sessions"], event_date=event["event_date"],
                cutoff_id=cutoff["cutoff_id"], source_session_ceiling=cutoff["source_session_ceiling"],
                n_prior=0)
            blocks = {}
            for block, names in table.BLOCKS.items():
                blocks[block] = {"sources": [{"source_date": cutoff["source_session_ceiling"],
                    "available_as_of": None, "knowledge_mode": "reconstructed",
                    "row_id": "synthetic-" + block, "fragment_id": "synthetic-" + block,
                    "content_hash": "sha256:" + "a" * 64}],
                    "missing": {name: "unavailable" for name in names if row[name] is None},
                    "quote_status": "absent" if block == "daily_market" else "not_applicable"}
            row["lineage"] = canonical_json(blocks)
            result.append(_hash(row, "row_id"))
    return result


def _lineage(row, block, edit):
    blocks = json.loads(row["lineage"])
    edit(blocks[block])
    row["lineage"] = canonical_json(blocks)
    _hash(row, "row_id")


def _metadata_only(monkeypatch, repository):
    original = repository.scan
    scans = []

    def scan(request, *, table_name):
        scans.append((table_name, request.columns))
        assert table_name not in ("computed_moves", "research_magnitude_history", table.TABLE)
        assert table_name != "daily_market" or request.columns == ("date",)
        yield from original(request, table_name=table_name)

    monkeypatch.setattr(repository, "scan", scan)
    return scans


def test_dictionary_schema_order_hash_and_subset_isolation():
    assert table.CONTRACT.contract_id == "research_magnitude_panel.v1"
    assert table.CONTRACT.definition_hash == table_contract_hash(table.CONTRACT)
    assert table.CONTRACT.primary_key == ("event_id", "context_id", "horizon_sessions")
    assert tuple(c.name for c in table.CONTRACT.columns) == (*table.METADATA, *table.FEATURES)
    assert tuple(table.FEATURES) == tuple(name for name, _, _ in _EXPECTED_FIELDS)
    hashes = {
        "H": "sha256:5feb7179a6b0082d30ab3966ef478d5ca9a4b03a814afaa9ba3a2284211a6340",
        "HM": "sha256:8af610ba8d07c3e5642bf134fd8b247bd136b108b2b673453df9ef166242d652",
        "HMO": "sha256:37a166f072a30f4ac9ee0986fd6db8b11e4099fb2503262e337af14bb6de58a3",
    }
    for subset in ("H", "HM", "HMO"):
        expected = _expected_recipe(subset)
        assert expected["dictionary_hash"] == hashes[subset]
        assert table.feature_dictionary(subset) == expected["dictionary"]
        assert table.recipe(subset) == expected
    forbidden = {"or_exern_z252", "move", "abs_move", "size_v1_4", "sector", "has_target"}
    assert not forbidden.intersection(table.FEATURES)
    first = table.recipe("H")
    first["dictionary"][0]["name"] = "poison"
    assert table.recipe("H")["dictionary"][0]["name"] == "n_prior"
    assert table.recipe("H")["dictionary_hash"] != table.recipe("HM")["dictionary_hash"]


@pytest.mark.parametrize("subset", ["unknown", "", None, [], {}, 1, True])
def test_unknown_or_untyped_subset_refuses(subset):
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.feature_dictionary(subset)


def test_metadata_plan_is_deterministic_detached_and_grants_no_raw_access(native, monkeypatch):
    source, output, binding, producer, context = native
    scans = _metadata_only(monkeypatch, source.repository)
    actual = plan.build_context(source.repository, source.snapshot, output, binding, producer=producer)
    assert actual == _expected_context(native)
    assert actual == context
    assert actual["context_id"] != binding["context_id"]
    assert actual["history_snapshot"] == to_document(output)
    assert actual["source_snapshot"] == to_document(source.snapshot)
    assert actual["admission"]["raw_payload_access"] == "not_authorized"
    assert plan.preflight_context(source.repository, source.snapshot, output, actual) == actual["admission"]
    assert scans and all(name != "computed_moves" for name, _ in scans)
    assert actual["history_context"] is not binding
    assert actual["producer"] is not producer
    assert len(actual["admission"]["history_fragments"]) == len(binding["temporal_plan"]["events"])
    forbidden = {event["event_id"] for event in source.forbidden}
    assert not forbidden.intersection(item["event_id"] for item in actual["admission"]["prior_outcome_fragments"])


def test_complete_nullable_rows_and_all_horizons_are_valid(native):
    context = native[-1]
    rows = _rows(context)
    assert table.validate_rows(rows, context) is None
    assert table.validate_rows(list(reversed(rows)), context) is None
    assert len(rows) == 3 * len(context["history_context"]["temporal_plan"]["events"])


@pytest.mark.parametrize("change", [
    {"schema_version": "research_magnitude_panel.v2"}, {"extra": 1}, {"context_id": "wrong"},
    {"history_context_id": "wrong"}, {"event_id": "unknown"}, {"cutoff_id": "wrong"},
    {"n_prior": None}, {"n_prior": -1},
    {"n_prior": 1.5}, {"n_prior": True}, {"ret5": "1"}, {"implied_move": 0.0}, {"implied_move": -1.0},
    {"source_session_ceiling": "2099-01-01"}, {"row_id": "wrong"},
])
def test_rows_reject_unknown_identity_version_types_and_sentinels(native, change):
    rows = _rows(native[-1])
    rows[0].update(change)
    # Rehash semantic corruptions so these guards do not pass only on digest mismatch.
    if "row_id" not in change:
        _hash(rows[0], "row_id")
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(rows, native[-1])


@pytest.mark.parametrize("kind", ["duplicate", "missing", "missing_column"])
def test_exact_cohort_and_closed_columns(native, kind):
    rows = _rows(native[-1])
    if kind == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif kind == "missing":
        rows.pop()
    else:
        del rows[0]["iv30"]
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(rows, native[-1])


@pytest.mark.parametrize("edit", [
    lambda b: b.update(extra=True),
    lambda b: b["sources"][0].update(source_date="2099-01-01"),
    lambda b: b["sources"][0].update(source_date="2024-02-30"),
    lambda b: b["sources"][0].update(available_as_of="2099-01-01"),
    lambda b: b["sources"][0].update(knowledge_mode="guessed"),
    lambda b: b["sources"][0].update(fragment_id=""),
    lambda b: b["sources"][0].update(extra=True),
    lambda b: b["sources"].append(copy.deepcopy(b["sources"][0])),
    lambda b: b["missing"].clear(),
    lambda b: b["missing"].update(mean_prior_move="guessed"),
    lambda b: b.update(quote_status="present"),
])
def test_block_lineage_rejects_ambiguous_future_and_missing_metadata(native, edit):
    rows = _rows(native[-1])
    _lineage(rows[0], "history", edit)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(rows, native[-1])


def test_quote_zero_is_missing_not_numeric_zero_and_observed_requires_evidence(native):
    context = copy.deepcopy(native[-1])
    rows = _rows(context)
    _lineage(rows[0], "daily_market", lambda b: b.update(quote_status="zero"))
    assert table.validate_rows(rows, context) is None
    rows[0]["implied_move"] = 3.0
    _lineage(rows[0], "daily_market", lambda b: (
        b.update(quote_status="present"), b["missing"].pop("implied_move")))
    assert table.validate_rows(rows, context) is None
    _lineage(rows[0], "daily_market", lambda b: b["sources"][0].update(
        knowledge_mode="observed", available_as_of=rows[0]["source_session_ceiling"]))
    assert table.validate_rows(rows, context) is None
    for block in ("history", "prior_events"):
        changed = copy.deepcopy(rows)
        _lineage(changed[0], block, lambda b: b["sources"][0].update(
            knowledge_mode="observed", available_as_of=changed[0]["source_session_ceiling"]))
        with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
            table.validate_rows(changed, context)


def test_observed_lane_cannot_upgrade_reconstructed_history(native, monkeypatch):
    source, output, binding, producer, context = native
    _metadata_only(monkeypatch, source.repository)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.build_context(source.repository, source.snapshot, output, binding,
                           producer=producer, availability_lane="observed")
    changed = copy.deepcopy(context)
    changed["availability_lane"] = "observed"
    _context_hash(changed)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.preflight_context(source.repository, source.snapshot, output, changed)


@pytest.mark.parametrize("change", [{"extra": 1}, {"schema_version": "magnitude_panel_context.v2"},
    {"availability_lane": "latest"}, {"feature_recipe": {}}, {"producer": {}},
    {"producer": {"source_closure": True, "environment": True}},
    {"producer": {"source_closure": {"source.py": "not-a-hash"}, "environment": {"python": "3"}}}])
def test_rehashed_context_still_has_closed_semantics(native, change):
    context = copy.deepcopy(native[-1])
    context.update(change)
    _hash(context, "context_id")
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


@pytest.mark.parametrize("clock", ["2099-01-01", "2024-02-30", None])
def test_rehashed_invalid_or_future_clocks_refuse(native, clock):
    context = copy.deepcopy(native[-1])
    context["history_context"]["temporal_plan"]["events"][0]["forecasts"][0]["source_session_ceiling"] = clock
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


def test_snapshot_and_history_binding_mismatches_refuse_before_payload(native, monkeypatch):
    source, output, binding, producer, context = native
    _metadata_only(monkeypatch, source.repository)
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        plan.preflight_context(source.repository, output, source.snapshot, context)
    altered = replace(output, calendar_version="different")
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        plan.build_context(source.repository, source.snapshot, altered, binding, producer=producer)
    changed = copy.deepcopy(binding)
    changed["source_snapshot"] = to_document(output)
    _hash(changed, "context_id")
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        plan.build_context(source.repository, source.snapshot, output, changed, producer=producer)


def test_protected_requested_event_refuses_before_payload(native, monkeypatch):
    source, output, binding, producer, _ = native
    _metadata_only(monkeypatch, source.repository)
    changed = copy.deepcopy(binding)
    changed["temporal_plan"]["events"][0]["event_id"] = source.forbidden[0]["event_id"]
    _hash(changed["temporal_plan"], "plan_id")
    _hash(changed, "context_id")
    with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
        plan.build_context(source.repository, source.snapshot, output, changed, producer=producer)


@pytest.mark.parametrize("target", ["computed_moves", "research_magnitude_history"])
def test_mixed_physical_fragment_refuses_before_payload(native, monkeypatch, target):
    source, output, binding, producer, _ = native
    _metadata_only(monkeypatch, source.repository)
    original = source.repository.fragment_records

    def records(snapshot, table_name):
        result = list(original(snapshot, table_name))
        if table_name == target:
            result = [replace(record, row_count=record.row_count + 1) for record in result]
        return tuple(result)

    monkeypatch.setattr(source.repository, "fragment_records", records)
    with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
        plan.build_context(source.repository, source.snapshot, output, binding, producer=producer)


@pytest.mark.parametrize("kind", ["duplicate_event", "duplicate_horizon", "unknown_history_key",
                                 "duplicate_dictionary", "wrong_dictionary_type", "cutoff_version",
                                 "unknown_cutoff_key", "unknown_event_key"])
def test_rehashed_nested_contract_corruption_refuses(native, kind):
    context = copy.deepcopy(native[-1])
    temporal = context["history_context"]["temporal_plan"]
    if kind == "duplicate_event":
        temporal["events"].append(copy.deepcopy(temporal["events"][0]))
    elif kind == "duplicate_horizon":
        temporal["events"][0]["forecasts"][0] = copy.deepcopy(temporal["events"][0]["forecasts"][1])
    elif kind == "unknown_history_key":
        context["history_context"]["allow_protected"] = True
    elif kind == "cutoff_version":
        temporal["events"][0]["forecasts"][0]["schema_version"] = "earnings_forecast_cutoff.v999"
    elif kind == "unknown_cutoff_key":
        temporal["events"][0]["forecasts"][0]["allow_future"] = True
    elif kind == "unknown_event_key":
        temporal["events"][0]["allow_protected"] = True
    elif kind == "duplicate_dictionary":
        context["feature_recipe"]["dictionary"].append(context["feature_recipe"]["dictionary"][0])
    else:
        context["feature_recipe"]["dictionary"][0]["physical_type"] = "string"
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


def test_stale_admission_and_untyped_history_context_refuse(native, monkeypatch):
    source, output, binding, producer, context = native
    _metadata_only(monkeypatch, source.repository)
    changed = copy.deepcopy(context)
    changed["admission"]["prior_outcome_fragments"] = []
    _hash(changed, "context_id")
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        plan.preflight_context(source.repository, source.snapshot, output, changed)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.build_context(source.repository, source.snapshot, output, None, producer=producer)


def test_panel_contract_sources_and_tests_select_research_mutations():
    from tools import mutation_pilot

    config = mutation_pilot.load_config()
    names = mutation_pilot.enabled_modules(config)
    paths = ["engine/v2/research/magnitude_panel_table.py", "engine/v2/research/magnitude_panel_plan.py",
             "tests/v2/research/test_magnitude_panel_contract.py"]
    assert paths[-1] in config["modules"]["research"]["tests"]
    for path in paths:
        assert mutation_pilot.changed_modules(config, names, [path]) == ["research"]
    assert mutation_pilot.changed_modules(config, names, ["README.md"]) == []
    assert mutation_pilot.changed_modules(config, names, []) == []


@pytest.mark.parametrize("values", [
    {"n_prior": 0, "mean_prior_move": 1.0, "mean_prior_abs_move": 1.0},
    {"n_prior": 1, "mean_prior_move": 1.0, "mean_prior_abs_move": -1.0},
    {"n_prior": 4},
])
def test_history_feature_invariants_are_not_weakened(native, values):
    rows = _rows(native[-1])
    rows[0].update(values)
    _lineage(rows[0], "history", lambda b: b.update(missing={
        name: "insufficient_history" for name in table.HISTORY if rows[0][name] is None}))
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(rows, native[-1])


@pytest.mark.parametrize("kind", ["swap_horizons", "wrong_ceiling", "wrong_preclose", "fake_calendar_id"])
def test_rehashed_causal_but_wrong_forecast_semantics_refuse_with_matching_rows(native, kind):
    context = copy.deepcopy(native[-1])
    forecasts = context["history_context"]["temporal_plan"]["events"][-1]["forecasts"]
    if kind == "swap_horizons":
        forecasts[0]["horizon_sessions"], forecasts[1]["horizon_sessions"] = (
            forecasts[1]["horizon_sessions"], forecasts[0]["horizon_sessions"])
    elif kind == "wrong_ceiling":
        forecasts[0]["source_session_ceiling"] = forecasts[1]["source_session_ceiling"]
    elif kind == "wrong_preclose":
        forecasts[0]["pre_announcement_close"] = forecasts[1]["source_session_ceiling"]
    else:
        forecasts[0]["calendar_id"] = "sha256:" + "0" * 64
    _context_hash(context)
    # All nested digests and row bindings agree with the forgery. Calendar semantics must reject it.
    rows = _rows(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(rows, context)


def test_none_rows_is_a_typed_contract_refusal(native):
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(None, native[-1])


def _expected_context(native):
    source, output, binding, producer, context = native
    records = source.repository.fragment_records(output, "research_magnitude_history")
    history_fragments = [{"fragment_id": r.fragment_id, "object_id": r.object_ref.object_id,
                          "content_hash": r.object_ref.content_hash} for r in records]
    source_records = {record.primary_key_min: record for record in
                      source.repository.fragment_records(source.snapshot, "computed_moves")}
    prior = []
    # The fixture has exactly four admitted prior events before the final forecast ceiling.
    for event in source.events[:-1]:
        record = source_records[(event["ticker"], event["event_date"].date().isoformat())]
        prior.append({"event_id": event["event_id"], "fragment_id": record.fragment_id,
                      "object_id": record.object_ref.object_id, "content_hash": record.object_ref.content_hash})
    expected = {"schema_version": "magnitude_panel_context.v1",
        "source_snapshot": to_document(source.snapshot), "history_snapshot": to_document(output),
        "history_context": binding, "feature_recipe": _expected_recipe("HMO"), "producer": producer,
        "availability_lane": "reconstructed",
        "forecast_calendar": {"days": list(calendar_sessions("2022-05-25", "2024-06-18").days),
                              "observed_through": "2024-06-18"},
        "forecast_calendar_source": {"snapshot_id": source.snapshot.snapshot_id,
            "calendar_version": source.snapshot.calendar_version, "table": "daily_market", "columns": ["date"],
            "dataset_version": source.snapshot.table_versions["daily_market"].dataset_version_id,
            "objects": [{"fragment_id": r.fragment_id, "object_id": r.object_ref.object_id,
                         "content_hash": r.object_ref.content_hash}
                        for r in source.repository.fragment_records(source.snapshot, "daily_market")]},
        "admission": {"raw_payload_access": "not_authorized", "raw_policy_status": "not_authorized_by_this_plan",
            "required_later": ["permitted_raw_read_contract", "approved_protected_event_policy",
                               "per_observation_availability", "bounded_batched_materializer"],
            "history_fragments": history_fragments, "prior_outcome_fragments": prior}}
    return {**expected, "context_id": content_hash(expected)}


def test_admission_identity_matches_independent_metadata_oracle(native):
    assert native[-1] == _expected_context(native)


@pytest.mark.parametrize("kind", ["bad_digest", "snapshot_id", "source_snapshot", "event_ticker",
                                 "population_use", "random_membership_version", "population_input"])
def test_public_history_preflight_rederives_identity_before_payloads(native, monkeypatch, kind):
    source, output, binding, _, _ = native
    _metadata_only(monkeypatch, source.repository)
    changed = copy.deepcopy(binding)
    temporal = changed["temporal_plan"]
    if kind == "bad_digest":
        changed["context_id"] = "sha256:" + "0" * 64
    else:
        if kind == "snapshot_id":
            temporal["snapshot_id"] = "wrong-committed-source"
        elif kind == "source_snapshot":
            changed["source_snapshot"] = to_document(output)
        elif kind == "random_membership_version":
            temporal["random_membership_version"] = "sha256:" + "0" * 64
        elif kind == "population_input":
            temporal["population_input"]["objects"] = []
        elif kind == "event_ticker":
            temporal["events"][0]["ticker"] = "FORGED"
        else:
            temporal["events"][0]["population_use"] = "protected"
        _hash(temporal, "plan_id")
        _hash(changed, "context_id")
    with pytest.raises(DataError) as caught:
        history.preflight_source(source.repository, source.snapshot, changed)
    assert caught.value.code == "IDENTITY_CONFLICT"


def test_public_history_preflight_preserves_valid_admitted_metadata(native, monkeypatch):
    source, _, binding, _, _ = native
    scans = _metadata_only(monkeypatch, source.repository)
    population, candidates = history.preflight_source(source.repository, source.snapshot, binding)
    assert scans
    assert {event["event_id"] for event in source.events} <= set(population.index)
    assert [event.event_id for event, _, _ in candidates] == [event["event_id"] for event in source.events[:-1]]


@pytest.mark.parametrize("row,count", [({}, 0), (None, 0), ([], 0), ({"n_prior": 0}, 0)])
def test_public_history_feature_validation_malformed_rows_are_typed(row, count):
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        history.validate_features(row, count)


@pytest.mark.parametrize("count", [None, True, False, 0.0, 1.0, "0", -1])
def test_public_history_feature_validation_requires_integer_count(count):
    row = {name: None for name, _, _ in _EXPECTED_FIELDS[:11]}
    row["n_prior"] = 1 if count is True or (type(count) is float and count == 1.0) else 0
    if row["n_prior"] == 1:
        row["mean_prior_move"], row["mean_prior_abs_move"] = -2.0, 2.0
    # Validate the independent row before challenging only the public count's type/range.
    assert history.validate_features(row, row["n_prior"]) is None
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        history.validate_features(row, count)


@pytest.mark.parametrize("digest", [None, "", "sha256:abc", "sha256:" + "A" * 64,
                                     "sha256:" + "g" * 64, "md5:" + "a" * 64, True])
def test_lineage_rejects_missing_or_malformed_content_hash(native, digest):
    rows = _rows(native[-1])
    def edit(block):
        if digest is None:
            block["sources"][0].pop("content_hash")
        else:
            block["sources"][0]["content_hash"] = digest
    _lineage(rows[0], "history", edit)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(rows, native[-1])


def test_content_digest_is_bound_into_row_identity(native):
    rows = _rows(native[-1])
    original = rows[0]["row_id"]
    _lineage(rows[0], "history", lambda block: block["sources"][0].update(content_hash="sha256:" + "b" * 64))
    assert rows[0]["row_id"] != original
    assert table.validate_rows(rows, native[-1]) is None
    rows[0]["row_id"] = original
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(rows, native[-1])


@pytest.mark.parametrize("quote,value", [("present", None), ("zero", 3.0), ("absent", 3.0),
                                         ("unavailable", 3.0), ("not_applicable", None)])
def test_daily_quote_status_cannot_disagree_with_implied_move(native, quote, value):
    rows = _rows(native[-1])
    rows[0]["implied_move"] = value
    def edit(block):
        block["quote_status"] = quote
        if value is not None:
            block["missing"].pop("implied_move")
    _lineage(rows[0], "daily_market", edit)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(rows, native[-1])


@pytest.mark.parametrize("kind", ["none", "mapping", "forged_version"])
def test_public_calendar_requires_exact_snapshot_before_any_scan(native, monkeypatch, kind):
    source = native[0]
    scans = _metadata_only(monkeypatch, source.repository)
    supplied = {"none": None, "mapping": to_document(source.snapshot),
                "forged_version": replace(source.snapshot, calendar_version="forged")}[kind]
    with pytest.raises(DataError, match="SNAPSHOT_UNRESOLVED"):
        magnitude_features.calendar_input(source.repository, supplied)
    assert scans == []


def test_public_calendar_and_history_plans_have_full_independent_metadata(native, monkeypatch):
    source, _, binding, _, _ = native
    _metadata_only(monkeypatch, source.repository)
    calendar, metadata = magnitude_features.calendar_input(source.repository, source.snapshot)
    assert calendar == calendar_sessions("2022-05-25", "2024-06-18")
    records = source.repository.fragment_records(source.snapshot, "daily_market")
    assert metadata == {"snapshot_id": source.snapshot.snapshot_id,
        "calendar_version": source.snapshot.calendar_version, "table": "daily_market", "columns": ["date"],
        "dataset_version": source.snapshot.table_versions["daily_market"].dataset_version_id,
        "objects": [{"fragment_id": r.fragment_id, "object_id": r.object_ref.object_id,
                     "content_hash": r.object_ref.content_hash} for r in records]}
    _, candidates = history.preflight_source(source.repository, source.snapshot, binding)
    records = {r.primary_key_min: r for r in source.repository.fragment_records(source.snapshot, "computed_moves")}
    for (event, request, record), expected_event in zip(candidates, source.events[:-1], strict=True):
        day = expected_event["event_date"].date()
        assert (event.event_id, event.ticker, event.event_date, event.session) == (
            expected_event["event_id"], "SYNTH", expected_event["event_date"], "AMC")
        assert record == records[("SYNTH", day.isoformat())]
        assert request == DataQuery(snapshot_id=source.snapshot.snapshot_id,
            table_contract_ref=source.snapshot.table_versions["computed_moves"].table_contract_ref,
            columns=("event_date", "realized_move_pct", "available_as_of_date", "skipped"),
            key_filter=(KeyPredicate(column="ticker", operator="eq", values=("SYNTH",)),),
            time_interval=TimeInterval(column="event_date", start_inclusive=day.isoformat(),
                                      end_exclusive=(day + timedelta(days=1)).isoformat()),
            order_by=("ticker", "event_date"), max_batch_rows=1, max_result_rows=1)


def test_independent_physical_table_contract():
    metadata = ("schema_version", "event_id", "context_id", "history_context_id", "horizon_sessions",
                "event_date", "cutoff_id", "source_session_ceiling", "lineage", "row_id")
    columns = tuple(ColumnContract(name=name, physical_type="int64" if name == "horizon_sessions"
                                   else "string", nullable=False) for name in metadata)
    columns += tuple(ColumnContract(name=f["name"], physical_type=f["physical_type"],
        nullable=f["nullable"], unit=f["unit"], null_policy=f["on_missing"])
        for f in _expected_recipe("HMO")["dictionary"])
    expected = TableContract(contract_id="research_magnitude_panel.v1", table_name="research_magnitude_panel",
        semantic_version="1.0.0", definition_hash="sha256:" + "0" * 64, columns=columns,
        primary_key=("event_id", "context_id", "horizon_sessions"), duplicate_policy="none_by_construction",
        foreign_keys=(), partition_columns=("context_id",), filterable_columns=("context_id", "event_id"),
        orderable_columns=("event_id", "context_id", "horizon_sessions"), observation_time_column="event_date",
        publication_time_column=None, receipt_time_column=None,
        finality_semantics="immutable research reconstruction; session dates are not publication proof",
        provenance_semantics="explicit history binding; per-block source and availability lineage",
        coverage_semantics="exact admitted target set, with horizons 1, 3 and 5 for every event",
        schema_evolution_policy="new contract identity for any semantic or column change",
        maximum_batch_rows=65536, legacy_mapping_ref=None)
    assert table.CONTRACT == replace(expected, definition_hash=table_contract_hash(expected))


def _populated_rows(context):
    rows = _rows(context)
    for row in rows:
        row.update({name: 2.5 for name, _, _ in _EXPECTED_FIELDS})
        row["n_prior"] = 12
        blocks = json.loads(row["lineage"])
        for block in blocks.values():
            block["missing"] = {}
        blocks["daily_market"]["quote_status"] = "present"
        row["lineage"] = canonical_json(blocks)
        _hash(row, "row_id")
    return rows


@pytest.mark.parametrize("field,value", [("ret5", True), ("ret5", "2.5"), ("ret5", float("inf")),
    ("ret5", float("nan")), ("n_prior", 12.0), ("n_prior", True),
    ("implied_move", 0.0), ("implied_move", -1.0)])
def test_populated_numeric_guard_isolated_from_missingness(native, field, value):
    rows = _populated_rows(native[-1])
    assert table.validate_rows(rows, native[-1]) is None
    rows[0][field] = value
    _hash(rows[0], "row_id")
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(rows, native[-1])


@pytest.mark.parametrize("kind", ["no_sources", "observed_without_availability", "availability_before_source",
                                 "missing_map_extra", "bad_source_type", "blank_row_id", "noncanonical_json"])
def test_lineage_guards_with_populated_positive_control(native, kind):
    rows = _populated_rows(native[-1])
    assert table.validate_rows(rows, native[-1]) is None
    def edit(block):
        if kind == "no_sources":
            block["sources"] = []
        elif kind == "observed_without_availability":
            block["sources"][0]["knowledge_mode"] = "observed"
        elif kind == "availability_before_source":
            block["sources"][0]["available_as_of"] = "2000-01-01"
        elif kind == "missing_map_extra":
            block["missing"]["ret5"] = "unavailable"
        elif kind == "bad_source_type":
            block["sources"][0] = None
        elif kind == "blank_row_id":
            block["sources"][0]["row_id"] = " "
    _lineage(rows[0], "runup", edit)
    if kind == "noncanonical_json":
        rows[0]["lineage"] = json.dumps(json.loads(rows[0]["lineage"]), indent=2)
        _hash(rows[0], "row_id")
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(rows, native[-1])


@pytest.mark.parametrize("producer", [
    {"source_closure": {}, "environment": {"python": "3"}},
    {"source_closure": {"source.py": "sha256:" + "a" * 64}, "environment": {}},
    {"source_closure": {" ": "sha256:" + "a" * 64}, "environment": {"python": "3"}},
    {"source_closure": {"source.py": "sha256:" + "A" * 64}, "environment": {"python": "3"}},
    {"source_closure": {"source.py": "sha256:" + "a" * 64}, "environment": {"python": "3"}, "extra": 1},
])
def test_isolated_producer_guards(native, producer):
    context = copy.deepcopy(native[-1])
    assert plan.validate_context(context) is None
    context["producer"] = producer
    _hash(context, "context_id")
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


def test_builder_deeply_detaches_nested_caller_objects(native):
    source, output, binding, producer, _ = native
    binding, producer = copy.deepcopy(binding), copy.deepcopy(producer)
    context = plan.build_context(source.repository, source.snapshot, output, binding, producer=producer)
    original = copy.deepcopy(context)
    binding["temporal_plan"]["events"][0]["forecasts"][0]["source_session_ceiling"] = "2099-01-01"
    producer["environment"]["libraries"]["numpy"] = "caller-poison"
    assert context == original
    assert plan.validate_context(context) is None
    context["history_context"]["temporal_plan"]["events"].clear()
    context["producer"]["environment"]["libraries"]["numpy"] = "result-poison"
    assert binding["temporal_plan"]["events"]
    assert producer["environment"]["libraries"]["numpy"] == "caller-poison"


@pytest.mark.parametrize("missing", ["source", "history"])
def test_preflight_rejects_untyped_snapshot_arguments(native, monkeypatch, missing):
    source, output, _, _, context = native
    scans = _metadata_only(monkeypatch, source.repository)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.preflight_context(source.repository, None if missing == "source" else source.snapshot,
                               None if missing == "history" else output, context)
    assert scans == []


def test_validation_progress_ticks_once_per_event_and_row(native, monkeypatch):
    counts, totals = {}, {}
    class CountingProgress:
        def __init__(self, label, total):
            self.label = label
            counts[label], totals[label] = 0, total

        def tick(self):
            counts[self.label] += 1

    monkeypatch.setattr(plan, "Progress", CountingProgress)
    monkeypatch.setattr(table, "Progress", CountingProgress)
    rows = _rows(native[-1])
    table.validate_rows(rows, native[-1])
    expected = {"panel clocks": len(rows) // 3, "panel rows": len(rows)}
    assert counts == totals == expected


@pytest.mark.parametrize("value", [None, 20240101, {}, "2024-02-30", "2024-01-01T12:00:00", "2024-1-1"])
def test_public_canonical_day_typed_refusal(value):
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.canonical_day(value)


def test_public_canonical_day_preserves_valid_date():
    assert table.canonical_day("2024-02-29") == "2024-02-29"


@pytest.mark.parametrize("field", ["snapshot_id", "calendar_version", "table", "columns", "dataset_version", "objects"])
@pytest.mark.parametrize("side", ["history", "panel", "both"])
def test_detached_calendar_source_binding_refuses_rehashed_mutations(native, field, side):
    context = copy.deepcopy(native[-1])
    sources = {"history": context["history_context"]["temporal_plan"]["calendar_input"],
               "panel": context["forecast_calendar_source"]}
    for name in sources if side == "both" else [side]:
        sources[name][field] = None
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(_rows(context), context)


@pytest.mark.parametrize("field", ["history_fragments", "prior_outcome_fragments"])
@pytest.mark.parametrize("value", [None, {}, "fragments", [None], [{}], [{"fragment_id": "x"}]])
def test_detached_admission_closed_lists(native, field, value):
    context = copy.deepcopy(native[-1])
    context["admission"][field] = value
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(_rows(context), context)


@pytest.mark.parametrize("field", ["history_fragments", "prior_outcome_fragments"])
@pytest.mark.parametrize("key,value", [("fragment_id", " "), ("object_id", None),
                                       ("content_hash", "sha256:bad"), ("extra", "unknown")])
def test_detached_admission_closed_entries(native, field, key, value):
    context = copy.deepcopy(native[-1])
    context["admission"][field][0][key] = value
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


def test_detached_calendar_is_consistency_not_repository_authentication(native, monkeypatch):
    source, output, _, _, original = native
    context = copy.deepcopy(original)
    # Well-formed but counterfeit membership cannot be authenticated without the repository.
    for metadata in (context["forecast_calendar_source"],
                     context["history_context"]["temporal_plan"]["calendar_input"]):
        metadata["objects"][0]["content_hash"] = "sha256:" + "f" * 64
    _context_hash(context)
    plan.validate_context(context)
    _metadata_only(monkeypatch, source.repository)
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        plan.preflight_context(source.repository, source.snapshot, output, context)


def test_panel_validation_stage_start_and_end_are_visible(native, capsys):
    table.validate_rows(_rows(native[-1]), native[-1])
    captured = capsys.readouterr()
    assert captured.out == ""
    for label in ("panel clocks", "panel rows"):
        assert f"[{label}] start" in captured.err
        assert f"[{label}] done in " in captured.err


@pytest.mark.parametrize("side", ["history", "panel"])
def test_detached_calendar_rejects_well_formed_but_mismatched_object(native, side):
    context = copy.deepcopy(native[-1])
    metadata = (context["forecast_calendar_source"] if side == "panel" else
                context["history_context"]["temporal_plan"]["calendar_input"])
    metadata["objects"][0]["content_hash"] = "sha256:" + "f" * 64
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(_rows(context), context)


@pytest.mark.parametrize("horizon", [True, 1.0])
def test_row_horizon_type_guard_with_otherwise_matching_identity(native, horizon):
    rows = _rows(native[-1])
    assert table.validate_rows(rows, native[-1]) is None
    row = next(row for row in rows if row["horizon_sessions"] == 1)
    # bool/float compare equal to 1: all event/cutoff/ceiling pins remain valid.
    row["horizon_sessions"] = horizon
    _hash(row, "row_id")
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(rows, native[-1])


@pytest.mark.parametrize("field,value", [
    ("event_id", True), ("event_id", None), ("event_id", ""), ("event_id", " padded "),
    ("ticker", None), ("ticker", 1), ("ticker", ""), ("ticker", " padded "),
    ("session", None), ("session", "UNKNOWN"), ("population_use", None),
    ("population_use", "protected"), ("event_date", "2024-6-1"),
])
def test_closed_event_fields_with_rehashed_matching_fold_membership(native, field, value):
    context = copy.deepcopy(native[-1])
    temporal = context["history_context"]["temporal_plan"]
    event = temporal["events"][0]
    original_id = event["event_id"]
    event[field] = value
    if field == "event_id":
        for fold in temporal["folds"]:
            for key in ("training_candidate_event_ids", "evaluation_event_ids"):
                fold[key] = [value if item == original_id else item for item in fold[key]]
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        table.validate_rows(_rows(context), context)


@pytest.mark.parametrize("field,value", [
    ("horizons", None), ("horizons", {}), ("horizons", [5, 3, True]), ("horizons", [5, 3, 1.0]),
    ("horizons", [1, 3, 5]), ("folds", None), ("folds", {}), ("folds", []),
    ("folds", [None]), ("folds", [{}]), ("target_recipe", {}), ("target_recipe", None),
    ("fold_policy", {}), ("availability", {}), ("population_input", None),
    ("training_start", None), ("training_start", "2023-1-1"),
    ("evaluation_years", None), ("evaluation_years", []), ("evaluation_years", [2024]),
    ("evaluation_years", [True]), ("holdout_as_of_month", None),
    ("holdout_as_of_month", "2024-1"), ("random_membership_version", None),
    ("rolling_membership_version", True),
])
def test_closed_temporal_field_schema_after_rehash(native, field, value):
    context = copy.deepcopy(native[-1])
    context["history_context"]["temporal_plan"][field] = value
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


@pytest.mark.parametrize("field", ["year", "training_labels_before",
    "training_candidate_event_ids", "evaluation_event_ids", "extra"])
@pytest.mark.parametrize("value", [None, True, {}, [], [True], ["unknown"]])
def test_closed_fold_fields_and_membership_after_rehash(native, field, value):
    context = copy.deepcopy(native[-1])
    context["history_context"]["temporal_plan"]["folds"][0][field] = value
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


@pytest.mark.parametrize("field,value", [
    ("table", None), ("table", "computed_moves"), ("columns", None), ("columns", ["event_id"]),
    ("dataset_version", None), ("dataset_version", "forged"), ("objects", None),
    ("objects", {}), ("objects", [None]), ("objects", [{}]), ("extra", True),
])
def test_closed_population_metadata_schema_after_rehash(native, field, value):
    context = copy.deepcopy(native[-1])
    context["history_context"]["temporal_plan"]["population_input"][field] = value
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


@pytest.mark.parametrize("field", ["target_recipe", "fold_policy", "availability"])
def test_closed_fixed_temporal_documents_reject_unknown_keys(native, field):
    context = copy.deepcopy(native[-1])
    context["history_context"]["temporal_plan"][field]["extra"] = True
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


@pytest.mark.parametrize("producer", [None, {}, {"source_closure": True, "environment": {}},
    {"source_closure": {"source.py": "not-a-hash"}, "environment": {"python": "3"}}])
def test_closed_history_producer_after_rehash(native, producer):
    context = copy.deepcopy(native[-1])
    context["history_context"]["producer"] = producer
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


@pytest.mark.parametrize("value", [object(), {"not-json"}, complex(1, 2)])
def test_builder_non_json_producer_is_typed_refusal(native, monkeypatch, value):
    source, output, binding, producer, _ = native
    _metadata_only(monkeypatch, source.repository)
    producer = copy.deepcopy(producer)
    producer["environment"]["unsupported"] = value
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.build_context(source.repository, source.snapshot, output, binding, producer=producer)


@pytest.mark.parametrize("path,value", [
    (("history_start",), None), (("history_start",), "2022-1-1"),
    (("temporal_plan", "events"), None), (("temporal_plan", "events"), {}),
    (("temporal_plan", "events"), []), (("temporal_plan", "events"), [None]),
    (("temporal_plan", "events", 0, "forecasts"), None),
    (("temporal_plan", "events", 0, "forecasts"), {}),
    (("temporal_plan", "events", 0, "forecasts"), [None]),
])
def test_closed_history_dates_and_event_containers(native, path, value):
    context = copy.deepcopy(native[-1])
    parent = context["history_context"]
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    # Do not inspect malformed events while recomputing the three enclosing digests.
    _hash(context["history_context"]["temporal_plan"], "plan_id")
    _hash(context["history_context"], "context_id")
    _hash(context, "context_id")
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


@pytest.mark.parametrize("field,value", [("horizon_sessions", True), ("horizon_sessions", 1.0),
                                         ("horizon_sessions", None), ("extra", True)])
def test_closed_forecast_fields_with_rehashed_matching_one_session_clock(native, field, value):
    context = copy.deepcopy(native[-1])
    cutoff = next(cutoff for cutoff in context["history_context"]["temporal_plan"]["events"][0]["forecasts"]
                  if cutoff["horizon_sessions"] == 1)
    cutoff[field] = value
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


@pytest.mark.parametrize("path,value", [(("quote_indicator",), 1),
    (("dictionary", 0, "nullable"), 0), (("dictionary", 1, "nullable"), 1)])
@pytest.mark.parametrize("rehash_dictionary", [False, True])
def test_recipe_bool_fields_do_not_accept_equal_integers(native, path, value, rehash_dictionary):
    context = copy.deepcopy(native[-1])
    parent = context["feature_recipe"]
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    if rehash_dictionary:
        context["feature_recipe"]["dictionary_hash"] = content_hash(context["feature_recipe"]["dictionary"])
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), ("tuple",), {1: "numeric-key"}])
def test_producer_environment_requires_finite_plain_json(native, value):
    source, output, binding, producer, original = native
    producer = copy.deepcopy(producer)
    producer["environment"]["bad"] = value
    context = copy.deepcopy(original)
    context["producer"] = producer
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.build_context(source.repository, source.snapshot, output, binding, producer=producer)


def test_detached_plan_without_evaluation_events_has_typed_refusal(native):
    context = copy.deepcopy(native[-1])
    context["history_context"]["temporal_plan"]["evaluation_years"] = ["2025"]
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


def test_public_history_identity_serialization_failure_is_typed(native):
    class Unsupported:
        def __str__(self):
            raise TypeError("not serializable")

    binding = copy.deepcopy(native[2])
    assert history.validate_context_identity(binding) is None
    binding["producer"]["environment"]["unsupported"] = Unsupported()
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        history.validate_context_identity(binding)


def _refresh_fold_memberships(temporal):
    # Preserve derived membership order, so order/uniqueness controls cannot fail on stale folds.
    for fold in temporal["folds"]:
        fold["training_candidate_event_ids"] = [event["event_id"] for event in temporal["events"]
            if event["event_date"] < fold["training_labels_before"]]
        fold["evaluation_event_ids"] = [event["event_id"] for event in temporal["events"]
            if event["event_date"][:4] == fold["year"]]


@pytest.mark.parametrize("kind", ["history_window", "event_order", "declared_window", "duplicate_event"])
def test_isolated_temporal_guards_with_consistent_folds(native, kind):
    context = copy.deepcopy(native[-1])
    binding = context["history_context"]
    temporal = binding["temporal_plan"]
    if kind == "history_window":
        # Every cutoff remains native and causal; only the history start is too late.
        binding["history_start"] = temporal["events"][0]["event_date"]
    elif kind == "event_order":
        temporal["events"].reverse()
    elif kind == "declared_window":
        # A valid training date before the evaluation year, but after an included event.
        temporal["training_start"] = temporal["events"][0]["event_date"][:4] + "-02-01"
    else:
        temporal["events"].insert(0, copy.deepcopy(temporal["events"][0]))
    _refresh_fold_memberships(temporal)
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


@pytest.mark.parametrize("kind", ["unknown_key", "missing_interior_session"])
def test_isolated_calendar_structure_and_session_contents(native, kind):
    context = copy.deepcopy(native[-1])
    calendar = context["forecast_calendar"]
    if kind == "unknown_key":
        calendar["extra"] = True
    else:
        # Modify actual calendar contents, not just a digest. All declared calendar IDs agree.
        event = context["history_context"]["temporal_plan"]["events"][0]
        calendar["days"].remove(event["forecasts"][1]["source_session_ceiling"])
        calendar_id = content_hash(calendar)
        for event in context["history_context"]["temporal_plan"]["events"]:
            for cutoff in event["forecasts"]:
                cutoff["calendar_id"] = calendar_id
    _context_hash(context)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan.validate_context(context)


@pytest.mark.parametrize("boundary", ["validate_context", "validate_rows", "preflight_context", "build_context"])
def test_excessive_document_depth_is_a_typed_public_refusal(native, boundary):
    import sys

    source, output, binding, producer, original = native
    nested = leaf = {}
    for _ in range(sys.getrecursionlimit() + 100):
        leaf["next"] = {}
        leaf = leaf["next"]
    context = copy.deepcopy(original)
    context["producer"]["environment"]["nested"] = nested
    producer = copy.deepcopy(producer)
    producer["environment"]["nested"] = nested
    # Deliberately neither deepcopy nor hash the over-depth document at the test boundary.
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        if boundary == "validate_context":
            plan.validate_context(context)
        elif boundary == "validate_rows":
            table.validate_rows([], context)
        elif boundary == "preflight_context":
            plan.preflight_context(source.repository, source.snapshot, output, context)
        else:
            plan.build_context(source.repository, source.snapshot, output, binding, producer=producer)
