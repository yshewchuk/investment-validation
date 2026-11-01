"""Real synthetic native source, admissions, fits, quotes, targets and reports."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.models.no_fit import no_fit_guard as legacy_no_fit
from engine.v2.contracts import ArtifactRef
from engine.v2.data import objects
from engine.v2.foundation import SystemClock, from_document, to_document
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.models.payoff_artifact import make_payoff_line_artifact, serialize_payoff_artifact
from engine.v2.research._pricing import trading_calendar_from_snapshot
from engine.v2.research.event_history import MAX_OUTCOME_OBJECT_BYTES
from engine.v2.research.magnitude_panel_qualified_table import recipe
from experiments import distribution_market as market
from experiments import distribution_pipeline as pipeline
from experiments import eligible_history, size_panel, size_search_binding
from experiments.distribution_candidate_recipe import candidate_recipe
from experiments.distribution_fitted import DistributionFits
from experiments.distribution_run_plan import DistributionRunPlan
from experiments.distribution_run_request import PRIMARY_ARMS
from experiments.distribution_trials import DistributionTrials
from experiments.magnitude_search_plan import seal_search_plan
from experiments.magnitude_search_shards import iter_section
from experiments.native_registration import prepare_native_registration, reserve_native_registration
from experiments.v2_magnitude_search_run import seal_search_run
from tests.data_scan_support import (
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.v2.research.magnitude_trial_support import prediction_spec
from tests.v2.research.test_distribution_run_request import request as run_request
from tests.v2.research.test_magnitude_panel_qualified import panel_data
from tests.v2.research.test_magnitude_search_request import request as search_request

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def native_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("distribution-native-pipeline")
    generator = panel_data.__wrapped__(root, SimpleNamespace(param={
        "days": ["2022-06-02", "2022-09-02", "2023-06-02", "2024-06-17"], "values": [2., 22., 6., 8.]}))
    data = next(generator)
    ids = [event["event_id"] for event in data.events]
    history = eligible_history.materialize(data.repository, data.panel_source,
        **{**data.request, "limits": data.panel_request["limits"], "event_ids": ids})
    panel_request = {**data.panel_request, "event_ids": ids, "history_snapshot": history["feature_snapshot"]}
    panel = size_panel.materialize(data.repository, data.panel_source, **panel_request)
    panel_request.update(feature_snapshot_id=panel["feature_snapshot"].snapshot_id,
        history_snapshot=history["feature_snapshot"].snapshot_id, price_history_snapshot=data.prices.snapshot_id)
    request = search_request()
    request.update(feature_dictionary_hash=recipe("HMO")["dictionary_hash"],
        feature_snapshot_id=panel["feature_snapshot"].snapshot_id, history_snapshot_id=history["feature_snapshot"].snapshot_id,
        outer_years=["2024"], inner_years={"2024": ["2023"]}, max_fits=102)
    request["eligibility_policy_ref"]["plan_id"] = panel["context"]["eligibility_plan"]["plan_id"]
    request["limits"] = {"child": {"memory_bytes": 2 << 30, "cpu_seconds": 30, "wall_seconds": 60},
        "parent": {"memory_bytes": 3 << 30, "wall_seconds": 600},
        "build": {"memory_bytes": 3 << 30, "wall_seconds": 60}, "job": {"wall_seconds": 600}}
    spec = prediction_spec(experiment_id="synthetic-distribution-source", folds=("2024",), runner="experiments/size_search.py")
    registration, spec = size_search_binding.prepare(data.repository, spec, request, panel_request,
        scope="panel-source", training_start="2022-01-01", labels_before="2025-01-01")
    store = data.repository._store
    reserve_native_registration(data.conn, store, registration, spec)
    sealed = seal_search_plan(data.conn, store, registration, request)
    ref = from_document(ArtifactRef, sealed["plan_ref"])
    seal_search_run(data.conn, store, data.repository, registration, request, ref, clock=SystemClock())
    case = SimpleNamespace(**{**vars(data), "root": root, "store": store, "registration": registration,
        "request": request, "plan": sealed["plan"], "ref": ref, "spec": spec, "panel_request": panel_request})
    case.quotes, case.target_objects = quote_snapshot(case)
    yield case
    try:
        next(generator)
    except StopIteration:
        pass


def quote_snapshot(case):
    """Real pinned quote/calendar fragments; no replacement quote or price reader."""
    contract = contract_for("option_chains")
    records = []
    for day, premium in (("2024-06-14", 1.), ("2024-06-17", 1.), ("2024-06-18", 1.5)):
        rows = [dict(ticker="SYNTH", obs_date=pd.Timestamp(day), year=2024,
            expiry=pd.Timestamp("2024-06-21"), dte=(pd.Timestamp("2024-06-21") - pd.Timestamp(day)).days,
            strike=100., right=right, bid=premium, ask=premium, delta=.5, spot=100., quote_repaired=False)
            for right in ("C", "P")]
        records.append(publish_and_inspect(case.store, contract, contract_ref_for(contract), rows, "2024"))
    source = case.repository.resolve_full(case.panel_source.snapshot_id)
    daily = case.repository.table_contract(case.panel_source, "daily_market")
    quote_hash = objects.partition_logical_hash(case.store, [row.object_ref for row in records],
        contract, contract_ref_for(contract), "2024")
    snapshot = commit_tables(case.conn, SystemClock(),
        {"option_chains": records, "daily_market": list(case.repository.fragment_records(case.panel_source, "daily_market"))},
        {"option_chains": contract, "daily_market": daily}, store=case.store,
        scope="distribution-quotes", receipt_id="distribution-quotes", attempt_id="distribution-quotes",
        partition_logical_hashes={"option_chains": {"2024": quote_hash},
            "daily_market": source.table_manifests["daily_market"].partition_logical_hashes})
    targets = {record.object_ref.object_id for record in case.repository.fragment_records(case.panel_source, "computed_moves")
        if record.time_min == "2024-06-17"}
    return snapshot, targets


def make_run(case, *, run_type=DistributionRunPlan, experiment_id="synthetic-distribution-pipeline",
             runner="experiments/distribution_trials.py"):
    source = case.registration.document
    spec = prediction_spec(experiment_id=experiment_id, folds=("2024",), runner=runner,
        hypothesis="Compare a raw magnitude histogram with its matched point model.", primary_arm_id="raw_pair", arms=("raw_pair",))
    own, spec = prepare_native_registration(case.repository, spec, code_root=ROOT,
        as_of_month=source["holdouts"]["holdout_as_of_month"], scope=source["scope"],
        event_ids=source["event_ids"], snapshot=case.panel_source)
    reserve_native_registration(case.conn, case.store, own, spec)
    adapter = DistributionTrials(conn=case.conn, store=case.store, repository=case.repository,
        own_registration=own, source_registration=case.registration, source_request=case.request,
        source_plan_ref=case.ref, recipe=candidate_recipe("H"), budget_caps={"max_fits": 12,
            "max_trees": 49200, "child_limits": dict(case.request["limits"]["child"])})
    policy = run_request()
    policy.update(quote_snapshot_id=case.quotes.snapshot_id, bridges=[], max_opportunities=1)
    for alpha in policy["alphas"]:
        artifact = make_payoff_line_artifact(dict(n=3, resid_sd=0., residuals=[0.], intercept=1., slope=1.),
            strategy="STR-THRU", driver="abs_move", alpha=alpha, cutoff="2023-12-31", window=("2022-01-01", "2023-12-30"))
        ref = case.store.publish_bytes(serialize_payoff_artifact(artifact), schema_ref=artifact.schema_version)
        policy["bridges"].append(dict(outer_year="2024", alpha=alpha, ref=to_document(ref)))
    return run_type(adapter, policy)


def test_native_fit_pool_quote_target_and_report_composition(native_case, monkeypatch):
    case, run = native_case, make_run(native_case)
    plan = run.seal(clock=SystemClock())
    fits = DistributionFits(run.trials)
    run.trials.start("inner", "2024", "2023")
    for horizon in (5, 3, 1):
        fits.fit("inner", "2024", horizon, "2023")
    run.trials.start("outer", "2024")
    results = {horizon: fits.fit("outer", "2024", horizon) for horizon in (5, 3, 1)}
    assert all(result["tail"]["policy"] == "empirical_excess" for result in results.values())
    original, opened = objects.read_verified_bytes, set()
    original_path = objects.verify_object_path
    def verify(store, ref):
        assert ref.object_id not in case.target_objects, "outer targets require a byte-bounded read"
        return original_path(store, ref)
    def read_bytes(store, ref, *, max_bytes):
        if ref.object_id in case.target_objects:
            assert max_bytes == MAX_OUTCOME_OBJECT_BYTES
            assert all(run.record(f"intents:2024:{alpha}") is not None for alpha in run.request["alphas"])
            opened.add(ref.object_id)
        return original(store, ref, max_bytes=max_bytes)
    monkeypatch.setattr(objects, "read_verified_bytes", read_bytes)
    monkeypatch.setattr(objects, "verify_object_path", verify)
    original_chains, requests = market.load_chain_index, []
    def chains(repository, snapshot, keys):
        keys = tuple(keys)
        requests.append(tuple(day.date().isoformat() for _, day in keys))
        if any(day >= pd.Timestamp("2024-06-17") for _, day in keys):
            assert all(run.record(f"intents:2024:{alpha}") is not None for alpha in run.request["alphas"])
        return original_chains(repository, snapshot, keys)
    monkeypatch.setattr(market, "load_chain_index", chains)
    books = {alpha: {arm: [] for arm in (*PRIMARY_ARMS, "incumbent", "gate")} for alpha in run.request["alphas"]}
    quotes, opportunities, sections = {alpha: {} for alpha in books}, [], []
    calendar = trading_calendar_from_snapshot(case.repository, case.quotes, extend_days=0)
    with no_fit_guard(), legacy_no_fit():
        pipeline.evaluate_outer(run, case.plan["folds"][0], fits, results, plan, books, quotes, opportunities, sections, calendar)
        report = pipeline.finish_reports(run, books, quotes, opportunities, sections)
    assert opened == case.target_objects and len(opened) == 1
    assert requests == [("2024-06-14",)] + [("2024-06-17", "2024-06-18")] * 3
    assert len(opportunities) == 1
    scores = [row for section in sections if section["name"].startswith("scores:")
        for row in iter_section(case.store, section["name"], section["descriptor"], lambda row: (row["position"],))]
    assert len(scores) == 9 and {row["horizon"] for row in scores} == {1, 3, 5}
    assert all(row["observation"] == 8 and row["status"] == "available" for row in scores)
    for item in report["reports"]:
        assert item["report"]["arms"]["histogram"]["funded"] == 1
        assert item["report"]["arms"]["histogram"]["settled_net_profit"] == 100
        assert item["report"]["contrasts"]["same_mean"]["net_delta_per_opportunity"] == 0
    assert report["promotion_authority"] == "none" and report["current_strategy_superiority_claim"] is False
