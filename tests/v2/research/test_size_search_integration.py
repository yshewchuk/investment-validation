"""Synthetic native source→stored panel→v2 journal→bounded child→comparison path."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import SystemClock, content_hash, from_document
from engine.v2.models.adapters import RuntimeFitForbidden
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.errors import OpsError
from engine.v2.research.magnitude_panel_qualified_table import recipe
from experiments import eligible_history, size_panel
from experiments import magnitude_search_plan as planner
from experiments import size_search_binding as binding
from experiments.native_registration import reserve_native_registration, verify_native_registration
from experiments.size_search_report import collect
from experiments.v2_magnitude_search_run import run_search, seal_search_run
from tests.v2.research.magnitude_trial_support import poison_outcomes, prediction_spec
from tests.v2.research.test_magnitude_panel_qualified import panel_data
from tests.v2.research.test_magnitude_search_request import request as request_fixture

# Share the bounded native journal fixture instead of rebuilding it on each xdist worker.
pytestmark = pytest.mark.xdist_group("size_search_integration")


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    root = tmp_path_factory.mktemp("size-v2")
    generator = panel_data.__wrapped__(root, SimpleNamespace())
    data = next(generator)
    ids = [e["event_id"] for e in data.events if str(e["event_date"].date()) in
           ("2022-06-02", "2023-01-03", "2023-06-02", "2024-06-17")]
    history_request = {**data.request, "limits": data.panel_request["limits"], "event_ids": ids}
    history = eligible_history.materialize(data.repository, data.panel_source, **history_request)
    panel_request = {**data.panel_request, "event_ids": ids, "history_snapshot": history["feature_snapshot"]}
    panel = size_panel.materialize(data.repository, data.panel_source, **panel_request)
    panel_request = {**panel_request, "feature_snapshot_id": panel["feature_snapshot"].snapshot_id,
        "history_snapshot": history["feature_snapshot"].snapshot_id,
        "price_history_snapshot": data.prices.snapshot_id}
    request = request_fixture()
    request.update(feature_dictionary_hash=recipe("HMO")["dictionary_hash"],
        feature_snapshot_id=panel["feature_snapshot"].snapshot_id, history_snapshot_id=history["feature_snapshot"].snapshot_id,
        outer_years=["2024"], inner_years={"2024": ["2023"]}, max_fits=102)
    request["eligibility_policy_ref"].update(plan_id=panel["context"]["eligibility_plan"]["plan_id"])
    request["limits"] = {"child": {"memory_bytes": 2 << 30, "cpu_seconds": 30, "wall_seconds": 60},
        "parent": {"memory_bytes": 3 << 30, "wall_seconds": 600},
        "build": {"memory_bytes": 3 << 30, "wall_seconds": 60}, "job": {"wall_seconds": 600}}
    spec = prediction_spec(experiment_id="synthetic-qualified-size", folds=("2024",), runner="experiments/size_search.py")
    args = dict(scope="panel-source", training_start="2022-01-01", labels_before="2025-01-01")
    registration, spec = binding.prepare(data.repository, spec, request, panel_request, **args)
    store = data.repository._store
    reserve_native_registration(data.conn, store, registration, spec)
    sealed = planner.seal_search_plan(data.conn, store, registration, request)
    ref = from_document(ArtifactRef, sealed["plan_ref"])
    context = (data.conn, store, data.repository, registration, request, ref)
    seal_search_run(*context, clock=SystemClock())
    result = SimpleNamespace(**{**vars(data), "root": root})
    result.__dict__.update(panel=panel, request=request, panel_request=panel_request,
        spec=spec, registration=registration, store=store, plan=sealed["plan"], ref=ref, context=context, args=args)
    yield result
    try:
        next(generator)
    except StopIteration:
        pass


def test_native_v16_anchor_and_explicit_v2_fitted_identities(case):
    verify_native_registration(case.conn, case.store, case.registration)
    plan = case.plan
    assert plan["schema_version"] == "magnitude_search_plan.v2"
    assert plan["candidate_count"] == 33 and plan["planned_fit_count"] == 102
    assert plan["counts"]["total_component_fits"] == 108
    assert plan["horizons"] == [5, 3, 1]
    assert case.registration.document["size_comparison"]["component_parameters"]["mlp"]["random_state"] == 20260829
    assert len({c["candidate_id"] for c in plan["candidates"]}) == 33
    assert all(c["candidate_id"] == content_hash({k: v for k, v in c.items() if k != "candidate_id"})
               and c["source_binding_id"] == content_hash(case.registration.document["size_panel"]) for c in plan["candidates"])
    assert {len(c["recipe"]["features"]) for c in plan["candidates"]} == {11, 28, 43}


def test_preflight_missing_inner_year_refuses_before_reservation(case):
    changed = deepcopy(case.request)
    changed["inner_years"] = {"2024": ["2020"]}
    before = case.conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0]
    with pytest.raises(Exception):
        binding.prepare(case.repository, case.spec, changed, case.panel_request, **case.args)
    assert case.conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == before


def test_replay_stale_producer_and_request_mismatch_refuse(case, monkeypatch):
    monkeypatch.setattr(binding, "source_closure", lambda *a: {})
    with pytest.raises(OpsError, match="differs"):
        planner.read_search_plan(case.conn, case.store, case.registration, case.request, case.ref)


def test_no_fit_probe_does_not_claim_comparator(case):
    from experiments.size_comparison_run import execute
    before = case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0]
    fold = {**case.plan["folds"][0]["outer"], "evaluation_labels_before": "2025-01-01"}
    with no_fit_guard(), pytest.raises(RuntimeFitForbidden):
        execute(case.conn, case.store, case.repository, case.registration, case.plan, fold,
                limits=case.request["limits"]["child"])
    assert case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0] == before


def test_complete_synthetic_serial_search_child_diagnostics(case, monkeypatch):
    poison_outcomes(monkeypatch, case.repository, case.panel_source, ["2023-03-20", "2023-04-03", "2023-05-15"])
    for producer in (size_panel, eligible_history):
        monkeypatch.setattr(producer, "materialize", lambda *a, **k: pytest.fail("hidden feature rebuild"))
    import experiments.magnitude_search_outer as outer
    import experiments.magnitude_search_trials as trials
    import experiments.size_comparison_run as comparison
    jobs = {"inner_candidates": 0, "outer_challengers": 0, "outer_comparators": 0}
    for module, role in ((trials, "inner_candidates"), (outer, "outer_challengers"), (comparison, "outer_comparators")):
        original = module.run_fitted_trial
        def counted(*args, _original=original, _role=role, **kwargs):
            jobs[_role] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, "run_fitted_trial", counted)
    first = run_search(*case.context, limits=case.request["limits"]["child"])
    report = collect(*case.context)
    assert report["search"] == first
    assert jobs == {"inner_candidates": 33, "outer_challengers": 1, "outer_comparators": 1}
    assert report["preflight"]["child_jobs"] == jobs
    assert report["preflight"]["maximum_child_invocations"] == sum(jobs.values()) == 35
    assert len(report["inner_fits"]) == 33
    assert sum(f["completed_fit_count"] for f in report["inner_fits"]) == 99
    assert sum(f["component_fits"] for f in report["comparison_fits"]) == 0
    assert sum(len(f["diagnostics"]) for f in report["inner_fits"]) == 99
    assert report["paired"]["eligible_events"] == 1
    assert report["paired"]["comparison_status"] == "unavailable_no_complete_matched_events"
    assert report["paired"]["broad_coverage_metrics"]["history_mean"]["count"] == 3
    from experiments.magnitude_search_records import read_record
    diagnostics = []
    for candidate in case.plan["candidates"]:
        key = content_hash([case.plan["plan_id"], "2024", "2023", candidate["candidate_id"]])
        receipt = read_record(case.conn, case.store, case.registration.run_id, "execution:" + key)
        assert receipt["fit_attempt_count"] == 3
        if candidate["recipe"]["model"] == "elasticnet":
            diagnostics.extend(receipt["diagnostics"])
    assert len(diagnostics) == 54 and all(d["converged"] for d in diagnostics)


def test_completed_search_exact_and_expired_replay(completed_case, monkeypatch):
    case = completed_case
    poison_outcomes(monkeypatch, case.repository, case.panel_source, ["2023-03-20", "2023-04-03", "2023-05-15"])
    for producer in (size_panel, eligible_history):
        monkeypatch.setattr(producer, "materialize", lambda *a, **k: pytest.fail("hidden feature rebuild"))
    report = collect(*case.context)
    first = report["search"]
    import experiments.magnitude_search_outer as outer
    import experiments.magnitude_search_trials as trials
    import experiments.size_comparison_run as comparison
    from experiments.magnitude_search_records import read_record
    for module in (trials, outer, comparison):
        monkeypatch.setattr(module, "run_fitted_trial", lambda *a, **k: pytest.fail("retry fit"))
    assert run_search(*case.context, limits=case.request["limits"]["child"]) == first
    assert collect(*case.context) == report
    from datetime import timedelta

    from engine.v2.foundation import parse_timestamp
    from experiments import size_job_budget, size_search
    saved = read_record(case.conn, case.store, case.registration.run_id, "size-job-deadline")
    frozen = SimpleNamespace(now=lambda: parse_timestamp(saved["deadline"]) + timedelta(seconds=1))
    monkeypatch.setattr(size_job_budget, "SystemClock", lambda: frozen)
    monkeypatch.setattr(size_search, "run_search", lambda *a, **k: pytest.fail("completed CLI dispatched"))
    before = case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0]
    assert size_search._run(case.context) == report
    assert case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0] == before



@pytest.mark.parametrize("field", ["size_request", "size_panel_request", "size_panel", "size_comparison"])
def test_native_v16_missing_or_nondictionary_additive_field_refuses(case, field):
    from engine.v2.ops.native_registration import _magnitude_binding
    for bad in (None, [], "untrusted"):
        document = case.registration.document
        document[field] = bad
        assert _magnitude_binding(document) is False
    document = case.registration.document
    del document[field]
    assert _magnitude_binding(document) is False


@pytest.mark.parametrize("field", ["memory_bytes", "cpu_seconds", "wall_seconds"])
def test_public_comparator_limit_drift_refuses_before_inputs(case, monkeypatch, field):
    from experiments import size_comparison_run as comparison
    monkeypatch.setattr(comparison, "fold_inputs", lambda *a: pytest.fail("opened inputs"))
    limits = dict(case.request["limits"]["child"])
    limits[field] += 1
    fold = {**case.plan["folds"][0]["outer"], "evaluation_labels_before": "2025-01-01"}
    before = case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0]
    with pytest.raises(OpsError, match="differs"):
        comparison.execute(case.conn, case.store, case.repository, case.registration, case.plan, fold, limits=limits)
    assert case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0] == before


@pytest.fixture(scope="module")
def completed_case(case):
    """Explicit setup: standalone poison selections also get a completed journal."""
    run_search(*case.context, limits=case.request["limits"]["child"])
    return case


@pytest.mark.parametrize("slot,mutation", [("start", "missing"), ("start", "changed"),
    ("inputs", "missing"), ("inputs", "changed"), ("terminal", "missing"),
    ("terminal", "result"), ("started:1", "missing"), ("completed:1", "missing"),
    ("started:1", "changed"), ("completed:1", "changed"),
    ("terminal", "missing_result"), ("terminal", "result_none")])
def test_report_replays_comparator_evidence(completed_case, monkeypatch, slot, mutation):
    case = completed_case
    from datetime import timedelta

    from engine.v2.foundation import parse_timestamp
    from experiments import size_job_budget
    from experiments.magnitude_search_records import read_record
    saved = read_record(case.conn, case.store, case.registration.run_id, "size-job-deadline")
    frozen = SimpleNamespace(now=lambda: parse_timestamp(saved["deadline"]) + timedelta(seconds=1))
    monkeypatch.setattr(size_job_budget, "SystemClock", lambda: frozen)

    from experiments import magnitude_search_outer as outer
    from experiments import magnitude_search_trials as trials
    from experiments import size_comparison_run as comparison
    key = content_hash([case.plan["plan_id"], "2024", "comparison"])
    target = "comparison-" + slot + ":" + key
    original = comparison.read_record
    observed = []
    def read(*args):
        value = original(*args)
        if args[-1] == target:
            observed.append(target)
            if mutation == "missing":
                return None
            value = deepcopy(value)
            if mutation == "missing_result":
                del value["result"]
            elif mutation == "result_none":
                value["result"] = None
            elif mutation == "result":
                value["result"]["forecasts"][0]["history_mean"] = 987654.
            else:
                value["planted_mismatch"] = True
        return value
    monkeypatch.setattr(comparison, "read_record", read)
    for module in (comparison, outer, trials):
        monkeypatch.setattr(module, "run_fitted_trial", lambda *a, **k: pytest.fail("report launched fit"))
    for name in ("append_record", "publish_inputs", "resolved_limits"):
        monkeypatch.setattr(comparison, name, lambda *a, **k: pytest.fail("report wrote or claimed budget"))
    before = case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0]
    with pytest.raises(OpsError) as caught:
        collect(*case.context)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert observed == [target]
    assert case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0] == before


@pytest.mark.parametrize("headroom", [-1, 0, 1])
def test_prepared_registration_byte_boundary_matches_reserve_and_verify(case, monkeypatch, headroom):
    from engine.v2.foundation import canonical_json
    from experiments.native_registration import read_native_registration
    expected = case.registration.binding_json
    budget = len(expected) + headroom
    monkeypatch.setattr(binding, "MAX_NATIVE_EVIDENCE_BYTES", budget)
    before = case.conn.total_changes
    if headroom < 0:
        monkeypatch.setattr(binding, "_folds", lambda *a: pytest.fail("oversize binding reached fold work"))
        for action in (
            lambda: binding.prepare(case.repository, case.spec, case.request, case.panel_request, **case.args),
            lambda: reserve_native_registration(case.conn, case.store, case.registration, case.spec,
                                                 max_evidence_bytes=budget),
        ):
            with pytest.raises(OpsError) as caught:
                action()
            assert caught.value.code == "RESOURCE_LIMIT_EXCEEDED"
            assert caught.value.problem.message == "native registration exceeds its UTF-8 byte budget"
            assert caught.value.problem.details == {"registration_bytes": len(expected), "max_registration_bytes": budget}
        for action in (
            lambda: verify_native_registration(case.conn, case.store, case.registration, max_evidence_bytes=budget),
            lambda: read_native_registration(case.conn, case.store, case.spec, max_evidence_bytes=budget),
        ):
            with pytest.raises(OpsError) as caught:
                action()
            assert caught.value.code == "RESOURCE_LIMIT_EXCEEDED"
            assert caught.value.problem.message == "native registration exceeds its UTF-8 byte budget"
            assert caught.value.problem.details == {"registration_bytes": len(expected), "max_registration_bytes": budget}
    else:
        prepared, spec = binding.prepare(case.repository, case.spec, case.request, case.panel_request, **case.args)
        assert prepared.binding_json == expected == canonical_json(prepared.document).encode()
        assert reserve_native_registration(case.conn, case.store, prepared, spec,
                                            max_evidence_bytes=budget) == prepared
        verify_native_registration(case.conn, case.store, prepared, max_evidence_bytes=budget)
        assert read_native_registration(case.conn, case.store, spec, max_evidence_bytes=budget) == prepared
        assert binding.prepare_plan(case.conn, case.store, prepared, case.request) == case.plan
    assert case.conn.total_changes == before


@pytest.mark.parametrize("kind", ["execution", "terminal", "outer-forecasts", "comparison-terminal"])
def test_report_native_absent_record_refuses_before_presentation(completed_case, monkeypatch, kind):
    from experiments import magnitude_search_outer as outer
    from experiments import magnitude_search_records as records
    from experiments import magnitude_search_trials as trials
    from experiments import size_comparison_run as comparison
    from experiments import size_search_report as presentation
    case = completed_case
    group, candidate = case.plan["folds"][0], case.plan["candidates"][0]
    identity = [case.plan["plan_id"], group["outer"]["year"]]
    if kind in ("execution", "terminal"):
        identity += [group["inner"][0]["year"], candidate["candidate_id"]]
    elif kind == "comparison-terminal":
        identity += ["comparison"]
    target = kind + ":" + content_hash(identity)
    native_read, seen = records.read_record, []
    assert native_read(case.conn, case.store, case.registration.run_id, target) is not None
    assert native_read(case.conn, case.store, case.registration.run_id, "never-published") is None
    def read(conn, store, run_id, slot):
        if slot == target:
            seen.append(slot)
            slot = "never-published"
        return native_read(conn, store, run_id, slot)
    def forbidden(*a, **k):
        pytest.fail("read-only report performed work or reached unvalidated presentation")
    for module in (outer, trials, comparison, presentation):
        monkeypatch.setattr(module, "read_record", read)
    for module in (outer, trials, comparison):
        for name in ("run_fitted_trial", "publish_inputs"):
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(records, "transaction", forbidden)
    monkeypatch.setattr(case.store, "publish_bytes", forbidden)
    monkeypatch.setattr(presentation, "paired_report", forbidden)
    before = case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0]
    with pytest.raises(OpsError) as caught:
        collect(*case.context)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert seen == [target]
    assert case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0] == before


@pytest.mark.parametrize("kind", ["inner", "outer"])
def test_public_fold_inputs_accepts_exact_registered_fold(case, monkeypatch, kind):
    poison_outcomes(monkeypatch, case.repository, case.panel_source, ["2023-03-20", "2023-04-03", "2023-05-15"])
    group = case.plan["folds"][0]
    fold = group["inner"][0] if kind == "inner" else {**group["outer"],
        "evaluation_labels_before": case.registration.document["magnitude_execution"]["evaluation_labels_before"]}
    inputs = binding.fold_inputs(case.repository, case.registration.document, fold)
    training = {row["event_id"] for row in inputs["training"]}
    assert training == set(fold["training_candidate_event_ids"])
    assert training.isdisjoint(fold["evaluation_event_ids"])
    assert {row["event_id"] for row in inputs["features"]} == training | set(fold["evaluation_event_ids"])


@pytest.mark.parametrize("change", ["cutoff", "training", "evaluation", "order", "year", "evaluation_cutoff", "extra", "missing", "none", "evaluation_in_training"])
def test_public_fold_inputs_refuses_caller_drift_before_dependent_reads(case, monkeypatch, change):
    fold = {**deepcopy(case.plan["folds"][0]["outer"]),
        "evaluation_labels_before": case.registration.document["magnitude_execution"]["evaluation_labels_before"]}
    if change in ("cutoff", "evaluation_in_training"):
        fold["training_labels_before"] = "2025-01-01"
    if change in ("training", "evaluation_in_training"):
        fold["training_candidate_event_ids"] += fold["evaluation_event_ids"]
    elif change == "evaluation":
        fold["evaluation_event_ids"] = fold["training_candidate_event_ids"][:1]
    elif change == "order":
        assert len(fold["training_candidate_event_ids"]) > 1
        fold["training_candidate_event_ids"].reverse()
    elif change == "year":
        fold["year"] = "2025"
    elif change == "evaluation_cutoff":
        fold["evaluation_labels_before"] = "2026-01-01"
    elif change == "extra":
        fold["caller_authority"] = True
    elif change == "missing":
        del fold["training_labels_before"]
    elif change == "none":
        fold = None
    def forbidden(*a, **k):
        pytest.fail("unregistered fold reached dependent panel or outcome reads")
    for name in ("_read_panel", "preflight_magnitude_targets", "qualify_magnitude_training_targets"):
        monkeypatch.setattr(binding, name, forbidden)
    before = case.conn.total_changes
    with pytest.raises(OpsError) as caught:
        binding.fold_inputs(case.repository, case.registration.document, fold)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.retryable is False
    assert case.conn.total_changes == before
