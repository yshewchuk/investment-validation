"""Native signed fold admission and existing sharded transport, without fitting."""
from copy import deepcopy

import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import eligible_history, iv_crush_search, size_panel
from experiments import magnitude_search_evidence as evidence
from experiments import magnitude_search_inputs as inputs
from experiments import magnitude_search_plan as plans
from experiments import native_registration as native
from tests.v2.research.magnitude_trial_support import catalog_rows, poison_outcomes
from tests.v2.research.test_iv_crush_inputs import poison, target
from tests.v2.research.test_iv_crush_registration import prepared as prepare_fixture
from tests.v2.research.test_iv_crush_search_plan import request as search_request
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401
from tests.v2.research.test_native_registration import artifact_object_bytes


@pytest.fixture
def fold_case(panel_data, monkeypatch, request):  # noqa: F811
    with monkeypatch.context() as admission:
        admission.setattr("tests.v2.research.test_iv_crush_registration.target",
            lambda event: target(event, value=-140., post_iv30=-20.))
        data = prepare_fixture.__wrapped__(panel_data, admission)
    # Explicit development cohort excludes the fixture's missing bound and random-window embargo.
    ids = [event for event in data.registration.document["event_ids"]
           if not any(day in event for day in (":2023-05-15:", ":2023-03-20:"))]
    history_request = {**data.request, "event_ids": ids, "limits": data.panel_request["limits"]}
    history = eligible_history.materialize(data.repository, data.panel_source, **history_request)
    data.panel_request = {**data.panel_request, "event_ids": ids, "history_snapshot": history["feature_snapshot"]}
    panel = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    data.features = panel["feature_snapshot"]
    data.registration, data.spec = iv_crush_search.prepare(data.repository, data.spec,
        **{**data.options, "feature_snapshot": data.features, "event_ids": ids})
    data.registration = native.reserve_native_registration(data.conn, data.store, data.registration, data.spec)
    document = search_request(data)
    policy = panel["context"]["eligibility_plan"]
    document["eligibility_policy_ref"] = {"schema_ref": "group_eligibility_plan.v1",
        "plan_id": policy["plan_id"], "policy_id": policy["policy"]["policy_id"]}
    data.plan = plans.seal_search_plan(data.conn, data.store, data.registration, document)["plan"]
    group = data.plan["folds"][0]
    data.fold = ({**group["outer"], "evaluation_labels_before": "2025-01-01"}
                 if getattr(request, "param", None) == "outer" else group["inner"][0])
    poison_outcomes(monkeypatch, data.repository, data.panel_source, (), all_outcomes=True)
    training = set(data.fold["training_candidate_event_ids"])
    protected_days = [event["event_date"].date().isoformat() for event in data.events if event["event_id"] not in training]
    poison(monkeypatch, data.repository, data.targets, protected_days)
    return data


def admit(data, **changes):
    return evidence.admit_fold_inputs(data.repository, data.registration.document, data.fold,
        history_failure=fail("EXPERIMENT_IDENTITY_CONFLICT", "history changed"), plan=data.plan, **changes)


def test_real_fold_roundtrip_and_target_projection(fold_case, monkeypatch):
    data = fold_case
    cohort = admit(data)
    assert cohort["target"]["column"] == "crush_pct_iv30"
    assert cohort["training"] == [{"event_id": data.fold["training_candidate_event_ids"][0], "crush_pct_iv30": -140.}]
    assert cohort["preflight"]["snapshot_id"] == data.targets.snapshot_id != data.panel_source.snapshot_id
    assert cohort["training_inputs"]["labels_before"] == data.fold["training_labels_before"]
    assert {row["event_id"] for row in cohort["features"]} == set(data.fold["training_candidate_event_ids"] + data.fold["evaluation_event_ids"])
    assert "candidate_event_ids" not in cohort["training_inputs"]
    manifest, identity = inputs.publish_inputs(data.store, cohort)
    assert inputs.load_inputs(data.store, manifest, identity) == cohort
    assert set(manifest["sections"]) == {"features", "training", "preflight.event_ids", "preflight.planned_inputs",
        "training_inputs.consumed_inputs", "training_inputs.excluded_future_events", "training_inputs.holdout_exclusions"}
    blocked = {entry["ref"]["artifact_id"] for name, section in manifest["sections"].items()
               if name not in {"features", "training"} for entry in section["shards"]}
    original = data.store.read_verified
    def only_fit(ref):
        assert ref.artifact_id not in blocked, "fit projection opened evidence-only shard"
        return original(ref)
    monkeypatch.setattr(data.store, "read_verified", only_fit)
    projected = inputs.load_fit_inputs(data.store, manifest, identity, data.fold["evaluation_event_ids"],
        [5, 3, 1], ["n_prior"], target_column="crush_pct_iv30")
    assert projected["training"] == cohort["training"]
    assert all(set(row) == {"event_id", "context_id", "horizon_sessions", "n_prior"} for row in projected["features"])


@pytest.mark.parametrize("fold_case", ["outer"], indirect=True)
def test_normalized_outer_fold_reads_only_its_earlier_training(fold_case):
    data = fold_case
    cohort = admit(data)
    assert cohort["training_inputs"]["labels_before"] == "2024-01-01"
    assert [row["event_id"] for row in cohort["training"]] == data.fold["training_candidate_event_ids"]
    assert len(cohort["training"]) == 3 and all(row["crush_pct_iv30"] == -140. for row in cohort["training"])
    assert len(cohort["features"]) == 12


@pytest.mark.parametrize("change", ["fold", "target", "policy"])
def test_changed_fold_or_binding_refuses_before_training(fold_case, monkeypatch, change):
    data = fold_case
    from engine.v2.research import iv_crush_inputs
    monkeypatch.setattr(iv_crush_inputs, "load_iv_crush_targets", lambda *a, **k: pytest.fail("training opened"))
    if change == "fold":
        data.fold = {**data.fold, "training_labels_before": "2024-01-01"}
    else:
        data.plan = deepcopy(data.plan)
        if change == "target":
            data.plan["target"]["domain"] = "nonnegative"
        else:
            data.plan["request"]["eligibility_policy_ref"]["plan_id"] = "sha256:" + "0" * 64
    with pytest.raises(OpsError) as caught:
        admit(data)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("reader", ["fit", "whole"])
def test_wrong_target_pin_refuses_before_any_shard(fold_case, monkeypatch, missing, reader):
    data = fold_case
    manifest, identity = inputs.publish_inputs(data.store, admit(data))
    before, saved = catalog_rows(data.conn), artifact_object_bytes(data.store)
    monkeypatch.setattr(data.store, "read_verified", lambda *a: pytest.fail("shard opened"))
    monkeypatch.setattr(data.store, "publish_bytes", lambda *a, **k: pytest.fail("CAS write"))
    expected = fail("EXPERIMENT_IDENTITY_CONFLICT", "search fit target differs from its cohort").problem
    if reader == "fit":
        for column in ("absolute_move_pct", "other"):
            with pytest.raises(OpsError) as caught:
                inputs.load_fit_inputs(data.store, manifest, identity, [], [], [], target_column=column)
            assert caught.value.problem == expected
    altered = deepcopy(manifest)
    if missing:
        del altered["inline"]["target.column"]
    else:
        altered["inline"]["target.column"] = "absolute_move_pct"
    with pytest.raises(OpsError) as caught:
        if reader == "fit":
            inputs.load_fit_inputs(data.store, altered, content_hash(altered), [], [], [], target_column="crush_pct_iv30")
        else:
            inputs.load_inputs(data.store, altered, content_hash(altered))
    assert caught.value.problem == expected and caught.value.problem.retryable is False
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == saved


@pytest.mark.parametrize("target_pin", [
    pytest.param("absent", id="missing-target"),
    pytest.param({}, id="missing-column"),
    pytest.param({"column": "absolute_move_pct"}, id="wrong-column"),
    pytest.param({"column": None}, id="null-column"),
    pytest.param(None, id="null-target"),
    pytest.param("crush_pct_iv30", id="scalar-target"),
    pytest.param([], id="list-target"),
])
def test_invalid_iv_target_pin_refuses_before_publication(fold_case, monkeypatch, target_pin):
    data = fold_case
    cohort = admit(data)
    if target_pin == "absent":
        del cohort["target"]
    else:
        cohort["target"] = target_pin
    before, saved = catalog_rows(data.conn), artifact_object_bytes(data.store)
    monkeypatch.setattr(data.store, "read_verified", lambda *a, **k: pytest.fail("CAS read"))
    monkeypatch.setattr(data.store, "publish_bytes", lambda *a, **k: pytest.fail("CAS write"))
    with pytest.raises(OpsError) as caught:
        inputs.publish_inputs(data.store, cohort)
    assert caught.value.problem == fail(
        "EXPERIMENT_IDENTITY_CONFLICT", "search fit target differs from its cohort").problem
    assert caught.value.problem.retryable is False
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == saved


@pytest.mark.parametrize("drift", ["environment", "runner", "worker", "sealed_implementation"])
def test_installed_plan_drift_refuses_before_any_input_read(fold_case, monkeypatch, drift):
    data = fold_case
    from engine.v2.research import iv_crush_inputs
    from experiments import regression_search_binding
    def forbidden(*args, **kwargs):
        pytest.fail("input read preceded installation validation")
    for name in ("preflight_iv_crush_targets", "load_iv_crush_targets"):
        monkeypatch.setattr(iv_crush_inputs, name, forbidden)
    monkeypatch.setattr(regression_search_binding, "read_stored_panel", forbidden)
    if drift == "environment":
        monkeypatch.setattr(plans, "environment_identity", lambda *a: {"changed": True})
        message = "search anchor sources or environment differ from this installation"
    elif drift == "sealed_implementation":
        data.plan = deepcopy(data.plan)
        data.plan["implementation"]["source_closure"] = {}
        data.plan["plan_id"] = content_hash({k: v for k, v in data.plan.items() if k != "plan_id"})
        message = "regression search differs from its native binding"
    else:
        original = plans.source_closure
        path = (data.registration.document["execution_plan"]["runner"] if drift == "runner"
                else "experiments/magnitude_search_worker.py")
        def changed(root, entries):
            result = original(root, entries)
            if path in result:
                result[path] = "sha256:" + "0" * 64
            return result
        monkeypatch.setattr(plans, "source_closure", changed)
        message = ("search anchor sources or environment differ from this installation" if drift == "runner"
                   else "regression search differs from its native binding")
    before = (data.conn.total_changes, sorted(str(p) for p in data.store.root.rglob("*")))
    with pytest.raises(OpsError) as caught:
        admit(data)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", message).problem
    assert before == (data.conn.total_changes, sorted(str(p) for p in data.store.root.rglob("*")))
