"""Native selected outer fits, temporal ordering and pooled independent oracles."""
import hashlib
import math
import shutil
from contextlib import contextmanager
from functools import cache
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from engine.models.no_fit import RuntimeFitForbidden as LegacyRuntimeFitForbidden
from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.contracts import ArtifactRef
from engine.v2.data.repository import Repository
from engine.v2.foundation import (
    ArtifactStore,
    SystemClock,
    canonical_json,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.models.adapters import RuntimeFitForbidden
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.research.magnitude_history_store import read_stored
from experiments import magnitude_search_outer as outer
from experiments import magnitude_search_trials as trials
from experiments.magnitude_search_plan import seal_search_plan
from tests.v2.research.magnitude_trial_support import (
    ROOT,
    history_event,
    poison_outcomes,
    trial_cases,
)
from tests.v2.research.test_magnitude_search_execution import LIMITS, inject_child
from tests.v2.research.test_magnitude_search_plan import source as base_source  # noqa: F401

HORIZONS = [5, 3, 1]
CONFLICT = "outer evaluation evidence differs from its sealed inputs"
HOLDOUTS = {"random": "excluded; not evaluated", "rolling": "excluded; not evaluated"}
TARGET_RECIPE = {"schema_version": "earnings_magnitude.v1", "units": "percentage_points",
    "target": "absolute_earnings_close_to_close_return", "price_basis": "split_neutral_non_dividend_adjusted",
    "estimand": "conditional_mean", "primary_loss": "pooled_rmse",
    "secondary_diagnostics": ["mae", "quantile_calibration"]}


@contextmanager
def outer_problem(code, message, *, category="validation", retryable=False, details=None):
    with pytest.raises(OpsError) as caught:
        yield
    assert to_document(caught.value.problem) == {
        "schema_version": "problem.v1.0", "code": code, "category": category,
        "retryable": retryable, "message": message, "stage": None, "trace_id": None,
        "dependency_refs": [], "retry_after_seconds": None, "diagnostic_ref": None, "details": details or {}}


@cache
def known_targets():
    """Fixture-owned truth, independent of recorded validation or returned predictions."""
    days = ("2022-06-02", "2022-12-22", "2022-12-23", "2023-01-03", "2023-06-02",
            "2024-06-12", "2024-06-17", "2024-12-27", "2024-12-30", "2025-01-07")
    values = {"SYNTH_A": (2, 3, 3, 4, 6, 10, 8, 3, 3, 12),
              "SYNTH_B": (2, 3, 3, 8, 12, 20, 4, 3, 3, 16)}
    return {history_event(day, ticker=ticker)["event_id"]: float(value)
            for ticker, moves in values.items() for day, value in zip(days, moves)}


def journal_slots(case):
    return [tuple(row) for row in case.conn.execute(
        "SELECT slot, ref_json FROM magnitude_search_records ORDER BY slot")]


def assert_source_seals(case, year="2024"):
    shared = {"experiments/magnitude_search_evidence.py", "experiments/magnitude_search_trials.py",
              "experiments/magnitude_search_execution.py", "experiments/magnitude_search_worker.py"}
    for sources, required in ((case.recorder._read("seal")["recorder_sources"], shared),
            (slot(case, "start", year)["outer_sources"], shared | {"experiments/magnitude_search_outer.py"})):
        assert sources.keys() >= required
        assert sources == {name: "sha256:" + hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                           for name in sources}


def independent_metrics(rows):
    errors = [abs(row["predicted_absolute_move_pct"] - known_targets()[row["event_id"]]) for row in rows]
    return {"count": len(errors), "mae": pytest.approx(math.fsum(errors) / len(errors)),
            "rmse": pytest.approx(math.sqrt(math.fsum(error ** 2 for error in errors) / len(errors)))}


def expected_training_receipt(case, year):
    """Pin fixture membership and physical identities without journal or qualifier output."""
    days = ["2022-06-02", "2022-12-22", "2022-12-23", "2023-01-03", "2023-06-02"]
    if year == "2025":
        days += ["2024-06-12", "2024-06-17"]
    events = sorted([history_event(day, ticker=ticker) for day in days for ticker in ("SYNTH_A", "SYNTH_B")],
                    key=lambda event: (event["event_date"], event["event_id"]))
    records = {(record.partition_key, record.time_min): record for record in
               case.recorder.repository.fragment_records(case.snapshot, "computed_moves")}
    physical = []
    for event in events:
        record = records[event["ticker"], event["event_date"].date().isoformat()]
        physical.append({"event_id": event["event_id"], "fragment_id": record.fragment_id,
            "object_id": record.object_ref.object_id, "content_hash": record.object_ref.content_hash})
    keys = [event["event_id"] for event in events]
    identity = {"schema_version": "research_magnitude_training_targets.v1",
        "snapshot_id": case.snapshot.snapshot_id, "holdout_as_of_month": "2026-01",
        "random_membership_version": "canonical-event-sha256.v1", "rolling_membership_version": "calendar-months.v1",
        "population_use": "post-release selection", "purpose": "selection",
        "computed_moves_version": case.snapshot.table_versions["computed_moves"].dataset_version_id,
        "labels_before": {"2024": "2024-01-01", "2025": "2024-12-27"}[year],
        "target_recipe": TARGET_RECIPE, "target_recipe_id": content_hash(TARGET_RECIPE),
        "candidate_event_ids": keys, "qualified_event_ids": keys, "read_inputs": physical, "omitted_labels": []}
    return {**identity, "inputs_id": content_hash(identity), "holdout_exclusions": []}


def assert_fold_oracle(case, result, year):
    targets = {event: target for event, target in known_targets().items() if event.split(":")[1][:4] == year}
    forecasts = slot(case, "forecasts", year)
    indexed = {(row["event_id"], row["horizon_sessions"]): row["predicted_absolute_move_pct"] for row in forecasts}
    expected_predictions = [{"event_id": event, "horizon_sessions": horizon,
        "predicted_absolute_move_pct": indexed[event, horizon], "absolute_move_pct": targets[event]}
        for event in sorted(targets) for horizon in sorted(HORIZONS)]
    assert forecasts == [{key: value for key, value in row.items() if key != "absolute_move_pct"}
                         for row in expected_predictions]
    selection = next(item for item in case.selected["selections"] if item["outer_year"] == year)
    candidate = next(item for item in case.plan["candidates"]
                     if item["candidate_id"] == selection["selected_candidate_id"])
    started = slot(case, "start", year)
    assert_source_seals(case, year)
    expected_start = {"outer_id": content_hash([case.plan["plan_id"], year]), "plan_id": case.plan["plan_id"],
        "outer_year": year, "selection_id": content_hash(case.selected), "candidate": candidate,
        "limits": LIMITS, "outer_sources": started["outer_sources"],
        "prospective_candidate_id": case.selected["prospective_candidate_id"]}
    assert canonical_json(started) == canonical_json(expected_start)
    inputs, execution, validation = (slot(case, name, year) for name in ("inputs", "execution", "validation"))
    assert execution["fits"] == expected_fits(case, year)
    assert {row["event_id"]: row["absolute_move_pct"] for row in validation["targets"]} == targets
    training_inputs = expected_training_receipt(case, year)
    assert inputs["training_inputs"] == training_inputs
    assert validation["evaluation_inputs"]["labels_before"] == "2025-02-01"
    assert validation["evaluation_inputs"]["purpose"] == "selection"
    assert case.plan["horizons"] == HORIZONS
    expected = {"schema_version": "magnitude_search_outer_result.v1",
        "outer_id": content_hash([case.plan["plan_id"], year]), "plan_id": case.plan["plan_id"],
        "outer_year": year, "candidate": candidate, "selection_id": content_hash(case.selected),
        "inputs_id": content_hash(inputs), "execution_id": content_hash(execution),
        "validation_id": content_hash(validation), "training_inputs": training_inputs,
        "evaluation_inputs": validation["evaluation_inputs"], "fit_attempt_count": 3,
        "event_count": {"2024": 8, "2025": 2}[year], "forecast_count": {"2024": 24, "2025": 6}[year],
        "predictions": expected_predictions, "metrics": independent_metrics(expected_predictions),
        "per_horizon": [{"horizon_sessions": horizon, **independent_metrics([
            row for row in expected_predictions if row["horizon_sessions"] == horizon])} for horizon in HORIZONS],
        "prospective_candidate_id": case.selected["prospective_candidate_id"],
        "evaluation_authority": "selected_outer_fold_only", "final_read_authority": "none",
        "holdout_results": HOLDOUTS}
    assert result == {**expected, "result_id": content_hash({key: value for key, value in result.items()
                                                          if key != "result_id"})}
    return expected_predictions


def native_matrices(case, year="2024"):
    """Rebuild matrices from native rows captured before any outer admission."""
    fold = next(item["outer"] for item in case.plan["folds"] if item["outer"]["year"] == year)
    train = fold["training_candidate_event_ids"]
    features = case.recorder.registration.document["magnitude_execution"]["features"]
    indexed = {(row["event_id"], row["horizon_sessions"]): row for row in case.native_inputs["rows"]}
    result = []
    for horizon in HORIZONS:
        matrices = [{"columns": features,
            "row_keys": [[event, indexed[event, horizon]["context_id"], horizon] for event in events],
            "values": [[indexed[event, horizon][name] for name in features] for event in events]}
            for events in (train, fold["evaluation_event_ids"])]
        matrices[0]["absolute_move_pct"] = [known_targets()[event] for event in train]
        result.append(matrices)
    return result


def expected_fits(case, year="2024"):
    """Pin journal inputs and fit identities to independently read native evidence."""
    inputs = slot(case, "inputs", year)
    fold = next(item["outer"] for item in case.plan["folds"] if item["outer"]["year"] == year)
    train = fold["training_candidate_event_ids"]
    participants = set(train + fold["evaluation_event_ids"])
    assert inputs["stored_inputs_id"] == case.native_inputs["inputs_id"]
    assert canonical_json(inputs["features"]) == canonical_json([
        row for row in case.native_inputs["rows"] if row["event_id"] in participants])
    assert inputs["training"] == [{"event_id": event, "absolute_move_pct": known_targets()[event]} for event in train]
    result = []
    for horizon, matrices in zip(HORIZONS, native_matrices(case, year)):
        result.append({"horizon_sessions": horizon, "training_event_ids": train,
            "evaluation_event_ids": fold["evaluation_event_ids"], "training_matrix_id": content_hash(matrices[0]),
            "prediction_matrix_id": content_hash(matrices[1])})
    return result


@pytest.fixture(scope="module")
def native_outer_source(base_source, tmp_path_factory):  # noqa: F811
    path = tmp_path_factory.mktemp("outer-native")
    conn = connect(path / "catalog.sqlite")
    base_source.conn.backup(conn)
    shutil.copytree(base_source.store.root, path / "store")
    store = ArtifactStore(path / "store")
    request = {"schema_version": "magnitude_search_request.v1",
        "inner_years": {"2024": ["2023"], "2025": ["2023", "2024"]},
        "space": {"history_ridge": {"alpha": [0.5, 2]}}, "max_candidates": 2, "max_fits": 24}
    sealed = seal_search_plan(conn, store, base_source.registration, request)
    recorder = trials.SearchTrials(conn, store, Repository(conn, store), base_source.registration,
                                  request, from_document(ArtifactRef, sealed["plan_ref"]))
    recorder.seal(clock=SystemClock())
    native_inputs = read_stored(base_source.repository, base_source.feature_snapshot, base_source.build["context"])
    case = SimpleNamespace(recorder=recorder, plan=sealed["plan"], native_inputs=native_inputs)
    for args in trial_cases(case):
        recorder.execute(*args, limits=LIMITS)
    selected = recorder.freeze()
    yield SimpleNamespace(**vars(case), conn=conn, store=store, request=request, selected=selected,
                          snapshot=base_source.snapshot)
    conn.close()


@pytest.fixture
def outer_case(native_outer_source, tmp_path):
    original = native_outer_source
    conn = connect(tmp_path / "catalog.sqlite")
    original.conn.backup(conn)
    shutil.copytree(original.store.root, tmp_path / "store")
    store = ArtifactStore(tmp_path / "store")
    recorder = trials.SearchTrials(conn, store, Repository(conn, store), original.recorder.registration,
                                  original.request, original.recorder.plan_ref)
    yield SimpleNamespace(**{**vars(original), "conn": conn, "store": store, "recorder": recorder})
    conn.close()


def run(case, year="2024", **kwargs):
    r = case.recorder
    return outer.execute_outer_fold(case.conn, case.store, r.repository, r.registration,
                                    r.request, r.plan_ref, year, limits=kwargs.get("limits", LIMITS))


def collect(case):
    r = case.recorder
    return outer.collect_outer_results(case.conn, case.store, r.repository, r.registration, r.request, r.plan_ref)


def slot(case, name, year="2024"):
    key = content_hash([case.plan["plan_id"], year])
    return case.recorder._read("outer-" + name + ":" + key)


def forbid_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("replay attempted fitting or native input/label access")
    for module in (outer, trials):
        monkeypatch.setattr(module, "run_fitted_trial", forbidden)
        monkeypatch.setattr(module, "load_magnitude_targets", forbidden)
    monkeypatch.setattr(outer, "_inputs", forbidden)
    monkeypatch.setattr(outer, "execution_limits", forbidden)


def test_real_outer_selected_only_forecasts_precede_labels_and_replay(outer_case, monkeypatch):
    case = outer_case
    poison_outcomes(monkeypatch, case.recorder.repository, case.snapshot, ("2023-04-03", "2025-12-03"))
    original = outer.load_magnitude_targets
    def labels(*args, **kwargs):
        assert slot(case, "forecasts") is not None
        assert slot(case, "execution")["completed_fit_count"] == 3
        assert kwargs["labels_before"] == "2025-02-01"
        assert not case.conn.in_transaction
        return original(*args, **kwargs)
    monkeypatch.setattr(outer, "load_magnitude_targets", labels)
    result = run(case)
    assert_source_seals(case)
    scores = case.selected["selections"][0]["scores"]
    assert scores[0]["metrics"] == scores[1]["metrics"]
    assert result["candidate"]["candidate_id"] == min(item["candidate_id"] for item in scores)
    assert_fold_oracle(case, result, "2024")
    before = case.conn.total_changes
    forbid_work(monkeypatch)
    with no_fit_guard(), legacy_guard():
        assert run(case) == result
    assert case.conn.total_changes == before
    with outer_problem("EXPERIMENT_IDENTITY_CONFLICT", CONFLICT):
        run(case, limits={**LIMITS, "cpu_seconds": 31})


def test_collect_requires_all_folds_and_pools_errors_without_outer_selection(outer_case, monkeypatch):
    case = outer_case
    first = run(case)
    before = case.conn.total_changes, journal_slots(case)
    public_replay = Mock(side_effect=AssertionError("incomplete collect attempted public outer replay"))
    with monkeypatch.context() as patch:
        patch.setattr(outer, "execute_outer_fold", public_replay)
        with outer_problem("EXPERIMENT_IDENTITY_CONFLICT", "outer report requires every completed outer fold"):
            collect(case)
    public_replay.assert_not_called()
    assert (case.conn.total_changes, journal_slots(case)) == before
    assert slot(case, "start", "2025") is None
    second = run(case, "2025")
    expected_predictions = [{**row, "outer_year": year} for year, result in (("2024", first), ("2025", second))
                            for row in assert_fold_oracle(case, result, year)]
    forbid_work(monkeypatch)
    before = case.conn.total_changes, journal_slots(case)
    with no_fit_guard(), legacy_guard():
        report = collect(case)
    expected = {"schema_version": "magnitude_search_outer_report.v1", "plan_id": case.plan["plan_id"],
        "folds": [first, second], "predictions": expected_predictions, "metrics": independent_metrics(expected_predictions),
        "event_count": 10, "forecast_count": 30,
        "per_horizon": [{"horizon_sessions": horizon, **independent_metrics([
            row for row in expected_predictions if row["horizon_sessions"] == horizon])} for horizon in HORIZONS],
        "prospective_candidate_id": case.selected["selections"][-1]["selected_candidate_id"],
        "winner_policy": "latest_outer_inner_selection_before_its_evaluation",
        "final_read_authority": "none", "holdout_results": HOLDOUTS}
    assert report == {**expected, "result_id": content_hash({key: value for key, value in report.items()
                                                          if key != "result_id"})}
    assert report["metrics"]["rmse"] != pytest.approx((first["metrics"]["rmse"] + second["metrics"]["rmse"]) / 2)
    assert (case.conn.total_changes, journal_slots(case)) == before


@pytest.mark.parametrize("guard, exception", [(no_fit_guard, RuntimeFitForbidden), (legacy_guard, LegacyRuntimeFitForbidden)])
def test_no_fit_guards_before_outer_claim(outer_case, monkeypatch, guard, exception):
    affinity = Mock(side_effect=AssertionError("guarded call resolved affinity"))
    monkeypatch.setattr(outer, "execution_limits", affinity)
    before = outer_case.conn.total_changes, journal_slots(outer_case)
    with guard(), pytest.raises(exception) as caught:
        run(outer_case)
    assert type(caught.value) is exception
    assert caught.value.args == ("no-fit guard active: execute_outer_fold",)
    affinity.assert_not_called()
    assert (outer_case.conn.total_changes, journal_slots(outer_case)) == before
    assert slot(outer_case, "start") is None


@pytest.mark.parametrize("number", [1, 2, 3])
def test_outer_fit_failure_retains_exact_prefix_without_replacement_or_retry(outer_case, monkeypatch, number):
    case = outer_case
    children = inject_child(monkeypatch, "count = 0\ndef injected(recipe, train, labels, test):\n"
        "    global count\n    count += 1\n"
        f"    if count == {number}: raise ValueError('synthetic estimator failure')\n"
        "    return [2.0] * len(test)\n")
    monkeypatch.setattr(outer, "load_magnitude_targets", lambda *a, **kw: pytest.fail("failed fit opened outer labels"))
    with outer_problem("EXPERIMENT_VARIANT_FAILED",
            "selected outer candidate failed; replacement and retry are forbidden", category="internal"):
        run(case)
    fits = expected_fits(case)
    assert slot(case, "execution") == {"status": "failed", "fits": fits[:number - 1],
        "fit_attempt_count": number, "completed_fit_count": number - 1, "forecasts_id": None}
    for index, fit in enumerate(fits):
        assert slot(case, "fit-started:" + str(fit["horizon_sessions"])) == (fit if index < number else None)
        assert slot(case, "fit-completed:" + str(fit["horizon_sessions"])) == (fit if index < number - 1 else None)
    assert all(slot(case, name) is None for name in ("forecasts", "validation", "terminal"))
    candidate = next(item for item in case.plan["candidates"]
                     if item["candidate_id"] == case.selected["selections"][0]["selected_candidate_id"])
    assert slot(case, "start")["candidate"] == candidate
    before = case.conn.total_changes, journal_slots(case)
    monkeypatch.setattr(outer, "run_fitted_trial", lambda *a: pytest.fail("retried outer fit"))
    monkeypatch.setattr(outer, "_inputs", lambda *a: pytest.fail("retried outer inputs"))
    with outer_problem("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed"):
        run(case)
    assert case.recorder.freeze() == case.selected
    assert (case.conn.total_changes, journal_slots(case)) == before
    assert len(children) == 1
    assert all(child.poll() is not None and not path.exists() for child, path in children)


@pytest.mark.parametrize("stage", ["labels", "input"])
def test_outer_shared_failure_never_replaces_candidate_or_retries(outer_case, monkeypatch, stage):
    case = outer_case
    def refused(*args, **kwargs):
        raise fail("RESOURCE_UNAVAILABLE", "synthetic shared failure")
    monkeypatch.setattr(outer, "load_magnitude_targets" if stage == "labels" else "_inputs", refused)
    with outer_problem("RESOURCE_UNAVAILABLE", "synthetic shared failure", category="resource", retryable=True):
        run(case)
    assert slot(case, "start") is not None and slot(case, "terminal") is None
    assert (slot(case, "forecasts") is not None) == (stage == "labels")
    before = case.conn.total_changes, journal_slots(case)
    monkeypatch.setattr(outer, "run_fitted_trial", lambda *a: pytest.fail("retried outer fit"))
    with outer_problem("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed"):
        run(case)
    assert case.recorder.freeze() == case.selected
    assert (case.conn.total_changes, journal_slots(case)) == before


@pytest.mark.parametrize("year, limits", [("2026", LIMITS), (2024, LIMITS), ("2024", {}),
    ("2024", {**LIMITS, "wall_seconds": True})])
def test_invalid_request_never_claims(outer_case, year, limits):
    before = outer_case.conn.total_changes
    message = ("outer fold is outside the sealed search plan" if year != "2024" else
               "search execution requires explicit positive integer resource limits")
    with outer_problem("INVALID_EXPERIMENT_SPEC", message):
        run(outer_case, year, limits=limits)
    assert outer_case.conn.total_changes == before
