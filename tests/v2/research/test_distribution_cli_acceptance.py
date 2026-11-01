"""Real CLI dispatch/run/pipeline/CAS; only native inputs and OS containment doubled."""
import json
import sqlite3
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import pytest

from engine.v2.foundation import ArtifactStore, artifact_reference, to_document
from engine.v2.ops.errors import fail
from experiments import distribution_cli as cli
from experiments import distribution_run as runner
from experiments.distribution_candidate_recipe import candidate_recipe
from experiments.distribution_run_request import PRIMARY_ARMS, validate_request
from experiments.magnitude_search_records import read_record
from tests.v2.research.magnitude_trial_support import prediction_spec
from tests.v2.research.test_distribution_run import case as case
from tests.v2.research.test_distribution_run_acceptance import rows, snapshot


def durable(case):
    tables = tuple((name, tuple(tuple(row) for row in case.conn.execute('SELECT * FROM "' + name + '"')))
        for name, in case.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"))
    return snapshot(case), tables


@pytest.fixture
def operator(case, monkeypatch, tmp_path, capsys):
    documents = {"spec": to_document(prediction_spec(experiment_id="cli-own", runner="experiments/distribution_run.py")),
        "source-spec": to_document(prediction_spec(experiment_id="cli-source")), "source-request": {},
        "source-plan-ref": to_document(artifact_reference(b"source", "source.v1")),
        "recipe": candidate_recipe("H"), "budget-caps": {}, "request": case.request}
    arguments = []
    for name, document in documents.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(document), encoding="utf-8")
        arguments.extend(("--" + name, str(path)))
    arguments.extend(("--catalog", str(tmp_path / "catalog.sqlite"), "--store-root", str(case.store.root), "--no-ledger"))
    active, calls, instances, connections, published, evaluations, finalized = [], [], [], [], [], [], []
    @contextmanager
    def contained(limits, seconds):
        assert not active
        calls.append((deepcopy(limits), seconds))
        active.append(True)
        try:
            yield
        finally:
            active.pop()
    def trials(conn, store, repository, own, source, *args):
        assert active and own is case.trials.own and source is case.trials.source
        connections.append(conn)
        # The existing native-input adapter's closures share the same physical DB/CAS.
        # CLI-created connections/stores still own all run records and publications.
        case.trials.conn, case.trials.store = conn, store
        return case.trials
    initialize, evaluate, finish, publish = (runner.DistributionRun.__init__, runner.evaluate_outer,
        runner.finish_reports, runner.DistributionRun.publish)
    def initialized(run, *args):
        assert active and all(run is not previous for previous in instances)
        instances.append(run)
        initialize(run, *args)
    def evaluated(*args):
        assert active and set(args[3]) == {1, 3, 5}
        evaluations.append(args[1]["outer"]["year"])
        return evaluate(*args)
    def finished(*args):
        finalized.append(True)
        return finish(*args)
    def published_section(run, *args):
        assert active
        section = publish(run, *args)
        published.append(section)
        return section
    monkeypatch.setattr(cli, "contained", contained)
    monkeypatch.setattr(cli, "DistributionTrials", trials)
    monkeypatch.setattr(cli, "read_native_registration", lambda conn, store, spec, **kw:
        case.trials.own if spec.experiment_id == "cli-own" else case.trials.source)
    monkeypatch.setattr(runner, "contained", lambda *args: pytest.fail("CLI reset containment through run()"))
    monkeypatch.setattr(runner.DistributionRun, "__init__", initialized)
    monkeypatch.setattr(runner.DistributionRun, "publish", published_section)
    monkeypatch.setattr(runner, "evaluate_outer", evaluated)
    monkeypatch.setattr(runner, "finish_reports", finished)
    # Independent module clocks: CLI begins at100, finishes bootstrap at103.5;
    # the actual run reaches its started record at107, under the original deadline.
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: 107.0))
    def invoke(action):
        ticks = iter((100.0, 103.5))
        monkeypatch.setattr(cli, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
        before = len(calls)
        status = cli.main([action, *arguments])
        captured = capsys.readouterr()
        assert len(calls) == before + 1 and not active
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connections[-1].execute("SELECT 1")
        return status, json.loads(captured.out)
    def record(slot):
        return read_record(case.conn, case.store, "distribution-own", "distribution:run-" + slot)
    return SimpleNamespace(invoke=invoke, record=record, calls=calls, instances=instances,
        published=published, evaluations=evaluations, finalized=finalized,
        request_path=tmp_path / "request.json")


def test_cli_fresh_runner_shares_original_deadline_and_replays_real_cas(case, operator):
    assert operator.invoke("seal")[0] == 0
    status, result = operator.invoke("run")
    assert status == 0 and result == operator.record("completed")
    assert operator.evaluations == ["2023", "2024"] and operator.finalized == [True]
    assert operator.published == result["sections"] and len(result["sections"]) == 36
    assert all(rows(case, section) for section in result["sections"])
    assert operator.calls == [(case.request["limits"]["parent"], 56.5)] * 2
    started = operator.record("started")
    assert (datetime.fromisoformat(started["deadline"]) - datetime.fromisoformat(started["started_at"])).total_seconds() == 53
    assert result["elapsed_seconds"] == 7
    before, calls = durable(case), list(case.calls)
    for action in ("collect", "report", "run"):
        assert operator.invoke(action) == (0, result)
        assert durable(case) == before and case.calls == calls
    assert len(operator.instances) == 6  # Fresh execution reopens after its read-only completion probe.


@pytest.mark.parametrize("changed", ["parent_limit", "job_limit", "alpha"])
def test_changed_valid_request_file_refuses_before_started_or_fits(case, operator, changed):
    assert operator.invoke("seal")[0] == 0
    request = deepcopy(case.request)
    if changed == "parent_limit":
        request["limits"]["parent"]["cpu_seconds"] -= 1
    elif changed == "job_limit":
        request["limits"]["job_wall_seconds"] -= 1
    else:
        request["alphas"].append(.75)
        request["bridges"].extend({**entry, "alpha": .75} for entry in case.request["bridges"] if entry["alpha"] == 0)
    assert validate_request(request) == request
    operator.request_path.write_text(json.dumps(request), encoding="utf-8")
    before, calls = durable(case), list(case.calls)
    status, result = operator.invoke("run")
    assert status == 2 and result["refused"]["code"] == "INVALID_EXPERIMENT_SPEC"
    assert durable(case) == before and case.calls == calls
    assert operator.record("started") is None and operator.record("completed") is None
    assert not operator.evaluations and not operator.published and not operator.finalized
    # This is NOT authenticated sealed-cap enforcement of pre-admission work (S1):
    # valid file limits govern bootstrap containment before seal mismatch refusal.
    assert operator.calls[-1] == (request["limits"]["parent"], request["limits"]["job_wall_seconds"] - 3.5)


def test_cli_missing_outer_result_refuses_before_evaluation_or_publication(case, operator):
    assert operator.invoke("seal")[0] == 0
    # Impossible native recipe; deliberately corrupt only the fake post-seal adapter.
    case.trials.recipe["horizons"] = [5, 3]
    status, result = operator.invoke("run")
    expected = fail("INVALID_EXPERIMENT_SPEC", "distribution outer results require horizons 1,3,5")
    assert status == 2 and result["refused"] == to_document(expected.problem)
    assert not operator.evaluations and not operator.published and not operator.finalized
    assert not any(call[0] in ("decision", "outcome", "targets") for call in case.calls)
    assert operator.record("failed")["code"] == expected.code and operator.record("completed") is None


def test_failure_journal_io_is_terminal_and_fresh_cli_retry_cannot_write_or_fit(case, operator, monkeypatch):
    assert operator.invoke("seal")[0] == 0
    original, failures = ArtifactStore.publish_bytes, []
    def publish(store, payload, **kwargs):
        if kwargs.get("schema_ref") == "magnitude_search_record.v1" and json.loads(payload).get("slot") == "distribution:run-failed":
            failures.append(True)
            raise OSError("synthetic failure-journal storage outage")
        return original(store, payload, **kwargs)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", publish)
    case.state.update(failure=fail("RESOURCE_LIMIT_EXCEEDED", "synthetic fit exhaustion"), fail_on=8)
    status, result = operator.invoke("run")
    assert status == 2 and result["refused"]["code"] == "RESOURCE_UNAVAILABLE"  # S2 masks the primary error.
    assert failures == [True] and operator.evaluations == ["2023"] and not operator.finalized
    assert len(operator.published) == 9 and all(rows(case, section) for section in operator.published)
    assert sum(call[0] == "fit" for call in case.calls) == 8
    assert operator.record("started") is not None
    assert operator.record("failed") is None and operator.record("completed") is None
    before, calls = durable(case), list(case.calls)
    case.state["failure"] = None
    status, result = operator.invoke("run")
    assert status == 2 and result["refused"]["code"] == "EXPERIMENT_IDENTITY_CONFLICT"
    assert operator.invoke("report")[0] == 2
    assert durable(case) == before and case.calls == calls and failures == [True]
    assert len(operator.instances) == 6 and not operator.finalized


def test_zero_funded_cli_reports_are_diagnostics_without_promotion_authority(case, operator):
    assert operator.invoke("seal")[0] == 0
    case.state["missing_pool"] = True
    status, result = operator.invoke("run")
    assert status == 0
    assert result["promotion_authority"] == result["final_read_authority"] == "none"
    assert result["current_strategy_superiority_claim"] is False
    assert result["calibration_identity"] == "uncalibrated" and result["capacity"] == "not_established_no_depth"
    for item in result["reports"]:
        report = item["report"]
        assert report["opportunities"] == 4 and report["uplift_met"] is True
        for arm in PRIMARY_ARMS:
            assert report["arms"][arm]["funded"] == 0
            assert report["risk_checks"][arm] == {"loss_limit_met": True, "settled_drawdown_met": True, "completeness_met": None}
    books = [section for section in result["sections"] if section["name"].startswith("book:")]
    assert len(books) == 9 and all(row["status"] == "no_trade" for section in books for row in rows(case, section))
    assert operator.invoke("report") == (0, result)
