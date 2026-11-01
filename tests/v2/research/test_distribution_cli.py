"""Operator boundaries: explicit lifecycle, one budget and no hidden input creation."""
import json
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.foundation import ArtifactError, DocumentError
from engine.v2.ops.errors import OpsError, fail
from experiments import distribution_cli as cli
from tests.v2.research.magnitude_trial_support import prediction_spec
from tests.v2.research.test_distribution_run_request import request


def arguments(tmp_path, action="preflight"):
    return [action, *[value for name in ("spec", "source-spec", "source-request", "source-plan-ref",
        "recipe", "budget-caps", "request", "catalog") for value in ("--" + name, str(tmp_path / name))],
        "--store-root", str(tmp_path), "--no-ledger"]


def setup(monkeypatch, tmp_path, action="preflight"):
    calls, active = [], []
    @contextmanager
    def contained(limits, seconds):
        active.append(True)
        calls.append(("limits", limits, seconds))
        try:
            yield
        finally:
            active.pop()
    def connect(*args, **kwargs):
        assert active
        calls.append(("connect", args, kwargs))
        return SimpleNamespace(close=lambda: calls.append(("close",)))
    def dispatch(args, *values):
        assert active
        calls.append(("dispatch", args.action))
        return {"schema_version": "test.v1", "action": args.action}
    monkeypatch.setattr(cli, "contained", contained)
    monkeypatch.setattr(cli, "_read_document", lambda path: request())
    monkeypatch.setattr(cli.sqlite3, "connect", connect)
    monkeypatch.setattr(cli, "connect", connect)
    monkeypatch.setattr(cli, "Repository", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_dispatch", dispatch)
    return calls, arguments(tmp_path, action)


@pytest.mark.parametrize("action", ["prepare", "register", "preflight", "seal", "run", "collect", "report"])
def test_all_actions_are_contained_and_readonly_actions_use_existing_catalog(monkeypatch, tmp_path, capsys, action):
    calls, argv = setup(monkeypatch, tmp_path, action)
    assert cli.main(argv) == 0
    assert json.loads(capsys.readouterr().out) == {"schema_version": "test.v1", "action": action}
    assert len([call for call in calls if call[0] == "limits"]) == 1
    connection = next(call for call in calls if call[0] == "connect")
    if action in ("prepare", "preflight", "collect", "report"):
        assert connection[1][0].endswith("?mode=ro") and connection[2]["uri"] is True
    else:
        assert connection[2] == {"must_exist": True}
    assert calls[-1] == ("close",)


def test_invalid_request_refuses_before_containment_or_catalog(monkeypatch, tmp_path, capsys):
    calls, argv = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_read_document", lambda path: {})
    assert cli.main(argv) == 2
    assert json.loads(capsys.readouterr().out)["refused"]["code"] == "INVALID_EXPERIMENT_SPEC"
    assert calls == []


@pytest.mark.parametrize("error,code", [
    (MemoryError("private"), "RESOURCE_LIMIT_EXCEEDED"),
    (OSError("private"), "RESOURCE_UNAVAILABLE"),
    (ArtifactError("IO_ERROR", "private"), "RESOURCE_UNAVAILABLE"),
    (DocumentError("TYPE_MISMATCH", "private", "private"), "INVALID_EXPERIMENT_SPEC"),
    (fail("HOLDOUT_ACCESS_DENIED", "protected input"), "HOLDOUT_ACCESS_DENIED"),
])
def test_refusals_have_one_machine_readable_problem_and_cleanup(monkeypatch, tmp_path, capsys, error, code):
    calls, argv = setup(monkeypatch, tmp_path)
    def broken(*args):
        raise error
    monkeypatch.setattr(cli, "_dispatch", broken)
    assert cli.main(argv) == 2
    output = capsys.readouterr().out
    assert json.loads(output)["refused"]["code"] == code and "private" not in output
    assert calls[-1] == ("close",)


def test_programming_failure_is_not_reclassified(monkeypatch, tmp_path):
    calls, argv = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_dispatch", lambda *args: (_ for _ in ()).throw(TypeError("programming bug")))
    with pytest.raises(TypeError, match="programming bug"):
        cli.main(argv)
    assert calls[-1] == ("close",)


def test_no_ledger_acknowledgment_is_required(tmp_path):
    with pytest.raises(SystemExit) as caught:
        cli.main(arguments(tmp_path)[:-1])
    assert caught.value.code == 2


def test_collect_missing_catalog_does_not_create_one(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_read_document", lambda path: request())
    monkeypatch.setattr(cli, "contained", lambda *args: nullcontext())
    assert cli.main(arguments(tmp_path, "collect")) == 2
    assert json.loads(capsys.readouterr().out)["refused"]["code"] == "RESOURCE_UNAVAILABLE"
    assert not (tmp_path / "catalog").exists()


@pytest.mark.parametrize("edit", [dict(runner="experiments/size_search.py"), dict(seed=1),
    dict(price_source="options"), dict(economic_params={"fee": 1})])
def test_registration_static_policy_precedes_source_reads(monkeypatch, edit):
    spec = prediction_spec(runner="experiments/distribution_run.py")
    monkeypatch.setattr(cli, "read_search_plan", lambda *args: pytest.fail("invalid policy read source"))
    with pytest.raises(OpsError) as caught:
        cli._prepare(None, None, None, replace(spec, **edit), None, {}, None, None, {})
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_cli_request_bootstrap_consumes_parent_deadline(monkeypatch, tmp_path):
    calls, argv = setup(monkeypatch, tmp_path)
    values = iter([10.0, 13.5])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(values))
    monkeypatch.setattr(cli, "stage", lambda *args: __import__("contextlib").nullcontext())
    assert cli.main(argv) == 0
    assert calls[0][2] == request()["limits"]["job_wall_seconds"] - 3.5


def test_expired_bootstrap_refuses_before_process_or_catalog_access(monkeypatch, tmp_path, capsys):
    calls, argv = setup(monkeypatch, tmp_path)
    values = iter([10., 10. + request()["limits"]["job_wall_seconds"]])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(values))
    assert cli.main(argv) == 2
    assert json.loads(capsys.readouterr().out)["refused"]["code"] == "RESOURCE_LIMIT_EXCEEDED"
    assert calls == []
