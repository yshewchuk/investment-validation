"""Native CLI callers expose the same finite registration-read budget."""
import json
from types import SimpleNamespace

import pytest

from engine.v2.ops.errors import fail
from experiments import native_registration as native
from experiments import v2_gate, v2_magnitude_search, v2_prediction

CAP = 16 * 1024 * 1024
CLIS = (v2_gate, v2_prediction, v2_magnitude_search)


def arguments(cli, tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    args = ["prepare" if cli is v2_magnitude_search else "run", "--spec", str(spec),
            "--catalog", str(tmp_path / "catalog.sqlite"), "--store-root", str(tmp_path), "--no-ledger"]
    if cli is v2_magnitude_search:
        args += ["--request", str(spec)]
    else:
        args += ["--as-of-month", "2026-01"]
    return args


@pytest.mark.parametrize("cli,register", [(cli, False) for cli in CLIS] + [(v2_gate, True), (v2_prediction, True)])
@pytest.mark.parametrize("budget", [None, 1, CAP + 1, 2**62 - 1])
def test_cli_forwards_default_or_explicit_finite_budget_and_closes(tmp_path, monkeypatch, capsys, cli, register, budget):
    conn, store, spec = object(), object(), object()
    closed, calls = [], []
    connection = SimpleNamespace(close=lambda: closed.append(conn))
    monkeypatch.setattr(cli, "connect", lambda *args, **kwargs: connection)
    monkeypatch.setattr(cli, "experiment_spec_from_document", lambda document: spec)
    if cli is v2_magnitude_search:
        monkeypatch.setattr(cli, "ArtifactStore", lambda path: store)
    else:
        monkeypatch.setattr(cli.foundation, "ArtifactStore", lambda path: store)
        monkeypatch.setattr(cli, "Repository", lambda *args, **kwargs: None)
        owner = cli.gate if cli is v2_gate else cli.prediction
        name = "validate_gate_spec" if cli is v2_gate else "validate_prediction_spec"
        monkeypatch.setattr(owner, name, lambda value: None)
        monkeypatch.setattr(cli.outcomes, "publish_native_refusal", lambda *args, **kwargs: None)

    def registration_boundary(actual_conn, actual_store, *positional, max_evidence_bytes, **kwargs):
        actual_spec = positional[-1]
        assert (actual_conn, actual_store, actual_spec) == (connection, store, spec)
        calls.append(max_evidence_bytes)
        raise fail("RESOURCE_LIMIT_EXCEEDED", "native registration exceeds its UTF-8 byte budget",
                   details={"registration_bytes": max_evidence_bytes + 1, "max_registration_bytes": max_evidence_bytes})

    monkeypatch.setattr(cli if cli is v2_magnitude_search else native,
                        "register_native" if register else "read_native_registration", registration_boundary)
    args = arguments(cli, tmp_path)
    if register:
        args[0] = "register"
    if budget is not None:
        args += ["--max-registration-bytes", str(budget)]
    assert cli.main(args) == 2
    output = json.loads(capsys.readouterr().out)
    problem = output["refused"]
    assert (problem["code"] if isinstance(problem, dict) else problem) == "RESOURCE_LIMIT_EXCEEDED"
    limit = CAP if budget is None else budget
    assert (problem if isinstance(problem, dict) else output)["details"] == {
        "registration_bytes": limit + 1, "max_registration_bytes": limit}
    assert calls == [CAP if budget is None else budget] and closed == [conn]


@pytest.mark.parametrize("cli,register", [(cli, False) for cli in CLIS] + [(v2_gate, True), (v2_prediction, True)])
@pytest.mark.parametrize("budget", ["0", "-1", "1.5", "inf", "None", str(2**62), str(2**100)])
def test_cli_rejects_invalid_budget_before_catalog_or_store(tmp_path, monkeypatch, capsys, cli, register, budget):
    def forbid(*args, **kwargs):
        pytest.fail("invalid registration budget reached storage")
    args = arguments(cli, tmp_path) + ["--max-registration-bytes", budget]
    if register:
        args[0] = "register"
    monkeypatch.setattr(cli, "connect", forbid)
    monkeypatch.setattr(cli if cli is v2_magnitude_search else cli.foundation, "ArtifactStore", forbid)
    monkeypatch.setattr(cli.Path, "read_text", forbid)
    if cli is v2_magnitude_search:
        monkeypatch.setattr(cli, "_read_document", forbid)
        monkeypatch.setattr(cli.Path, "is_dir", forbid)
    with pytest.raises(SystemExit) as captured:
        cli.main(args)
    assert type(captured.value) is SystemExit and captured.value.code == 2
    diagnostic = capsys.readouterr().err.splitlines()[-1].split(": error: ", 1)[-1]
    if budget in ("1.5", "inf", "None"):
        assert diagnostic == f"argument --max-registration-bytes: invalid int value: {budget!r}"
    else:
        assert diagnostic == "--max-registration-bytes must be a positive SQLite-safe byte budget"


@pytest.mark.parametrize("route", ["gate", "positive", "magnitude"])
@pytest.mark.parametrize("budget", [None, CAP + 1])
def test_cli_carries_initial_read_budget_into_current_request_revalidation(tmp_path, monkeypatch, capsys, route, budget):
    cli = v2_gate if route == "gate" else v2_prediction
    calls, closed = [], []
    connection = SimpleNamespace(close=lambda: closed.append(True))
    registration = SimpleNamespace(document={})
    monkeypatch.setattr(cli, "connect", lambda *a, **k: connection)
    monkeypatch.setattr(cli, "experiment_spec_from_document", lambda document: None)
    monkeypatch.setattr(cli.foundation, "ArtifactStore", lambda path: None)
    monkeypatch.setattr(cli, "Repository", lambda *a, **k: None)
    owner = cli.gate if route == "gate" else cli.magnitude_report if route == "magnitude" else cli.prediction
    monkeypatch.setattr(owner, "validate_gate_spec" if route == "gate" else "validate_prediction_spec", lambda spec: None)
    def initial(*args, **kwargs):
        calls.append(("read", kwargs["max_evidence_bytes"]))
        return registration
    def revalidate(*args, **kwargs):
        calls.append(("revalidate", kwargs["max_evidence_bytes"]))
        raise fail("RESOURCE_LIMIT_EXCEEDED", "native registration exceeds its UTF-8 byte budget",
                   details={"registration_bytes": limit + 1, "max_registration_bytes": limit})
    monkeypatch.setattr(native, "read_native_registration", initial)
    monkeypatch.setattr(cli if route == "magnitude" else native,
                        "admit_magnitude_inputs" if route == "magnitude" else "require_native_registration", revalidate)
    monkeypatch.setattr(cli.outcomes, "publish_native_refusal", lambda *a, **k: None)
    args = arguments(cli, tmp_path)
    if route == "magnitude":
        args += ["--target", "absolute_move", "--magnitude-training-start", "2020-01-01",
                 "--magnitude-labels-before", "2025-01-01"]
    limit = CAP if budget is None else budget
    if budget is not None:
        args += ["--max-registration-bytes", str(budget)]
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "RESOURCE_LIMIT_EXCEEDED", "refusal_ref": None,
        "details": {"registration_bytes": limit + 1, "max_registration_bytes": limit}}
    assert calls == [("read", limit), ("revalidate", limit)] and closed == [True]
