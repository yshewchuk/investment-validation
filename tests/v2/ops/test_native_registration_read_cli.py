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


@pytest.mark.parametrize("cli", CLIS, ids=lambda cli: cli.__name__.rsplit(".", 1)[-1])
@pytest.mark.parametrize("budget", [None, 1, CAP + 1, 2**62 - 1])
def test_cli_forwards_default_or_explicit_finite_budget_and_closes(tmp_path, monkeypatch, capsys, cli, budget):
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

    def read_registration(actual_conn, actual_store, actual_spec, *, max_evidence_bytes):
        assert (actual_conn, actual_store, actual_spec) == (connection, store, spec)
        calls.append(max_evidence_bytes)
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "stop after exercising the read boundary")

    monkeypatch.setattr(cli if cli is v2_magnitude_search else native, "read_native_registration", read_registration)
    args = arguments(cli, tmp_path)
    if budget is not None:
        args += ["--max-registration-bytes", str(budget)]
    assert cli.main(args) == 2
    output = json.loads(capsys.readouterr().out)
    problem = output["refused"]
    assert (problem["code"] if isinstance(problem, dict) else problem) == "EXPERIMENT_IDENTITY_CONFLICT"
    assert calls == [CAP if budget is None else budget] and closed == [conn]


@pytest.mark.parametrize("cli", CLIS, ids=lambda cli: cli.__name__.rsplit(".", 1)[-1])
@pytest.mark.parametrize("budget", ["0", "-1", "1.5", "inf", "None", str(2**62), str(2**100)])
def test_cli_rejects_invalid_budget_before_catalog_or_store(tmp_path, monkeypatch, capsys, cli, budget):
    def forbid(*args, **kwargs):
        pytest.fail("invalid registration budget reached storage")
    args = arguments(cli, tmp_path) + ["--max-registration-bytes", budget]
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
