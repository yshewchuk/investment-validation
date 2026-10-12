"""Offline operator submission, allowlist and source-registration contracts."""
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.ops import massive_acquire as cli
from engine.v2.ops import providers
from engine.v2.ops.bootstrap import open_catalog
from tests.ops_support import FakeClock


@pytest.fixture
def isolated(monkeypatch):
    marker = "synthetic-massive-key-never-persist"
    monkeypatch.setattr(providers, "os", SimpleNamespace(environ={"MASSIVE_API_KEY": marker,
                                                                "ORATS_API_KEY": "unrelated-synthetic"}))
    def forbidden(*_args, **_kwargs):
        raise AssertionError("submission attempted provider I/O")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(cli, "worker_source_manifest", lambda _: {"synthetic_source": "v1"})
    monkeypatch.setattr(cli, "environment_identity", lambda n: {"synthetic_threads": n})
    return marker


def arguments(root):
    return ["--root", str(root), "--idempotency-key", "synthetic-one", "--ticker", "SYNTH",
            "--dataset", "daily", "--start", "2026-08-01", "--end", "2026-08-31",
            "--as-of", "2026-09-12", "--adjustment", "raw"]


def test_submission_is_idempotent_and_never_fetches_or_persists_credentials(tmp_path, isolated, capsys):
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=FakeClock())
    try:
        assert cli.main(arguments(tmp_path)) == 0
        first = capsys.readouterr()
        assert first.err == ""
        receipt = json.loads(first.out)
        assert receipt["kind"] == "massive_acquisition"
        assert receipt["state"] == "queued"
        assert cli.main(arguments(tmp_path)) == 0
        repeated = json.loads(capsys.readouterr().out)
        assert repeated["job_id"] == receipt["job_id"]
        assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
        row = conn.execute("SELECT * FROM jobs").fetchone()
        spec = json.loads(row["spec_json"])
        assert spec["provider_budget_ref"] == "massive-stocks"
        assert spec["parameters"]["provider_calls"] == 1
        assert spec["parameters"]["ticker"] == "SYNTH"
        assert row["max_attempts"] == 1
        assert conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM provider_accounts").fetchone()[0] == 0
        assert isolated not in first.out + str(tuple(conn.iterdump()))
        assert "MASSIVE_API_KEY" not in str(tuple(conn.iterdump()))
    finally:
        conn.close()


@pytest.mark.parametrize("mode", ["missing", "bad_dataset", "bad_adjustment", "extra", "missing_catalog"])
def test_cli_refusals_are_redacted_and_do_not_create_catalogs(tmp_path, isolated, capsys, mode):
    args = arguments(tmp_path)
    if mode == "missing":
        args = []
    elif mode == "bad_dataset":
        args[args.index("--dataset") + 1] = isolated
    elif mode == "bad_adjustment":
        args[args.index("--adjustment") + 1] = isolated
    elif mode == "extra":
        args += ["--unexpected", isolated]
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {"refused": "RESOURCE_UNAVAILABLE" if mode == "missing_catalog" else "INVALID_REQUEST"}
    assert isolated not in captured.out
    assert not (tmp_path / "catalog.sqlite").exists()


def test_executor_credential_allowlist_is_account_specific(isolated):
    assert providers.provider_credentials({"provider_account": "massive-stocks"}) == {"MASSIVE_API_KEY": isolated}
    assert providers.provider_credentials({"provider_account": "other"}) == {}
    assert providers.provider_credentials({}) == {}


def test_production_source_closure_contains_new_worker_and_http_edge():
    from engine.v2.ops.fingerprints import worker_source_manifest
    manifest = worker_source_manifest(Path(__file__).resolve().parents[3])
    paths = set(manifest)
    assert "engine/v2/ops/stores/massive/acquisition.py" in paths
    assert "engine/v2/ops/providers/massive.py" in paths
    assert "engine/v2/ops/providers/massive_policy.py" in paths
