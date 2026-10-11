"""End-to-end parameter-binding checks for the v2 replay CLI."""
from __future__ import annotations

import json

import pytest

from tools import v2_replay
from engine.v2.research.replay import ALPHA_GRID
from tests.v2.research.replay_cli_support import replay_snapshot


def test_replay_cli_alpha_list_changes_real_snapshot_result(tmp_path, monkeypatch):
    real_run = v2_replay._replay_run.run
    omitted = object()
    forwarded_alphas = []

    def recording_run(*args, **kwargs):
        forwarded_alphas.append(kwargs.get("alphas", omitted))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(v2_replay._replay_run, "run", recording_run)
    snapshot = replay_snapshot(tmp_path)

    def run_report(name: str, alpha_args: list[str]) -> dict:
        reports_dir = tmp_path / name
        argv = [
            "--catalog",
            str(tmp_path / "catalog.sqlite"),
            "--store-root",
            str(tmp_path / "store"),
            "--scope",
            "shadow",
            "--snapshot-id",
            snapshot.snapshot_id,
            "--strategy",
            "STR-THRU",
            "--reports-dir",
            str(reports_dir),
            *alpha_args,
        ]
        assert v2_replay.main(argv) == 0
        report_path = next(reports_dir.glob("replay_*.json"))
        return json.loads(report_path.read_text())

    one_alpha = run_report("one-alpha", ["--alpha", "0"])
    end_alpha = run_report("end-alpha", ["--alpha", "1"])
    two_alphas = run_report("two-alphas", ["--alpha", "0", "--alpha", "1"])
    duplicate_alphas = run_report("duplicate-alphas", ["--alpha", "0", "--alpha", "0"])
    default_alphas = run_report("default-alphas", [])
    grid_alphas = [
        run_report(f"grid-alpha-{alpha}", ["--alpha", str(alpha)])
        for alpha in ALPHA_GRID
    ]

    assert forwarded_alphas[:5] == [
        [0.0], [1.0], [0.0, 1.0], [0.0, 0.0], omitted
    ]

    one_rows = one_alpha["results"][0]["rows"]
    end_rows = end_alpha["results"][0]["rows"]
    assert one_rows > 0
    assert two_alphas["results"][0]["rows"] == one_rows + end_rows
    assert two_alphas["results"][0]["rows"] > one_rows
    assert duplicate_alphas["results"][0]["rows"] == 2 * one_rows
    assert default_alphas["results"][0]["rows"] == sum(
        report["results"][0]["rows"] for report in grid_alphas
    )


@pytest.mark.parametrize("alpha", ["abc", "nan", "inf", "-0.1", "1.1"])
def test_replay_cli_rejects_invalid_alpha_before_opening_catalog(alpha, tmp_path, capsys):
    catalog_path = tmp_path / "unused-catalog.sqlite"
    store_path = tmp_path / "unused-store"
    with pytest.raises(SystemExit) as exc_info:
        v2_replay.main([
            "--catalog", str(catalog_path),
            "--store-root", str(store_path),
            "--alpha", alpha,
        ])

    assert exc_info.value.code == 2
    assert "alpha must be a finite number in [0, 1]" in capsys.readouterr().err
    assert not catalog_path.exists()
    assert not store_path.exists()
