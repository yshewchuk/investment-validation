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
    replay_outputs = []

    def recording_run(*args, **kwargs):
        forwarded_alphas.append(kwargs.get("alphas", omitted))
        result = real_run(*args, **kwargs)
        replay_outputs.append(result)
        return result

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

    one_trades = replay_outputs[0]["trades"]
    end_trades = replay_outputs[1]["trades"]
    two_trades = replay_outputs[2]["trades"]
    duplicate_trades = replay_outputs[3]["trades"]
    assert two_trades["fill_alpha"].tolist() == [0.0, 1.0]
    assert duplicate_trades["fill_alpha"].tolist() == [0.0, 0.0]

    def assert_expected_prices(trades, alpha, entry_cost, exit_value):
        matched = trades.loc[trades["fill_alpha"] == alpha]
        assert len(matched) == 1
        row = matched.iloc[0]
        assert row["entry_cost"] == pytest.approx(entry_cost)
        assert row["exit_value"] == pytest.approx(exit_value)
        assert row["ret"] == pytest.approx((exit_value - entry_cost) / entry_cost)

    # Synthetic ATM call+put sums: May 2 ask/bid are 3.8/3.0; May 3
    # bid/ask are 5.0/5.8. Alpha 0 crosses entry ask and exit bid;
    # alpha 1 uses entry bid and exit ask.
    assert_expected_prices(one_trades, 0.0, 3.8, 5.0)
    assert_expected_prices(end_trades, 1.0, 3.0, 5.8)

    corrupted_trades = end_trades.copy()
    corrupted_trades.loc[corrupted_trades["fill_alpha"] == 1.0, "exit_value"] += 1.0
    with pytest.raises(AssertionError):
        assert_expected_prices(corrupted_trades, 1.0, 3.0, 5.8)


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
