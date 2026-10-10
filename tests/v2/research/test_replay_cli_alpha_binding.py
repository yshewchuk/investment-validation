"""End-to-end parameter-binding checks for the v2 replay CLI."""
from __future__ import annotations

import json

from tools import v2_replay
from tests.data_scan_support import catalog_and_store
from tests.test_v2_research_replay import _chain_rows, _commit, _event_rows


def test_replay_cli_alpha_list_changes_real_snapshot_result(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit(
        conn,
        clock,
        store,
        chain_rows=_chain_rows(),
        event_rows=_event_rows(),
        receipt_id="alpha-binding",
    )
    conn.close()

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
    two_alphas = run_report("two-alphas", ["--alpha", "0", "--alpha", "1"])

    assert one_alpha["results"][0]["rows"] == 1
    assert two_alphas["results"][0]["rows"] == 2
