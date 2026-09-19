"""P6-5 export-offline-file: validation, scan wiring, receipt shape and CLI.

The release tree is a REAL ``render_bundle`` output (never hand-rolled fixture
files), the same way ``tests/test_dashboard.py::TestSingleFile`` builds one.
Only ``export_offline_file``'s own behaviour is under test here --
``write_single_file`` itself is covered by ``tests/test_dashboard.py``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from engine.dashboard.render import render_bundle
from engine.score import ScoreResult
from tools.v2_dashboard_offline_file import OfflineFileError, export_offline_file, main

AS_OF = pd.Timestamp("2026-08-10")
EVENT = pd.Timestamp("2026-08-12")


def _result(ticker="AAA", strategy="STR-THRU", **kw):
    base = dict(as_of=AS_OF, event_date=EVENT, strike=100.0, expiry=EVENT, gate_score=0.5,
                gate_threshold=0.3, gate_pass=True, exp_pnl_model=0.01, n_analogs=10,
                snapshot_hash="snap-test", flags=[])
    base.update(kw)
    return ScoreResult(ticker=ticker, strategy=strategy, **base)


def _scores():
    return pd.DataFrame([_result().as_dict() | {"strike_offset": None}])


def _release_tree(tmp_path: Path, release_id: str = "r1") -> Path:
    release_root = tmp_path / "release_root"
    render_bundle(_scores(), release_root / "releases" / release_id, as_of=AS_OF)
    (release_root / "CURRENT").write_text(release_id + "\n")
    return release_root


def test_builds_a_single_file_that_inlines_the_release(tmp_path):
    receipt = export_offline_file(_release_tree(tmp_path), "r1")

    assert receipt["release_id"] == "r1"
    assert receipt["secret_scan_hits"] == 0
    assert receipt["external_refs"] == 0
    out = Path(receipt["path"])
    assert out.is_file()
    text = out.read_text()
    assert "window.BOARD" in text
    assert 'src="http' not in text
    assert 'href="http' not in text


def test_content_hash_matches_the_written_bytes(tmp_path):
    receipt = export_offline_file(_release_tree(tmp_path), "r1")

    out = Path(receipt["path"])
    assert receipt["content_hash"] == "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest()
    assert receipt["byte_size"] == out.stat().st_size


def test_writes_an_evidence_receipt_when_artifact_root_given(tmp_path):
    evidence = tmp_path / "evidence"
    receipt = export_offline_file(_release_tree(tmp_path), "r1", artifact_root=evidence)

    written = sorted(evidence.glob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text()) == receipt


def test_traversing_release_id_is_refused(tmp_path):
    release_root = _release_tree(tmp_path)

    for release_id in ("../etc", "r1/../../etc"):
        with pytest.raises(OfflineFileError):
            export_offline_file(release_root, release_id)


def test_release_id_with_embedded_slash_is_refused(tmp_path):
    with pytest.raises(OfflineFileError):
        export_offline_file(_release_tree(tmp_path), "a/b")


def test_missing_release_is_refused(tmp_path):
    with pytest.raises(OfflineFileError):
        export_offline_file(_release_tree(tmp_path), "does-not-exist")


def test_symlinked_release_directory_is_refused(tmp_path):
    release_root = _release_tree(tmp_path)
    (release_root / "releases" / "r2").symlink_to(
        release_root / "releases" / "r1", target_is_directory=True)

    with pytest.raises(OfflineFileError):
        export_offline_file(release_root, "r2")


def test_secret_shaped_content_is_refused(tmp_path):
    release_root = _release_tree(tmp_path, "r3")
    release_dir = release_root / "releases" / "r3"
    (release_dir / "data" / "tickers" / "extra.json").write_text(
        '{"api_key":"abcd1234efgh5678"}')
    out_path = tmp_path / "should-not-exist.html"

    with pytest.raises(OfflineFileError):
        export_offline_file(release_root, "r3", out=out_path)

    assert not out_path.exists()


def test_cli_writes_json_receipt_to_stdout(tmp_path, capsys):
    release_root = _release_tree(tmp_path)

    assert main(["--release-root", str(release_root), "--release-id", "r1",
                 "--out", str(tmp_path / "cli-out.html")]) == 0

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["schema_version"] == "offline_file_receipt.v1.0"


def test_cli_exits_2_on_refusal(tmp_path):
    release_root = _release_tree(tmp_path)

    assert main(["--release-root", str(release_root), "--release-id", "../etc"]) == 2