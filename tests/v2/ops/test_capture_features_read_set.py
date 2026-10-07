"""``capture`` enumerates ``legacy_features``'s data-dependent read set.

Synthetic fixture trees only. ``engine/data/features/panel.py`` globs
``moves_*.json`` under the oquants and computed-moves directories and reads a
price file per covered ticker; the worker's staged tree is built only from the
manifest, so these tests check the manifest lists exactly that set, is
deterministic, refuses indirection, and that a tree staged from it satisfies
panel's own discovery.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from engine.data.features import panel
from engine.v2.data.legacy_nightly_read_plan import manifest_problems
from engine.v2.foundation import to_document
from engine.v2.ops.capture_inputs import capture
from engine.v2.ops.errors import OpsError
from engine.v2.ops.legacy_adapter import copy_read_set
from tests.test_v2_ops_capture_inputs import SESSION, TICKER, _build_fixture, _write_px_csv

OQUANTS = "earnings_predictions/data/raw/oquants/moves"
COMPUTED = "data/raw/computed_moves"
PX = "earnings_predictions/data/raw/yfinance"
DAYS = pd.bdate_range("2023-01-02", periods=310)


def _moves_doc(ticker: str, n: int = 6) -> dict:
    dates = [f"2023-{m:02d}-15" for m in range(1, n + 1)]
    return {"ticker": ticker,
            "data": {"dates": dates, "realized_moves": [0.01] * n,
                     "abs_realized_moves": [0.01] * n,
                     "quarters": [f"Q{m}" for m in range(1, n + 1)]}}


def _put(root, directory: str, name: str, doc: dict):
    path = root / directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def _features_tree(root) -> None:
    """oquants: AAA (from ``_build_fixture``), BBB, CCC whose document ticker is
    CCX; computed: DDD. Decoys: state files, a non-matching suffix, a directory
    named like a moves file, and a price file for a ticker no moves file covers.
    Prices exist only for BBB and CCX; AAA and DDD have none."""
    root.mkdir(parents=True, exist_ok=True)
    _build_fixture(root)
    _put(root, OQUANTS, "moves_BBB.json", _moves_doc("BBB"))
    _put(root, OQUANTS, "moves_CCC.json", _moves_doc("CCX"))
    _put(root, OQUANTS, "oquants_state.json", {"ticker": "STATE"})
    _put(root, COMPUTED, "moves_DDD.json", _moves_doc("DDD"))
    _put(root, COMPUTED, "state.json", {"ticker": "STATE"})
    _put(root, COMPUTED, "moves_DDD.json.tmp", {"ticker": "TMP"})
    (root / COMPUTED / "moves_dir.json").mkdir()
    for ticker in ("BBB", "CCX", "ZZZ"):
        _write_px_csv(root, ticker, DAYS)


def _capture(root):
    return capture(root, as_of=SESSION, tickers=[TICKER], year_start=2024, year_end=2024)


def test_capture_lists_exactly_the_moves_and_price_files_panel_reads(tmp_path):
    _features_tree(tmp_path)
    manifest = _capture(tmp_path)
    paths = {ref.path for ref in manifest.file_refs}
    assert {p for p in paths if p.startswith((OQUANTS + "/", COMPUTED + "/"))} == {
        f"{OQUANTS}/moves_AAA.json", f"{OQUANTS}/moves_BBB.json",
        f"{OQUANTS}/moves_CCC.json", f"{COMPUTED}/moves_DDD.json"}
    # document ticker (CCX) wins over the file-name stem (CCC); AAA and DDD have
    # no price file and are simply absent; ZZZ is not covered by any moves file.
    assert {p for p in paths if p.startswith(PX + "/")} == {
        f"{PX}/px_BBB.csv", f"{PX}/px_CCX.csv"}
    assert manifest_problems(to_document(manifest)) == []


def test_capture_of_the_features_read_set_is_deterministic(tmp_path):
    _features_tree(tmp_path)
    first, second = _capture(tmp_path), _capture(tmp_path)
    assert first.manifest_id == second.manifest_id
    assert first.file_refs == second.file_refs
    assert [r.path for r in first.file_refs] == sorted(r.path for r in first.file_refs)


def test_capture_refuses_when_no_moves_file_exists(tmp_path):
    _build_fixture(tmp_path)
    (tmp_path / OQUANTS / "moves_AAA.json").unlink()
    _put(tmp_path, COMPUTED, "state.json", {"ticker": "STATE"})  # a decoy is not a moves file
    with pytest.raises(OpsError) as excinfo:
        _capture(tmp_path)
    assert excinfo.value.code == "INPUT_CHANGED"
    assert excinfo.value.problem.details["family"] == "features_moves"


def test_one_moves_directory_is_enough(tmp_path):
    _build_fixture(tmp_path)
    (tmp_path / OQUANTS / "moves_AAA.json").unlink()
    _put(tmp_path, COMPUTED, "moves_DDD.json", _moves_doc("DDD"))
    paths = {ref.path for ref in _capture(tmp_path).file_refs}
    assert f"{COMPUTED}/moves_DDD.json" in paths
    assert not any(p.startswith(OQUANTS + "/") for p in paths)


def test_symlinked_moves_file_is_refused(tmp_path):
    _features_tree(tmp_path)
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps(_moves_doc("LNK")))
    (tmp_path / OQUANTS / "moves_LNK.json").symlink_to(target)
    with pytest.raises(OpsError) as excinfo:
        _capture(tmp_path)
    assert excinfo.value.code == "INPUT_CHANGED"


def test_symlinked_moves_directory_is_refused(tmp_path):
    _build_fixture(tmp_path)
    real = tmp_path / "real_computed"
    _put(tmp_path, "real_computed", "moves_DDD.json", _moves_doc("DDD"))
    (tmp_path / COMPUTED).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / COMPUTED).symlink_to(real, target_is_directory=True)
    with pytest.raises(OpsError) as excinfo:
        _capture(tmp_path)
    assert excinfo.value.code == "INPUT_CHANGED"


def test_staged_features_read_set_satisfies_panel_discovery(tmp_path):
    source, staging = tmp_path / "source", tmp_path / "staging"
    _features_tree(source)
    manifest = _capture(source)
    copy_read_set(source, staging, [ref.path for ref in manifest.file_refs])

    events = panel.build_events(staging / OQUANTS, extra_moves_dirs=(staging / COMPUTED,))
    assert sorted(set(events["ticker"])) == ["BBB", "CCX", "DDD"]
    assert (staging / PX / "px_BBB.csv").is_file()

    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(FileNotFoundError):
        panel.build_events(bare / OQUANTS, extra_moves_dirs=(bare / COMPUTED,))


def test_manifest_problems_flags_a_manifest_without_the_moves_family(tmp_path):
    _features_tree(tmp_path)
    document = to_document(_capture(tmp_path))
    assert manifest_problems(document) == []
    document["file_refs"] = [ref for ref in document["file_refs"]
                             if not ref["path"].startswith((OQUANTS + "/", COMPUTED + "/"))]
    problems = manifest_problems(document)
    assert [(p["kind"], p["family"]) for p in problems] == [("legacy_features", "features_moves")]


def test_unparseable_moves_file_is_captured_with_its_stem_ticker(tmp_path):
    _features_tree(tmp_path)
    (tmp_path / OQUANTS / "moves_BAD.json").write_text("{not json")
    _write_px_csv(tmp_path, "BAD", DAYS)
    paths = {ref.path for ref in _capture(tmp_path).file_refs}
    assert f"{OQUANTS}/moves_BAD.json" in paths
    assert f"{PX}/px_BAD.csv" in paths


def test_manifest_problems_refuses_a_moves_directory_holding_only_a_state_file(tmp_path):
    _features_tree(tmp_path)
    document = to_document(_capture(tmp_path))
    state_ref = {"path": f"{COMPUTED}/state.json", "content_hash": "sha256:" + "0" * 64, "byte_size": 1}
    document["file_refs"] = [ref for ref in document["file_refs"]
                             if not ref["path"].startswith((OQUANTS + "/", COMPUTED + "/"))] + [state_ref]
    problems = manifest_problems(document)
    assert [(p["kind"], p["family"]) for p in problems] == [("legacy_features", "features_moves")]
