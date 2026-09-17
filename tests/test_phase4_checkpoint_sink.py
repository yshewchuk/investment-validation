from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.phase4_checkpoint_sink import (
    CheckpointSinkError,
    DiskCheckpointSink,
    SCHEMA_VERSION,
)


def _resource(root: Path, name: str = "model.bin", data: bytes = b"model") -> Path:
    directory = root / "resources"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(data)
    return path


def test_first_write_persists_case_resource_and_manifest(tmp_path: Path) -> None:
    resource = _resource(tmp_path)
    sink = DiskCheckpointSink(tmp_path)
    reference = sink.write_resource("champion", resource)
    case_hash = sink.write_case("STR-THRU_001", {"score": 1.25, "ready": True})
    manifest = sink.finalize({"release_id": "release-1"})

    assert json.loads((tmp_path / "cases" / "STR-THRU_001.json").read_text()) == {
        "ready": True,
        "score": 1.25,
    }
    assert manifest == json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["resources"] == [reference]
    assert manifest["cases"] == [{"case_id": "STR-THRU_001", "sha256": case_hash}]
    assert reference["path"] == "resources/model.bin"


def test_resume_finds_complete_unmanifested_cases_and_preserves_them(tmp_path: Path) -> None:
    first = DiskCheckpointSink(tmp_path)
    first_hash = first.write_case("case-1", {"value": 1})

    resumed = DiskCheckpointSink(tmp_path)
    assert resumed.completed_case_ids() == ("case-1",)
    resumed.write_case("case-2", {"value": 2})
    manifest = resumed.finalize({"run": "resumed"})

    assert manifest["cases"] == [
        {"case_id": "case-1", "sha256": first_hash},
        {"case_id": "case-2", "sha256": resumed.write_case("case-2", {"value": 2})},
    ]
    assert DiskCheckpointSink(tmp_path).completed_case_ids() == ("case-1", "case-2")


def test_identical_case_duplicate_is_idempotent_and_conflict_raises(tmp_path: Path) -> None:
    sink = DiskCheckpointSink(tmp_path)
    first = sink.write_case("same", {"nested": [1, 2]})
    assert sink.write_case("same", {"nested": [1, 2]}) == first
    with pytest.raises(CheckpointSinkError, match="conflicting duplicate"):
        sink.write_case("same", {"nested": [1, 3]})


def test_resource_deduplication_and_conflicting_redefinition(tmp_path: Path) -> None:
    first = _resource(tmp_path, "first.bin", b"same")
    second = _resource(tmp_path, "second.bin", b"same")
    sink = DiskCheckpointSink(tmp_path)
    reference = sink.write_resource("shared", first)
    assert sink.write_resource("shared", first) == reference
    with pytest.raises(CheckpointSinkError, match="conflicting redefinition"):
        sink.write_resource("shared", second)

    first.write_bytes(b"changed")
    with pytest.raises(CheckpointSinkError, match="hash conflict"):
        sink.finalize({})


def test_rejects_traversal_symlink_escape_and_malformed_ids(tmp_path: Path) -> None:
    outside = tmp_path.parent / (tmp_path.name + "-outside.bin")
    outside.write_bytes(b"outside")
    sink = DiskCheckpointSink(tmp_path)
    with pytest.raises(CheckpointSinkError, match="safe root-relative|escapes bundle root"):
        sink.write_resource("outside", Path("../" + outside.name))

    link = tmp_path / "escape.bin"
    link.symlink_to(outside)
    with pytest.raises(CheckpointSinkError, match="escapes bundle root"):
        sink.write_resource("link", link)

    for case_id in ("../case", "nested/case", "", "."):
        with pytest.raises(CheckpointSinkError, match="safe identifier"):
            sink.write_case(case_id, {})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_values(tmp_path: Path, value: float) -> None:
    sink = DiskCheckpointSink(tmp_path)
    with pytest.raises(CheckpointSinkError, match="non-finite"):
        sink.write_case("bad", {"nested": [value]})
    with pytest.raises(CheckpointSinkError, match="non-finite"):
        sink.finalize({"bad": value})


def test_manifest_hash_is_deterministic_across_order_and_reruns(tmp_path: Path) -> None:
    roots = [tmp_path / "one", tmp_path / "two"]
    manifests = []
    for index, root in enumerate(roots):
        resource = _resource(root, data=b"shared")
        sink = DiskCheckpointSink(root)
        case_ids = ("b", "a") if index == 0 else ("a", "b")
        for case_id in case_ids:
            sink.write_case(case_id, {"case": case_id})
        sink.write_resource("resource", resource)
        first = sink.finalize({"release": "fixed", "number": 1})
        second = sink.finalize({"number": 1, "release": "fixed"})
        assert first == second
        manifests.append(first)

    assert manifests[0] == manifests[1]


@pytest.mark.parametrize("kind", ["malformed", "bad_hash"])
def test_rejects_malformed_or_hash_invalid_manifest(tmp_path: Path, kind: str) -> None:
    manifest_path = tmp_path / "manifest.json"
    if kind == "malformed":
        manifest_path.write_text("{not-json", encoding="utf-8")
    else:
        manifest_path.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "metadata": {},
            "resources": [],
            "cases": [],
            "manifest_hash": "sha256:" + "0" * 64,
        }), encoding="utf-8")
    with pytest.raises(CheckpointSinkError):
        DiskCheckpointSink(tmp_path)


def test_stale_temp_files_are_removed_without_touching_completed_cases(tmp_path: Path) -> None:
    cases = tmp_path / "cases"
    cases.mkdir(parents=True)
    root_temp = tmp_path / ".manifest.json.dead.tmp"
    case_temp = cases / ".case.json.dead.tmp"
    root_temp.write_bytes(b"partial")
    case_temp.write_bytes(b"partial")

    sink = DiskCheckpointSink(tmp_path)
    sink.write_case("complete", {"ok": True})

    assert not root_temp.exists()
    assert not case_temp.exists()
    assert sink.completed_case_ids() == ("complete",)
