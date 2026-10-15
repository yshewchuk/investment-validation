from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from engine.v2.data.legacy_nightly_read_plan import NIGHTLY_CAPTURE_IMPLEMENTATION_REF
from engine.v2.foundation import SystemClock
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.workflows.commands import _read_input_manifest_ref
from engine.v2.ops.errors import OpsError


def _ops_root(tmp_path):
    root = tmp_path / "ops"
    root.mkdir(parents=True, exist_ok=True)
    clock = SystemClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    return root, conn, clock


def _incomplete_manifest_path(tmp_path):
    document = {"capture_implementation_ref": NIGHTLY_CAPTURE_IMPLEMENTATION_REF,
                "file_refs": []}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    return path


def test_nightly_plan_manifest_is_still_gated_on_barrier_families(tmp_path):
    root, conn, clock = _ops_root(tmp_path)
    args = SimpleNamespace(input_manifest=_incomplete_manifest_path(tmp_path))
    with pytest.raises(OpsError) as excinfo:
        _read_input_manifest_ref(args, root, conn, clock)
    assert excinfo.value.code == "INPUT_CHANGED"
    conn.close()


def test_training_plan_manifest_is_not_gated_on_nightly_barrier_families(tmp_path):
    root, conn, clock = _ops_root(tmp_path)
    args = SimpleNamespace(input_manifest=_incomplete_manifest_path(tmp_path))
    manifest_ref = _read_input_manifest_ref(args, root, conn, clock, nightly=False)
    assert manifest_ref
    conn.close()


def test_an_oversize_symlinked_or_missing_manifest_is_input_changed(tmp_path, monkeypatch):
    root, conn, clock = _ops_root(tmp_path)
    import engine.v2.ops.workflows.commands as commands

    missing = tmp_path / "nope.json"
    real = _incomplete_manifest_path(tmp_path)
    link = tmp_path / "link.json"
    link.symlink_to(real)
    monkeypatch.setattr(commands, "MAX_INPUT_MANIFEST_BYTES", 4)

    args = SimpleNamespace(input_manifest=missing)
    with pytest.raises(OpsError) as excinfo:
        _read_input_manifest_ref(args, root, conn, clock, nightly=False)
    assert excinfo.value.code == "INPUT_CHANGED"
    assert "input manifest is missing" in str(excinfo.value)

    for path in (link, real):
        args = SimpleNamespace(input_manifest=path)
        with pytest.raises(OpsError) as excinfo:
            _read_input_manifest_ref(args, root, conn, clock, nightly=False)
        assert excinfo.value.code == "INPUT_CHANGED"
        assert "not a bounded regular file" in str(excinfo.value)
    conn.close()
