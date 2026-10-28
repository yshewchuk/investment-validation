from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.ops import cli
from engine.v2.ops.errors import OpsError
from engine.v2.ops.plans import nightly_plan
from engine.v2.ops.workflows import commands
from tests.ops_support import FakeClock


def test_nightly_plan_hashes_pinned_snapshot_id_for_legacy_and_native_modes():
    source_root = str(Path(__file__).resolve().parents[3])
    common = {
        "manifest_ref": "artifact:manifest",
        "expected_population": ("FAKE|TWIN-P|2026-10-28",),
    }
    legacy_a = nightly_plan(source_root, "2026-10-28", snapshot_id="snap-a", **common)
    legacy_b = nightly_plan(source_root, "2026-10-28", snapshot_id="snap-b", **common)
    assert legacy_a["snapshot_id"] == "snap-a"
    assert legacy_a["plan_hash"] != legacy_b["plan_hash"]

    native_inputs = {
        "snapshot_id": "snap-native",
        "snapshot_ref_artifact_id": "artifact:snapshot",
        "materialization_request_ref": "artifact:request",
        "scratch_estimate_bytes": 1,
    }
    native = nightly_plan(source_root, "2026-10-28", input_mode="snapshot",
                          snapshot_inputs=native_inputs, **common)
    assert native["snapshot_id"] == "snap-native"
    assert native["snapshot_inputs"] == native_inputs
    with pytest.raises(OpsError) as raised:
        nightly_plan(source_root, "2026-10-28", input_mode="snapshot",
                     snapshot_inputs=native_inputs, snapshot_id="snap-other", **common)
    assert raised.value.problem.code == "INVALID_REQUEST"


def test_legacy_ops_plan_resolves_shadow_head_once_and_saves_its_id(monkeypatch, tmp_path):
    args = cli.parser().parse_args(["plan", "nightly", "--as-of", "2026-10-28"])
    resolved = []

    class Store:
        def __init__(self, root):
            self.root = root

        def read_verified(self, ref):
            assert ref.artifact_id == "snapshot-ref"
            return b'{"snapshot_id":"snap-shadow"}'

    def resolve_head(conn, store, scope, *, clock):
        resolved.append(scope)
        assert scope == "shadow"
        return SimpleNamespace(artifact_id="snapshot-ref")

    monkeypatch.setattr(commands, "ArtifactStore", Store)
    monkeypatch.setattr(commands, "_planned_population",
                        lambda *args: (("FAKE|TWIN-P|2026-10-28",), None, ()))
    monkeypatch.setattr(commands, "_snapshot_inputs", lambda *args: None)
    monkeypatch.setattr(commands, "_read_input_manifest_ref", lambda *args: "artifact:manifest")
    monkeypatch.setattr(commands, "_read_refresh_plan", lambda args: None)
    monkeypatch.setattr(commands, "_retrain_section", lambda *args: None)
    monkeypatch.setattr(commands, "save_plan",
                        lambda conn, root, plan, *, clock: SimpleNamespace(artifact_id="plan-ref"))
    monkeypatch.setattr("engine.v2.ops.snapshots.resolve_snapshot_head", resolve_head)
    monkeypatch.setattr("engine.v2.ops.snapshot_stages._catalog_path",
                        lambda conn: "catalog.sqlite")

    result = commands._plan_command(args, tmp_path, object(), FakeClock())

    assert resolved == ["shadow"]
    assert result["plan"]["snapshot_id"] == "snap-shadow"
