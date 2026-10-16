"""P7-1 assembler: the manifest it writes passes the checker; the job graph
follows the live registry; a planted unaccounted job and bad specs fail."""
from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from checks import phase7_candidate_manifest as checker
from engine.v2.ops import stages, submission
from tools import phase7_candidate_manifest as asm

COMMIT = "a" * 40


def _write(path, doc):
    path.write_text(json.dumps(doc))
    return str(path)


@pytest.fixture
def spec(tmp_path):
    evidence = []
    for phase, schema in (("3B", "phase3b.x"), ("4", "phase4.x"), ("5", "phase5.x"),
                          ("6", "phase6.x")):
        doc = {"schema_version": schema, "status": "PASS", "commit": COMMIT}
        if phase == "5":
            doc["release_id"] = "rel-1"
        evidence.append({"phase": phase, "schema_version": schema, "status": "PASS",
                         "path": _write(tmp_path / f"ev{phase}.json", doc)})
    return {
        "commit": COMMIT, "registry_owner": "native-ops",
        "config": [{"name": "cfg", "path": _write(tmp_path / "cfg.json", {"a": 1})}],
        "data_snapshot": {"snapshot_id": "snap-1",
                          "path": _write(tmp_path / "snap.json", {"snapshot_id": "snap-1"})},
        "model_deployment": {"release_id": "rel-1",
                             "path": _write(tmp_path / "rel.json", {"release_id": "rel-1"})},
        "phase_evidence": evidence,
        "extra_jobs": [
            {"id": "legacy-nightly", "schedule": "legacy-nightly", "kind": "scheduled_writer",
             "owner": "legacy-writer", "writes_official": True},
            {"id": "native-official", "schedule": "native-official", "kind": "scheduled_writer",
             "owner": "native-writer", "writes_official": True}],
        "authority": {
            "old": {"schedule": "legacy-nightly", "writer": "legacy-writer",
                    "credential_owner": "legacy-cred"},
            "proposed": {"schedule": "native-official", "writer": "native-writer",
                         "credential_owner": "native-cred"},
            "retained_old_deployment": {"ref": "legacy-deploy-1", "rollback_owner": "ops"}},
    }


def test_assembled_manifest_passes_checker(spec, tmp_path):
    manifest = asm.assemble(spec, tmp_path / "out")
    result = checker.validate(manifest, root=tmp_path)
    assert result["ok"], result["findings"]
    ids = {j["id"] for j in manifest["job_graph"]["jobs"]}
    assert set(stages.registry().names()) <= ids


def test_new_registry_kind_appears_without_edit(spec, tmp_path, monkeypatch):
    real = stages.registry()
    base = real.get(real.names()[0])
    extra = dataclasses.replace(base, name="serving_projection_test")
    monkeypatch.setattr(stages, "registry", lambda: submission.KindRegistry(
        [real.get(n) for n in real.names()] + [extra]))
    manifest = asm.assemble(spec, tmp_path / "out")
    assert "serving_projection_test" in {j["id"] for j in manifest["job_graph"]["jobs"]}
    assert checker.validate(manifest, root=tmp_path)["ok"]


def test_planted_unaccounted_job_fails(spec, tmp_path):
    manifest = asm.assemble(spec, tmp_path / "out")
    graph = tmp_path / "out" / "job_graph.json"
    doc = json.loads(graph.read_text())
    doc["jobs"].append({"id": "rogue", "schedule": "rogue-cron", "writes_official": True})
    graph.write_text(json.dumps(doc))
    manifest["job_graph"]["sha256"] = hashlib.sha256(graph.read_bytes()).hexdigest()
    assert checker.UNACCOUNTED_JOB in checker.validate(manifest, root=tmp_path)["finding_codes"]


def test_missing_referenced_file_is_spec_error(spec, tmp_path):
    spec["config"][0]["path"] = str(tmp_path / "nope.json")
    with pytest.raises(asm.SpecError, match="config"):
        asm.assemble(spec, tmp_path / "out")


def test_extra_job_colliding_with_registry_is_spec_error(spec, tmp_path):
    spec["extra_jobs"].append({"id": stages.registry().names()[0], "schedule": "x",
                               "kind": "reader", "owner": "o", "writes_official": False})
    with pytest.raises(asm.SpecError, match="collides"):
        asm.assemble(spec, tmp_path / "out")


def test_writes_only_inside_out_dir(spec, tmp_path):
    before = set(tmp_path.iterdir())
    asm.assemble(spec, tmp_path / "out")
    assert set(tmp_path.iterdir()) - before == {tmp_path / "out"}