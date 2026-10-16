"""P7-1 assembler: the manifest it writes passes the checker; the job graph
follows the live registry; a planted unaccounted job and bad specs fail."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess

import pytest

from checks import phase7_candidate_manifest as checker
from engine.v2.ops import stages, submission
from tools import phase7_candidate_manifest as asm

COMMIT = subprocess.run(["git", "-C", str(asm.ROOT), "rev-parse", "HEAD"], capture_output=True,
                        text=True, check=True).stdout.strip()


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


def _out(tmp_path):
    return tmp_path / "out"


def test_written_manifest_passes_checker(spec, tmp_path):
    returned = asm.assemble(spec, _out(tmp_path))
    on_disk = json.loads((_out(tmp_path) / "manifest.json").read_text())
    assert on_disk == returned
    assert on_disk["authority"] == spec["authority"]
    result = checker.validate(on_disk, root=tmp_path)
    assert result["ok"], result["findings"]
    assert set(stages.registry().names()) <= {j["id"] for j in on_disk["job_graph"]["jobs"]}


def test_new_registry_kind_appears_without_edit(spec, tmp_path, monkeypatch):
    real = stages.registry()
    base = real.get(real.names()[0])
    extra = dataclasses.replace(base, name="serving_projection_test")
    monkeypatch.setattr(stages, "registry", lambda: submission.KindRegistry(
        [real.get(n) for n in real.names()] + [extra]))
    manifest = asm.assemble(spec, _out(tmp_path))
    assert "serving_projection_test" in {j["id"] for j in manifest["job_graph"]["jobs"]}
    assert checker.validate(manifest, root=tmp_path)["ok"]


def test_official_kinds_are_scheduled_writers(monkeypatch):
    real = stages.registry()
    base = real.get(real.names()[0])
    official = dataclasses.replace(base, name="off_ns", namespaces=frozenset({"official"}))
    reader = dataclasses.replace(base, name="plain_reader", effects=())
    monkeypatch.setattr(stages, "registry", lambda: submission.KindRegistry([official, reader]))
    rows = {j["id"]: j for j in asm.registry_jobs("o", {"plain_reader"})}
    assert rows["off_ns"] == {"id": "off_ns", "schedule": "native:off_ns", "owner": "o",
                              "kind": "scheduled_writer", "writes_official": True}
    assert rows["plain_reader"]["kind"] == "scheduled_writer"
    assert rows["plain_reader"]["writes_official"] is True
    unlisted = {j["id"]: j for j in asm.registry_jobs("o")}
    assert unlisted["plain_reader"]["kind"] == "reader"
    assert unlisted["plain_reader"]["writes_official"] is False


def test_background_kind_with_effects(monkeypatch):
    real = stages.registry()
    base = dataclasses.replace(real.get(real.names()[0]), name="bg", effects=("staged",),
                               namespaces=frozenset({"shadow"}))
    monkeypatch.setattr(stages, "registry", lambda: submission.KindRegistry([base]))
    assert asm.registry_jobs("o")[0] == {"id": "bg", "schedule": "native:bg", "owner": "o",
                                         "kind": "background", "writes_official": False}


def test_planted_unaccounted_job_fails(spec, tmp_path):
    manifest = asm.assemble(spec, _out(tmp_path))
    graph = _out(tmp_path) / "job_graph.json"
    doc = json.loads(graph.read_text())
    doc["jobs"].append({"id": "rogue", "schedule": "rogue-cron", "writes_official": True})
    graph.write_text(json.dumps(doc))
    manifest["job_graph"]["sha256"] = hashlib.sha256(graph.read_bytes()).hexdigest()
    assert checker.UNACCOUNTED_JOB in checker.validate(manifest, root=tmp_path)["finding_codes"]


@pytest.mark.parametrize("mutate, match", [
    (lambda s, t: s["config"][0].update(path=str(t / "nope.json")), r"config\[0\]"),
    (lambda s, t: s.pop("commit"), "commit"),
    (lambda s, t: s.update(commit="b" * 40), "HEAD"),
    (lambda s, t: s.update(config={"name": "x"}), "config"),
    (lambda s, t: s.update(extra_jobs=["x"]), r"extra_jobs\[0\]"),
    (lambda s, t: s.update(data_snapshot=None), "data_snapshot"),
    (lambda s, t: s.update(registry_owner=""), "registry_owner"),
    (lambda s, t: s["extra_jobs"].append({"id": stages.registry().names()[0]}), "duplicate"),
    (lambda s, t: s["extra_jobs"].append({"id": "legacy-nightly"}), "duplicate"),
])
def test_bad_spec_is_spec_error(spec, tmp_path, mutate, match):
    mutate(spec, tmp_path)
    with pytest.raises(asm.SpecError, match=match):
        asm.assemble(spec, _out(tmp_path))


def test_out_dir_inside_repo_is_refused(spec):
    with pytest.raises(asm.SpecError, match="outside the repository"):
        asm.assemble(spec, asm.ROOT / "scratch_out")


def test_oversize_file_is_refused(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(asm, "MAX_REF_BYTES", 4)
    with pytest.raises(asm.SpecError, match="exceeds"):
        asm.assemble(spec, _out(tmp_path))


def test_writes_only_inside_out_dir(spec, tmp_path):
    before = set(tmp_path.iterdir())
    asm.assemble(spec, _out(tmp_path))
    assert set(tmp_path.iterdir()) - before == {_out(tmp_path)}


def test_main_exit_codes(spec, tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    assert asm.main(["--spec", str(path), "--out-dir", str(tmp_path / "o0")]) == 0
    spec["authority"]["proposed"] = dict(spec["authority"]["old"])
    path.write_text(json.dumps(spec))
    assert asm.main(["--spec", str(path), "--out-dir", str(tmp_path / "o1")]) == 1
    path.write_text("[]")
    assert asm.main(["--spec", str(path), "--out-dir", str(tmp_path / "o2")]) == 2
