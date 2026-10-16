"""P7-1 assembler: the manifest it writes passes the checker; the job graph
follows the live registry; a planted unaccounted job and bad specs fail."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from checks import phase7_candidate_manifest as checker
from engine.v2.ops import stages, submission
from tools import phase7_candidate_manifest as asm

COMMIT = "a" * 40
_REAL_COMMIT = asm.check_commit


@pytest.fixture(autouse=True)
def _stub_commit(monkeypatch):
    """Bind the spec commit without consulting the (possibly dirty) checkout."""
    monkeypatch.setattr(asm, "check_commit", lambda spec, root: spec["commit"]
                        if isinstance(spec.get("commit"), str) and spec["commit"]
                        else _REAL_COMMIT(spec, root))


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
        "non_official_registry_kinds": ["experiment", "models_promote", "models_rollback"],
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
    reader = dataclasses.replace(base, name="plain_reader", effects=(),
                                 namespaces=frozenset({"shadow"}))
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
    (lambda s, t: s.update(commit=5), "required"),
    (lambda s, t: s.pop("commit"), "required"),
    (lambda s, t: s["extra_jobs"].append({"id": "z1", "schedule": "", "owner": "o",
                                          "kind": "reader", "writes_official": False}),
     r"extra_jobs\[z1\]\.schedule"),
    (lambda s, t: s["extra_jobs"].append({"id": "z2", "schedule": "s", "owner": "o",
                                          "writes_official": False}),
     r"extra_jobs\[z2\]\.kind"),
    (lambda s, t: s.update(model_deployment=[]), "model_deployment"),
    (lambda s, t: s["config"][0].update(name=""), r"config\[0\]\.name"),
    (lambda s, t: s["phase_evidence"][0].update(phase=""), r"phase_evidence\[0\]\.phase"),
    (lambda s, t: s["phase_evidence"][0].update(schema_version=None),
     r"phase_evidence\[0\]\.schema_version"),
    (lambda s, t: s.pop("non_official_registry_kinds"), "experiment"),
    (lambda s, t: s.update(official_registry_kinds="publication"), "official_registry_kinds"),
    (lambda s, t: s.update(official_registry_kinds=["no_such_kind"]), "official_registry_kinds"),
    (lambda s, t: s.update(official_registry_kinds=None), "official_registry_kinds"),
    (lambda s, t: s["extra_jobs"].append({"id": "x", "schedule": "s", "kind": "reader",
                                          "writes_official": False}), r"extra_jobs\[x\]\.owner"),
    (lambda s, t: s["extra_jobs"].append({"id": "y", "schedule": "s", "kind": "reader",
                                          "owner": "o", "writes_official": "false"}),
     "writes_official"),
    (lambda s, t: s["phase_evidence"][0].update(status=5), r"phase_evidence\[0\]\.status"),
    (lambda s, t: s["data_snapshot"].pop("snapshot_id"), "snapshot_id"),
    (lambda s, t: s["model_deployment"].update(release_id=""), "release_id"),
    (lambda s, t: s.update(authority=[]), "authority"),
    (lambda s, t: s["config"][0].update(path=5), "path must be a string"),

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


def test_cap_is_inclusive_and_shared_with_the_checker(tmp_path, monkeypatch):
    path = tmp_path / "f.bin"
    path.write_bytes(b"12345")
    monkeypatch.setattr(asm, "MAX_REF_BYTES", 5)
    assert asm.hash_ref("f", path)[1] == hashlib.sha256(b"12345").hexdigest()
    monkeypatch.setattr(asm, "MAX_REF_BYTES", 4)
    with pytest.raises(asm.SpecError, match="exceeds"):
        asm.hash_ref("f", path)
    assert checker.MAX_REF_BYTES == 256 * 1024 * 1024
    monkeypatch.undo()
    assert asm.MAX_REF_BYTES == checker.MAX_REF_BYTES
    assert asm.MAX_SPEC_BYTES == 1024 * 1024


def test_checker_reference_cap_boundary(spec, tmp_path, monkeypatch):
    manifest = asm.assemble(spec, _out(tmp_path))
    refs = [manifest["candidate"]["config"][0], manifest["data_snapshot"],
            manifest["model_deployment"], manifest["job_graph"], manifest["consumer_inventory"],
            *manifest["phase_evidence"]]
    biggest = max(Path(r["path"]).stat().st_size for r in refs)
    monkeypatch.setattr(checker, "MAX_REF_BYTES", biggest)
    assert checker.REF_MALFORMED not in checker.validate(manifest, root=tmp_path)["finding_codes"]
    monkeypatch.setattr(checker, "MAX_REF_BYTES", biggest - 1)
    assert checker.REF_MALFORMED in checker.validate(manifest, root=tmp_path)["finding_codes"]


def test_spec_cap_boundary(tmp_path, monkeypatch):
    path = tmp_path / "spec.json"
    path.write_bytes(b'{"a":1}')
    monkeypatch.setattr(asm, "MAX_SPEC_BYTES", 7)
    assert asm.load_spec(path) == {"a": 1}
    monkeypatch.setattr(asm, "MAX_SPEC_BYTES", 6)
    with pytest.raises(asm.SpecError, match="spec exceeds"):
        asm.load_spec(path)


def test_overlapping_registry_kind_lists_are_refused(spec, tmp_path):
    spec["official_registry_kinds"] = ["experiment"]
    with pytest.raises(asm.SpecError, match="overlap"):
        asm.assemble(spec, _out(tmp_path))


def test_uncreatable_out_dir_exits_2(spec, tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert asm.main(["--spec", str(path), "--out-dir", str(blocker / "sub")]) == 2


def _with_consumers(manifest, tmp_path, names):
    doc = tmp_path / "inv.json"
    doc.write_text(json.dumps({"consumers": [{"id": n} for n in names]}))
    manifest["consumer_inventory"] = {
        "path": str(doc), "sha256": hashlib.sha256(doc.read_bytes()).hexdigest(),
        "consumers": names}
    return checker.validate(manifest, root=tmp_path)["finding_codes"]


def test_consumer_names_are_labels_but_not_blank(spec, tmp_path):
    manifest = asm.assemble(spec, _out(tmp_path))
    codes = _with_consumers(manifest, tmp_path, ["GET /release/current", "latest"])
    assert checker.LATEST_REF not in codes and checker.MISSING_FIELD not in codes
    assert checker.MISSING_FIELD in _with_consumers(manifest, tmp_path, ["   "])
    assert checker.MISSING_FIELD in _with_consumers(manifest, tmp_path, [""])


def _git(repo, *args):
    env = {k: v for k, v in os.environ.items()
           if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")}
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-C", str(repo), *args],
                   check=True, capture_output=True, env=env)


def test_dirty_tracked_tree_is_refused(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("1")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "x")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert _REAL_COMMIT({"commit": head}, repo) == head
    with pytest.raises(asm.SpecError, match="HEAD"):
        _REAL_COMMIT({"commit": "b" * 40}, repo)
    (repo / "untracked.txt").write_text("u")
    assert _REAL_COMMIT({"commit": head}, repo) == head
    (repo / "a.txt").write_text("2")
    with pytest.raises(asm.SpecError, match="uncommitted"):
        _REAL_COMMIT({"commit": head}, repo)


def test_undecided_primary_kind_is_refused():
    with pytest.raises(asm.SpecError, match="experiment"):
        asm.registry_jobs("o")
    decided = ("experiment", "models_promote", "models_rollback")
    rows = {j["id"]: j for j in asm.registry_jobs("o", (), decided)}
    assert rows["experiment"]["writes_official"] is False
    rows = {j["id"]: j for j in asm.registry_jobs("o", ("experiment",), decided[1:])}
    assert rows["experiment"]["kind"] == "scheduled_writer"


def test_store_write_lease_kind_needs_a_decision(monkeypatch):
    real = stages.registry()
    base = real.get(real.names()[0])
    ptr = dataclasses.replace(base, name="ptr_writer", namespaces=frozenset({"shadow"}),
                              store_domains=(("deployment_pointer", "write"),))
    monkeypatch.setattr(stages, "registry", lambda: submission.KindRegistry([ptr]))
    with pytest.raises(asm.SpecError, match="ptr_writer"):
        asm.registry_jobs("o")
    assert asm.registry_jobs("o", ("ptr_writer",))[0]["kind"] == "scheduled_writer"
    assert asm.registry_jobs("o", (), ("ptr_writer",))[0]["writes_official"] is False


def test_writes_exactly_three_files_inside_out_dir(spec, tmp_path):
    before = set(tmp_path.iterdir())
    asm.assemble(spec, _out(tmp_path))
    assert set(tmp_path.iterdir()) - before == {_out(tmp_path)}
    assert sorted(p.name for p in _out(tmp_path).iterdir()) == [
        "consumer_inventory.json", "job_graph.json", "manifest.json"]


def test_spec_listed_official_kind_is_a_writer_in_the_graph(spec, tmp_path):
    name = stages.registry().names()[0]
    spec["official_registry_kinds"] = [name]
    asm.assemble(spec, _out(tmp_path))
    rows = {j["id"]: j for j in json.loads((_out(tmp_path) / "job_graph.json").read_text())["jobs"]}
    assert rows[name]["writes_official"] is True
    manifest = json.loads((_out(tmp_path) / "manifest.json").read_text())
    assert {j["id"]: j for j in manifest["job_graph"]["jobs"]}[name]["kind"] == "scheduled_writer"


def test_main_exit_codes(spec, tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    assert asm.main(["--spec", str(path), "--out-dir", str(tmp_path / "o0")]) == 0
    spec["authority"]["proposed"] = dict(spec["authority"]["old"])
    path.write_text(json.dumps(spec))
    assert asm.main(["--spec", str(path), "--out-dir", str(tmp_path / "o1")]) == 1
    assert (tmp_path / "o1" / "manifest.json").is_file()
    path.write_text("[]")
    assert asm.main(["--spec", str(path), "--out-dir", str(tmp_path / "o2")]) == 2
    assert not (tmp_path / "o2").exists()


def test_main_unusable_spec_files_exit_2(tmp_path):
    out = str(tmp_path / "o")
    assert asm.main(["--spec", str(tmp_path / "missing.json"), "--out-dir", out]) == 2
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe\x00")
    assert asm.main(["--spec", str(bad), "--out-dir", out]) == 2
    bad.write_text("{not json")
    assert asm.main(["--spec", str(bad), "--out-dir", out]) == 2


def test_spec_over_cap_is_refused_with_its_message(spec, tmp_path, monkeypatch, capsys):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(asm, "MAX_SPEC_BYTES", 100)
    assert asm.main(["--spec", str(path), "--out-dir", str(tmp_path / "o")]) == 2
    assert "spec exceeds" in capsys.readouterr().err


def test_nested_out_dir_and_rerun(spec, tmp_path):
    out = tmp_path / "a" / "b" / "out"
    asm.assemble(spec, out)
    asm.assemble(spec, out)
    assert sorted(p.name for p in out.iterdir()) == [
        "consumer_inventory.json", "job_graph.json", "manifest.json"]


@pytest.mark.parametrize("mutate, match", [
    (lambda s, t: s["data_snapshot"].update(snapshot_id="other"), r"data_snapshot\.snapshot_id"),
    (lambda s, t: s["model_deployment"].update(release_id="other"),
     r"model_deployment\.release_id"),
    (lambda s, t: s["phase_evidence"][0].update(status="FAIL"), r"phase_evidence\[0\]\.status"),
    (lambda s, t: s["phase_evidence"][0].update(schema_version="phase3b.y"),
     r"phase_evidence\[0\]\.schema_version"),
])
def test_spec_identity_must_match_the_document(spec, tmp_path, mutate, match):
    mutate(spec, tmp_path)
    with pytest.raises(asm.SpecError, match=match):
        asm.assemble(spec, _out(tmp_path))


def test_real_commit_check_runs_inside_assemble_and_main(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(asm, "check_commit", _REAL_COMMIT)
    spec["commit"] = "b" * 40
    with pytest.raises(asm.SpecError, match="HEAD"):
        asm.assemble(spec, _out(tmp_path))
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    assert asm.main(["--spec", str(path), "--out-dir", str(tmp_path / "o")]) == 2
    assert not (tmp_path / "o").exists()


def test_written_files_are_canonical_and_sorted(spec, tmp_path):
    asm.assemble(spec, _out(tmp_path))
    for name in ("manifest.json", "job_graph.json", "consumer_inventory.json"):
        text = (_out(tmp_path) / name).read_text()
        assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
    manifest = json.loads((_out(tmp_path) / "manifest.json").read_text())
    ids = [j["id"] for j in manifest["job_graph"]["jobs"]]
    assert ids == sorted(ids)
    consumers = manifest["consumer_inventory"]["consumers"]
    assert consumers == sorted(consumers) and len(consumers) > 1
    inventory = json.loads((_out(tmp_path) / "consumer_inventory.json").read_text())
    assert [c["id"] for c in inventory["consumers"]] == consumers


def test_manifest_digests_match_the_files_as_written(spec, tmp_path):
    manifest = asm.assemble(spec, _out(tmp_path))
    for section in ("job_graph", "consumer_inventory"):
        data = Path(manifest[section]["path"]).read_bytes()
        assert manifest[section]["sha256"] == hashlib.sha256(data).hexdigest()


def test_discovery_failure_is_exit_2_and_creates_nothing(spec, tmp_path, monkeypatch):
    def boom(root):
        raise SyntaxError("bad module")
    monkeypatch.setattr(asm, "discover_entrypoints", boom)
    with pytest.raises(asm.SpecError, match="consumer discovery failed"):
        asm.assemble(spec, _out(tmp_path))
    assert not _out(tmp_path).exists()
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    assert asm.main(["--spec", str(path), "--out-dir", str(tmp_path / "o")]) == 2
