"""P7-1 candidate/authority manifest validation: schema, identity and
ownership rules, with synthetic fixtures proving every negative control.

No real evidence, deployment or credential is read or written: the checker
takes a tree of small JSON files under ``tmp_path``. One test proves the run
is read-only, one proves it is deterministic.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from checks import phase7_candidate_manifest as p7

COMMIT = "a" * 40


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ref(path: Path, doc) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(doc, indent=2, sort_keys=True).encode()
    path.write_bytes(blob)
    return {"path": str(path), "sha256": _sha(blob)}


def good_tree(tmp_path: Path) -> dict:
    """A fully valid manifest plus its referenced files."""
    cfg = _ref(tmp_path / "candidate" / "supervisor.json", {"profile": "eod"})
    snapshot = _ref(tmp_path / "data" / "snapshot.json", {"snapshot_id": "snap-2026-09-24"})
    release = _ref(tmp_path / "models" / "manifest.json",
                   {"release_id": "rel-0007", "schema_version": "phase5_staged_release.v1.0"})
    graph = _ref(tmp_path / "ops" / "job_graph.json", {"jobs": [
        {"id": "native-eod", "schedule": "eod-v2", "writes_official": True},
        {"id": "preview-eod", "schedule": "eod-v1", "writes_official": True},
        {"id": "catalog-backup", "schedule": "nightly-backup", "writes_official": False},
        {"id": "board-reads", "schedule": "serve", "writes_official": False},
    ]})
    inventory = _ref(tmp_path / "ops" / "consumers.json", {"rows": [
        {"id": "board"}, {"id": "detail"}, {"id": "ledger-export"}, {"id": "research-replay"},
    ]})
    evidence = {
        "3B": {"schema_version": "phase3b_acceptance.v1.0"},
        "4": {"schema_version": "phase4_native_release.v1.0"},
        "5": {"schema_version": "phase5_acceptance.v1.0", "release_id": "rel-0007"},
        "6": {"schema_version": "phase6_acceptance.v1.0"},
    }
    phase_rows = []
    for phase, extra in evidence.items():
        doc = {"status": "PASS", "candidate_commit": COMMIT, **extra}
        ref = _ref(tmp_path / "evidence" / f"{phase}.json", doc)
        phase_rows.append({"phase": phase, "schema_version": extra["schema_version"],
                           "status": "PASS", "path": ref["path"], "sha256": ref["sha256"]})
    return {
        "schema_version": p7.MANIFEST_SCHEMA,
        "candidate": {"commit": COMMIT, "config": [
            {"name": "supervisor", "path": cfg["path"], "sha256": cfg["sha256"]}]},
        "data_snapshot": {"snapshot_id": "snap-2026-09-24",
                          "path": snapshot["path"], "sha256": snapshot["sha256"]},
        "model_deployment": {"release_id": "rel-0007",
                             "path": release["path"], "sha256": release["sha256"]},
        "job_graph": {"path": graph["path"], "sha256": graph["sha256"], "jobs": [
            {"id": "native-eod", "schedule": "eod-v2", "kind": "scheduled_writer",
             "owner": "v2-supervisor"},
            {"id": "preview-eod", "schedule": "eod-v1", "kind": "scheduled_writer",
             "owner": "legacy-supervisor"},
            {"id": "catalog-backup", "schedule": "nightly-backup", "kind": "background",
             "owner": "ops"},
            {"id": "board-reads", "schedule": "serve", "kind": "reader", "owner": "ops"},
        ]},
        "consumer_inventory": {"path": inventory["path"], "sha256": inventory["sha256"],
                               "consumers": ["board", "detail", "ledger-export",
                                             "research-replay"]},
        "authority": {
            "old": {"schedule": "eod-v1", "writer": "legacy-supervisor",
                    "credential_owner": "legacy-cred"},
            "proposed": {"schedule": "eod-v2", "writer": "v2-supervisor",
                         "credential_owner": "v2-cred"},
            "retained_old_deployment": {"ref": "deploy/legacy-2026-09-01",
                                        "rollback_owner": "ops-oncall"},
        },
        "phase_evidence": phase_rows,
    }


def findings_for(manifest: dict, tmp_path: Path) -> set[tuple[str, str]]:
    evidence = p7.validate(manifest, tmp_path)
    return {(f["code"], f["subject"]) for f in evidence["findings"]}


def codes_for(manifest: dict, tmp_path: Path) -> set[str]:
    return {f["code"] for f in p7.validate(manifest, tmp_path)["findings"]}


# ---------------------------------------------------------------------------
# the valid shape
# ---------------------------------------------------------------------------


def test_valid_manifest_has_no_findings(tmp_path):
    evidence = p7.validate(good_tree(tmp_path), tmp_path)
    assert evidence["findings"] == []
    assert evidence["ok"] and evidence["status"] == "MANIFEST_VALID"
    assert "not a Phase 7 readiness claim" in evidence["note"]


def test_deterministic_and_sorted(tmp_path):
    manifest = good_tree(tmp_path)
    first = json.dumps(p7.validate(manifest, tmp_path), sort_keys=True)
    second = json.dumps(p7.validate(copy.deepcopy(manifest), tmp_path), sort_keys=True)
    assert first == second
    bad = copy.deepcopy(manifest)
    bad["schema_version"] = "nope"
    rows = p7.validate(bad, tmp_path)["findings"]
    assert rows == sorted(rows, key=lambda r: (r["code"], r["subject"], r["detail"]))


def test_read_only(tmp_path):
    manifest = good_tree(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    p7.validate(manifest, tmp_path)
    bad = copy.deepcopy(manifest)
    bad["authority"]["old"]["writer"] = "latest"
    p7.validate(bad, tmp_path)
    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert before == after
    assert not (tmp_path / "authority").exists()


# ---------------------------------------------------------------------------
# negative controls: implicit/latest and identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mutate", [
    lambda m: m["data_snapshot"].update(snapshot_id="latest"),
    lambda m: m["model_deployment"].update(release_id="current"),
    lambda m: m["authority"]["retained_old_deployment"].update(ref="prod"),
    lambda m: m["authority"]["proposed"].update(credential_owner=""),
    lambda m: m["job_graph"]["jobs"][0].update(owner="whatever:latest"),
    lambda m: m["candidate"]["config"][0].update(name="default"),
])
def test_implicit_refs_rejected(tmp_path, mutate):
    manifest = good_tree(tmp_path)
    mutate(manifest)
    assert p7.LATEST_REF in codes_for(manifest, tmp_path)


def test_short_or_non_hex_commit_rejected(tmp_path):
    manifest = good_tree(tmp_path)
    manifest["candidate"]["commit"] = "e67868a"
    assert p7.BAD_CODE_IDENTITY in codes_for(manifest, tmp_path)
    manifest = good_tree(tmp_path)
    del manifest["candidate"]["commit"]
    assert p7.MISSING_FIELD in codes_for(manifest, tmp_path)


# ---------------------------------------------------------------------------
# negative controls: referenced files verified, not trusted
# ---------------------------------------------------------------------------


def test_hash_tampering_detected(tmp_path):
    manifest = good_tree(tmp_path)
    target = Path(manifest["phase_evidence"][0]["path"])
    doc = json.loads(target.read_text())
    doc["status"] = "PASS"  # unchanged
    target.write_text(json.dumps({**doc, "quiet_edit": True}, indent=2, sort_keys=True))
    assert p7.REF_HASH_MISMATCH in codes_for(manifest, tmp_path)


def test_missing_referenced_file_detected(tmp_path):
    manifest = good_tree(tmp_path)
    Path(manifest["data_snapshot"]["path"]).unlink()
    assert p7.REF_MISSING in codes_for(manifest, tmp_path)


def test_booleans_never_trusted(tmp_path):
    """An evidence file that says ``ok: true`` but disagrees on status fails."""
    manifest = good_tree(tmp_path)
    row = manifest["phase_evidence"][2]  # the Phase 5 row
    target = Path(row["path"])
    blob = json.dumps({"ok": True, "passed": True, "status": "FAIL",
                       "schema_version": row["schema_version"],
                       "candidate_commit": COMMIT, "release_id": "rel-0007"},
                      indent=2, sort_keys=True).encode()
    target.write_bytes(blob)
    row["sha256"] = _sha(blob)
    assert p7.EVIDENCE_STALE in codes_for(manifest, tmp_path)


def test_evidence_bound_to_other_candidate_is_stale(tmp_path):
    manifest = good_tree(tmp_path)
    row = manifest["phase_evidence"][1]  # the Phase 4 row
    target = Path(row["path"])
    blob = json.dumps({"status": "PASS", "schema_version": row["schema_version"],
                       "candidate_commit": "b" * 40}, indent=2, sort_keys=True).encode()
    target.write_bytes(blob)
    row["sha256"] = _sha(blob)
    assert p7.EVIDENCE_STALE in codes_for(manifest, tmp_path)


def test_malformed_recorded_hash_rejected(tmp_path):
    manifest = good_tree(tmp_path)
    manifest["data_snapshot"]["sha256"] = "not-a-hash"
    assert (p7.BAD_HASH, "data_snapshot") in findings_for(manifest, tmp_path)


def test_malformed_referenced_json_detected(tmp_path):
    manifest = good_tree(tmp_path)
    target = Path(manifest["job_graph"]["path"])
    blob = b"{not json"
    target.write_bytes(blob)
    manifest["job_graph"]["sha256"] = _sha(blob)
    assert p7.REF_MALFORMED in codes_for(manifest, tmp_path)


# ---------------------------------------------------------------------------
# negative controls: job graph accounting
# ---------------------------------------------------------------------------


def test_unaccounted_graph_job_rejected(tmp_path):
    manifest = good_tree(tmp_path)
    graph_file = Path(manifest["job_graph"]["path"])
    doc = json.loads(graph_file.read_text())
    doc["jobs"].append({"id": "shadow-promoter", "schedule": "eod-v2",
                        "writes_official": False})
    blob = json.dumps(doc, indent=2, sort_keys=True).encode()
    graph_file.write_bytes(blob)
    manifest["job_graph"]["sha256"] = _sha(blob)
    found = findings_for(manifest, tmp_path)
    assert (p7.UNACCOUNTED_JOB, "job:shadow-promoter") in found


def test_phantom_accounted_job_rejected(tmp_path):
    """A job the manifest accounts for that the graph does not run is a lie."""
    manifest = good_tree(tmp_path)
    manifest["job_graph"]["jobs"].append({"id": "retired-cron", "schedule": "old-cron",
                                          "kind": "background", "owner": "ops"})
    assert (p7.PHANTOM_JOB, "job:retired-cron") in findings_for(manifest, tmp_path)


def test_unknown_job_kind_rejected(tmp_path):
    manifest = good_tree(tmp_path)
    for job in manifest["job_graph"]["jobs"]:
        if job["id"] == "catalog-backup":
            job["kind"] = "cron"
    assert (p7.BAD_JOB_KIND, "job:catalog-backup") in findings_for(manifest, tmp_path)


def test_duplicate_job_ids_rejected(tmp_path):
    manifest = good_tree(tmp_path)
    manifest["job_graph"]["jobs"].append(copy.deepcopy(manifest["job_graph"]["jobs"][0]))
    assert (p7.DUPLICATE, "job:native-eod") in findings_for(manifest, tmp_path)


def test_hidden_background_writer_rejected(tmp_path):
    """A background job the graph actually lets write officially competes."""
    manifest = good_tree(tmp_path)
    for job in manifest["job_graph"]["jobs"]:
        if job["id"] == "catalog-backup":
            job["kind"] = "background"
    graph_file = Path(manifest["job_graph"]["path"])
    doc = json.loads(graph_file.read_text())
    for job in doc["jobs"]:
        if job["id"] == "catalog-backup":
            job["writes_official"] = True
    blob = json.dumps(doc, indent=2, sort_keys=True).encode()
    graph_file.write_bytes(blob)
    manifest["job_graph"]["sha256"] = _sha(blob)
    found = findings_for(manifest, tmp_path)
    assert (p7.WRITER_CONFLICT, "job:catalog-backup") in found


def test_official_writer_job_owned_by_undeclared_writer_rejected(tmp_path):
    manifest = good_tree(tmp_path)
    for job in manifest["job_graph"]["jobs"]:
        if job["id"] == "native-eod":
            job["owner"] = "third-supervisor"
    found = findings_for(manifest, tmp_path)
    assert (p7.WRITER_CONFLICT, "job:native-eod") in found
    assert (p7.WRITER_NO_JOB, "authority.proposed.writer") in found


def test_duplicate_official_writer_per_side_rejected(tmp_path):
    manifest = good_tree(tmp_path)
    manifest["job_graph"]["jobs"][0]["schedule"] = "eod-v2-a"
    extra = {"id": "native-eod-2", "schedule": "eod-v2-b", "kind": "scheduled_writer",
             "owner": "v2-supervisor"}
    manifest["job_graph"]["jobs"].append(extra)
    graph_file = Path(manifest["job_graph"]["path"])
    doc = json.loads(graph_file.read_text())
    doc["jobs"].append({"id": "native-eod-2", "schedule": "eod-v2-b",
                        "writes_official": True})
    blob = json.dumps(doc, indent=2, sort_keys=True).encode()
    graph_file.write_bytes(blob)
    manifest["job_graph"]["sha256"] = _sha(blob)
    manifest["authority"]["proposed"]["schedule"] = "eod-v2-a"
    found = findings_for(manifest, tmp_path)
    assert (p7.DUPLICATE_OFFICIAL_WRITER, "authority.proposed.writer") in found


def test_duplicate_schedule_rejected(tmp_path):
    manifest = good_tree(tmp_path)
    manifest["job_graph"]["jobs"][2]["schedule"] = "eod-v1"  # backup on the writer schedule
    graph_file = Path(manifest["job_graph"]["path"])
    doc = json.loads(graph_file.read_text())
    for job in doc["jobs"]:
        if job["id"] == "catalog-backup":
            job["schedule"] = "eod-v1"
    blob = json.dumps(doc, indent=2, sort_keys=True).encode()
    graph_file.write_bytes(blob)
    manifest["job_graph"]["sha256"] = _sha(blob)
    assert (p7.DUPLICATE, "schedule:eod-v1") in findings_for(manifest, tmp_path)


# ---------------------------------------------------------------------------
# negative controls: ownership, retention and consumers
# ---------------------------------------------------------------------------


def test_missing_old_or_proposed_ownership_rejected(tmp_path):
    manifest = good_tree(tmp_path)
    del manifest["authority"]["old"]["credential_owner"]
    found = codes_for(manifest, tmp_path)
    assert p7.OWNERSHIP_INCOMPLETE in found and p7.MISSING_FIELD in found


def test_old_deployment_retention_required(tmp_path):
    manifest = good_tree(tmp_path)
    del manifest["authority"]["retained_old_deployment"]
    found = codes_for(manifest, tmp_path)
    assert p7.OLD_DEPLOYMENT_MISSING in found and p7.MISSING_FIELD in found


def test_no_op_switch_rejected(tmp_path):
    """old == proposed on every role is not a cutover manifest."""
    manifest = good_tree(tmp_path)
    manifest["authority"]["proposed"] = copy.deepcopy(manifest["authority"]["old"])
    assert p7.WRITER_NO_SWITCH in codes_for(manifest, tmp_path)


def test_unlisted_consumer_rejected(tmp_path):
    manifest = good_tree(tmp_path)
    manifest["consumer_inventory"]["consumers"].append("phone-offline")
    found = findings_for(manifest, tmp_path)
    assert (p7.CONSUMER_UNACCOUNTED, "consumer:phone-offline") in found


def test_empty_consumer_list_rejected(tmp_path):
    manifest = good_tree(tmp_path)
    manifest["consumer_inventory"]["consumers"] = []
    assert p7.BAD_LIST in codes_for(manifest, tmp_path)


# ---------------------------------------------------------------------------
# negative controls: phase evidence
# ---------------------------------------------------------------------------


def test_missing_phase_evidence_rejected(tmp_path):
    manifest = good_tree(tmp_path)
    manifest["phase_evidence"] = [r for r in manifest["phase_evidence"] if r["phase"] != "6"]
    found = findings_for(manifest, tmp_path)
    assert (p7.MISSING_PHASE_EVIDENCE, "phase:6") in found


def test_non_required_phase_row_rejected(tmp_path):
    """A 3A preview row (or any phase outside 3B/4/5/6) is not evidence P7-1 wants."""
    manifest = good_tree(tmp_path)
    manifest["phase_evidence"][0]["phase"] = "3A"
    found = codes_for(manifest, tmp_path)
    assert p7.BAD_PHASE in found and p7.MISSING_PHASE_EVIDENCE in found


def test_preview_evidence_cannot_substitute(tmp_path):
    """A retained 3A preview receipt filed as Phase 3B fails the schema binding."""
    manifest = good_tree(tmp_path)
    for row in manifest["phase_evidence"]:
        if row["phase"] == "3B":
            doc = json.loads(Path(row["path"]).read_text())
            doc["schema_version"] = "phase3_evidence.v1.0"
            blob = json.dumps(doc, indent=2, sort_keys=True).encode()
            Path(row["path"]).write_bytes(blob)
            row["sha256"] = _sha(blob)
            row["schema_version"] = "phase3_evidence.v1.0"
    found = findings_for(manifest, tmp_path)
    assert (p7.SCHEMA_VERSION, "phase:3B") in found


def test_phase_five_release_binding(tmp_path):
    """Phase 5 evidence must name the deployed release, not an older one."""
    manifest = good_tree(tmp_path)
    row = next(r for r in manifest["phase_evidence"] if r["phase"] == "5")
    target = Path(row["path"])
    blob = json.dumps({"status": "PASS", "schema_version": row["schema_version"],
                       "candidate_commit": COMMIT, "release_id": "rel-0005"},
                      indent=2, sort_keys=True).encode()
    target.write_bytes(blob)
    row["sha256"] = _sha(blob)
    assert (p7.REF_MISMATCH, "phase:5") in findings_for(manifest, tmp_path)


def test_unknown_and_missing_top_level_keys_rejected(tmp_path):
    manifest = good_tree(tmp_path)
    manifest["phase7_approved"] = True
    del manifest["data_snapshot"]
    found = codes_for(manifest, tmp_path)
    assert p7.UNKNOWN_KEY in found and p7.MISSING_FIELD in found


def test_bad_manifest_schema_version_rejected(tmp_path):
    manifest = good_tree(tmp_path)
    manifest["schema_version"] = "phase7_candidate_manifest.v0.9"
    assert (p7.SCHEMA_VERSION, "manifest") in findings_for(manifest, tmp_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_human_and_json_modes(tmp_path, capsys):
    manifest = good_tree(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert p7.main([str(path), "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "MANIFEST_VALID (0 findings)" in out and "not a Phase 7 readiness claim" in out

    manifest["candidate"]["commit"] = "latest"
    path.write_text(json.dumps(manifest))
    assert p7.main([str(path), "--root", str(tmp_path), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "MANIFEST_INVALID"
    assert any(f["code"] == p7.LATEST_REF for f in payload["findings"])


def test_cli_unreadable_manifest_exits_two(tmp_path, capsys):
    assert p7.main([str(tmp_path / "nope.json")]) == 2
    assert "cannot read manifest" in capsys.readouterr().err
