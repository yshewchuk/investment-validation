#!/usr/bin/env python3
"""P7-1 assembler: build the candidate/authority manifest the checker validates.

Operator-run only (no production caller). Writes three JSON files into
``--out-dir`` (refused when it is inside the repository) and nothing else.
One operator spec file (kept outside the repository: it holds the legacy
scheduler entry, credential owners and local evidence paths) names the
commit, config, snapshot, release, phase evidence, ``authority`` and any
extra jobs the registry cannot know. The commit is required and must equal
this checkout's ``HEAD``, because the job graph (live ``stages.registry()``)
and the consumer inventory both come from this checkout. A job kind added
later (#726's ``serving_projection``) appears with no edit here. Exit 0 = the
checker passes, 1 = findings, 2 = unusable spec.

    python3 tools/phase7_candidate_manifest.py --spec SPEC.json --out-dir DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checks import phase7_candidate_manifest as checker  # noqa: E402
from engine.v2.ops import stages  # noqa: E402
from tools.phase6_inventory import discover_entrypoints  # noqa: E402

MAX_REF_BYTES = checker.MAX_REF_BYTES
MAX_SPEC_BYTES = 1024 * 1024
_JOB_FIELDS = ("id", "schedule", "owner", "kind", "writes_official")


class SpecError(Exception):
    """The operator spec (or a file it names) cannot be used."""


def _sha256(field: str, path) -> tuple[str, str]:
    if not isinstance(path, (str, Path)):
        raise SpecError(f"{field}: path must be a string")
    target = Path(path).resolve()
    if not target.is_file():
        raise SpecError(f"{field}: referenced file does not exist: {target}")
    digest, size = hashlib.sha256(), 0
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            size += len(chunk)
            if size > MAX_REF_BYTES:
                raise SpecError(f"{field}: file exceeds {MAX_REF_BYTES} bytes")
            digest.update(chunk)
    return str(target), digest.hexdigest()


def _obj(field: str, value) -> dict:
    if not isinstance(value, dict):
        raise SpecError(f"{field}: must be a JSON object")
    return value


def _rows(field: str, value) -> list[dict]:
    if not isinstance(value, list):
        raise SpecError(f"{field}: must be a list")
    return [_obj(f"{field}[{i}]", row) for i, row in enumerate(value)]


def _strings(field: str, row: dict, *keys: str) -> None:
    for key in keys:
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise SpecError(f"{field}.{key}: required non-empty string")


def _ref(field: str, row: dict, *keys: str) -> dict:
    path, digest = _sha256(field, row.get("path"))
    out = {k: row.get(k) for k in keys}
    out.update(path=path, sha256=digest)
    return out


def _write(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def _commit(spec: dict, root: Path) -> str:
    commit = spec.get("commit")
    if not isinstance(commit, str) or not commit:
        raise SpecError("commit: required in the spec")
    done = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=False, timeout=30)
    if done.returncode != 0 or done.stdout.strip() != commit:
        raise SpecError("commit: does not equal this checkout's HEAD, so the registry and "
                        "inventory would describe a different commit")
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                            "--untracked-files=no"], capture_output=True, text=True,
                           check=False, timeout=30)
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise SpecError("commit: the checkout has uncommitted tracked changes, so the registry "
                        "and inventory would not describe that commit")
    return commit


#: Namespaces every kind may run in without being a writer of anything official.
_SAFE_NAMESPACES = frozenset({"shadow", "smoke"})


def registry_jobs(owner: str, official_kinds=(), non_official_kinds=()) -> list[dict]:
    """One row per job kind in the live registry. A kind is an official writer when
    its namespaces include ``official`` or the operator lists it in
    ``official_registry_kinds``. A kind that can run in any other namespace (today ``experiment`` in ``primary``) or
    holds a store ``write`` lease (today ``models_promote`` and ``models_rollback`` on
    ``deployment_pointer``) must be decided explicitly in ``official_registry_kinds`` or
    ``non_official_registry_kinds``; an undecided one is refused (fail closed)."""
    reg, rows = stages.registry(), []
    for name in reg.names():
        kind = reg.get(name)
        writes_store = any(mode == "write" for _, mode in kind.store_domains)
        if ((kind.namespaces - _SAFE_NAMESPACES - {"official"} or writes_store)
                and name not in official_kinds and name not in non_official_kinds):
            raise SpecError(f"{name}: can run in {sorted(kind.namespaces - _SAFE_NAMESPACES)} "
                            f"or holds a store write lease ({writes_store}); list it in "
                            "official_registry_kinds or non_official_registry_kinds")
        official = "official" in kind.namespaces or name in official_kinds
        rows.append({"id": name, "schedule": f"native:{name}", "owner": owner,
                     "kind": ("scheduled_writer" if official
                              else "background" if kind.effects else "reader"),
                     "writes_official": official})
    return rows


def _kind_list(spec: dict, key: str) -> set:
    value, known = spec.get(key, []), set(stages.registry().names())
    if not isinstance(value, list) or not all(isinstance(k, str) and k in known for k in value):
        raise SpecError(f"{key}: must be a list of registry job kind names")
    return set(value)


def assemble(spec: dict, out_dir: Path, root: Path = ROOT) -> dict:
    spec = _obj("spec", spec)
    out_dir = Path(out_dir).resolve()
    if out_dir.is_relative_to(root.resolve()):
        raise SpecError("out-dir: must be outside the repository")
    commit = _commit(spec, root)
    owner = spec.get("registry_owner")
    if not isinstance(owner, str) or not owner:
        raise SpecError("registry_owner: required string")
    official_kinds = _kind_list(spec, "official_registry_kinds")
    non_official_kinds = _kind_list(spec, "non_official_registry_kinds")
    if official_kinds & non_official_kinds:
        raise SpecError("official_registry_kinds and non_official_registry_kinds overlap: "
                        f"{sorted(official_kinds & non_official_kinds)}")
    jobs = registry_jobs(owner, official_kinds, non_official_kinds)
    seen = {j["id"] for j in jobs}
    for extra in _rows("extra_jobs", spec.get("extra_jobs", [])):
        jid = extra.get("id")
        if not isinstance(jid, str) or jid in seen:
            raise SpecError(f"extra_jobs: id {jid!r} is not a unique string "
                            "(duplicate or registry collision)")
        seen.add(jid)
        _strings(f"extra_jobs[{jid}]", extra, "schedule", "owner", "kind")
        if not isinstance(extra.get("writes_official"), bool):
            raise SpecError(f"extra_jobs[{jid}].writes_official: must be a JSON boolean")
        jobs.append({k: extra.get(k) for k in _JOB_FIELDS})
    jobs.sort(key=lambda j: j["id"])
    config = _rows("config", spec.get("config", []))
    evidence = _rows("phase_evidence", spec.get("phase_evidence", []))
    snapshot = _obj("data_snapshot", spec.get("data_snapshot"))
    deployment = _obj("model_deployment", spec.get("model_deployment"))
    for i, row in enumerate(config):
        _strings(f"config[{i}]", row, "name")
    for i, row in enumerate(evidence):
        _strings(f"phase_evidence[{i}]", row, "phase", "schema_version", "status")
    _strings("data_snapshot", snapshot, "snapshot_id")
    _strings("model_deployment", deployment, "release_id")
    _obj("authority", spec.get("authority"))
    manifest = {
        "schema_version": checker.MANIFEST_SCHEMA,
        "candidate": {"commit": commit,
                      "config": [_ref(f"config[{i}]", r, "name") for i, r in enumerate(config)]},
        "data_snapshot": _ref("data_snapshot", snapshot, "snapshot_id"),
        "model_deployment": _ref("model_deployment", deployment, "release_id"),
        "authority": _obj("authority", spec.get("authority")),
        "phase_evidence": [_ref(f"phase_evidence[{i}]", r, "phase", "schema_version", "status")
                           for i, r in enumerate(evidence)],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path, inventory_path = out_dir / "job_graph.json", out_dir / "consumer_inventory.json"
    found = discover_entrypoints(root)
    _write(graph_path, {"jobs": [{"id": j["id"], "schedule": j["schedule"],
                                  "writes_official": j["writes_official"]} for j in jobs]})
    _write(inventory_path, {"consumers": [{"id": i, "kind": m["kind"], "source": m["source"]}
                                          for i, m in sorted(found.items())]})
    manifest["job_graph"] = {**_ref("job_graph", {"path": graph_path}),
                             "jobs": [{k: j[k] for k in ("id", "schedule", "kind", "owner")}
                                      for j in jobs]}
    manifest["consumer_inventory"] = {**_ref("consumer_inventory", {"path": inventory_path}),
                                      "consumers": sorted(found)}
    _write(out_dir / "manifest.json", manifest)
    return manifest


def _load_spec(path: Path) -> dict:
    with path.open("rb") as handle:
        data = handle.read(MAX_SPEC_BYTES + 1)
    if len(data) > MAX_SPEC_BYTES:
        raise SpecError(f"spec exceeds {MAX_SPEC_BYTES} bytes")
    return json.loads(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = assemble(_load_spec(args.spec), args.out_dir)
    except (SpecError, OSError, ValueError) as exc:
        print(f"spec unusable: {exc}", file=sys.stderr)
        return 2
    result = checker.validate(manifest, root=ROOT)
    print(f"{result['status']}: {len(manifest['job_graph']['jobs'])} jobs, "
          f"{len(manifest['consumer_inventory']['consumers'])} consumers, "
          f"codes={result['finding_codes']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
