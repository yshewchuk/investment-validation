#!/usr/bin/env python3
"""P7-1 assembler: build the candidate/authority manifest the checker validates.

Read-only except for three JSON files written into ``--out-dir``. One operator
spec file (kept outside the repository: it holds the legacy scheduler entry,
credential owners and local evidence paths) names the commit, config,
snapshot, release, phase evidence, ``authority`` and any extra jobs the
registry cannot know. The job-graph rows come from the live
``stages.registry()``, so a job kind added later (``serving_projection``,
#726) appears with no edit here. Exit 0 = the checker passes, 1 = findings,
2 = unusable spec.

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


class SpecError(Exception):
    """The operator spec (or a file it names) cannot be used."""


def _sha256(field: str, path) -> tuple[str, str]:
    target = Path(str(path)).resolve()
    if not target.is_file():
        raise SpecError(f"{field}: referenced file does not exist: {target}")
    return str(target), hashlib.sha256(target.read_bytes()).hexdigest()


def _ref(field: str, row: dict, *keys: str) -> dict:
    path, digest = _sha256(field, row.get("path"))
    out = {k: row.get(k) for k in keys}
    out.update(path=path, sha256=digest)
    return out


def _write(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def _commit(spec: dict, root: Path) -> str:
    if spec.get("commit"):
        return spec["commit"]
    done = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SpecError("commit: not in the spec and git rev-parse HEAD failed")
    return done.stdout.strip()


def registry_jobs(owner: str) -> list[dict]:
    """One row per job kind in the live registry; never an official writer unless
    the kind itself is registered for the ``official`` namespace."""
    reg, rows = stages.registry(), []
    for name in reg.names():
        kind = reg.get(name)
        official = "official" in kind.namespaces
        rows.append({"id": name, "schedule": f"native:{name}", "owner": owner,
                     "kind": ("scheduled_writer" if official
                              else "background" if kind.effects else "reader"),
                     "writes_official": official})
    return rows


def assemble(spec: dict, out_dir: Path, root: Path = ROOT) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = registry_jobs(str(spec.get("registry_owner") or ""))
    seen = {j["id"] for j in jobs}
    for extra in spec.get("extra_jobs", []):
        if extra.get("id") in seen:
            raise SpecError(f"extra_jobs: id {extra.get('id')!r} collides with a registry job")
        seen.add(extra.get("id"))
        jobs.append({k: extra.get(k) for k in ("id", "schedule", "owner", "kind",
                                               "writes_official")})
    jobs.sort(key=lambda j: str(j["id"]))
    graph_path = out_dir / "job_graph.json"
    _write(graph_path, {"jobs": [{"id": j["id"], "schedule": j["schedule"],
                                  "writes_official": j["writes_official"]} for j in jobs]})
    found = discover_entrypoints(root)
    inventory_path = out_dir / "consumer_inventory.json"
    _write(inventory_path, {"consumers": [{"id": i, "kind": m["kind"], "source": m["source"]}
                                          for i, m in sorted(found.items())]})
    manifest = {
        "schema_version": checker.MANIFEST_SCHEMA,
        "candidate": {"commit": _commit(spec, root),
                      "config": [_ref(f"config[{i}]", r, "name")
                                 for i, r in enumerate(spec.get("config", []))]},
        "data_snapshot": _ref("data_snapshot", spec.get("data_snapshot", {}), "snapshot_id"),
        "model_deployment": _ref("model_deployment", spec.get("model_deployment", {}),
                                 "release_id"),
        "job_graph": {**_ref("job_graph", {"path": graph_path}),
                      "jobs": [{k: j[k] for k in ("id", "schedule", "kind", "owner")}
                               for j in jobs]},
        "consumer_inventory": {**_ref("consumer_inventory", {"path": inventory_path}),
                               "consumers": sorted(found)},
        "authority": spec.get("authority", {}),
        "phase_evidence": [_ref(f"phase_evidence[{i}]", r, "phase", "schema_version", "status")
                           for i, r in enumerate(spec.get("phase_evidence", []))],
    }
    _write(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        manifest = assemble(json.loads(args.spec.read_text()), args.out_dir, args.root)
    except (SpecError, OSError, ValueError) as exc:
        print(f"spec unusable: {exc}", file=sys.stderr)
        return 2
    result = checker.validate(manifest, root=args.root)
    print(f"{result['status']}: {len(manifest['job_graph']['jobs'])} jobs, "
          f"{len(manifest['consumer_inventory']['consumers'])} consumers, "
          f"codes={result['finding_codes']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
