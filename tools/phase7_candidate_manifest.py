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
later (#726's ``serving_projection``) appears with no edit here unless it can run outside
``shadow``/``smoke``/``official``, in which case the spec must decide it. Exit 0 = the
checker passes, 1 = findings, 2 = unusable spec.

Spec keys: ``commit`` (required, must equal ``HEAD``), ``registry_owner``, ``config``
[{name, path}], ``data_snapshot`` {snapshot_id, path}, ``model_deployment`` {release_id,
path}, ``phase_evidence`` [{phase, schema_version, status, path}], ``authority``,
``extra_jobs`` [{id, schedule, owner, kind, writes_official}] and the optional
``official_registry_kinds`` / ``non_official_registry_kinds``. The spec is capped at
``MAX_SPEC_BYTES`` (1 MiB) and each referenced file at ``MAX_REF_BYTES`` (256 MiB).

    python3 tools/phase7_candidate_manifest.py --spec SPEC.json --out-dir DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Deliberately this checkout, not engine.paths.ROOT / INVESTING_PLAN_ROOT: the registry is
# imported from this tree and the spec commit is checked against this tree's HEAD, so the
# job graph, the consumer inventory and the commit all describe the same checkout.
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


def hash_ref(field: str, path) -> tuple[str, str, bytes]:
    """Read the referenced file once (at most ``MAX_REF_BYTES`` + 1 bytes) and return
    ``(resolved path, sha256, the bytes hashed)`` so callers parse what they hashed."""
    if not isinstance(path, (str, Path)):
        raise SpecError(f"{field}: path must be a string")
    target = Path(path).resolve()
    if not target.is_file():
        raise SpecError(f"{field}: referenced file does not exist: {target}")
    with target.open("rb") as handle:
        data = handle.read(MAX_REF_BYTES + 1)
    if len(data) > MAX_REF_BYTES:
        raise SpecError(f"{field}: file exceeds {MAX_REF_BYTES} bytes")
    return str(target), hashlib.sha256(data).hexdigest(), data


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


def _ref(field: str, row: dict, *keys: str, agree: tuple[str, ...] = ()) -> dict:
    """Hash the referenced file; every key in ``agree`` must equal the value the
    (size-capped) document itself carries, so the manifest never records an
    identity or status the file does not hold."""
    path, digest, data = hash_ref(field, row.get("path"))
    if agree:
        try:
            doc = json.loads(data)
        except ValueError as exc:
            raise SpecError(f"{field}: referenced file is not JSON ({exc})") from exc
        if not isinstance(doc, dict):
            raise SpecError(f"{field}: referenced document must be a JSON object")
        for key in agree:
            if doc.get(key) != row.get(key):
                raise SpecError(f"{field}.{key}: spec says {row.get(key)!r} but the document "
                                f"says {doc.get(key)!r}")
    out = {k: row.get(k) for k in keys}
    out.update(path=path, sha256=digest)
    return out


def _write(path: Path, doc: dict) -> None:
    """Write via a temp file in the same directory, then ``os.replace``."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


#: Trees that ``stages.registry()`` imports from or ``phase6_inventory.discover_entrypoints``
#: globs; an untracked file there would enter the manifest of a commit that does not contain it.
_SCANNED_TREES = ("engine/", "tools/", "dashboard/", "checks/", "experiments/")


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        # ``-C`` does not override these, so an inherited value could make git inspect a
        # different repository than the one whose registry and inventory are described.
        env = {k: v for k, v in os.environ.items()
               if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")}
        return subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args],
                              capture_output=True, text=True,
                              check=False, timeout=30, env=env)
    except (subprocess.SubprocessError, OSError) as exc:
        raise SpecError(f"commit: git {args[0]} failed ({type(exc).__name__})") from exc


def check_commit(spec: dict, root: Path) -> str:
    commit = spec.get("commit")
    if not isinstance(commit, str) or not commit:
        raise SpecError("commit: required in the spec")
    done = _git(root, "rev-parse", "HEAD")
    if done.returncode != 0:
        raise SpecError(f"commit: git rev-parse HEAD failed (exit {done.returncode})")
    if done.stdout.strip() != commit:
        raise SpecError("commit: does not equal this checkout's HEAD, so the registry and "
                        "inventory would describe a different commit")
    # Plumbing, not a hand-parsed status: untracked files under the scanned trees, and any
    # difference between the work tree and HEAD (git diff compares content, so a file with only stale stat data is clean; diff-index would flag it).
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z", "--",
                     *_SCANNED_TREES)
    if untracked.returncode != 0:
        raise SpecError(f"commit: git ls-files failed (exit {untracked.returncode})")
    # discover_entrypoints globs *.py without applying .gitignore, so an ignored source
    # file would enter the inventory of a commit that does not contain it.
    ignored = _git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--",
                   *_SCANNED_TREES)
    if ignored.returncode != 0:
        raise SpecError(f"commit: git ls-files failed (exit {ignored.returncode})")
    ignored_sources = [p for p in ignored.stdout.split("\0")
                       if p.endswith(".py") and "__pycache__/" not in p]
    tracked = _git(root, "diff", "--quiet", "HEAD", "--")
    if tracked.returncode not in (0, 1):
        raise SpecError(f"commit: git diff failed (exit {tracked.returncode})")
    changed = bool(untracked.stdout.strip("\0")) or bool(ignored_sources) or tracked.returncode == 1
    if changed:
        raise SpecError("commit: the checkout has uncommitted tracked changes or untracked "
                        "files under the scanned trees, so the registry and inventory would "
                        "not describe that commit")
    return commit


#: Namespaces every kind may run in without being a writer of anything official.
_SAFE_NAMESPACES = frozenset({"shadow", "smoke"})


def _kind_set(key: str, value) -> set:
    known = set(stages.registry().names())
    if not isinstance(value, (list, tuple, set, frozenset)) \
            or not all(isinstance(k, str) and k in known for k in value):
        raise SpecError(f"{key}: must be a list of registry job kind names")
    return set(value)


def registry_jobs(owner: str, official_kinds=(), non_official_kinds=()) -> list[dict]:
    """One row per job kind in the live registry. A kind is an official writer when
    its namespaces include ``official`` or the operator lists it in
    ``official_registry_kinds``. A kind that can run in any other namespace (today
    ``experiment`` in ``primary``) must be decided explicitly in
    ``official_registry_kinds`` or ``non_official_registry_kinds``; an undecided one is
    refused (fail closed). Listing a kind non-official records ``writes_official`` false by
    the operator's declaration, even for a ``primary`` kind.

    User decision (2026-10-16): a kind holding a store ``write`` lease (today
    ``models_promote`` and ``models_rollback`` on ``deployment_pointer``) is an operator
    action of its side, not an official scheduled writer, so it never counts against the
    one-official-writer rule. It is allowed only while it can touch the shadow/non-official
    pointer, i.e. its namespaces are within ``shadow``/``smoke``; one that can run in any
    other namespace is refused unless listed in ``official_registry_kinds`` (then it is the
    official writer), and a shadow-only one may not be listed official."""
    reg, rows = stages.registry(), []
    if not isinstance(owner, str) or not owner:
        raise SpecError("registry_owner: required string")
    official_kinds = _kind_set("official_registry_kinds", official_kinds)
    non_official_kinds = _kind_set("non_official_registry_kinds", non_official_kinds)
    if official_kinds & non_official_kinds:
        raise SpecError("official_registry_kinds and non_official_registry_kinds overlap: "
                        f"{sorted(official_kinds & non_official_kinds)}")
    for name in reg.names():
        kind = reg.get(name)
        writes_store = any(mode == "write" for _, mode in kind.store_domains)
        shadow_only = kind.namespaces <= _SAFE_NAMESPACES
        listed = name in official_kinds or name in non_official_kinds
        if writes_store and shadow_only and name in official_kinds:
            raise SpecError(f"{name}: an operator action on the shadow pointer cannot be an "
                            "official writer; remove it from official_registry_kinds")
        if writes_store and not shadow_only and name not in official_kinds:
            raise SpecError(f"{name}: holds a store write lease and can run in "
                            f"{sorted(kind.namespaces - _SAFE_NAMESPACES)}; only "
                            "official_registry_kinds may list it")
        if not writes_store and kind.namespaces - _SAFE_NAMESPACES - {"official"} and not listed:
            raise SpecError(f"{name}: can run in {sorted(kind.namespaces - _SAFE_NAMESPACES)}; "
                            "list it in official_registry_kinds or non_official_registry_kinds")
        official = "official" in kind.namespaces or name in official_kinds
        rows.append({"id": name, "schedule": f"native:{name}", "owner": owner,
                     "kind": ("scheduled_writer" if official
                              else "background" if kind.effects else "reader"),
                     "writes_official": official})
    return rows


_OUTPUT_NAMES = ("job_graph.json", "consumer_inventory.json", "manifest.json")


def assemble(spec: dict, out_dir: Path, spec_path: Path | None = None) -> dict:
    spec = _obj("spec", spec)
    out_dir = Path(out_dir).resolve()
    if out_dir.is_relative_to(ROOT.resolve()):
        raise SpecError("out-dir: must be outside the repository")
    commit = check_commit(spec, ROOT)  # the checkout whose registry is imported above
    owner = spec.get("registry_owner")
    jobs = registry_jobs(owner, spec.get("official_registry_kinds", []),
                         spec.get("non_official_registry_kinds", []))
    seen = {j["id"] for j in jobs}
    for extra in _rows("extra_jobs", spec.get("extra_jobs", [])):
        jid = extra.get("id")
        if not isinstance(jid, str) or not jid.strip() or jid in seen:
            raise SpecError(f"extra_jobs: id {jid!r} is not a unique string "
                            "(duplicate or registry collision)")
        seen.add(jid)
        _strings(f"extra_jobs[{jid}]", extra, "schedule", "owner", "kind")
        if extra["kind"] not in checker.JOB_KINDS:
            raise SpecError(f"extra_jobs[{jid}].kind: must be one of {checker.JOB_KINDS}")
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
    authority = _obj("authority", spec.get("authority"))
    for side in ("old", "proposed", "retained_old_deployment"):
        _obj(f"authority.{side}", authority.get(side))
    manifest = {
        "schema_version": checker.MANIFEST_SCHEMA,
        "candidate": {"commit": commit,
                      "config": [_ref(f"config[{i}]", r, "name") for i, r in enumerate(config)]},
        "data_snapshot": _ref("data_snapshot", snapshot, "snapshot_id", agree=("snapshot_id",)),
        "model_deployment": _ref("model_deployment", deployment, "release_id",
                                 agree=("release_id",)),
        "authority": _obj("authority", spec.get("authority")),
        "phase_evidence": [_ref(f"phase_evidence[{i}]", r, "phase", "schema_version", "status",
                                agree=("schema_version", "status"))
                           for i, r in enumerate(evidence)],
    }
    reserved = {(out_dir / name).resolve() for name in _OUTPUT_NAMES}
    inputs = [Path(ref["path"]).resolve() for ref in (
        *manifest["candidate"]["config"], manifest["data_snapshot"],
        manifest["model_deployment"], *manifest["phase_evidence"])]
    if spec_path is not None:
        inputs.append(Path(spec_path).resolve())
    clash = sorted(str(p) for p in reserved.intersection(inputs))
    if clash:
        raise SpecError(f"out-dir: output file(s) {clash} are also the spec or a referenced "
                        "input; writing them would destroy the pinned evidence")
    try:
        found = discover_entrypoints(ROOT)
    except (SyntaxError, OSError, ValueError) as exc:
        raise SpecError(f"consumer discovery failed in this checkout: "
                        f"{type(exc).__name__}: {exc}") from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path, inventory_path = out_dir / "job_graph.json", out_dir / "consumer_inventory.json"
    _write(graph_path, {"jobs": [{k: j[k] for k in ("id", "schedule", "kind", "owner",
                                                     "writes_official")} for j in jobs]})
    _write(inventory_path, {"consumers": [{"id": i, "kind": m["kind"], "source": m["source"]}
                                          for i, m in sorted(found.items())]})
    manifest["job_graph"] = {**_ref("job_graph", {"path": graph_path}),
                             "jobs": [{k: j[k] for k in ("id", "schedule", "kind", "owner")}
                                      for j in jobs]}
    manifest["consumer_inventory"] = {**_ref("consumer_inventory", {"path": inventory_path}),
                                      "consumers": sorted(found)}
    _write(out_dir / "manifest.json", manifest)
    return manifest


def load_spec(path: Path) -> dict:
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
        manifest = assemble(load_spec(args.spec), args.out_dir, spec_path=args.spec)
    except (SpecError, OSError, ValueError) as exc:
        print(f"spec unusable: {exc}", file=sys.stderr)
        return 2
    result = checker.validate(manifest, root=ROOT)
    print(f"{result['status']}: {len(manifest['job_graph']['jobs'])} jobs, "
          f"{len(manifest['consumer_inventory']['consumers'])} consumers, "
          f"codes={result['finding_codes']}")
    print("note: models_promote/models_rollback are recorded as non-official operator "
          "actions on the shadow pointer (issue #808 tracks the plan builders' production "
          "default)", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
