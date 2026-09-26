#!/usr/bin/env python3
"""D19: the real render-parity comparator.

"The v2 render stage's bundle, compared with ``render_bundle`` invoked the
legacy way on the same scores, ladder, model evidence, ledger generation,
meta, and health, is identical in every serialized view except declared
execution-metadata fields; the serialized selfcheck passes in a separate
bounded process." (D19, quoted in the brief.)

This is the OPERATOR tool: it reads a real ``legacy_render`` job's recorded
input bindings and committed outputs from the ops catalog at ``--root``,
rebuilds the legacy-way bundle in a bounded subprocess against
``--legacy-root`` (the private root the render job itself used -- its
staging copy, or a snapshot materialization root), runs the legacy
serialized selfcheck on the v2 bundle in a second bounded subprocess, and
publishes a kind-``render_bundle_parity`` :class:`ComparisonReceipt` under
``--artifact-root``. Nothing here runs against real data on its own; the
synthetic tests seed a small catalog and monkeypatch the heavy legacy
loaders INSIDE the bounded subprocess via the ``PHASE2_RENDER_PARITY_TEST_PATCH``
hook below.

Usage (operator)::

    python3 checks/rearchitecture_phase2_render_parity.py \\
        --root data/operations --render-job job_abc123 \\
        --legacy-root /private/legacy/root --artifact-root evidence/ \\
        --max-rss-gb 6.5

Subprocess worker mode (internal; never invoked directly by an operator)::

    python3 checks/rearchitecture_phase2_render_parity.py \\
        --worker oracle|selfcheck <job.json> <out_dir>
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.phase2_render_oracle import legacy_way_bundle
from checks.rearchitecture_phase1_gate import source_files, source_hash
from checks.rearchitecture_phase2_evidence import RENDER_PARITY_KIND
from checks.rearchitecture_phase2_gate import environment_hash
from engine.v2.contracts import JobSpec, SnapshotRef
from engine.v2.diagnosis import (
    AGREE,
    DIFFER,
    ComparisonReceipt,
    Finding,
    StagePlan,
    TolerancePolicy,
    compare_records,
    content_hash,
    merge_receipts,
    problem,
)
from engine.v2.foundation import ArtifactStore, SystemClock, to_document
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import load_json
from engine.v2.ops.checkpoints import artifact as resolve_artifact
from engine.v2.ops.input_bindings import recorded_bindings
from engine.v2.ops.render_inputs import EXECUTION_METADATA_FIELDS
from engine.v2.ops.session_resolution import resolve_effective_session

CATALOG_FILENAME = "catalog.sqlite"
#: Env var a synthetic test sets to a dotted module path whose ``patch()``
#: monkeypatches ``FeatureContext.load``/``Scorer`` INSIDE the bounded
#: subprocess, before it loads anything. Absent in every operator run.
TEST_PATCH_ENV = "PHASE2_RENDER_PARITY_TEST_PATCH"
_REQUIRED_BINDINGS = ("score.json", "finality.json", "model_evidence.json",
                      "ledger_generation.tar")
#: A comparator over plain JSON-shaped records needs no stage graph or
#: tolerance -- every render bundle file is an exact-serialization check
#: (render_inputs.py's own D19 docstring), so both are declared empty here
#: rather than reused from SCORER_V1/SCORE_RECORD_V1, which belong to a
#: different comparison (ScoreResult records, not bundle files).
_FILE_PLAN = StagePlan(plan_id="render_bundle_file.v1", stages=())
_FILE_POLICY = TolerancePolicy(policy_id="render_bundle_file.exact.v1")


# --------------------------------------------------------------------------
# catalog reads
# --------------------------------------------------------------------------


def _open(root):
    conn = open_catalog(Path(root) / CATALOG_FILENAME, clock=SystemClock())
    return conn, ArtifactStore(root)


def _latest_succeeded_attempt(conn, job_id):
    row = conn.execute(
        "SELECT attempt_id FROM attempts WHERE job_id=? AND state='succeeded' "
        "ORDER BY attempt_number DESC LIMIT 1", (job_id,)).fetchone()
    if row is None:
        raise SystemExit(f"no succeeded attempt for render job {job_id!r}")
    return row["attempt_id"]


def _job_spec(conn, job_id):
    row = conn.execute("SELECT spec_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if row is None:
        raise SystemExit(f"unknown job {job_id!r}")
    return load_json(JobSpec, row["spec_json"])


def _load_bound_bytes(conn, store, attempt_id):
    """The four required bindings, plus ``prior_selfcheck.json`` when the
    render job bound one (D19/C08 decision 2: optional, off by default)."""
    bindings = recorded_bindings(conn, attempt_id)
    missing = [name for name in _REQUIRED_BINDINGS if name not in bindings]
    if missing:
        raise SystemExit(f"render attempt is missing recorded bindings: {missing}")
    names = list(_REQUIRED_BINDINGS)
    if "prior_selfcheck.json" in bindings:
        names.append("prior_selfcheck.json")
    out = {}
    for name in names:
        ref = resolve_artifact(conn, store, bindings[name].artifact_id)
        out[name] = store.read_verified(ref)
    return out


def _load_committed_bundle(conn, store, attempt_id, dest_dir):
    """The render job's sole checkpointed output: ``bundle.tar`` under the
    name ``legacy_render`` (``worker.py::dispatch`` registers exactly one
    named output per legacy action; ``render.json``/``meta.json``/
    ``health.json`` are plain staging files, not separately checkpointed --
    ``meta``/``health`` are read back out of ``bundle/data/`` instead, which
    is where ``render_bundle`` always writes them)."""
    row = conn.execute(
        "SELECT artifact_id FROM attempt_outputs WHERE attempt_id=? AND name='legacy_render'",
        (attempt_id,)).fetchone()
    if row is None:
        raise SystemExit("render attempt has no committed 'legacy_render' output")
    ref = resolve_artifact(conn, store, row["artifact_id"])
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "bundle.tar"
    tar_path.write_bytes(store.read_verified(ref))
    with tarfile.open(tar_path) as archive:
        archive.extractall(dest_dir)
    return dest_dir / "bundle"


def _resolve_snapshot(conn, store, job_spec):
    """The snapshot this render job's PLAN ran under, when one exists.

    ``render``/``projection`` never binds ``snapshot_ref.json`` itself (only
    ``score``/``decision_replay``/``materialize`` do -- P2-6 §9.3
    ``SNAPSHOT_STAGES``); every stage in a snapshot-mode plan graph DOES
    carry ``snapshot_generation_id`` on its own parameters
    (``nightly._stage_parameters``). So: read the id from this job's own
    parameters, then resolve the real ``SnapshotRef`` (for its
    ``manifest_hash``) off whichever declared dependency bound it. A
    legacy-mode nightly leaves ``snapshot_generation_id`` empty and binds no
    snapshot at all -- this returns ``(None, None)``, and the evidence
    validator's binding check is a no-op when the evidence document itself
    names no ``snapshot_ref`` either (``_check_bindings`` only fires when
    ``snapshot_ref_obj is not None``).
    """
    generation_id = job_spec.parameters.get("snapshot_generation_id")
    if not generation_id:
        return None, None
    for dependency_job_id in job_spec.dependency_job_ids:
        dep_attempt = conn.execute(
            "SELECT attempt_id FROM attempts WHERE job_id=? AND state='succeeded' "
            "ORDER BY attempt_number DESC LIMIT 1", (dependency_job_id,)).fetchone()
        if dep_attempt is None:
            continue
        binding = recorded_bindings(conn, dep_attempt["attempt_id"]).get("snapshot_ref.json")
        if binding is None:
            continue
        ref = resolve_artifact(conn, store, binding.artifact_id)
        snapshot_ref = load_json(SnapshotRef, store.read_verified(ref).decode())
        return snapshot_ref.snapshot_id, snapshot_ref.manifest_hash
    return generation_id, None


# --------------------------------------------------------------------------
# bounded subprocess workers (internal entry points)
# --------------------------------------------------------------------------


def _apply_test_patch():
    hook = os.environ.get(TEST_PATCH_ENV)
    if hook:
        importlib.import_module(hook).patch()


def _oracle_worker(job_file, out_dir):
    from engine.features import FeatureContext
    from engine.score import Scorer
    from engine.v2.ops.render_inputs import (
        assemble_scores,
        stage_ledger_generation,
        stage_model_evidence,
    )

    job = json.loads(Path(job_file).read_text())
    legacy_root = Path(os.environ["INVESTING_PLAN_ROOT"])
    evidence_path = Path(job_file).parent / "model_evidence.json"
    evidence_path.write_text(json.dumps(job["evidence"]))
    stage_model_evidence(evidence_path, legacy_root)
    stage_ledger_generation(Path(job["ledger_tar_path"]), legacy_root)
    scores = assemble_scores(job["score_document"])
    years = range(int(job["year_start"]), int(job["year_end"]) + 1)
    scorer = Scorer(context=FeatureContext.load(list(job["tickers"]), years=years))
    legacy_way_bundle(
        Path(out_dir) / "bundle", scores=scores, panel=scorer.context.panel,
        trades=scorer.trades, registry=scorer.registry, finality=job["finality"],
        requested_as_of=job["requested_as_of"], resolved_as_of=job["resolved_as_of"],
        tickers=tuple(job["tickers"]), horizon_days=int(job["horizon_days"]),
        evidence=job["evidence"], selfcheck_report=job.get("prior_selfcheck"))


def _selfcheck_worker(job_file, out_dir):
    from engine.dashboard.selfcheck import DEFAULT_N, selfcheck
    from engine.features import FeatureContext
    from engine.score import Scorer

    job = json.loads(Path(job_file).read_text())
    years = range(int(job["year_start"]), int(job["year_end"]) + 1)
    scorer = Scorer(context=FeatureContext.load(list(job["context_tickers"]), years=years))
    result = selfcheck(Path(job["bundle_dir"]), n=int(job.get("sample", DEFAULT_N)), scorer=scorer)
    value = result.as_dict() if hasattr(result, "as_dict") else vars(result)
    (Path(out_dir) / "selfcheck.json").write_text(json.dumps(value, default=str))


_WORKERS = {"oracle": _oracle_worker, "selfcheck": _selfcheck_worker}


def _run_worker(mode, job_file, out_dir):
    """The bounded subprocess's own entry point: never raises out of main."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    try:
        _apply_test_patch()
        _WORKERS[mode](job_file, out_dir)
        (Path(out_dir) / "result.json").write_text(json.dumps({"ok": True}))
        return 0
    except Exception as exc:  # noqa: BLE001 -- reported via result.json, not a bare traceback
        (Path(out_dir) / "result.json").write_text(
            json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1


# --------------------------------------------------------------------------
# bounded subprocess launcher (parent side)
# --------------------------------------------------------------------------


def run_bounded(argv, *, max_rss_gb, cwd, env, poll_s=2, timeout=1800):
    """Invoke ``tools/bounded_run.py`` and return its ``CompletedProcess``.

    A memory-cap breach is ``tools/bounded_run.py``'s own clean, explained
    exit 137 -- never a hang: the watchdog polls every ``poll_s`` seconds and
    kills the tree itself. This call returns within ``timeout`` because the
    resource wait is bounded below it: ``--max-wait-s`` is ``timeout`` less
    ten minutes, floored at 0 (never negative, and always strictly less than
    ``timeout`` since 600 > 0), so a job that never gets its slot exits 75
    before ``subprocess.run``'s own deadline can SIGKILL it -- even for a
    short ``timeout`` such as 60 s, where a 60 s floor would have made the
    wait equal to (not below) the outer timeout.
    """
    command = [sys.executable, str(ROOT / "tools" / "bounded_run.py"),
              "--max-rss-gb", str(max_rss_gb), "--poll-s", str(poll_s),
              "--max-wait-s", str(max(0, timeout - 600)), "--", *argv]
    return subprocess.run(command, cwd=str(cwd), env=env, capture_output=True, text=True,
                          timeout=timeout)


def _run_worker_subprocess(mode, job_file, out_dir, *, max_rss_gb, env, repo_root):
    argv = [sys.executable, str(repo_root / "checks" / "rearchitecture_phase2_render_parity.py"),
           "--worker", mode, str(job_file), str(out_dir)]
    proc = run_bounded(argv, max_rss_gb=max_rss_gb, cwd=repo_root, env=env)
    result_path = Path(out_dir) / "result.json"
    ok = False
    if proc.returncode == 0 and result_path.is_file():
        try:
            ok = bool(json.loads(result_path.read_text()).get("ok"))
        except ValueError:
            ok = False
    return ok, proc


# --------------------------------------------------------------------------
# per-file comparison (record comparison for JSON, byte comparison else)
# --------------------------------------------------------------------------


def _relative_paths(bundle_dir):
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        return set()
    return {p.relative_to(bundle_dir).as_posix() for p in bundle_dir.rglob("*") if p.is_file()}


#: The declared metadata files also ship a browser-fetchable ``window.X = ...;``
#: sibling (``meta.js``/``health.js``) carrying the SAME embedded fields
#: (``render_inputs.py``'s own ``_normalized_js``/``_JS_ASSIGNMENT``, mirrored
#: here rather than imported since it is private to that module). Comparing
#: it as an opaque byte blob -- the plain "byte comparison for other files"
#: rule -- would report the two sides' genuinely-different ``generated_at``
#: as a false disagreement, exactly the failure D19's own metadata exclusion
#: exists to prevent; unwrapping it here is what lets that JS sibling be
#: compared "in every serialized view" per the D19 requirement, not just the
#: ``.json`` one.
_JS_ASSIGNMENT = re.compile(r"^window\.[A-Za-z0-9_]+ = ")


def _strip_metadata(stem, payload):
    fields = EXECUTION_METADATA_FIELDS.get(stem)
    if not fields or not isinstance(payload, dict):
        return payload
    return {k: v for k, v in payload.items() if k not in fields}


def _js_record(path, stem):
    text = path.read_text()
    match = _JS_ASSIGNMENT.match(text)
    if not match:
        return None
    body = text[match.end():]
    body = body[:-2] if body.endswith(";\n") else body
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    return _strip_metadata(stem, payload)


def _file_record(path, stem):
    """A JSON (or JS-wrapped metadata) file's stripped payload, or a byte
    -comparison record for everything else."""
    if path.suffix == ".json":
        return _strip_metadata(stem, json.loads(path.read_text()))
    if path.suffix == ".js" and stem in EXECUTION_METADATA_FIELDS:
        record = _js_record(path, stem)
        if record is not None:
            return record
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "byte_size": len(data)}


def _per_file_receipt(relpath, left_dir, right_dir):
    left_path, right_path = Path(left_dir) / relpath, Path(right_dir) / relpath
    stem = Path(relpath).stem
    left = _file_record(left_path, stem) if left_path.is_file() else None
    right = _file_record(right_path, stem) if right_path.is_file() else None
    if left is None or right is None:
        left, right = {"present": left is not None}, {"present": right is not None}
    receipt = compare_records(left, right, comparison_kind=RENDER_PARITY_KIND, tier=2,
                              left_ref=f"v2:{relpath}", right_ref=f"legacy:{relpath}",
                              stage_plan=_FILE_PLAN, tolerance_policy=_FILE_POLICY)
    named = tuple(dataclasses.replace(f, field_path=f"{relpath}:{f.field_path}")
                 for f in receipt.findings)
    return dataclasses.replace(receipt, findings=named)


def compare_bundles(bundle_v2, bundle_legacy) -> ComparisonReceipt:
    """Every serialized file, findings naming file and JSON path only."""
    relpaths = sorted(_relative_paths(bundle_v2) | _relative_paths(bundle_legacy))
    receipts = [_per_file_receipt(rel, bundle_v2, bundle_legacy) for rel in relpaths]
    if not receipts:
        return merge_receipts([], comparison_kind=RENDER_PARITY_KIND, tier=2, expected=0)
    return merge_receipts(receipts, comparison_kind=RENDER_PARITY_KIND, tier=2,
                          expected=len(relpaths))


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def _write_oracle_job(scratch, *, score_document, finality, evidence, ledger_tar_bytes,
                      tickers, year_start, year_end, horizon_days, requested_as_of,
                      resolved_as_of, prior_selfcheck=None):
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "ledger_generation.tar").write_bytes(ledger_tar_bytes)
    job = {"score_document": score_document, "finality": finality, "evidence": evidence,
          "ledger_tar_path": str(scratch / "ledger_generation.tar"), "tickers": list(tickers),
          "year_start": year_start, "year_end": year_end, "horizon_days": horizon_days,
          "requested_as_of": str(requested_as_of), "resolved_as_of": str(resolved_as_of),
          "prior_selfcheck": prior_selfcheck}
    job_path = scratch / "job.json"
    job_path.write_text(json.dumps(job, default=str))
    return job_path


def _selfcheck_finding(observed_ok):
    return Finding(finding_id=content_hash(["selfcheck", observed_ok])[7:19],
                  first_differing_stage="unassigned", field_path="__selfcheck__.ok",
                  left_value=True, right_value=observed_ok, kind="value",
                  owning_stage="unassigned")


def _run_oracle(*, job_spec, bound, legacy_root, scratch, max_rss_gb, repo_root):
    score_document = json.loads(bound["score.json"])
    finality = json.loads(bound["finality.json"])
    evidence = json.loads(bound["model_evidence.json"])
    requested_as_of = job_spec.parameters["session"]
    resolved_as_of = resolve_effective_session(finality, requested_as_of)
    prior_selfcheck = json.loads(bound["prior_selfcheck.json"]) if "prior_selfcheck.json" in bound \
        else None
    job_file = _write_oracle_job(
        Path(scratch) / "oracle_job", score_document=score_document, finality=finality,
        evidence=evidence, ledger_tar_bytes=bound["ledger_generation.tar"],
        tickers=sorted(set(job_spec.parameters["tickers"])),
        year_start=job_spec.parameters["year_start"], year_end=job_spec.parameters["year_end"],
        horizon_days=job_spec.parameters.get("horizon_days", 35),
        requested_as_of=requested_as_of, resolved_as_of=resolved_as_of,
        prior_selfcheck=prior_selfcheck)
    out_dir = Path(scratch) / "oracle_out"
    env = dict(os.environ, INVESTING_PLAN_ROOT=str(legacy_root))
    ok, proc = _run_worker_subprocess("oracle", job_file, out_dir, max_rss_gb=max_rss_gb,
                                      env=env, repo_root=repo_root)
    return ok, proc, out_dir / "bundle"


def _run_selfcheck(*, job_spec, bundle_v2, legacy_root, scratch, max_rss_gb, repo_root):
    job = {"tickers": sorted(set(job_spec.parameters["tickers"])),
          "context_tickers": sorted(set(job_spec.parameters["context_tickers"])),
          "year_start": job_spec.parameters["year_start"],
          "year_end": job_spec.parameters["year_end"], "bundle_dir": str(bundle_v2)}
    if "sample" in job_spec.parameters:
        job["sample"] = job_spec.parameters["sample"]
    job_dir = Path(scratch) / "selfcheck_job"
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / "job.json"
    job_path.write_text(json.dumps(job, default=str))
    out_dir = Path(scratch) / "selfcheck_out"
    env = dict(os.environ, INVESTING_PLAN_ROOT=str(legacy_root))
    ok, proc = _run_worker_subprocess("selfcheck", job_path, out_dir, max_rss_gb=max_rss_gb,
                                      env=env, repo_root=repo_root)
    report_path = out_dir / "selfcheck.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    return bool(ok and report.get("ok")), report, proc


def build_render_comparison_receipt(*, root, render_job, legacy_root, max_rss_gb=6.5,
                                    repo_root=ROOT):
    """Assemble the full D19 receipt for one real ``legacy_render`` attempt."""
    conn, store = _open(root)
    try:
        attempt_id = _latest_succeeded_attempt(conn, render_job)
        job_spec = _job_spec(conn, render_job)
        bound = _load_bound_bytes(conn, store, attempt_id)
        snapshot_id, manifest_hash = _resolve_snapshot(conn, store, job_spec)
        scratch = Path(tempfile.mkdtemp(prefix="render-parity-"))
        bundle_v2 = _load_committed_bundle(conn, store, attempt_id, scratch / "v2")
        ok, oracle_proc, bundle_legacy = _run_oracle(
            job_spec=job_spec, bound=bound, legacy_root=legacy_root, scratch=scratch,
            max_rss_gb=max_rss_gb, repo_root=repo_root)
        file_receipt = compare_bundles(bundle_v2, bundle_legacy)
        selfcheck_ok, selfcheck_report, selfcheck_proc = _run_selfcheck(
            job_spec=job_spec, bundle_v2=bundle_v2, legacy_root=legacy_root, scratch=scratch,
            max_rss_gb=max_rss_gb, repo_root=repo_root)
    finally:
        conn.close()
    return _finish_receipt(file_receipt, ok, oracle_proc, selfcheck_ok, selfcheck_report,
                           selfcheck_proc, repo_root=repo_root, snapshot_id=snapshot_id,
                           manifest_hash=manifest_hash, render_job=render_job)


def _finish_receipt(file_receipt, oracle_ok, oracle_proc, selfcheck_ok, selfcheck_report,
                    selfcheck_proc, *, repo_root, snapshot_id, manifest_hash, render_job):
    findings, problems = file_receipt.findings, list(file_receipt.problems)
    if not oracle_ok:
        problems.append(problem(
            "ORACLE_SUBPROCESS_FAILED", "the bounded legacy-way rebuild subprocess did not "
            "succeed", category="dependency", retryable=True,
            dependency_refs=(oracle_proc.stderr[-500:],)))
    if not selfcheck_ok:
        problems.append(problem(
            "SELFCHECK_FAILED", "the bounded legacy selfcheck on the v2 bundle did not pass",
            category="validation"))
        findings = findings + (_selfcheck_finding(selfcheck_report.get("ok")),)
    verdict = AGREE if (oracle_ok and selfcheck_ok and not findings) else DIFFER
    receipt = dataclasses.replace(file_receipt, findings=findings, verdict=verdict,
                                  problems=tuple(problems))
    code_hash = source_hash(source_files(repo_root))
    env_hash, _ = environment_hash(repo_root)
    envelope = dataclasses.replace(receipt.envelope, code_hash=code_hash,
                                   environment_hash=env_hash, snapshot_id=snapshot_id,
                                   snapshot_manifest_hash=manifest_hash)
    # D19 is evidence about this committed render attempt, not an anonymous
    # replay label.  The Phase 3 verifier binds this exact job to its retained
    # output and dependencies.
    return dataclasses.replace(receipt, envelope=envelope, right_ref=render_job)


def publish_receipt(receipt: ComparisonReceipt, artifact_root):
    """Write the receipt under ``artifact_root``; return the evidence ref shape."""
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    data = json.dumps(to_document(receipt), sort_keys=True, allow_nan=False).encode()
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    rel_path = f"render_comparison_receipt_{digest.split(':')[1][:16]}.json"
    (artifact_root / rel_path).write_bytes(data)
    return {"path": rel_path, "content_hash": digest}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--worker":
        return _run_worker(argv[1], argv[2], argv[3])
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--render-job", required=True)
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--max-rss-gb", type=float, default=6.5)
    args = parser.parse_args(argv)
    receipt = build_render_comparison_receipt(
        root=args.root, render_job=args.render_job, legacy_root=args.legacy_root,
        max_rss_gb=args.max_rss_gb)
    ref = publish_receipt(receipt, args.artifact_root)
    print(json.dumps({"path": ref["path"], "content_hash": ref["content_hash"],
                      "verdict": receipt.verdict}))
    return 0 if receipt.verdict == AGREE else 1


if __name__ == "__main__":
    sys.exit(main())
