#!/usr/bin/env python3
"""P5-6 integrated acceptance gate over one staged model release.

Given a release root written by ``tools/phase5_prepare_release.py`` (layout:
:mod:`checks.phase5_release`), the gate:

1. **Members.** Every expected model binding and every catalog state
   (:data:`checks.phase5_release.STATE_SPECS`) has a staged member whose
   object bytes hash to the recorded ``content_hash``, and a typed loader
   accepts it where one exists. A MISSING or PENDING member is a finding, not
   a skip.
2. **Consumers.** Each v2 score consumer (:data:`CONSUMERS`) resolves its
   members from the staged release, and refuses ``MODEL_NOT_READY`` when the
   member is taken away. A consumer with no probe yet is a PENDING finding.
3. **No fitting.** Step 2 runs as one scoring pass under both no-fit guards
   (legacy ``engine.models.no_fit`` and v2 ``engine.v2.models.no_fit``). A
   profile hook records every call to either ``forbid_fitting`` (so a caller
   that swallows the exception is still caught), ``joblib.dump`` is rigged,
   and the watched directories are listed before and after. A planted fit
   and a planted cache write must both be detected, or the watch itself fails.
4. **Rollback.** On a scratch copy of the deployment pointer state, promote
   the candidate, roll back, and check the pointer resolves to the incumbent
   again with byte-identical manifest bytes and untouched history.
5. **Phase 4.** With ``--phase4-corpus``, every traced pair is verified by
   Phase 4's own trace verifier, its frozen model bindings are rebound by
   content hash to the staged release's objects, and it is scored with
   ``application.score_frozen`` under both no-fit guards; the runtime stage
   receipts must equal the captured ones
   (:mod:`checks.phase5_phase4_replay`). Without a corpus the subject is
   ``NOT_RUN`` and the best status is ``RELEASE_PASS``, never ``PASS``.
6. **Report.** A value-free private report (ids, statuses, codes and counts
   only) through ``engine.report.Report``, refused inside the repo.

Usage::

    python3 checks/phase5_acceptance.py --release-root /root/p5-6/release-A \\
        --artifact-root /root/p5-6/acceptance-A [--phase4-corpus <corpus>] \\
        [--phase4-checkpoint <resumable jsonl path>]
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.phase5_consumers import CONSUMERS, ReleaseContext  # noqa: E402
from checks.phase5_phase4_replay import REPLAY_CODES, replay_corpus  # noqa: E402
from checks.phase5_release import (  # noqa: E402
    EXPECTED_MODEL_BINDINGS,
    MISSING,
    PENDING,
    STAGED,
    STATE_SPECS,
    ReleaseLayoutError,
    StateSpec,
    deployment_root,
    modules_available,
    read_manifest,
    sha256_bytes,
)
from engine.v2.foundation import content_hash, to_document  # noqa: E402
from engine.v2.models import deployment  # noqa: E402
from engine.v2.models.contracts import ModelRelease  # noqa: E402

PHASE5_EVIDENCE_SCHEMA = "phase5_acceptance.v1.0"

# Finding codes. Every one is a gate failure.
RELEASE_LAYOUT = "P5_RELEASE_LAYOUT"
MODEL_RELEASE_INVALID = "P5_MODEL_RELEASE_INVALID"
MEMBER_MISSING = "P5_MEMBER_MISSING"
MEMBER_PENDING = "P5_MEMBER_PENDING"
MEMBER_UNKNOWN = "P5_MEMBER_UNKNOWN"
MEMBER_OBJECT_ABSENT = "P5_MEMBER_OBJECT_ABSENT"
MEMBER_HASH_MISMATCH = "P5_MEMBER_HASH_MISMATCH"
MEMBER_UNLOADABLE = "P5_MEMBER_UNLOADABLE"
MEMBER_IDENTITY = "P5_MEMBER_IDENTITY"
LINEAGE_INVALID = "P5_LINEAGE_INVALID"
CONSUMER_UNRESOLVED = "P5_CONSUMER_UNRESOLVED"
CONSUMER_NO_REFUSAL = "P5_CONSUMER_NO_REFUSAL"
CONSUMER_ERROR = "P5_CONSUMER_ERROR"
CONSUMER_PENDING = "P5_CONSUMER_PENDING"
CONSUMER_BLOCKED = "P5_CONSUMER_BLOCKED"
RUNTIME_FIT = "P5_RUNTIME_FIT"
MODEL_CACHE_WRITE = "P5_MODEL_CACHE_WRITE"
GUARD_CONTROL_FAILED = "P5_GUARD_CONTROL_FAILED"
ROLLBACK_NO_INCUMBENT = "P5_ROLLBACK_NO_INCUMBENT"
ROLLBACK_CANDIDATE_LIVE = "P5_ROLLBACK_CANDIDATE_ALREADY_DEPLOYED"
ROLLBACK_NOT_EXACT = "P5_ROLLBACK_NOT_EXACT"
PROMOTE_REFUSED = "P5_PROMOTE_REFUSED"
REPORT_INCOMPLETE = "P5_REPORT_INCOMPLETE"

FINDING_CODES = (
    RELEASE_LAYOUT, MODEL_RELEASE_INVALID, MEMBER_MISSING, MEMBER_PENDING,
    MEMBER_UNKNOWN, MEMBER_OBJECT_ABSENT, MEMBER_HASH_MISMATCH, MEMBER_UNLOADABLE,
    MEMBER_IDENTITY, LINEAGE_INVALID,
    CONSUMER_UNRESOLVED, CONSUMER_NO_REFUSAL, CONSUMER_ERROR, CONSUMER_PENDING,
    CONSUMER_BLOCKED, RUNTIME_FIT, MODEL_CACHE_WRITE, GUARD_CONTROL_FAILED,
    ROLLBACK_NO_INCUMBENT, ROLLBACK_CANDIDATE_LIVE, ROLLBACK_NOT_EXACT,
    PROMOTE_REFUSED, *REPLAY_CODES, REPORT_INCOMPLETE,
)


def _rss_gb() -> float:
    """Resident set of this process, in GB -- same helper and unit as
    ``tools/capture_tier0_corpus.py::_rss_gb`` and ``tools/mem_sampler.py``."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def _progress(tag: str, message: str, started: float) -> None:
    """One phase-boundary progress line: wall-clock time, elapsed since
    ``started``, RSS. The gate has no other output for the 10s of minutes a
    real release/corpus can take per phase (measured: a 133-minute run that
    never printed anything past its own startup banner and never created an
    artifact directory), so every ``build_evidence``/``replay_corpus`` phase
    boundary calls this -- the same ``[tag] message`` idiom
    ``tools/capture_tier0_corpus.py``'s ``[corpus] ...`` lines use, extended
    with a wall-clock timestamp so a supervisor tailing the log with `date`
    can tell how far along a run is, not just that it is still alive."""
    print(f"[{tag} {time.strftime('%H:%M:%S')}] {message} "
          f"(+{time.perf_counter() - started:.1f}s, rss {_rss_gb():.2f}G)", flush=True)


class ReportPathError(ValueError):
    """The private report path resolves inside the repo."""

    code = "REPORT_PATH_INSIDE_REPO"


@dataclasses.dataclass
class _Findings:
    rows: list[dict] = dataclasses.field(default_factory=list)

    def add(self, code: str, subject: str, detail: str) -> None:
        self.rows.append({"code": code, "subject": subject, "detail": detail})



# --------------------------------------------------------------------------
# 1. members
# --------------------------------------------------------------------------


def _verify_object(release_root: Path, path: str, expected: str) -> str | None:
    """None if the object's bytes hash to ``expected``, else a finding code."""
    base = deployment_root(release_root).resolve()
    target = (base / path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return MEMBER_OBJECT_ABSENT
    if not target.is_file():
        return MEMBER_OBJECT_ABSENT
    return None if sha256_bytes(target.read_bytes()) == expected else MEMBER_HASH_MISMATCH


def _load_model_release(release_root: Path, release_id: str, findings: _Findings):
    root = deployment_root(release_root)
    try:
        manifest = deployment._read_manifest(root, release_id)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        findings.add(MODEL_RELEASE_INVALID, release_id, type(exc).__name__)
        return None
    if manifest is None:
        findings.add(MODEL_RELEASE_INVALID, release_id, "no staged manifest")
        return None
    # Verify the declared hash version. This accepts verified historical
    # member-only manifests while requiring semantic hashes for new releases.
    if not deployment._manifest_hash_matches(manifest):
        findings.add(MODEL_RELEASE_INVALID, release_id, "release_hash disagrees with manifest")
        return None
    return manifest.release


def _model_member_rows(release_root: Path, release: ModelRelease | None,
                       findings: _Findings) -> list[dict]:
    rows = []
    bindings = () if release is None else release.bindings
    for role, strategy in EXPECTED_MODEL_BINDINGS:
        member_id = f"model:{role}:{strategy}"
        matches = [b for b in bindings if (b.role, b.strategy_id) == (role, strategy)]
        if len(matches) != 1:
            findings.add(MEMBER_MISSING, member_id,
                         "no staged binding" if not matches else "ambiguous binding")
            rows.append({"member_id": member_id, "category": "model", "status": MISSING,
                         "objects": 0, "verdict": MEMBER_MISSING})
            continue
        verdict = "ok"
        for member in matches[0].members:
            code = _verify_object(release_root, member.path, member.content_hash)
            if code:
                findings.add(code, member_id, member.name)
                verdict = code
        rows.append({"member_id": member_id, "category": "model", "status": STAGED,
                     "objects": len(matches[0].members), "verdict": verdict,
                     "binding_id": matches[0].binding_id})
    return rows


def _typed_load(spec: StateSpec, release_root: Path, obj: Mapping[str, Any]):
    """Load a staged state through its own verified loader, if one exists."""
    root = deployment_root(release_root)
    if spec.member_id.startswith(("payoff_line:", "payoff_surface:")):
        from engine.v2.models.payoff_artifact import PayoffArtifactLoader, PayoffArtifactRef

        return PayoffArtifactLoader(root).load(
            PayoffArtifactRef(path=obj["path"], content_hash=obj["content_hash"]))
    if spec.member_id.startswith("recalibration_map:"):
        from engine.v2.models.recalibration_artifact import (
            RecalibrationArtifactLoader,
            RecalibrationArtifactRef,
        )

        return RecalibrationArtifactLoader(root).load(
            RecalibrationArtifactRef(path=obj["path"], content_hash=obj["content_hash"]))
    if "engine.v2.models.frozen_state" in spec.modules:
        from engine.v2.models.frozen_state import FrozenStateLoader, FrozenStateRef

        return FrozenStateLoader(root).load(
            FrozenStateRef(path=obj["path"], content_hash=obj["content_hash"]))
    return None  # raw members (Tier-4 folds, parquet pools): hash-verified only


def _state_member_rows(release_root: Path, manifest: Mapping[str, Any], findings: _Findings,
                       specs: tuple[StateSpec, ...]) -> tuple[list[dict], dict[str, dict]]:
    by_id = {row["member_id"]: row for row in manifest.get("members", ())}
    known = {spec.member_id for spec in specs}
    for unknown in sorted(set(by_id) - known):
        findings.add(MEMBER_UNKNOWN, unknown, "not in the P5-6 catalog")
    rows, loaded = [], {}
    for spec in specs:
        row = by_id.get(spec.member_id)
        available, missing_module = modules_available(spec.modules)
        status = (row or {}).get("status") or (MISSING if available else PENDING)
        out = {"member_id": spec.member_id, "category": "state", "kind": spec.kind,
               "status": status, "objects": len((row or {}).get("objects", ())),
               "verdict": "ok"}
        if status == PENDING:
            detail = (row or {}).get("detail") or f"{missing_module} not importable"
            findings.add(MEMBER_PENDING, spec.member_id, detail)
            out["verdict"] = MEMBER_PENDING
        elif status != STAGED or not row.get("objects"):
            findings.add(MEMBER_MISSING, spec.member_id, (row or {}).get("detail", "absent"))
            out["verdict"] = MEMBER_MISSING
        else:
            out["verdict"] = _verify_state_objects(spec, release_root, row, findings, loaded)
        rows.append(out)
    return rows, loaded


def _verify_state_objects(spec, release_root, row, findings, loaded) -> str:
    verdict = "ok"
    artifacts = []
    for obj in row["objects"]:
        code = _verify_object(release_root, obj["path"], obj["content_hash"])
        if code:
            findings.add(code, spec.member_id, obj["name"])
            verdict = code
            continue
        if not modules_available(spec.modules)[0]:
            continue  # a staged row built elsewhere; bytes are still verified
        try:
            artifact = _typed_load(spec, release_root, obj)
        except (ValueError, KeyError, TypeError) as exc:
            findings.add(MEMBER_UNLOADABLE, spec.member_id, type(exc).__name__)
            verdict = MEMBER_UNLOADABLE
            continue
        issue = _identity_issue(spec, artifact)
        if issue:
            findings.add(MEMBER_IDENTITY, spec.member_id, f"{obj['name']}: {issue}")
            verdict = MEMBER_IDENTITY
            continue
        artifacts.append(artifact)
    if verdict == "ok":
        loaded[spec.member_id] = {"row": row, "artifacts": artifacts}
    return verdict


def _identity_issue(spec: StateSpec, artifact) -> str | None:
    """A loaded state must be the kind, role and strategy its catalog row names."""
    if artifact is None:
        return None
    if "engine.v2.models.frozen_state" in spec.modules:
        from engine.v2.models.frozen_release import member_kind

        if member_kind(artifact) != spec.kind:
            return f"kind {member_kind(artifact)} != {spec.kind}"
        role = spec.member_id.partition(":")[2]
        if spec.member_id.startswith("driver_residual_pool:") and artifact.role != role:
            return f"role {artifact.role} != {role}"
        return None
    strategy = getattr(artifact, "strategy", None)
    if strategy is not None and (strategy,) != spec.strategies:
        return f"strategy {strategy} not {spec.strategies}"
    kind = spec.member_id.split(":")[0]
    if kind.startswith("payoff_") and _payoff_kind(artifact) != kind:
        return f"{_payoff_kind(artifact)} staged as {spec.member_id}"
    return None


def _payoff_kind(artifact) -> str:
    from engine.v2.models.payoff_artifact import PayoffLineArtifact

    return "payoff_line" if isinstance(artifact, PayoffLineArtifact) else "payoff_surface"


def _lineage_subject(loaded: Mapping[str, dict], findings: _Findings) -> dict:
    """Every staged frozen state declares a well-formed lineage graph.

    Node ids are ``<member_id>/<object name>``; a state's ``upstream`` must
    name another staged state by that id. ``propagate_corrections`` with no
    changesets runs exactly the graph check a correction would: undeclared
    lineage, an unknown upstream and a cycle raise ``LineageError``.
    """
    nodes = []
    for member_id, state in sorted(loaded.items()):
        objects = sorted(state["row"]["objects"], key=lambda row: row["name"])
        for obj, artifact in zip(objects, state["artifacts"]):
            if artifact is not None and hasattr(artifact, "lineage"):
                nodes.append((f"{member_id}/{obj['name']}", artifact))
    if not nodes:
        return {"status": "NOT_APPLICABLE", "states": 0}
    from engine.v2.models.lineage import LineageError, propagate_corrections, state_node

    try:
        report = propagate_corrections([state_node(i, a) for i, a in nodes], ())
    except LineageError as exc:
        findings.add(LINEAGE_INVALID, "lineage", str(exc))
        return {"status": LINEAGE_INVALID, "states": len(nodes), "code": exc.code}
    return {"status": "ok", "states": len(nodes), "valid": len(report.valid)}


# --------------------------------------------------------------------------
# 2. consumers
# --------------------------------------------------------------------------

def _consumer_members(consumer: str) -> list[str]:
    if consumer == "frozen_stage_executor":
        return [f"model:{r}:{s}" for r, s in EXPECTED_MODEL_BINDINGS]
    return [spec.member_id for spec in STATE_SPECS if spec.consumer == consumer]


def _run_consumers(ctx: ReleaseContext, consumers: Mapping[str, Any],
                   findings: _Findings) -> list[dict]:
    rows = []
    for consumer, probe in consumers.items():
        if probe is None:
            members = ",".join(_consumer_members(consumer)) or "-"
            findings.add(CONSUMER_PENDING, consumer, f"no probe yet; members {members}")
            rows.append({"consumer": consumer, "member_id": members, "status": "PENDING"})
            continue
        try:
            results = probe(ctx)
        except Exception as exc:  # the probe itself broke: report, never skip
            findings.add(CONSUMER_ERROR, consumer, type(exc).__name__)
            rows.append({"consumer": consumer, "member_id": "-", "status": CONSUMER_ERROR})
            continue
        for result in results:
            rows.append(_consumer_verdict(result, findings))
    return rows


def _consumer_verdict(result: dict, findings: _Findings) -> dict:
    consumer, member = result["consumer"], result["member_id"]
    if result.get("blocked"):
        findings.add(CONSUMER_BLOCKED, consumer, f"{member} is not staged and verified")
        return {"consumer": consumer, "member_id": member, "status": CONSUMER_BLOCKED}
    status = "ok"
    if not result["resolved"]:
        findings.add(CONSUMER_UNRESOLVED, consumer, f"{member}: {result.get('detail', '')}")
        status = CONSUMER_UNRESOLVED
    if not result["refused_when_missing"]:
        findings.add(CONSUMER_NO_REFUSAL, consumer, f"{member} removed, no MODEL_NOT_READY")
        status = CONSUMER_NO_REFUSAL
    return {"consumer": consumer, "member_id": member, "status": status}


# --------------------------------------------------------------------------
# 3. the no-fit scoring pass
# --------------------------------------------------------------------------


class ModelCacheWrite(RuntimeError):
    """``joblib.dump`` was called during a guarded scoring pass."""


def _listing(dirs: Iterable[Path]) -> dict[str, tuple[int, int]]:
    out = {}
    for base in dirs:
        base = Path(base)
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                stat = path.stat()
                out[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return out


@dataclasses.dataclass
class ScoringWatch:
    fit_paths: list[str] = dataclasses.field(default_factory=list)
    dump_calls: int = 0
    changed_files: list[str] = dataclasses.field(default_factory=list)


@contextlib.contextmanager
def guarded_scoring(watch_dirs: Iterable[Path]):
    """Both no-fit guards on, fit attempts recorded, cache writes recorded.

    The profile hook matches the code objects of both ``forbid_fitting``
    functions, so it sees a fit attempt even when the caller catches the
    ``RuntimeFitForbidden`` it raises. Only this thread is profiled.
    """
    import joblib

    from engine.models import no_fit as legacy_no_fit
    from engine.v2.models import no_fit as v2_no_fit

    watch = ScoringWatch()
    dirs = [Path(d) for d in watch_dirs]
    codes = {legacy_no_fit.forbid_fitting.__code__, v2_no_fit.forbid_fitting.__code__}

    def hook(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            watch.fit_paths.append(str(frame.f_locals.get("path")))

    real_dump = joblib.dump

    def rigged_dump(*args, **kwargs):
        watch.dump_calls += 1
        raise ModelCacheWrite("joblib.dump during a guarded scoring pass")

    before = _listing(dirs)
    previous = sys.getprofile()
    joblib.dump = rigged_dump
    try:
        with legacy_no_fit.no_fit_guard(), v2_no_fit.no_fit_guard():
            sys.setprofile(hook)
            try:
                yield watch
            finally:
                sys.setprofile(previous)
    finally:
        joblib.dump = real_dump
        after = _listing(dirs)
        watch.changed_files = sorted(
            path for path in set(before) | set(after) if before.get(path) != after.get(path))


def _guard_controls(scratch: Path) -> dict[str, bool]:
    """The watch must see a planted fit and a planted cache write."""
    from engine.models import no_fit as legacy_no_fit
    from engine.v2.models import no_fit as v2_no_fit

    controls = {}
    for name, module in (("legacy_fit_detected", legacy_no_fit), ("v2_fit_detected", v2_no_fit)):
        with guarded_scoring([scratch]) as watch:
            with contextlib.suppress(Exception):  # swallowed on purpose
                module.forbid_fitting("p5-6.planted_fit")
        controls[name] = watch.fit_paths == ["p5-6.planted_fit"]
    import joblib

    with guarded_scoring([scratch]) as watch:
        with contextlib.suppress(Exception):
            joblib.dump({"planted": 1}, scratch / "planted.joblib")
        (scratch / "planted.bin").write_bytes(b"x")
    controls["cache_write_detected"] = watch.dump_calls == 1 and bool(watch.changed_files)
    return controls


def _scoring_pass(ctx, consumers, watch_dirs, findings) -> tuple[list[dict], dict]:
    with guarded_scoring(watch_dirs) as watch:
        rows = _run_consumers(ctx, consumers, findings)
    for path in sorted(set(watch.fit_paths)):
        findings.add(RUNTIME_FIT, "scoring_pass", path)
    if watch.dump_calls:
        findings.add(MODEL_CACHE_WRITE, "scoring_pass", f"joblib.dump x{watch.dump_calls}")
    for path in watch.changed_files:
        findings.add(MODEL_CACHE_WRITE, "scoring_pass", path)
    with tempfile.TemporaryDirectory(prefix="p5-6-guard-") as scratch:
        controls = _guard_controls(Path(scratch))
    for name, ok in controls.items():
        if not ok:
            findings.add(GUARD_CONTROL_FAILED, "scoring_pass", name)
    summary = {"fit_attempts": len(watch.fit_paths), "joblib_dump_calls": watch.dump_calls,
               "changed_files": len(watch.changed_files),
               "watched_dirs": len(list(watch_dirs)), "controls": controls}
    return rows, summary


# --------------------------------------------------------------------------
# 4. promote -> rollback
# --------------------------------------------------------------------------


def _copy_pointer_state(source: Path, dest: Path) -> None:
    """Manifests, pointer and history only: promotion never reads objects."""
    for name in ("releases", "history"):
        if (source / name).is_dir():
            shutil.copytree(source / name, dest / name)
    if (source / "DEPLOYED").is_file():
        shutil.copy2(source / "DEPLOYED", dest / "DEPLOYED")


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "DEPLOYED"}


def _rollback_round_trip(release_root: Path, candidate: str, first_deployment: bool,
                         findings: _Findings) -> dict:
    with tempfile.TemporaryDirectory(prefix="p5-6-rollback-") as scratch:
        root = Path(scratch)
        _copy_pointer_state(deployment_root(release_root), root)
        return _round_trip(root, candidate, first_deployment, findings)


def _round_trip(root: Path, candidate: str, first_deployment: bool, findings) -> dict:
    incumbent = deployment.current_pointer(root)
    if incumbent is None and not first_deployment:
        findings.add(ROLLBACK_NO_INCUMBENT, "rollback", "no DEPLOYED pointer to return to")
        return {"status": ROLLBACK_NO_INCUMBENT}
    if incumbent is not None and incumbent.release_id == candidate:
        findings.add(ROLLBACK_CANDIDATE_LIVE, "rollback", candidate)
        return {"status": ROLLBACK_CANDIDATE_LIVE}
    before_files = _snapshot(root)
    before_target = None if incumbent is None else to_document(
        deployment.resolve_release(root, incumbent.release_id))
    try:
        promoted = deployment.promote(root, candidate)
    except deployment.DeploymentError as exc:
        findings.add(PROMOTE_REFUSED, "rollback", type(exc).__name__)
        return {"status": PROMOTE_REFUSED}
    live = deployment.current_release(root)
    checks = {"promoted_resolves_candidate": promoted.release_id == candidate
              and to_document(live) == to_document(deployment.resolve_release(root, candidate))}
    if incumbent is None:
        try:
            deployment.rollback(root)
            checks["first_deployment_rollback_refused"] = False
        except deployment.NoPriorRelease:
            checks["first_deployment_rollback_refused"] = True
        checks["pointer_unchanged_by_refusal"] = deployment.current_pointer(root) == promoted
    else:
        back = deployment.rollback(root)
        after_files = _snapshot(root)
        checks.update({
            "pointer_release_id_restored": back.release_id == incumbent.release_id,
            "target_release_byte_exact": to_document(deployment.resolve_release(
                root, back.release_id)) == before_target,
            "manifests_and_history_prefix_byte_exact": all(
                after_files.get(path) == data for path, data in before_files.items()),
            "history_appended_two": len(deployment.pointer_history(root))
            == incumbent.sequence + 3,
        })
    for name, ok in checks.items():
        if not ok:
            findings.add(ROLLBACK_NOT_EXACT, "rollback", name)
    return {"status": "ok" if all(checks.values()) else ROLLBACK_NOT_EXACT,
            "first_deployment": incumbent is None, "checks": checks,
            "incumbent": None if incumbent is None else incumbent.release_id}


# --------------------------------------------------------------------------
# 5. Phase 4 integration (counted, not yet executed)
# --------------------------------------------------------------------------


def _phase4_subject(corpus: Path | None, release_root: Path, model_release,
                    manifest: Mapping[str, Any], watch_dirs: list[Path],
                    findings: _Findings, checkpoint_path: Path | None = None) -> dict:
    """Replay the corpus's traced pairs from the staged release, no fitting."""
    if corpus is None:
        return {"status": "NOT_RUN"}
    if model_release is None:  # already a P5_MODEL_RELEASE_INVALID finding
        return {"status": "BLOCKED"}
    with guarded_scoring(watch_dirs) as watch:
        summary, rows = replay_corpus(corpus, release_root, model_release, manifest,
                                      checkpoint_path=checkpoint_path)
    for code, subject, detail in rows:
        findings.add(code, subject, detail)
    for path in sorted(set(watch.fit_paths)):
        findings.add(RUNTIME_FIT, "phase4_replay", path)
    if watch.dump_calls:
        findings.add(MODEL_CACHE_WRITE, "phase4_replay", f"joblib.dump x{watch.dump_calls}")
    for path in watch.changed_files:
        findings.add(MODEL_CACHE_WRITE, "phase4_replay", path)
    if watch.fit_paths or watch.dump_calls or watch.changed_files:
        summary["status"] = "FAIL"
    summary["fit_attempts"] = len(watch.fit_paths)
    return summary


# --------------------------------------------------------------------------
# 6. the private report
# --------------------------------------------------------------------------


def _report_rows(evidence: dict) -> dict[str, list[list[str]]]:
    return {
        "members": [[r["member_id"], r["category"], r["status"], str(r["objects"]),
                     r["verdict"]] for r in evidence["members"]],
        "consumers": [[r["consumer"], r["member_id"], r["status"]]
                      for r in evidence["consumers"]],
        "controls": [[name, "pass" if ok else "FAIL"] for name, ok in sorted(
            {**evidence["scoring_pass"].get("controls", {}),
             **evidence["rollback"].get("checks", {}),
             "lineage_graph_valid": evidence["lineage"].get("status")
             in ("ok", "NOT_APPLICABLE")}.items())],
        "phase4": [[name, str(count)] for name, count in
                   (evidence["phase4"].get("dispositions") or {}).items()]
        + [["status", str(evidence["phase4"].get("status"))]],
        "findings": [[f["code"], f["subject"], f["detail"]] for f in evidence["findings"]],
    }


def _section(title: str, columns: list[str], rows: list[list[str]]) -> dict:
    return {"title": title, "columns": columns, "align": ["---"] * len(columns),
            "rows": rows or [["-"] * len(columns)]}


def write_report(evidence: dict, report_dir: Path) -> Path:
    """Render the value-free private report. Refuses a path inside the repo."""
    resolved = Path(report_dir).resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ReportPathError(f"refusing to write a report inside the repo: {resolved}")
    from engine.report import Report, build_provenance

    rows = _report_rows(evidence)
    members = evidence["members"]
    context = {
        "kind": "audit",
        "spec": {"id": "REARCH-PHASE-5-ACCEPTANCE",
                 "title": "Rearchitecture Phase 5 -- integrated model-release acceptance",
                 "type": "descriptive",
                 "hypothesis": "descriptive: is every member of the staged release present, "
                               "hash-verified, consumed without fitting, and rolled back "
                               "exactly?"},
        "results": {"headline": {}, "stress": {}, "mc": {}},
        "headline": {}, "backtest": {}, "checklist": [],
        "provenance": build_provenance(seeds={}, input_files=[]),
        "survivorship_note": "", "calibration": None,
        "funnel": [
            {"stage": "required members", "events": len(members), "note": "models + states"},
            {"stage": "members staged and verified", "headline": True,
             "events": sum(1 for r in members if r["verdict"] == "ok"),
             "note": f"status={evidence['status']}"},
            {"stage": "findings", "events": len(evidence["findings"]), "note": "all fail"},
        ],
        "extra_sections": [
            _section("Release members", ["member", "category", "status", "objects", "verdict"],
                     rows["members"]),
            _section("Score consumers", ["consumer", "members", "status"], rows["consumers"]),
            _section("No-fit and rollback controls", ["control", "result"], rows["controls"]),
            _section("Phase 4 replay", ["disposition", "pairs"], rows["phase4"]),
            _section("Findings", ["code", "subject", "detail"], rows["findings"]),
        ],
    }
    return Report(context).write(resolved, filename="phase5_report.md")


def _report_is_complete(text: str, evidence: dict) -> bool:
    headers = ("## 0. Verdict", "## 8. Provenance", "Release members", "Score consumers",
               "No-fit and rollback controls", "Phase 4 replay", "Findings")
    if not all(header in text for header in headers):
        return False
    return all("| " + " | ".join(row) + " |" in text
               for table in _report_rows(evidence).values() for row in table)


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def _default_watch_dirs(release_root: Path) -> list[Path]:
    from engine import paths

    return [paths.DATA / "models", deployment_root(release_root)]


def build_evidence(release_root: Path, *, report_dir: Path | None = None,
                   phase4_corpus: Path | None = None,
                   phase4_checkpoint: Path | None = None,
                   watch_dirs: Iterable[Path] | None = None,
                   consumers: Mapping[str, Any] | None = None,
                   state_specs: tuple[StateSpec, ...] = STATE_SPECS,
                   first_deployment: bool = False) -> dict:
    """Run every subject; ``consumers``/``state_specs``/``watch_dirs`` are test
    seams. Production passes none of them: the full catalog and the real
    consumer table, where a probe that does not exist yet is PENDING."""
    started = time.perf_counter()
    release_root = Path(release_root)
    findings = _Findings()
    evidence: dict[str, Any] = {"schema_version": PHASE5_EVIDENCE_SCHEMA,
                                "release_root": str(release_root), "members": [],
                                "consumers": [], "scoring_pass": {}, "rollback": {},
                                "lineage": {"status": "NOT_RUN"},
                                "phase4": {"status": "NOT_RUN"}}
    _progress("p5-accept", f"start release_root={release_root}", started)
    try:
        manifest = read_manifest(release_root)
    except (ReleaseLayoutError, OSError, ValueError) as exc:
        findings.add(RELEASE_LAYOUT, "phase5_release.json", str(exc))
        return _finish(evidence, findings, report_dir, started)
    release_id = manifest["release_id"]
    evidence.update({"release_id": release_id, "manifest_hash": manifest["manifest_hash"]})
    _progress("p5-accept", f"manifest read, release_id={release_id}", started)

    _progress("p5-accept", "members: loading model release + state members", started)
    model_release = _load_model_release(release_root, release_id, findings)
    evidence["members"] = _model_member_rows(release_root, model_release, findings)
    state_rows, loaded = _state_member_rows(release_root, manifest, findings, state_specs)
    evidence["members"] += state_rows
    _progress("p5-accept", f"members: {len(evidence['members'])} rows verified", started)

    _progress("p5-accept", "lineage: checking release lineage", started)
    evidence["lineage"] = _lineage_subject(loaded, findings)
    _progress("p5-accept", f"lineage: status={evidence['lineage'].get('status')}", started)

    dirs = list(watch_dirs) if watch_dirs is not None else _default_watch_dirs(release_root)
    if model_release is not None:
        ctx = ReleaseContext(release_root=release_root, model_release=model_release,
                             states=loaded)
        active_consumers = CONSUMERS if consumers is None else consumers
        _progress("p5-accept",
                  f"consumers: scoring pass over {len(active_consumers)} consumers", started)
        evidence["consumers"], evidence["scoring_pass"] = _scoring_pass(
            ctx, active_consumers, dirs, findings)
        _progress("p5-accept", f"consumers: {len(evidence['consumers'])} scored", started)

        _progress("p5-accept", "rollback: running promote/rollback round trip", started)
        evidence["rollback"] = _rollback_round_trip(
            release_root, release_id, first_deployment, findings)
        _progress("p5-accept",
                  f"rollback: status={evidence['rollback'].get('status')}", started)

    if phase4_corpus is None:
        _progress("p5-accept", "phase4: no --phase4-corpus given, NOT_RUN", started)
    else:
        _progress("p5-accept", f"phase4: replaying corpus={phase4_corpus}", started)
    evidence["phase4"] = _phase4_subject(phase4_corpus, release_root, model_release,
                                         manifest, dirs, findings,
                                         checkpoint_path=phase4_checkpoint)
    _progress("p5-accept", f"phase4: status={evidence['phase4'].get('status')}", started)

    _progress("p5-accept", "writing evidence + report", started)
    result = _finish(evidence, findings, report_dir, started)
    _progress("p5-accept", f"done status={result['status']}", started)
    return result


def _finish(evidence, findings, report_dir, started) -> dict:
    release_ok = not findings.rows
    phase4_ok = evidence["phase4"].get("status") == "PASS"
    evidence["release_ok"] = release_ok
    evidence["status"] = ("PASS" if release_ok and phase4_ok
                          else "RELEASE_PASS" if release_ok else "FAIL")
    evidence["findings"] = findings.rows
    evidence["finding_codes"] = sorted({row["code"] for row in findings.rows})
    evidence["runtime_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    if report_dir is not None:
        report = write_report(evidence, report_dir)
        if not _report_is_complete(report.read_text(), evidence):
            findings.add(REPORT_INCOMPLETE, "report", str(report))
            evidence["release_ok"], evidence["status"] = False, "FAIL"
            evidence["finding_codes"] = sorted({row["code"] for row in findings.rows})
        evidence["report"] = str(report)
    evidence["evidence_hash"] = content_hash(
        {k: v for k, v in evidence.items() if k not in ("runtime_ms", "report")})
    return evidence


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path(tempfile.gettempdir()) / "phase5-acceptance",
                        help="private report + evidence dir; refused inside the repo")
    parser.add_argument("--phase4-corpus", type=Path)
    parser.add_argument("--phase4-checkpoint", type=Path,
                        help="resumable JSONL of already-replayed phase4 pairs "
                             "(same corpus only; a mismatched corpus_hash is discarded)")
    parser.add_argument("--watch-dir", type=Path, action="append",
                        help="extra model-cache dir to watch (default data/models + release)")
    parser.add_argument("--first-deployment", action="store_true",
                        help="no incumbent pointer exists; accept rollback-refusal instead")
    args = parser.parse_args(argv)
    watch = None
    if args.watch_dir:
        watch = _default_watch_dirs(args.release_root) + list(args.watch_dir)
    try:
        evidence = build_evidence(args.release_root, report_dir=args.artifact_root,
                                  phase4_corpus=args.phase4_corpus,
                                  phase4_checkpoint=args.phase4_checkpoint, watch_dirs=watch,
                                  first_deployment=args.first_deployment)
    except ReportPathError as exc:
        print(f"refusing to write report: {exc}", file=sys.stderr)
        return 2
    out = args.artifact_root / "evidence.json"
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": evidence["status"], "evidence": str(out),
                      "report": evidence.get("report"),
                      "finding_codes": evidence["finding_codes"]}, indent=2))
    return 0 if evidence["release_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
