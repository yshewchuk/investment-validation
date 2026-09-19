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
5. **Phase 4.** With ``--phase4-corpus``, count the corpus pairs carrying an
   input trace. Integrated frozen-release replay of those traces is not
   implemented yet, so a supplied corpus is always a PENDING finding. Without
   one the subject is ``NOT_RUN`` and the best status is ``RELEASE_PASS``,
   never ``PASS``.
6. **Report.** A value-free private report (ids, statuses, codes and counts
   only) through ``engine.report.Report``, refused inside the repo.

Usage::

    python3 checks/phase5_acceptance.py --release-root /root/p5-6/release-A \\
        --artifact-root /root/p5-6/acceptance-A [--phase4-corpus <corpus>]
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
from typing import Any, Callable, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
PHASE4_PENDING = "P5_PHASE4_PENDING"
REPORT_INCOMPLETE = "P5_REPORT_INCOMPLETE"

FINDING_CODES = (
    RELEASE_LAYOUT, MODEL_RELEASE_INVALID, MEMBER_MISSING, MEMBER_PENDING,
    MEMBER_UNKNOWN, MEMBER_OBJECT_ABSENT, MEMBER_HASH_MISMATCH, MEMBER_UNLOADABLE,
    CONSUMER_UNRESOLVED, CONSUMER_NO_REFUSAL, CONSUMER_ERROR, CONSUMER_PENDING,
    CONSUMER_BLOCKED, RUNTIME_FIT, MODEL_CACHE_WRITE, GUARD_CONTROL_FAILED,
    ROLLBACK_NO_INCUMBENT, ROLLBACK_CANDIDATE_LIVE, ROLLBACK_NOT_EXACT,
    PROMOTE_REFUSED, PHASE4_PENDING, REPORT_INCOMPLETE,
)


class ReportPathError(ValueError):
    """The private report path resolves inside the repo."""

    code = "REPORT_PATH_INSIDE_REPO"


@dataclasses.dataclass
class _Findings:
    rows: list[dict] = dataclasses.field(default_factory=list)

    def add(self, code: str, subject: str, detail: str) -> None:
        self.rows.append({"code": code, "subject": subject, "detail": detail})


@dataclasses.dataclass
class ReleaseContext:
    """What the consumer probes read: the resolved release, never data/."""

    release_root: Path
    model_release: ModelRelease
    states: dict[str, dict]


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
    # The staged manifest records its own member hash; recompute it with the
    # deployment module's own function so a hand-edited manifest is caught.
    if deployment._release_hash(manifest.release) != manifest.release_hash:
        findings.add(MODEL_RELEASE_INVALID, release_id, "release_hash disagrees with members")
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
            artifacts.append(_typed_load(spec, release_root, obj))
        except (ValueError, KeyError, TypeError) as exc:
            findings.add(MEMBER_UNLOADABLE, spec.member_id, type(exc).__name__)
            verdict = MEMBER_UNLOADABLE
    if verdict == "ok":
        loaded[spec.member_id] = {"row": row, "artifacts": artifacts}
    return verdict


# --------------------------------------------------------------------------
# 2. consumers
# --------------------------------------------------------------------------

ProbeResult = tuple[bool, bool, str]  # (resolved, refused_when_missing, detail)


def _probe_frozen_executor(ctx: ReleaseContext) -> list[dict]:
    """``FrozenStageExecutor`` over every staged model binding."""
    from engine.v2.models.loader import FrozenInference
    from engine.v2.scoring.frozen_executor import FrozenStageExecutor, FrozenStageRefusal

    rows = []
    root = deployment_root(ctx.release_root)
    for binding in ctx.model_release.bindings:
        features = {name: 0.0 for name in binding.feature_order}
        resolved, detail = False, ""
        try:
            result = FrozenStageExecutor(inference=FrozenInference(root),
                                         release=ctx.model_release,
                                         binding_id=binding.binding_id).execute(features)
            staged = tuple(member.content_hash for member in binding.members)
            resolved = tuple(result.artifact_hashes) == staged
            detail = "" if resolved else "artifact hashes differ from the staged binding"
        except FrozenStageRefusal as exc:
            detail = exc.code + ":" + ",".join(exc.reason_codes)
        broken = dataclasses.replace(binding, members=tuple(
            dataclasses.replace(member, path=f"objects/absent-{member.name}")
            for member in binding.members))
        stripped = dataclasses.replace(ctx.model_release, bindings=tuple(
            broken if item.binding_id == binding.binding_id else item
            for item in ctx.model_release.bindings))
        refused = False
        try:
            FrozenStageExecutor(inference=FrozenInference(root), release=stripped,
                                binding_id=binding.binding_id).execute(features)
        except FrozenStageRefusal as exc:
            refused = exc.code == "MODEL_NOT_READY"
        rows.append({"consumer": "frozen_stage_executor",
                     "member_id": f"model:{binding.role}:{binding.strategy_id}",
                     "resolved": resolved, "refused_when_missing": refused, "detail": detail})
    return rows


def _probe_request(strategy: str, alpha: float):
    from engine.v2.contracts import ScoreRequest

    return ScoreRequest(
        event_id="p5-6-probe", calendar_revision="probe", strategy_version=strategy,
        deployment_id="p5-6-probe", decision_clock_id="entry-close",
        requested_decision_at="2026-09-16", snapshot_id="probe", mode="replay",
        fill_model={"alpha": alpha},
    )


def _probe_bundle(strategy: str, artifact, cutoff):
    """A source-only probe bundle. The context is fixed and synthetic: this
    probe proves resolution and refusal, it is not a parity comparison."""
    from engine.v2.scoring.source_inputs import SourceBundle

    zero_pool = ({"prediction": 1.0, "residual": 0.0},)
    runup = strategy == "STR-RUNUP"
    context = {"ticker": "P5PROBE", "event_date": "2026-09-16", "entry_date": "2026-09-16",
               "exit_date": "2026-09-17", "expiry": "2026-09-18", "spot": 100.0}
    recipes = {"driver_prediction": {"intercept": 1.0, "coefficients": {}}}
    refs = {"driver_prediction": "sha256:probe-driver"}
    extra = {}
    if runup:
        context.update({"strike": 100.0, "days_before_print": 7.0})
        recipes["runup_move_prediction"] = {"intercept": 0.0, "coefficients": {}}
        refs["runup_move_prediction"] = "sha256:probe-move"
        extra = {"runup_move_residual_rows": zero_pool}
    recipe = {"seed": 1, "draw_count": 16}
    if cutoff is not None:
        recipe["before"] = cutoff
    return SourceBundle(
        source_ref="p5-6-probe", context=context,
        raw_quotes={("C", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 3.0},
                    ("P", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 3.0}},
        feature_vector={}, feature_missing_mask={},
        model_identity={"driver": {"model_id": "probe"}},
        forecast_recipes=recipes, model_artifact_refs=refs,
        residual_recipe={"terminal_spots": (95.0, 105.0), "weights": (0.5, 0.5),
                         "capital_at_risk": 1.0},
        analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
        strategy=strategy, payoff_artifact_recipe=recipe, payoff_artifact=artifact,
        model_residual_rows=zero_pool, **extra,
    )


def _score_flags(strategy: str, artifact, alpha: float, cutoff) -> tuple[str, ...]:
    from engine.v2.scoring import application
    from engine.v2.scoring.source_inputs import build_native_score_inputs

    inputs = build_native_score_inputs(_probe_bundle(strategy, artifact, cutoff))
    record = application.score_one(_probe_request(strategy, alpha), inputs)
    return tuple(record.reason_codes)


def _payoff_probe(member_id: str, strategy: str, consumer: str):
    def probe(ctx: ReleaseContext) -> list[dict]:
        state = ctx.states.get(member_id)
        if state is None:
            return [{"consumer": consumer, "member_id": member_id, "blocked": True}]
        rows = []
        for artifact in state["artifacts"]:
            flags = _score_flags(strategy, artifact, artifact.alpha, artifact.cutoff)
            resolved = not ({"MODEL_NOT_READY", "NO_PAYOFF_MAP"} & set(flags))
            missing = _score_flags(strategy, None, artifact.alpha, artifact.cutoff)
            rows.append({"consumer": consumer, "member_id": member_id, "resolved": resolved,
                         "refused_when_missing": "MODEL_NOT_READY" in missing,
                         "detail": "" if resolved else ",".join(flags)})
        return rows
    return probe


#: consumer id -> probe. ``None`` means no probe exists yet: PENDING.
CONSUMERS: dict[str, Callable[[ReleaseContext], list[dict]] | None] = {
    "frozen_stage_executor": _probe_frozen_executor,
    "model_stage.payoff_line": _payoff_probe(
        "payoff_line:STR-THRU", "STR-THRU", "model_stage.payoff_line"),
    "model_stage.payoff_surface": _payoff_probe(
        "payoff_surface:STR-RUNUP", "STR-RUNUP", "model_stage.payoff_surface"),
    "model_stage.driver_residual_pool": None,
    "simulation.paired_residual_pool": None,
    "model_stage.recalibration": None,
    "chooser.admissible_table": None,
    "gate.trailing_cutoff": None,
    "chooser.analog_pool": None,
    "analogs.board_analog_matcher": None,
    "features.tier4_serving_folds": None,
}


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


def _phase4_subject(corpus: Path | None, findings: _Findings) -> dict:
    if corpus is None:
        return {"status": "NOT_RUN"}
    from checks.tier0_corpus import load, resolve_corpus

    loaded = load(resolve_corpus(corpus))
    pairs = list(loaded.pairs.values())
    traced = sum(1 for pair in pairs if (pair.get("payload") or {}).get("input_trace"))
    findings.add(PHASE4_PENDING, "phase4",
                 f"{traced}/{len(pairs)} pairs carry input_trace; integrated frozen-release "
                 "replay is not implemented yet")
    return {"status": "PENDING", "pairs": len(pairs), "traced": traced,
            "corpus_hash": loaded.index.get("corpus_hash")}


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
             **evidence["rollback"].get("checks", {})}.items())],
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
            _section("Findings", ["code", "subject", "detail"], rows["findings"]),
        ],
    }
    return Report(context).write(resolved, filename="phase5_report.md")


def _report_is_complete(text: str, evidence: dict) -> bool:
    headers = ("## 0. Verdict", "## 8. Provenance", "Release members", "Score consumers",
               "No-fit and rollback controls", "Findings")
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
                                "phase4": {"status": "NOT_RUN"}}
    try:
        manifest = read_manifest(release_root)
    except (ReleaseLayoutError, OSError, ValueError) as exc:
        findings.add(RELEASE_LAYOUT, "phase5_release.json", str(exc))
        return _finish(evidence, findings, report_dir, started)
    release_id = manifest["release_id"]
    evidence.update({"release_id": release_id, "manifest_hash": manifest["manifest_hash"]})
    model_release = _load_model_release(release_root, release_id, findings)
    evidence["members"] = _model_member_rows(release_root, model_release, findings)
    state_rows, loaded = _state_member_rows(release_root, manifest, findings, state_specs)
    evidence["members"] += state_rows
    if model_release is not None:
        ctx = ReleaseContext(release_root=release_root, model_release=model_release,
                             states=loaded)
        dirs = list(watch_dirs) if watch_dirs is not None else _default_watch_dirs(release_root)
        evidence["consumers"], evidence["scoring_pass"] = _scoring_pass(
            ctx, CONSUMERS if consumers is None else consumers, dirs, findings)
        evidence["rollback"] = _rollback_round_trip(
            release_root, release_id, first_deployment, findings)
    evidence["phase4"] = _phase4_subject(phase4_corpus, findings)
    return _finish(evidence, findings, report_dir, started)


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
                                  phase4_corpus=args.phase4_corpus, watch_dirs=watch,
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
