"""``phase5_state_stage`` job (#192 slice 3): stage a candidate release with the nightly's advanced pools.

Thin job wrapper (parameters, validator, plan, kind, worker adapter); the staging logic is
``tools/phase5_state_stage_job.py``, as ``training.py`` wraps ``tools/phase5_training_job.py``.
Contract: ``engine/v2/models/ARCHITECTURE.md`` section 2.2 (``phase5_state_stage``). Never touches ``DEPLOYED``;
``models_promote`` is a separate job.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from engine.v2.foundation import ArtifactError, safe_relative_path
from engine.v2.ops.errors import fail
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.submission import JobKind, RetryPolicy

RESULT_SCHEMA = "phase5_state_stage_result.v1.0"


@dataclass(frozen=True, kw_only=True)
class StateStageParameters:
    expected_ids: tuple[str, ...]        # always ("phase5_state_stage",)
    #: The store's parent: the catalog and objects live under ``<release_root>/deployment``.
    release_root: str
    prior_release_id: str | None
    release_id: str
    cutoff: str
    #: Frozen-state file names; exactly the ``input_bindings`` keys other than
    #: the pinned ``legacy_manifest.json`` read-set binding.
    state_files: tuple[str, ...]
    input_bindings: dict[str, str] | None = None
    mode: str = "carry"
    #: Bootstrap-only assembly inputs: ``tier4_dir``, ``panel_catalog``,
    #: ``panel_snapshot_id`` and ``panel_artifact_root`` (all non-empty strings).
    bootstrap: dict | None = None


def _safe_id(value) -> bool:
    return (isinstance(value, str) and bool(value) and not value.startswith(".")
            and not any(char in value for char in ("/", "\\", "\0")))


def _identity_problems(params) -> list[str]:
    problems = []
    if params.expected_ids != ("phase5_state_stage",):
        problems.append("expected_ids must be exactly ('phase5_state_stage',)")
    if not isinstance(params.release_root, str) or not params.release_root:
        problems.append("release_root must be a non-empty string")
    if params.mode not in ("carry", "bootstrap"):
        problems.append("mode must be 'carry' or 'bootstrap'")
    elif params.mode == "bootstrap" and params.prior_release_id is None:
        if not _safe_id(params.release_id):
            problems.append("release_id must be a safe non-empty id")
    elif not _safe_id(params.prior_release_id) or not _safe_id(params.release_id):
        problems.append("prior_release_id and release_id must be safe non-empty ids")
    elif params.prior_release_id == params.release_id:
        problems.append("release_id must differ from prior_release_id")
    try:
        if params.cutoff != date.fromisoformat(params.cutoff).isoformat():
            raise ValueError("not extended ISO")
    except (TypeError, ValueError):
        problems.append("cutoff must be an extended ISO date (YYYY-MM-DD)")
    return problems


def _plain_json_name(name) -> bool:
    try:
        parts = safe_relative_path(name)
    except ArtifactError:
        return False
    return len(parts) == 1 and name.endswith(".json") and not name.endswith(".summary.json")


def _file_problems(params) -> list[str]:
    files = params.state_files
    if not isinstance(files, tuple) or not files or len(set(files)) != len(files):
        return ["state_files must be a non-empty tuple without duplicates"]
    problems = []
    if not all(_plain_json_name(name) for name in files):
        problems.append("state_files must be plain .json names, not summaries")
    if "legacy_manifest.json" in files:
        problems.append("state_files must not list legacy_manifest.json")
    expected = set(files) | {"legacy_manifest.json"}
    bindings = params.input_bindings
    if not isinstance(bindings, Mapping) or set(bindings) != expected:
        problems.append("input_bindings keys must equal state_files plus legacy_manifest.json")
    elif not isinstance(bindings.get("legacy_manifest.json"), str) \
            or not bindings["legacy_manifest.json"]:
        problems.append("input_bindings['legacy_manifest.json'] must be a non-empty string")
    return problems


_BOOTSTRAP_KEYS = frozenset({"tier4_dir", "panel_catalog", "panel_snapshot_id",
                             "panel_artifact_root"})


def _bootstrap_problems(params) -> list[str]:
    bootstrap = params.bootstrap
    if params.mode == "bootstrap":
        if not isinstance(bootstrap, Mapping):
            return ["bootstrap must be a mapping with tier4_dir, panel_catalog, "
                    "panel_snapshot_id and panel_artifact_root"]
        problems = []
        if set(bootstrap) != _BOOTSTRAP_KEYS:
            problems.append("bootstrap keys must be exactly tier4_dir, panel_catalog, "
                            "panel_snapshot_id and panel_artifact_root")
        elif not all(isinstance(bootstrap[key], str) and bootstrap[key]
                     for key in _BOOTSTRAP_KEYS):
            problems.append("bootstrap values must be non-empty strings")
        return problems
    if bootstrap is not None:
        return ["bootstrap must be present only when mode is bootstrap"]
    return []


def state_stage_parameter_problems(job, params) -> list[str]:
    """Pure validator for the ``phase5_state_stage`` kind (no data or artifact reads)."""
    return [*_identity_problems(params), *_file_problems(params),
            *_bootstrap_problems(params)]


def state_stage_job_kind() -> JobKind:
    """Candidate write only; the pointer lease serialises it against ``models_promote``."""
    return JobKind(name="phase5_state_stage", worker="phase5_state_stage",
                   parameters=StateStageParameters,
                   resource_classes=frozenset({"projection"}), effects=("staged",),
                   retry=RetryPolicy("none", 1),
                   checkpoint_contract=RESULT_SCHEMA,
                   namespaces=frozenset({"shadow", "smoke"}),
                   validate=state_stage_parameter_problems,
                   store_domains=(("deployment_pointer", "write"), ("legacy_store", "read")))


def state_stage_plan(*, release_root, prior_release_id, release_id, cutoff, state_refs,
                     manifest_ref, source_root, mode="carry", bootstrap=None) -> dict:
    """Build a ``phase5_state_stage`` plan; ``state_refs`` maps file name -> training output id.
    ``manifest_ref`` is the pinned ``legacy_input_manifest.v1.0`` artifact the supervisor's
    launch-time ``_pin_read_set`` stages as ``staging/legacy``. ``source_root`` is the code
    checkout the implementation fingerprint is taken from (the caller's code source).
    ``mode`` is ``"carry"`` (default) or ``"bootstrap"`` (#1102); ``bootstrap`` carries the
    bootstrap assembly inputs and is omitted from a carry plan's parameters."""
    from engine.v2.foundation import content_hash
    from engine.v2.models import deployment
    from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest

    if not release_root:
        try:
            release_root = deployment.production_release_root()
        except deployment.MissingReleaseRoot as exc:
            raise fail("INVALID_REQUEST", "no release root given and no production "
                       "release root is configured") from exc
    release_root = str(Path(release_root).expanduser().resolve())
    profile = profile_named(DEFAULT_POLICY, "projection")
    refs = dict(state_refs)
    params = StateStageParameters(
        expected_ids=("phase5_state_stage",), release_root=release_root,
        prior_release_id=prior_release_id, release_id=release_id, cutoff=cutoff,
        state_files=tuple(sorted(refs)),
        input_bindings={**refs, "legacy_manifest.json": manifest_ref}, mode=mode,
        bootstrap=dict(bootstrap) if bootstrap is not None else None)
    problems = state_stage_parameter_problems(None, params)
    if problems:
        raise fail("INVALID_REQUEST", "state stage plan is invalid",
                   details={"problems": list(problems)})
    parameters = vars(params)
    if parameters.get("bootstrap") is None:
        parameters = {key: value for key, value in parameters.items() if key != "bootstrap"}
    root3 = Path(source_root)
    return {"schema_version": "operations_plan.v1.0", "kind": "phase5_state_stage",
            "mode": "shadow", "effects": ["staged"], "parameters": parameters,
            "input_refs": [*sorted(refs.values()), manifest_ref], "blocked_prerequisites": [],
            "spec_hash": content_hash(parameters),
            "implementation_ref": content_hash(worker_source_manifest(root3)),
            "environment_ref": content_hash(
                environment_identity(profile.thread_count or profile.cpu_count)),
            "resource_class": "projection"}


def _worker_problems(parameters) -> list[str]:
    try:
        params = StateStageParameters(
            expected_ids=tuple(parameters["expected_ids"]), release_root=parameters["release_root"],
            prior_release_id=parameters["prior_release_id"], release_id=parameters["release_id"],
            cutoff=parameters["cutoff"], state_files=tuple(parameters["state_files"]),
            input_bindings=parameters["input_bindings"],
            mode=parameters.get("mode", "carry"),
            bootstrap=parameters.get("bootstrap"))
    except (KeyError, TypeError):
        return ["parameters are missing a field or have the wrong type"]
    return state_stage_parameter_problems(None, params)


def run_state_stage_worker(parameters, root) -> dict:
    """Stage the candidate inside the job's staging root; a refusal is a typed ``OpsError``."""
    from tools import phase5_state_stage_job as job

    root = Path(root)
    problems = _worker_problems(parameters)
    if problems:
        raise fail("VALIDATION_FAILED", "state stage parameters are invalid",
                   details={"problems": problems})
    try:
        summary = job.stage_candidate(parameters, root)
    except job.StageRefused as exc:
        raise fail("VALIDATION_FAILED", str(exc), details=exc.details) from exc
    (root / "stage_result.json").write_text(
        json.dumps({"schema_version": RESULT_SCHEMA, **summary}, allow_nan=False, sort_keys=True))
    return {"outputs": [{"name": "stage_result", "path": "stage_result.json",
                         "schema": RESULT_SCHEMA}],
            "completed_ids": list(parameters["expected_ids"]), "no_work": False}
