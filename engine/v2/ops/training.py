"""Supervised training jobs (P6 slice 5, training half).

``tools/phase5_training_job.py`` is the audited training entry point; this
module gives its four job functions the ``training`` kind's parameter schema,
pure validator, plan builder and worker. The promote half builds on the same
wiring: an operator-submitted ``models_promote`` pointer swap, never part of
the nightly DAG. No bespoke CLI verb: ``ops plan`` and ``ops submit`` are the
existing generic subcommands.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from engine.v2.foundation import ArtifactError, safe_relative_path
from engine.v2.ops.errors import fail
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.submission import JobKind, RetryPolicy

MODES = ("recipe", "state", "board_analog", "trailing_cutoff")


@dataclass(frozen=True, kw_only=True)
class TrainingParameters:
    expected_ids: tuple[str, ...]
    mode: str
    recipe: str = ""
    state: str = ""
    alpha: float | None = None
    cutoffs: tuple[str, ...] = ()
    strategies: tuple[str, ...] = ()
    pairs_path: str = ""
    ticker_chunk: int = 1000
    #: The pinned legacy read set, ``{"legacy_manifest.json": <artifact id>}``,
    #: exactly like every ``legacy_*`` stage's own binding (``nightly.
    #: _stage_inputs``): the supervisor's launch-time ``_pin_read_set`` copies
    #: the manifest's ``file_refs`` into ``staging/legacy`` so the dataset
    #: builders read the real panel/tier4 inputs through ``INVESTING_PLAN_ROOT``.
    #: ``None`` on a plan whose operator named no manifest -- such a plan is
    #: blocked for submission (see :func:`training_plan`) rather than failing
    #: ``INPUT_CHANGED`` at launch.
    input_bindings: dict[str, str] | None = None


@dataclass(frozen=True, kw_only=True)
class PromoteParameters:
    expected_ids: tuple[str, ...]        # always ("models_promote",)
    release_root: str
    release_id: str
    #: Optional expected-incumbent guard forwarded to
    #: ``deployment.promote``: when set, the worker's swap refuses
    #: ``CONCURRENT_PROMOTE`` unless the live pointer names this release.
    #: ``None`` keeps the unguarded behavior.
    expected_previous_release_id: str | None = None
    input_bindings: dict[str, str] | None = None


@dataclass(frozen=True, kw_only=True)
class RollbackParameters:
    expected_ids: tuple[str, ...]        # always ("models_rollback",)
    release_root: str
    #: The live incumbent pinned at plan time, plus the undo-stack target the
    #: operator intended.  Execution refuses a plan whose incumbent no longer
    #: matches the live pointer, or whose target has since changed, so a stale
    #: plan can never undo a promotion that happened after it was made.  A store
    #: with no prior incumbent pins ``target_release_id=None`` and is refused by
    #: the worker's own ``deployment.rollback`` call, keeping the typed
    #: ``VALIDATION_FAILED``/``NoPriorRelease`` path.
    incumbent_release_id: str | None = None
    incumbent_sequence: int | None = None
    target_release_id: str | None = None
    input_bindings: dict[str, str] | None = None


def _recipe_problems(params) -> list[str]:
    from tools import phase5_training_job as job

    if not params.recipe:
        return ["mode=recipe needs a recipe key"]
    problems = []
    if params.state:
        problems.append("mode=recipe must not name a state")
    try:
        recipe = job.current_recipes()[job._key(params.recipe)]
    except (KeyError, ValueError):
        return [*problems, "recipe is not a current training recipe"]
    if recipe.folds.kind == "request_cutoff":
        if params.alpha is None:
            problems.append("a request_cutoff recipe needs alpha")
        if not params.cutoffs:
            problems.append("a request_cutoff recipe needs at least one cutoff")
    elif params.alpha is not None or params.cutoffs:
        problems.append("alpha/cutoffs apply to request_cutoff recipes only")
    if params.pairs_path:
        try:
            safe_relative_path(params.pairs_path)
        except ArtifactError:
            problems.append("pairs_path must be a plain relative path")
    return problems


def _state_problems(params) -> list[str]:
    from tools import phase5_training_job as job

    problems = []
    if params.recipe:
        problems.append("mode=state must not name a recipe")
    if params.alpha is not None:
        problems.append("mode=state does not take alpha")
    if (params.state not in job.STATES
            or params.state in (job.BOARD_ANALOG_STATE, job.TRAILING_CUTOFF_STATE)):
        problems.append("mode=state needs a frozen P5-4 state")
    if len(params.cutoffs) > 1 or (params.cutoffs and params.state != "paired_residual_pool"):
        problems.append("at most one cutoff, only for paired_residual_pool")
    if params.pairs_path:
        problems.append("mode=state does not take pairs_path")
    return problems


def _board_analog_problems(params) -> list[str]:
    problems = []
    if params.alpha is None:
        problems.append("mode=board_analog needs alpha")
    if not params.cutoffs:
        problems.append("mode=board_analog needs at least one cutoff")
    if params.recipe or params.state:
        problems.append("mode=board_analog must not name a recipe or state")
    if params.pairs_path:
        problems.append("mode=board_analog does not take pairs_path")
    return problems


def _trailing_cutoff_problems(params) -> list[str]:
    problems = []
    if not params.cutoffs:
        problems.append("mode=trailing_cutoff needs at least one cutoff")
    if params.alpha is not None:
        problems.append("mode=trailing_cutoff does not take alpha")
    if params.strategies:
        problems.append("mode=trailing_cutoff does not take strategies")
    if params.recipe or params.state:
        problems.append("mode=trailing_cutoff must not name a recipe or state")
    if params.pairs_path:
        problems.append("mode=trailing_cutoff does not take pairs_path")
    return problems


def _shape_problems(params) -> list[str]:
    """Checks independent of ``mode`` -- a plan built directly via
    :func:`training_plan` skips ``from_document``'s own type/finite/bool checks entirely (see
    module docstring notes), so this is the only place a submitted ``ticker_chunk``, ``alpha`` or
    ``cutoffs`` entry is checked for being a sane VALUE, not just the right JSON type."""
    problems = []
    if (isinstance(params.ticker_chunk, bool) or not isinstance(params.ticker_chunk, int)
            or params.ticker_chunk <= 0):
        problems.append("ticker_chunk must be a positive int")
    if params.alpha is not None:
        if (isinstance(params.alpha, bool) or not isinstance(params.alpha, (int, float))
                or not math.isfinite(params.alpha) or params.alpha < 0):
            problems.append("alpha must be a finite, non-negative number")
    for cutoff in params.cutoffs:
        try:
            date.fromisoformat(cutoff)
        except (TypeError, ValueError):
            problems.append("cutoffs must be valid ISO dates (YYYY-MM-DD)")
            break
    return problems


def training_parameter_problems(job, params) -> list[str]:
    """Pure validator for the ``training`` kind (no data or artifact reads)."""
    problems = []
    if params.expected_ids != ("training",):
        problems.append("expected_ids must be exactly ('training',)")
    problems.extend(_shape_problems(params))
    if params.mode not in MODES:
        problems.append("mode must be one of " + ", ".join(MODES))
        return problems
    if params.mode == "recipe":
        problems.extend(_recipe_problems(params))
    elif params.mode == "state":
        problems.extend(_state_problems(params))
    elif params.mode == "board_analog":
        problems.extend(_board_analog_problems(params))
    else:
        problems.extend(_trailing_cutoff_problems(params))
    return problems


def training_job_kind() -> JobKind:
    return JobKind(name="training", worker="training", parameters=TrainingParameters,
                   resource_classes=frozenset({"experiment_heavy"}), effects=("staged",),
                   retry=RetryPolicy("bounded", 1, (60,)),
                   checkpoint_contract="training_job_result.v1.0",
                   namespaces=frozenset({"shadow", "smoke"}),
                   validate=training_parameter_problems,
                   store_domains=(("legacy_store", "read"),))


def promote_job_kind() -> JobKind:
    """The operator-submitted pointer swap. ``deployment.promote`` only
    touches the release-store path named by the job's own
    ``release_root`` parameter, never the shared legacy tree -- but its
    ``_swap_pointer`` read-modify-write (current pointer + next sequence,
    then an atomic pointer write, then an append-only history write) has
    no locking of its own, so a shared write lease on the
    ``deployment_pointer`` domain serializes every ``models_promote``
    claim against every other one, globally, regardless of which
    ``release_root`` each names (2026-09-26 CodeRabbit finding).

    ``effects=("staged",)`` understates this: the job swaps the live
    DEPLOYED pointer, not a staging artifact. Left as ``"staged"``
    deliberately rather than invented on the spot -- a repo-wide grep
    (``engine/``, ``checks/``, ``tools/``) for ``.effects`` finds every
    ``JobKind`` in the codebase (``stages.py``, ``incremental_data.py``,
    ``training.py``'s own ``training_job_kind``) declaring exactly
    ``("staged",)`` and no reader of the field anywhere (see
    ``tests/test_v2_ops_stages_core_kinds.py``'s module docstring); there is
    no existing ``"deployed"``/``"pointer_swap"`` vocabulary value to reach
    for instead, and coining a one-off value for this single kind would make
    it look like a real, consumed distinction when the field is otherwise
    pure documentation. Revisit if/when something starts reading
    ``JobKind.effects``."""
    return JobKind(name="models_promote", worker="models_promote", parameters=PromoteParameters,
                   resource_classes=frozenset({"delivery"}), effects=("staged",),
                   retry=RetryPolicy("bounded", 1, (30,)),
                   checkpoint_contract="promote_pointer_state.v1.0",
                   namespaces=frozenset({"shadow", "smoke"}),
                   store_domains=(("deployment_pointer", "write"),))


def rollback_job_kind() -> JobKind:
    """The operator-submitted pointer rollback, the mirror of
    :func:`promote_job_kind`. ``deployment.rollback`` resolves its target
    from the release store's own append-only pointer history and then runs
    the same unlocked ``_swap_pointer`` read-modify-write promote does, so it
    carries the same shared ``deployment_pointer`` write lease that serializes
    every pointer swap globally. ``effects=("staged",)`` is the same
    deliberate understatement promote documents; no reader consumes
    ``JobKind.effects`` today."""
    return JobKind(name="models_rollback", worker="models_rollback",
                   parameters=RollbackParameters,
                   resource_classes=frozenset({"delivery"}), effects=("staged",),
                   retry=RetryPolicy("bounded", 1, (30,)),
                   checkpoint_contract="promote_pointer_state.v1.0",
                   namespaces=frozenset({"shadow", "smoke"}),
                   store_domains=(("deployment_pointer", "write"),))


def training_plan(*, mode, recipe="", state="", alpha=None, cutoffs=(), strategies=(),
                  pairs_path="", ticker_chunk=1000, manifest_ref=None) -> dict:
    """Build an operator-submitted ``training`` plan.

    ``manifest_ref`` is the pinned ``legacy_input_manifest.v1.0`` artifact the
    worker's legacy read set is staged from, exactly like a nightly stage's own
    ``legacy_manifest.json`` binding. Without it the job has no declared read
    set and the supervisor's launch-time ``_pin_read_set`` refuses
    ``INPUT_CHANGED``, so such a plan carries the same blocked prerequisites a
    manifest-less nightly plan does and can never be submitted.
    """
    from engine.v2.foundation import content_hash
    from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest

    profile = profile_named(DEFAULT_POLICY, "experiment_heavy")
    params = TrainingParameters(
        expected_ids=("training",), mode=mode, recipe=recipe, state=state,
        alpha=alpha, cutoffs=tuple(cutoffs), strategies=tuple(strategies),
        pairs_path=pairs_path, ticker_chunk=ticker_chunk,
        input_bindings=({"legacy_manifest.json": manifest_ref} if manifest_ref else None))
    problems = training_parameter_problems(None, params)
    if problems:
        raise fail("INVALID_REQUEST", "training plan is invalid",
                   details={"problems": list(problems)})
    root3 = Path(__file__).resolve().parents[3]
    return {"schema_version": "operations_plan.v1.0", "kind": "training", "mode": "shadow",
            "effects": ["staged"], "parameters": vars(params),
            "input_refs": [manifest_ref] if manifest_ref else [],
            "blocked_prerequisites": [] if manifest_ref else [
                "frozen_legacy_input_manifest", "adapter_parity_receipt"],
            "spec_hash": content_hash(vars(params)),
            "implementation_ref": content_hash(worker_source_manifest(root3)),
            "environment_ref": content_hash(
                environment_identity(profile.thread_count or profile.cpu_count)),
            "resource_class": "experiment_heavy"}


def promote_plan(*, release_root, release_id,
                 expected_previous_release_id=None) -> dict:
    """Build an operator-submitted ``models_promote`` plan. An empty
    ``release_root`` resolves ``engine.v2.models.deployment.
    production_deployment_root()`` (config key ``MODEL_RELEASE_ROOT``,
    one level below the value that key names -- see that function's
    docstring) here, at plan time -- a missing key is ``INVALID_REQUEST``
    and never reaches the worker with an empty root. An operator may also
    pass ``expected_previous_release_id`` (the optional expected-incumbent
    guard); ``None`` keeps the unguarded behavior."""
    from engine.v2.foundation import content_hash
    from engine.v2.models import deployment
    from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest

    if not release_id:
        raise fail("INVALID_REQUEST", "promote plan needs a release id")
    if not release_root:
        try:
            release_root = deployment.production_deployment_root()
        except deployment.MissingReleaseRoot as exc:
            raise fail("INVALID_REQUEST", "no release root given and no production "
                       "release root is configured") from exc
    # Absolute at plan time: the worker's cwd is its code snapshot
    # (``executor.launch``), so a relative path would promote somewhere the
    # operator never named.
    release_root = str(Path(release_root).expanduser().resolve())
    profile = profile_named(DEFAULT_POLICY, "delivery")
    params = PromoteParameters(expected_ids=("models_promote",), release_root=release_root,
                               release_id=release_id,
                               expected_previous_release_id=expected_previous_release_id)
    root3 = Path(__file__).resolve().parents[3]
    return {"schema_version": "operations_plan.v1.0", "kind": "promote", "mode": "shadow",
            "effects": ["staged"], "parameters": vars(params), "input_refs": [],
            "blocked_prerequisites": [], "spec_hash": content_hash(vars(params)),
            "implementation_ref": content_hash(worker_source_manifest(root3)),
            "environment_ref": content_hash(
                environment_identity(profile.thread_count or profile.cpu_count)),
            "resource_class": "delivery"}


def rollback_plan(*, release_root="") -> dict:
    """Build an operator-submitted ``models_rollback`` plan.

    Rollback takes no ``release_id`` and no expected-incumbent guard: the
    target is resolved by ``deployment.rollback`` from the release store's own
    append-only pointer history. The plan pins the live incumbent release id
    and pointer sequence, plus the undo-stack target ``deployment``'s own
    resolver names, so execution can refuse a plan the store has moved past
    (``run_rollback_worker``). An empty ``release_root`` resolves
    ``production_deployment_root()`` here, at plan time, exactly like
    :func:`promote_plan`; a missing key is ``INVALID_REQUEST`` and never
    reaches the worker with an empty root. The path is made absolute at plan
    time because the worker's cwd is its code snapshot (``executor.launch``).
    A store with no prior incumbent still gets a plan (``target_release_id`` is
    ``None``) rather than a plan-time refusal, so the job fails through the
    existing typed ``VALIDATION_FAILED``/``NoPriorRelease`` path. Every other
    resolver refusal -- an unreadable, gapped or inconsistent pointer history --
    is raised here as typed ``VALIDATION_FAILED`` carrying the resolver's own
    exception class, so a plan is never saved with a bogus target.
    """
    from engine.v2.foundation import content_hash
    from engine.v2.models import deployment
    from engine.v2.ops.experiments import default_checkout_root
    from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest

    if not release_root:
        try:
            release_root = deployment.production_deployment_root()
        except deployment.MissingReleaseRoot as exc:
            raise fail("INVALID_REQUEST", "no release root given and no production "
                       "release root is configured") from exc
    store = Path(release_root).expanduser().resolve()
    try:
        pointer = deployment.current_pointer(store)
    except ValueError as exc:
        raise fail("VALIDATION_FAILED", "rollback pointer could not be read",
                   details={"exception_class": type(exc).__name__}) from exc
    try:
        target = deployment.rollback_target(store)
    except deployment.NoPriorRelease:
        target = None
    except deployment.DeploymentError as exc:
        raise fail("VALIDATION_FAILED", "rollback target could not be resolved",
                   details={"exception_class": type(exc).__name__}) from exc
    profile = profile_named(DEFAULT_POLICY, "delivery")
    params = RollbackParameters(
        expected_ids=("models_rollback",), release_root=str(store),
        incumbent_release_id=None if pointer is None else pointer.release_id,
        incumbent_sequence=None if pointer is None else pointer.sequence,
        target_release_id=target)
    root = default_checkout_root()
    return {"schema_version": "operations_plan.v1.0", "kind": "rollback", "mode": "shadow",
            "effects": ["staged"], "parameters": vars(params), "input_refs": [],
            "blocked_prerequisites": [], "spec_hash": content_hash(vars(params)),
            "implementation_ref": content_hash(worker_source_manifest(root)),
            "environment_ref": content_hash(
                environment_identity(profile.thread_count or profile.cpu_count)),
            "resource_class": "delivery"}


def _pinned_pairs_path(root: Path, pairs_path: str) -> str | None:
    """Resolve a recipe's ``pairs_path`` beneath the attempt's staged, pinned legacy read
    set (``root / "legacy"``, populated ONLY from the plan's pinned manifest ``file_refs`` --
    see ``supervisor.Service._populate_legacy_staging``), refusing anything that is not a
    plain relative path staying inside that root -- same idiom as
    ``experiments.py::input_manifest``. The symlink check runs on the PRE-resolve candidate:
    a post-resolve check can never fire, since resolving a path already follows every symlink
    component to its target, and it catches a same-root symlink substitution that the
    post-resolve prefix check alone would miss. ``None`` (no override; the tool's own default
    path applies) when ``pairs_path`` is empty."""
    if not pairs_path:
        return None
    try:
        safe_relative_path(pairs_path)
    except ArtifactError:
        raise fail("INPUT_CHANGED", "pairs_path is not a pinned legacy input",
                   details={"pairs_path": pairs_path}) from None
    base = (root / "legacy").resolve()
    candidate = base / pairs_path
    if candidate.is_symlink():
        raise fail("INPUT_CHANGED", "pairs_path is not a pinned legacy input",
                   details={"pairs_path": pairs_path})
    path = candidate.resolve()
    if not path.is_file() or not str(path).startswith(str(base) + "/"):
        raise fail("INPUT_CHANGED", "pairs_path is not a pinned legacy input",
                   details={"pairs_path": pairs_path})
    return str(path)


def _run_recipe(parameters, out_dir, root) -> dict:
    from tools import phase5_training_job as job

    try:
        recipe = job.current_recipes()[job._key(parameters["recipe"])]
    except (KeyError, ValueError):
        raise fail("VALIDATION_FAILED", "recipe is not a current training recipe") from None
    job._guard("tools.phase5_training_job._run_recipe:" + recipe.recipe_id)
    dataset = job.build_dataset(recipe, pairs_path=_pinned_pairs_path(root, parameters["pairs_path"]))
    extra = ({"cutoffs": tuple(parameters["cutoffs"]), "alpha": parameters["alpha"]}
             if recipe.folds.kind == "request_cutoff" else {})
    result = job.run_training_job(recipe, dataset, out_dir, plan_only=False, **extra)
    return {"mode": "recipe", "recipe_id": recipe.recipe_id,
            "folds": {o.fold_id: o.status for o in result.outcomes}}


def _check_board_analog_budget(job, trade_rows) -> None:
    estimate = job.board_analog_rss_estimate(trade_rows)["estimated_peak_gb"]
    limit_gib = profile_named(DEFAULT_POLICY, "experiment_heavy").memory_bytes / (1024 ** 3)
    if estimate > limit_gib:
        raise fail("RESOURCE_LIMIT_EXCEEDED",
                   "board analog build exceeds the training profile's memory",
                   details={"estimated_peak_gb": estimate, "limit_gib": limit_gib})


def _output_entries(out_dir: Path) -> list[dict]:
    entries = [{"name": "training_result", "path": "training_result.json",
                "schema": "training_job_result.v1.0"}]
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(out_dir).as_posix()
            entries.append({"name": "training/" + rel.replace("/", "__"),
                            "path": f"training/{rel}",
                            "schema": "training_job_output.v1.0"})
    return entries


def _tool_failure(error: SystemExit):
    """Map a ``tools.phase5_training_job`` ``SystemExit`` refusal to a typed
    ``OpsError``. Messages are written here, fixed, and never interpolate the
    tool's own text (it may name a path)."""
    text = str(error)
    if text.startswith("RESUME_MISMATCH"):
        return fail("CHECKPOINT_INCOMPATIBLE",
                    "training output already exists with different content")
    if "tier4_forecasts is missing" in text:
        return fail("FEATURES_MISSING",
                    "tier4 forecasts are missing; lineage cannot be attached")
    if "no dataset builder" in text:
        return fail("INVALID_REQUEST", "recipe has no dataset builder")
    return fail("VALIDATION_FAILED", "training tool refused the job")


def _dispatch_mode(job, mode, parameters, out_dir, root) -> dict:
    if mode == "recipe":
        return _run_recipe(parameters, out_dir, root)
    if mode == "state":
        return job.run_state_job(
            parameters["state"], out_dir, plan_only=False,
            cutoff=parameters["cutoffs"][0] if parameters["cutoffs"] else None,
            ticker_chunk=parameters["ticker_chunk"])
    if mode == "board_analog":
        _known, trade_rows = job._replay_strategies()
        _check_board_analog_budget(job, trade_rows)
        return job.run_board_analog_job(
            out_dir, alpha=parameters["alpha"], cutoffs=parameters["cutoffs"],
            strategies=parameters["strategies"] or None, plan_only=False)
    if mode == "trailing_cutoff":
        return job.run_trailing_cutoff_job(out_dir, as_of=parameters["cutoffs"],
                                           plan_only=False)
    raise fail("INVALID_REQUEST", "unknown training mode", details={"mode": mode})


def run_training_worker(parameters, root) -> dict:
    """Run one training job inside its staging root.

    Always ``plan_only=False``: an ops job never previews. ``--plan-only``
    remains the local CLI's escape hatch (it refuses for ``--state`` jobs).
    Every refusal the tool can raise is mapped to a typed ``OpsError`` here --
    a bare ``SystemExit`` would otherwise surface as an untyped
    ``WORKER_FAILED`` the operator cannot branch on.
    """
    from engine.v2.models import RuntimeFitForbidden
    from engine.v2.models.training import TrainingRefused
    from tools import phase5_training_job as job

    root = Path(root)
    out_dir = root / "training"
    try:
        summary = _dispatch_mode(job, parameters["mode"], parameters, out_dir, root)
    except SystemExit as exc:
        raise _tool_failure(exc) from exc
    except TrainingRefused as exc:
        raise fail("CHECKPOINT_INCOMPATIBLE", "training receipt refused",
                   details={"issues": sorted({issue.code for issue in exc.issues})}) from exc
    except RuntimeFitForbidden as exc:
        raise fail("VALIDATION_FAILED",
                   "runtime fitting is forbidden during a training job") from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    (root / "training_result.json").write_text(
        json.dumps(summary, allow_nan=False, sort_keys=True))
    return {"outputs": _output_entries(out_dir),
            "completed_ids": list(parameters["expected_ids"]), "no_work": False}


def run_promote_worker(parameters, root) -> dict:
    """Run one operator-submitted promote inside its staging root.

    Refusal is free: ``deployment.promote`` already raises
    ``ReleaseNotStaged`` for an unstaged ``release_id``, so this worker (and
    therefore the whole job) can only ever succeed against a release an
    operator staged first. There is no other caller of this worker at all --
    no nightly stage names ``"models_promote"``; the only path that creates
    one is a ``submit`` an operator ran by hand.
    """
    from engine.v2.foundation import to_document
    from engine.v2.models import deployment

    try:
        state = deployment.promote(
            Path(parameters["release_root"]), parameters["release_id"],
            expected_previous_release_id=parameters.get(
                "expected_previous_release_id"))
    except deployment.DeploymentError as exc:
        raise fail("VALIDATION_FAILED", "promote refused",
                   details={"exception_class": type(exc).__name__}) from exc
    (Path(root) / "pointer_state.json").write_text(json.dumps(to_document(state), sort_keys=True))
    return {"outputs": [{"name": "pointer_state", "path": "pointer_state.json",
                         "schema": "promote_pointer_state.v1.0"}],
            "completed_ids": list(parameters["expected_ids"]), "no_work": False}


def _recorded_rollback(deployment, store, incumbent_id, incumbent_sequence, target):
    """Return the pinned plan's own already-recorded rollback swap, if the
    store holds it, else ``None``.

    This is the recovery receipt: the original attempt swapped the pointer and
    crashed before recording it, so a resubmission under a new idempotency key
    must report that same swap rather than undo whatever the store has since
    done.  The state has to be exactly the pinned plan's swap -- a rollback
    from the pinned incumbent id to the pinned target at the pinned sequence
    plus one -- whether it lives in the append-only history or, for a crash
    between the pointer write and the history append, only in the live pointer.
    """
    if incumbent_id is None or incumbent_sequence is None or target is None:
        return None
    wanted = incumbent_sequence + 1
    try:
        history = deployment.pointer_history(store)
    except (OSError, ValueError):
        return None
    for entry in history:
        if entry.sequence == wanted:
            if (entry.action == "rollback" and entry.release_id == target
                    and entry.previous_release_id == incumbent_id):
                return entry
            return None
    pointer = deployment.current_pointer(store)
    if (pointer is not None and pointer.sequence == wanted
            and pointer.action == "rollback" and pointer.release_id == target
            and pointer.previous_release_id == incumbent_id):
        return pointer
    return None


def _rollback_for_pinned_plan(deployment, store, incumbent_id, incumbent_sequence, target):
    """Perform the plan's rollback, or recover its lost receipt, or refuse stale.

    The live pointer must still name the pinned incumbent at the pinned
    sequence, and the store's own target resolver must still name the pinned
    target, before ``deployment.rollback`` may swap.  Otherwise a newer
    promotion has advanced the store and the plan is stale: refuse typed
    ``VALIDATION_FAILED`` without touching the pointer or history.  A plan made
    against a completely empty store pins all three of incumbent id, sequence
    and target as ``None``: while the pointer is still absent it lets
    ``deployment.rollback`` raise its own typed ``NoPriorRelease``, but a
    pointer that has appeared since the plan makes the plan stale and is
    refused the same typed way, without mutation.  A store with exactly one
    incumbent pins no target and likewise reaches ``deployment.rollback``'s own
    ``NoPriorRelease``.
    """
    recorded = _recorded_rollback(deployment, store, incumbent_id, incumbent_sequence, target)
    if recorded is not None:
        return recorded
    pointer = deployment.current_pointer(store)
    if incumbent_id is None and incumbent_sequence is None and target is None:
        if pointer is None:
            return deployment.rollback(store)
        raise fail("VALIDATION_FAILED",
                   "rollback plan no longer matches the live pointer; refusing a stale swap")
    if (incumbent_id is None or incumbent_sequence is None or pointer is None
            or pointer.release_id != incumbent_id or pointer.sequence != incumbent_sequence):
        raise fail("VALIDATION_FAILED",
                   "rollback plan no longer matches the live pointer; refusing a stale swap")
    if target is not None and deployment.rollback_target(store) != target:
        raise fail("VALIDATION_FAILED",
                   "rollback target changed since the plan was made; refusing a stale swap")
    return deployment.rollback(store)


def run_rollback_worker(parameters, root) -> dict:
    """Run one operator-submitted rollback inside its staging root.

    ``deployment.rollback`` resolves its target from the release store's own
    append-only pointer history and raises ``NoPriorRelease`` (a
    ``DeploymentError``) when there is nothing to roll back to. That refusal
    maps to the same typed ``VALIDATION_FAILED`` every deployment refusal uses,
    never a bare worker failure, and -- because ``rollback`` refuses before any
    pointer/history write -- leaves no successful pointer state behind.  The
    swap runs only while the store still matches the incumbent sequence and
    target pinned in the plan; a plan the store has moved past is refused the
    same typed way and mutates nothing.  A plan whose swap is already recorded
    (the original attempt's receipt was lost) reports that exact recorded state
    without swapping again, so a later promotion is never undone.  There is no
    other caller of this worker: no nightly stage names ``"models_rollback"``,
    only an operator's own ``submit``.
    """
    from engine.v2.foundation import to_document
    from engine.v2.models import deployment

    store = Path(parameters["release_root"])
    try:
        state = _rollback_for_pinned_plan(
            deployment, store, parameters.get("incumbent_release_id"),
            parameters.get("incumbent_sequence"), parameters.get("target_release_id"))
    except deployment.DeploymentError as exc:
        raise fail("VALIDATION_FAILED", "rollback refused",
                   details={"exception_class": type(exc).__name__}) from exc
    (Path(root) / "pointer_state.json").write_text(json.dumps(to_document(state), sort_keys=True))
    return {"outputs": [{"name": "pointer_state", "path": "pointer_state.json",
                         "schema": "promote_pointer_state.v1.0"}],
            "completed_ids": list(parameters["expected_ids"]), "no_work": False}
