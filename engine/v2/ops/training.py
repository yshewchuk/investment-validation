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
from dataclasses import dataclass
from pathlib import Path

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
    #: Unused by the training worker; present for KindRegistry parity with the
    #: other operator-submitted kinds.
    input_bindings: dict[str, str] | None = None


@dataclass(frozen=True, kw_only=True)
class PromoteParameters:
    expected_ids: tuple[str, ...]        # always ("models_promote",)
    release_root: str
    release_id: str
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
    return problems


def _board_analog_problems(params) -> list[str]:
    problems = []
    if params.alpha is None:
        problems.append("mode=board_analog needs alpha")
    if not params.cutoffs:
        problems.append("mode=board_analog needs at least one cutoff")
    if params.recipe or params.state:
        problems.append("mode=board_analog must not name a recipe or state")
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
    return problems


def training_parameter_problems(job, params) -> list[str]:
    """Pure validator for the ``training`` kind (no data or artifact reads)."""
    problems = []
    if params.expected_ids != ("training",):
        problems.append("expected_ids must be exactly ('training',)")
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
    """The operator-submitted pointer swap. No ``store_domains``:
    ``deployment.promote`` only touches the release-store path named by the
    job's own ``release_root`` parameter, never the shared legacy tree."""
    return JobKind(name="models_promote", worker="models_promote", parameters=PromoteParameters,
                   resource_classes=frozenset({"delivery"}), effects=("staged",),
                   retry=RetryPolicy("bounded", 1, (30,)),
                   checkpoint_contract="promote_pointer_state.v1.0",
                   namespaces=frozenset({"shadow", "smoke"}))


def training_plan(*, mode, recipe="", state="", alpha=None, cutoffs=(), strategies=(),
                  pairs_path="", ticker_chunk=1000) -> dict:
    from engine.v2.foundation import content_hash
    from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest

    profile = profile_named(DEFAULT_POLICY, "experiment_heavy")
    params = TrainingParameters(expected_ids=("training",), mode=mode, recipe=recipe, state=state,
                               alpha=alpha, cutoffs=tuple(cutoffs), strategies=tuple(strategies),
                               pairs_path=pairs_path, ticker_chunk=ticker_chunk)
    problems = training_parameter_problems(None, params)
    if problems:
        raise fail("INVALID_REQUEST", "training plan is invalid",
                   details={"problems": list(problems)})
    root3 = Path(__file__).resolve().parents[3]
    return {"schema_version": "operations_plan.v1.0", "kind": "training", "mode": "shadow",
            "effects": ["staged"], "parameters": vars(params), "input_refs": [],
            "blocked_prerequisites": [], "spec_hash": content_hash(vars(params)),
            "implementation_ref": content_hash(worker_source_manifest(root3)),
            "environment_ref": content_hash(
                environment_identity(profile.thread_count or profile.cpu_count)),
            "resource_class": "experiment_heavy"}


def promote_plan(*, release_root, release_id) -> dict:
    from engine.v2.foundation import content_hash
    from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest

    if not release_root or not release_id:
        raise fail("INVALID_REQUEST", "promote plan needs a release root and a release id")
    profile = profile_named(DEFAULT_POLICY, "delivery")
    params = PromoteParameters(expected_ids=("models_promote",), release_root=release_root,
                               release_id=release_id)
    root3 = Path(__file__).resolve().parents[3]
    return {"schema_version": "operations_plan.v1.0", "kind": "promote", "mode": "shadow",
            "effects": ["staged"], "parameters": vars(params), "input_refs": [],
            "blocked_prerequisites": [], "spec_hash": content_hash(vars(params)),
            "implementation_ref": content_hash(worker_source_manifest(root3)),
            "environment_ref": content_hash(
                environment_identity(profile.thread_count or profile.cpu_count)),
            "resource_class": "delivery"}


def _run_recipe(parameters, out_dir) -> dict:
    from tools import phase5_training_job as job

    try:
        recipe = job.current_recipes()[job._key(parameters["recipe"])]
    except (KeyError, ValueError):
        raise fail("VALIDATION_FAILED", "recipe is not a current training recipe") from None
    dataset = job.build_dataset(recipe, pairs_path=parameters["pairs_path"] or None)
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


def run_training_worker(parameters, root) -> dict:
    """Run one training job inside its staging root.

    Always ``plan_only=False``: an ops job never previews. ``--plan-only``
    remains the local CLI's escape hatch (it refuses for ``--state`` jobs).
    """
    from tools import phase5_training_job as job

    root = Path(root)
    out_dir = root / "training"
    mode = parameters["mode"]
    if mode == "recipe":
        summary = _run_recipe(parameters, out_dir)
    elif mode == "state":
        summary = job.run_state_job(
            parameters["state"], out_dir, plan_only=False,
            cutoff=parameters["cutoffs"][0] if parameters["cutoffs"] else None,
            ticker_chunk=parameters["ticker_chunk"])
    elif mode == "board_analog":
        _known, trade_rows = job._replay_strategies()
        _check_board_analog_budget(job, trade_rows)
        summary = job.run_board_analog_job(
            out_dir, alpha=parameters["alpha"], cutoffs=parameters["cutoffs"],
            strategies=parameters["strategies"] or None, plan_only=False)
    elif mode == "trailing_cutoff":
        summary = job.run_trailing_cutoff_job(out_dir, as_of=parameters["cutoffs"],
                                              plan_only=False)
    else:
        raise fail("INVALID_REQUEST", "unknown training mode",
                   details={"mode": mode})
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
        state = deployment.promote(Path(parameters["release_root"]), parameters["release_id"])
    except deployment.DeploymentError as exc:
        raise fail("VALIDATION_FAILED", f"promote refused: {exc}") from exc
    (Path(root) / "pointer_state.json").write_text(json.dumps(to_document(state), sort_keys=True))
    return {"outputs": [{"name": "pointer_state", "path": "pointer_state.json",
                         "schema": "promote_pointer_state.v1.0"}],
            "completed_ids": list(parameters["expected_ids"]), "no_work": False}
