"""Explicit contained distribution lifecycle over an existing qualified SIZE anchor."""
import argparse
import shutil
import sqlite3
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

from engine.v2.contracts import ArtifactRef, SnapshotRef
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    DocumentError,
    SystemClock,
    from_document,
    to_document,
)
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import experiment_spec_from_document
from engine.v2.research.progress import stage
from experiments.distribution_candidate_recipe import fit_budget, validate_candidate_recipe
from experiments.distribution_run_plan import contained
from experiments.distribution_run_request import require, validate_request
from experiments.distribution_trials import DistributionTrials
from experiments.magnitude_search_plan import read_search_plan
from experiments.native_registration import (
    MAX_NATIVE_EVIDENCE_BYTES,
    prepare_native_registration,
    read_native_registration,
    reserve_native_registration,
)
from experiments.v2_magnitude_search_run import _read_document, _render


def _input_document(path):
    try:
        return _read_document(path)
    except OSError:
        raise fail("INVALID_EXPERIMENT_SPEC", "distribution input document is unavailable") from None


def _prepare(conn, store, repository, spec, source, source_request, ref, recipe, caps):
    """Prepare only the distinct distribution registration, never rebuild or run SIZE."""
    require(spec.runner == "experiments/distribution_run.py" and spec.seed == 0
            and type(spec.seed) is int and not spec.economic_params and spec.price_source == "computed_moves")
    native = read_search_plan(conn, store, source, source_request, ref)
    binding = source.document
    require(binding["schema_version"] == "native_experiment_registration.v1.7"
            and set(native["horizons"]) >= {5, 3, 1}
            and list(spec.folds) == [group["outer"]["year"] for group in native["folds"]])
    require(set(caps) == {"max_fits", "max_trees", "child_limits"})
    budget = fit_budget(recipe, inner_folds=sum(len(group["inner"]) for group in native["folds"]),
        outer_folds=len(native["folds"]), **caps)
    registration, spec = prepare_native_registration(repository, spec,
        code_root=Path(__file__).resolve().parents[1],
        as_of_month=binding["holdouts"]["holdout_as_of_month"], scope=binding["scope"],
        event_ids=binding["event_ids"], snapshot=from_document(SnapshotRef, binding["snapshot"]))
    if len(registration.binding_json) > MAX_NATIVE_EVIDENCE_BYTES:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "distribution registration exceeds native byte budget")
    own = registration.document
    require(registration.run_id != source.run_id and own["schema_version"] == "native_experiment_registration.v1.0"
            and "final_read" not in own and all(own[key] == binding[key] for key in ("snapshot", "scope", "event_ids", "holdouts")))
    result = {"schema_version": "distribution_preparation.v1", "run_id": registration.run_id,
        "variant_id": registration.variant_id, "source_plan_id": native["plan_id"],
        "fit_budget": budget, "final_read_authority": "none"}
    return registration, spec, result


def _dispatch(args, conn, store, repository, request, began, *, probe=False):
    from experiments.distribution_run import DistributionRun

    try:
        spec = experiment_spec_from_document(_input_document(args.spec))
        source_spec = experiment_spec_from_document(_input_document(args.source_spec))
        ref = from_document(ArtifactRef, _input_document(args.source_plan_ref))
    except (DocumentError, TypeError):
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution lifecycle metadata") from None
    except OpsError as error:
        if error.code != "INVALID_REQUEST":
            raise
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution lifecycle metadata") from None
    source_request = _input_document(args.source_request)
    recipe = validate_candidate_recipe(_input_document(args.recipe))
    caps = _input_document(args.budget_caps)
    source = read_native_registration(conn, store, source_spec, max_evidence_bytes=MAX_NATIVE_EVIDENCE_BYTES)
    if args.action in ("prepare", "register"):
        registration, spec, result = _prepare(conn, store, repository, spec, source, source_request, ref, recipe, caps)
        if args.action == "register":
            reserve_native_registration(conn, store, registration, spec, max_evidence_bytes=MAX_NATIVE_EVIDENCE_BYTES)
        return {**result, "registered": args.action == "register"}
    own = read_native_registration(conn, store, spec, max_evidence_bytes=MAX_NATIVE_EVIDENCE_BYTES)
    trials = DistributionTrials(conn, store, repository, own, source, source_request, ref, recipe, caps)
    run = DistributionRun(trials, request)
    if args.action == "preflight":
        return run.preflight()
    if args.action == "seal":
        return run.seal(clock=SystemClock())
    if args.action == "run":
        if probe:
            return None if run.record("completed") is None else run.collect()
        return run._run_contained(SystemClock(), began)
    return run.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "register", "preflight", "seal", "run", "collect", "report"))
    paths = ("spec", "source-spec", "source-request", "source-plan-ref", "recipe", "budget-caps", "request", "catalog", "store-root")
    for name in paths:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--no-ledger", action="store_true", required=True)
    args = parser.parse_args(argv)
    if any("\0" in getattr(args, name.replace("-", "_")) for name in paths):
        parser.error("distribution paths must not contain NUL characters")
    conn, status, began = None, 0, time.monotonic()
    try:
        request = validate_request(_input_document(args.request))
        seconds = request["limits"]["job_wall_seconds"] - (time.monotonic() - began)
        if seconds <= 0:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "distribution request exhausted whole-job deadline")
        with contained(request["limits"]["parent"], seconds), stage("distribution " + args.action), redirect_stdout(sys.stderr):
            if not Path(args.store_root).is_dir():
                raise fail("RESOURCE_UNAVAILABLE", "distribution requires an existing artifact store")
            if args.action not in ("register", "seal"):
                conn = sqlite3.connect(Path(args.catalog).resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None)
                conn.row_factory = sqlite3.Row
            else:
                conn = connect(args.catalog, must_exist=True)
            store = ArtifactStore(args.store_root)
            result = _dispatch(args, conn, store, Repository(conn, store, pin_snapshots=True), request, began,
                probe=args.action == "run")
            if args.action == "run" and result is None:
                conn.close()
                conn = None
                conn = connect(args.catalog, must_exist=True)
                result = _dispatch(args, conn, store, Repository(conn, store, pin_snapshots=True), request, began)
            output = _render(result)
    except (OpsError, DataError) as error:
        output, status = _render({"refused": to_document(error.problem)}, scratch=False), 2
    except MemoryError:
        error = fail("RESOURCE_LIMIT_EXCEEDED", "distribution parent memory exhausted")
        output, status = _render({"refused": to_document(error.problem)}, scratch=False), 2
    except DocumentError:
        error = fail("INVALID_EXPERIMENT_SPEC", "invalid distribution metadata")
        output, status = _render({"refused": to_document(error.problem)}, scratch=False), 2
    except ArtifactError as error:
        code = "INTEGRITY_FAILED" if error.code == "INTEGRITY_FAILED" else "RESOURCE_UNAVAILABLE"
        problem = fail(code, "distribution artifact integrity failed" if code == "INTEGRITY_FAILED" else "distribution storage unavailable")
        output, status = _render({"refused": to_document(problem.problem)}, scratch=False), 2
    except (OSError, sqlite3.Error):
        error = fail("RESOURCE_UNAVAILABLE", "distribution storage unavailable")
        output, status = _render({"refused": to_document(error.problem)}, scratch=False), 2
    finally:
        if conn is not None:
            conn.close()
    with output:
        shutil.copyfileobj(output, sys.stdout, length=65536)
    return status
