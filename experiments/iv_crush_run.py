"""Contained local IV lifecycle over existing native snapshots; no provider or build."""
import argparse
import resource
import shutil
import signal
import sqlite3
import sys
import time
from contextlib import ExitStack, redirect_stdout
from pathlib import Path

from engine.v2.contracts import ArtifactRef
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    DocumentError,
    SystemClock,
    from_document,
)
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import experiment_spec_from_document
from experiments.iv_crush_search import prepare
from experiments.iv_crush_search_report import collect, resource_report
from experiments.magnitude_search_plan import _nested, seal_search_plan
from experiments.magnitude_search_request_versions import compile_request
from experiments.memory_guard import rss_watchdog
from experiments.native_registration import read_native_registration, reserve_native_registration
from experiments.size_search import _contain, _run
from experiments.v2_magnitude_search_run import _read_document, _render, seal_search_run


def _prepare(repository, spec, request, args):
    """Cheap metadata preparation is repeated by register; no second receipt format."""
    panel = request["panel_request"]
    registration, spec = prepare(repository, spec, source=repository.resolve_pinned(args.scope),
        feature_snapshot=repository.resolve(request["feature_snapshot_id"]),
        target_snapshot=repository.resolve(args.target_snapshot_id), scope=args.scope,
        as_of_month=panel["as_of_month"], training_start=args.training_start,
        labels_before=args.labels_before, event_ids=panel["event_ids"])
    temporal = registration.document["regression_plan"]
    if temporal["evaluation_years"] != request["outer_years"]:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "IV request outer years differ from registration")
    _nested(registration.document, request, temporal=temporal)
    return registration, spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "register", "seal", "run", "report"))
    for name in ("spec", "request", "catalog", "store-root"):
        parser.add_argument("--" + name, required=True)
    for name in ("scope", "target-snapshot-id", "training-start", "labels-before", "plan-ref"):
        parser.add_argument("--" + name)
    parser.add_argument("--no-ledger", action="store_true", required=True)
    args = parser.parse_args(argv)
    fresh = args.action in ("preflight", "register")
    if fresh and not all((args.scope, args.target_snapshot_id, args.training_start, args.labels_before)):
        parser.error("preflight/register require scope, target-snapshot-id, training-start and labels-before")
    if not fresh and not args.plan_ref:
        parser.error("seal/run/report require plan-ref")
    if any("\0" in (getattr(args, name) or "") for name in ("spec", "request", "catalog", "store_root", "plan_ref")):
        parser.error("IV paths must not contain NUL characters")
    conn, started, stack = None, time.monotonic(), ExitStack()
    try:
        request = _read_document(args.request)
        if request.get("schema_version") != "regression_search_request.v1":
            raise fail("INVALID_EXPERIMENT_SPEC", "IV requires its closed regression request")
        schedule = compile_request(request)
        try:
            memory = _contain(request)
        except OverflowError:
            raise fail("INVALID_EXPERIMENT_SPEC", "IV wall budget exceeds platform alarm range") from None
        stack.enter_context(rss_watchdog(memory))
        print(f"[iv] stage={args.action} start rss_kib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}", file=sys.stderr)
        with redirect_stdout(sys.stderr):
            try:
                spec = experiment_spec_from_document(_read_document(args.spec))
            except TypeError:
                raise fail("INVALID_EXPERIMENT_SPEC", "invalid IV specification metadata") from None
            if spec.runner != "experiments/iv_crush_search.py" or spec.price_source != "iv_crush_targets":
                raise fail("INVALID_EXPERIMENT_SPEC", "IV requires its dedicated runner and target")
            if not Path(args.store_root).is_dir():
                raise fail("RESOURCE_UNAVAILABLE", "existing IV artifact store required")
            if args.action == "report":
                conn = sqlite3.connect(Path(args.catalog).resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None)
                conn.row_factory = sqlite3.Row
            else:
                conn = connect(args.catalog, must_exist=True)
            store = ArtifactStore(args.store_root)
            repository = Repository(conn, store, pin_snapshots=True)
            if fresh:
                registration, spec = _prepare(repository, spec, request, args)
                if args.action == "preflight":
                    result = {"preflight": resource_report({"counts": schedule["counts"], "request": request}),
                        "run_id": registration.run_id, "variant_id": registration.variant_id,
                        "qualification": "metadata_only; panel and causal labels admitted during execution"}
                else:
                    reserve_native_registration(conn, store, registration, spec)
                    result = seal_search_plan(conn, store, registration, request)
            else:
                registration = read_native_registration(conn, store, spec)
                ref = from_document(ArtifactRef, _read_document(args.plan_ref))
                context = (conn, store, repository, registration, request, ref)
                if args.action == "seal":
                    result = seal_search_run(*context, clock=SystemClock())
                else:
                    result = _run(context, collector=collect) if args.action == "run" else collect(*context)
        with _render(result) as rendered:
            shutil.copyfileobj(rendered, sys.stdout)
    except (OpsError, DataError) as error:
        with _render({"refused": error.code, "message": str(error)}, scratch=False) as rendered:
            shutil.copyfileobj(rendered, sys.stdout)
        return 2
    except MemoryError:
        print('{"refused":"RESOURCE_LIMIT_EXCEEDED","message":"IV parent memory exhausted"}')
        return 2
    except DocumentError:
        print('{"refused":"INVALID_EXPERIMENT_SPEC","message":"invalid IV metadata"}')
        return 2
    except (ArtifactError, OSError, sqlite3.Error):
        print('{"refused":"RESOURCE_UNAVAILABLE","message":"IV storage unavailable"}')
        return 2
    finally:
        stack.close()
        signal.alarm(0)
        if conn is not None:
            conn.close()
        print(f"[iv] stage={args.action} end elapsed={time.monotonic() - started:.3f}s rss_kib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
