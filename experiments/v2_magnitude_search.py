"""Prepare or verify a bounded nested magnitude search plan; never execute trials."""
import argparse
import json
import sqlite3
from pathlib import Path

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import ArtifactError, ArtifactStore, from_document, to_document
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import experiment_spec_from_document
from experiments.magnitude_search_plan import read_search_plan, seal_search_plan
from experiments.native_registration import read_native_registration


def main(argv=None):
    """Local metadata-artifact lifecycle; no repository reads or experiment attempts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    for name in ("spec", "catalog", "store-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--plan-ref")
    args = parser.parse_args(argv)
    if (args.action == "verify") != (args.plan_ref is not None):
        parser.error("verify requires --plan-ref in addition to --request; prepare accepts no --plan-ref")
    conn = None
    try:
        if not Path(args.store_root).is_dir():
            raise fail("RESOURCE_UNAVAILABLE", "search requires an existing artifact store")
        spec = experiment_spec_from_document(json.loads(Path(args.spec).read_text()))
        document = json.loads(Path(args.request).read_text())
        reference = None if args.plan_ref is None else from_document(
            ArtifactRef, json.loads(Path(args.plan_ref).read_text()))
        conn = connect(args.catalog, must_exist=True)
        store = ArtifactStore(args.store_root)
        registration = read_native_registration(conn, store, spec)
        if args.action == "prepare":
            sealed = seal_search_plan(conn, store, registration, document)
            result = {"plan_ref": sealed["plan_ref"], "plan_id": sealed["plan"]["plan_id"],
                      "candidate_count": sealed["plan"]["candidate_count"],
                      "planned_fit_count": sealed["plan"]["planned_fit_count"]}
        else:
            plan = read_search_plan(conn, store, registration, document, reference)
            result = {"verified": plan["plan_id"]}
    except OpsError as error:
        print(json.dumps({"refused": to_document(error.problem)}, sort_keys=True))
        return 2
    except (ArtifactError, OSError, sqlite3.Error, ValueError, TypeError, KeyError) as error:
        problem = fail("RESOURCE_UNAVAILABLE" if isinstance(error, (ArtifactError, OSError, sqlite3.Error))
                       else "INVALID_EXPERIMENT_SPEC", "search planning request could not be completed").problem
        print(json.dumps({"refused": to_document(problem)}, sort_keys=True))
        return 2
    finally:
        if conn is not None:
            conn.close()
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
