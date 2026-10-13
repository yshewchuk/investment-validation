"""Recheck a completed acquisition's saved mosaic, never today's coverage plan."""
import json
from datetime import date, timedelta

from engine.v2.data import massive_daily
from engine.v2.data.incremental import load_raw_receipt
from engine.v2.foundation import content_hash
from engine.v2.ops import checkpoints, input_bindings, massive_plan
from engine.v2.ops.errors import fail
from engine.v2.ops.stores.massive import acquisition, coverage, pages


def acquisition_input(conn, store, attempt_id, source_job):
    binding = input_bindings.recorded_bindings(conn, attempt_id).get("acquisition.json")
    if binding is None or binding.binding != source_job + "#massive_acquisition":
        raise fail("INTEGRITY_FAILED", "Massive snapshot input binding differs")
    ref = checkpoints.registered_artifact(conn, binding.artifact_id)
    if ref.schema_ref != acquisition.HISTORY_SCHEMA or ref.content_hash != binding.content_hash:
        raise fail("INTEGRITY_FAILED", "Massive snapshot input identity differs")
    if ref.byte_size > 4 << 20:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "Massive acquisition artifact exceeds byte bound")
    document = json.loads(store.read_verified(ref), object_pairs_hook=pages._unique_object)
    producer = conn.execute("SELECT a.attempt_id,a.fence,j.spec_json FROM attempts a JOIN jobs j ON j.job_id=a.job_id "
        "JOIN attempt_outputs o ON o.attempt_id=a.attempt_id WHERE j.job_id=? AND j.kind=? "
        "AND j.state='succeeded' AND a.state='succeeded' AND o.name=? AND o.artifact_id=?",
        (source_job, acquisition.HISTORY_KIND, acquisition.KIND, ref.artifact_id)).fetchone()
    if (producer is None or document["schema_version"] != acquisition.HISTORY_SCHEMA
            or (document["attempt_id"], document["fence"]) != tuple(producer)[:2]):
        raise fail("INTEGRITY_FAILED", "Massive acquisition producer differs")
    params = json.loads(producer["spec_json"])["parameters"]
    if (params["expected_ids"] != [content_hash(document["coverage"]["request"])]
            or params["generation"] != document["coverage"]["generation"]):
        raise fail("INTEGRITY_FAILED", "Massive saved scope differs from its producer")
    return ref.artifact_id, document["coverage"]


def _plan_identity(plan):
    request = plan["request"]
    canonical = massive_plan._request_descriptor("daily", request["ticker"], request["start"],
                                                  request["end"], request["adjustment_basis"])
    coverage.validate_generation(plan["generation"])
    if (request != canonical or plan["schema_version"] != coverage.VERSION or plan["gaps"]
            or plan["plan_hash"] != content_hash({k: v for k, v in plan.items() if k != "plan_hash"})
            or plan["candidate_limit"] != coverage.MAX_OCCURRENCES
            or type(plan["candidate_count"]) is not int or not 1 <= plan["candidate_count"] <= 100
            or not 1 <= len(plan["bindings"]) <= 2 * coverage.MAX_OCCURRENCES + 1):
        raise ValueError
    return request


def _bindings(plan):
    request = _plan_identity(plan)
    next_day = date.fromisoformat(request["start"])
    for binding in plan["bindings"]:
        original = binding["evidence"]["request"]
        if (binding["start"] != next_day.isoformat() or not binding["start"] <= binding["end"] <= request["end"]
                or not original["start"] <= binding["start"] <= binding["end"] <= original["end"]
                or coverage.family(original) != coverage.family(request)
                or binding["evidence"]["generation"] != plan["generation"]):
            raise ValueError
        next_day = date.fromisoformat(binding["end"]) + timedelta(days=1)
    if next_day != date.fromisoformat(request["end"]) + timedelta(days=1):
        raise ValueError
    return request


def normalized_rows(conn, store, plan):
    request = _bindings(plan)
    verified, remaining, decoded, result = {}, [coverage.MAX_VERIFIED_BYTES], 0, []
    for binding in plan["bindings"]:
        identity = binding["occurrence_id"]
        if identity not in verified:
            row = conn.execute("SELECT * FROM ops_massive_occurrences WHERE occurrence_id=?", (identity,)).fetchone()
            if row is None:
                raise ValueError
            verified[identity] = coverage._verified(conn, store, row, remaining)
        evidence = verified[identity]
        if evidence != binding["evidence"]:
            raise ValueError
        pages = [evidence] if "pages" not in evidence else [json.loads(conn.execute(
            "SELECT document_json FROM ops_massive_pages WHERE page_id=?", (page,)).fetchone()[0])
            for page in evidence["pages"]]
        for page in pages:
            body = load_raw_receipt(conn, store, page["raw_receipt_id"])
            decoded += len(body)
            if decoded > coverage.MAX_VERIFIED_BYTES:
                raise fail("RESOURCE_LIMIT_EXCEEDED", "Massive normalization exceeds raw byte bound")
            result.extend(massive_daily.normalize_rows(json.loads(body)["results"], request, binding, page))
            if len(result) > 1000:
                raise fail("RESOURCE_LIMIT_EXCEEDED", "Massive normalized row bound exceeded")
    if not result:
        raise fail("SOURCE_EMPTY", "Massive saved interval contains no selected daily rows")
    return result
