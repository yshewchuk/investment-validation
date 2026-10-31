"""Fixed CAS-only numeric child; no catalog, outcome reader or completion writer."""
import json
import os
import resource
import sys

from experiments.memory_guard import start_rss_watchdog


def main():
    try:
        limits = json.loads(sys.argv[2])
        # Match native scalar/affinity bounds without numerical imports before containment.
        if (type(limits) is not dict or any(type(limits.get(key)) is not int
                or not 1 <= limits[key] < sys.maxsize for key in ("memory_bytes", "cpu_seconds"))
                or type(limits.get("cpu_id")) is not int
                or not 0 <= limits["cpu_id"] < 2 ** 31 - 1):
            print('{"error":"INVALID_EXPERIMENT_SPEC"}', flush=True)
            return
        resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"] + 1))
        os.sched_setaffinity(0, {limits["cpu_id"]})
        start_rss_watchdog(limits["memory_bytes"])
        from engine.v2.foundation import ArtifactStore, canonical_json, content_hash, to_document
        from engine.v2.ops.errors import fail
        from experiments.distribution_candidate_fit import fit_candidate
        from experiments.distribution_candidate_recipe import validate_candidate_recipe
        from experiments.distribution_fitted import publish_fit_output, validate_fit_output
        from experiments.magnitude_search_evidence import fit_evidence
        from experiments.magnitude_search_inputs import load_fit_inputs
        from experiments.magnitude_search_records import MAX_BYTES

        with open(sys.argv[1], "rb") as stream:
            payload = stream.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise MemoryError
        request = json.loads(payload)
        if type(request) is not dict:
            raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution worker header")
        recipe = validate_candidate_recipe(request.get("recipe"))
        events, horizon, root = (request.get(key) for key in ("events", "horizon", "store_root"))
        if (type(horizon) is not int or horizon not in recipe["horizons"]
                or type(events) is not list or not events
                or any(type(event) is not str or not event for event in events)
                or len(set(events)) != len(events) or type(root) is not str or not root):
            raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution worker axes")
        store, columns = ArtifactStore(root), recipe["point_recipe"]["features"]
        inputs = load_fit_inputs(store, request["manifest"], request["inputs_id"], events, [horizon], columns)
        indexed = {row["event_id"]: row for row in inputs["features"]}
        if any(event not in indexed for event in events):
            raise fail("INVALID_EXPERIMENT_SPEC", "distribution worker event has no projected features")
        evidence = fit_evidence(inputs, {"evaluation_event_ids": events}, {"horizons": [horizon]}, {"features": columns})[0]
        matrices = [[[indexed[event][name] for name in columns] for event in ids]
                    for ids in (evidence["training_event_ids"], events)]
        labels = [row["absolute_move_pct"] for row in inputs["training"]]
        result = fit_candidate(recipe, matrices[0], labels, matrices[1], columns=columns)
        rows = validate_fit_output(recipe, matrices, labels, events, result)
        output = {**publish_fit_output(store, rows, result["tail"], result["diagnostics"]),
            "recipe_id": result["recipe_id"], "evidence": {key: evidence[key] for key in
                ("horizon_sessions", "training_matrix_id", "prediction_matrix_id")},
            "ordered_events_id": content_hash([evidence["training_event_ids"], events])}
        payload = canonical_json(output).encode()
        if len(payload) > MAX_BYTES:
            raise MemoryError
        response = to_document(store.publish_bytes(payload, schema_ref="distribution_fit_output.v1"))
    except MemoryError:
        response = {"error": "RESOURCE_LIMIT_EXCEEDED"}
    except Exception as error:
        code = getattr(error, "code", None)
        response = {"error": code if type(code) is str and code in {"EXPERIMENT_VARIANT_FAILED", "INVALID_EXPERIMENT_SPEC",
            "EXPERIMENT_IDENTITY_CONFLICT", "RESOURCE_LIMIT_EXCEEDED"} else "RESOURCE_UNAVAILABLE"}
    print(json.dumps(response, allow_nan=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
