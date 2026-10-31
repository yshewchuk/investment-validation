"""Fixed fit-only child: admitted features/training targets, never validation targets."""
import json
import os
import resource
import sys

from experiments.memory_guard import start_rss_watchdog


def emit(value):
    print(json.dumps(value, allow_nan=False, separators=(",", ":")), flush=True)


def main():
    limits = json.loads(sys.argv[2])
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"] + 1))
    os.sched_setaffinity(0, {limits["cpu_id"]})
    start_rss_watchdog(limits["memory_bytes"])
    try:
        import pandas as pd

        from engine.v2.foundation import ArtifactStore
        from engine.v2.ops.errors import OpsError
        from experiments.magnitude_fitted import fitted_forecasts
        from experiments.magnitude_search_inputs import load_fit_inputs

        # The header is small: limits-free request fields, the manifest, its expected id and the store root.
        # The child streams only manifest-named shards; it opens no repository, catalog or validation target.
        from experiments.magnitude_search_records import MAX_BYTES

        # The header holds a manifest, which the journal caps at MAX_BYTES, so a larger file is not a header.
        with open(sys.argv[1], "rb") as stream:
            payload = stream.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise MemoryError
        request = json.loads(payload)
        inputs = load_fit_inputs(ArtifactStore(request["store_root"]), request["manifest"], request["inputs_id"],
                                 request["inner"]["evaluation_event_ids"], request["plan"]["horizons"],
                                 request["execution"]["features"])
        indexed = {(row["event_id"], row["horizon_sessions"]): row for row in inputs["features"]}

        def progress(state, fit):
            emit({"event": state, "fit": fit})
            if sys.stdin.readline() != "continue\n":
                raise OSError

        if request["execution"].get("schema_version") == "size_comparison_execution.v1":
            from experiments.size_comparator import forecasts
            emit(forecasts(indexed, pd.DataFrame(inputs["training"]), request["inner"], request["plan"],
                           request["execution"], progress=progress))
            return
        if (request["execution"].get("schema_version") == "size_candidate_execution.v1"
                and request["execution"]["fit_recipe"].get("schema_version") == "size_causal_comparator.v1"):
            from engine.v2.foundation import content_hash
            from experiments.size_comparator import forecasts
            result = forecasts(indexed, pd.DataFrame(inputs["training"]), request["inner"], request["plan"],
                               request["execution"], progress=progress)
            recipe_id = content_hash(request["execution"]["fit_recipe"])
            emit({"event": result["event"],
                "forecasts": [{"event_id": row["event_id"], "horizon_sessions": row["horizon_sessions"],
                    "predicted_absolute_move_pct": row["predicted_absolute_move_pct"]}
                    for row in result["forecasts"]],
                "diagnostics": [{"schema_version": "size_v1_4_fit.v1", "recipe_id": recipe_id,
                    "horizon_sessions": detail["horizon_sessions"],
                    "training_event_ids": detail["training_event_ids"],
                    "logical_fits": detail["logical_fits"],
                    "component_fits": detail["component_fits"],
                    "abstention_counts": detail["abstention_counts"],
                    "training_abstention_counts": detail["training_abstention_counts"]}
                    for detail in result["fit_details"]]})
            return
        diagnostics = []
        try:
            predictions, _ = fitted_forecasts(indexed, pd.DataFrame(inputs["training"]),
                request["inner"], request["plan"], request["execution"], progress=progress, strict_failures=True, diagnostics=diagnostics)
        except OpsError as error:
            if error.code != "EXPERIMENT_VARIANT_FAILED":
                raise
            emit({"event": "failed"})
            return
        emit({"event": "succeeded", "forecasts": [{"event_id": event, "horizon_sessions": horizon,
            "predicted_absolute_move_pct": value} for (event, horizon), value in sorted(predictions.items())],
            **({"diagnostics": diagnostics} if request["execution"].get("schema_version") == "size_candidate_execution.v1" else {})})
    except MemoryError:
        emit({"event": "resource_limit"})
    except Exception:
        emit({"event": "unavailable"})


if __name__ == "__main__":
    main()
