"""Fixed fit-only child: admitted features/training targets, never validation targets."""
import json
import os
import resource
import sys
import time

from experiments.memory_guard import rss_bytes, start_rss_watchdog


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

        started = time.monotonic()
        rss_before_kib = rss_bytes() // 1024
        rss_before_maxrss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        def rss_delta_kib():
            current_kib = rss_bytes() // 1024
            current_ru_maxrss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return max(0, current_kib - rss_before_kib, current_ru_maxrss_kib - rss_before_maxrss_kib)

        # The header holds a manifest, which the journal caps at MAX_BYTES, so a larger file is not a header.
        with open(sys.argv[1], "rb") as stream:
            payload = stream.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise MemoryError
        request = json.loads(payload)
        regression = request["execution"].get("schema_version") == "regression_candidate_execution.v1"
        target = request["execution"]["target"] if regression else None
        inputs = load_fit_inputs(ArtifactStore(request["store_root"]), request["manifest"], request["inputs_id"],
                                 request["inner"]["evaluation_event_ids"], request["plan"]["horizons"],
                                 request["execution"]["features"],
                                 **({"target_column": target["column"]} if regression else {}))
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
            if regression:
                predictions = regression_forecasts(indexed, inputs["training"], request, progress, diagnostics)
            else:
                predictions, _ = fitted_forecasts(indexed, pd.DataFrame(inputs["training"]),
                    request["inner"], request["plan"], request["execution"], progress=progress, strict_failures=True, diagnostics=diagnostics)
        except OpsError as error:
            if error.code != "EXPERIMENT_VARIANT_FAILED":
                raise
            timing = ({"score_seconds": max(0.0, time.monotonic() - started),
                       "rss_delta_kib": rss_delta_kib()}
                      if not regression else {})
            emit({"event": "failed", **timing})
            return
        column = target["prediction_column"] if regression else "predicted_absolute_move_pct"
        schema_version = request["execution"].get("schema_version")
        timing = ({"score_seconds": max(0.0, time.monotonic() - started),
                  "rss_delta_kib": rss_delta_kib()}
                 if not regression else {})
        emit({"event": "succeeded", "forecasts": [{"event_id": event, "horizon_sessions": horizon,
            column: value} for (event, horizon), value in sorted(predictions.items())],
            **timing,
            **({"diagnostics": diagnostics} if regression or schema_version == "size_candidate_execution.v1" else {})})
    except MemoryError:
        emit({"event": "resource_limit"})
    except Exception:
        emit({"event": "unavailable"})


def regression_forecasts(indexed, training, request, progress, diagnostics):
    """Fit the closed explicit-target route; infrastructure failures are never candidate failures."""
    from engine.v2.ops.errors import fail
    from engine.v2.models.training.fit_routes import check_fitting_allowed
    from engine.v2.research.horizon_fitted import HorizonFitFailure, fitted_forecasts
    check_fitting_allowed("regression_forecasts")
    execution = request["execution"]
    target = execution["target"]
    try:
        predictions, _ = fitted_forecasts(indexed, [row["event_id"] for row in training],
            [row[target["column"]] for row in training], request["inner"]["evaluation_event_ids"],
            request["plan"]["horizons"], execution["features"], target_column=target["column"],
            target_domain=target["domain"], route="regression_candidate", recipe=execution["fit_recipe"],
            progress=progress, diagnostics=diagnostics)
    except HorizonFitFailure as failure:
        error = failure.__cause__
        if isinstance(error, (ValueError, ArithmeticError)):
            raise fail("EXPERIMENT_VARIANT_FAILED", "fitted regression estimator failed") from None
        raise error
    return predictions


if __name__ == "__main__":
    main()
