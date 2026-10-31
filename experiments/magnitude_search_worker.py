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
        diagnostics = []
        try:
            predictions, _ = fitted_forecasts(indexed, pd.DataFrame(inputs["training"]),
                request["inner"], request["plan"], request["execution"], progress=progress, strict_failures=True, diagnostics=diagnostics)
        except OpsError as error:
            if error.code != "EXPERIMENT_VARIANT_FAILED":
                raise
            emit({"event": "failed", "score_seconds": max(0.0, time.monotonic() - started),
                  "rss_delta_kib": rss_delta_kib()})
            return
        emit({"event": "succeeded", "forecasts": [{"event_id": event, "horizon_sessions": horizon,
            "predicted_absolute_move_pct": value} for (event, horizon), value in sorted(predictions.items())],
            "score_seconds": max(0.0, time.monotonic() - started), "rss_delta_kib": rss_delta_kib(),
            **({"diagnostics": diagnostics} if request["execution"].get("schema_version") == "size_candidate_execution.v1" else {})})
    except MemoryError:
        emit({"event": "resource_limit"})
    except Exception:
        emit({"event": "unavailable"})


if __name__ == "__main__":
    main()
