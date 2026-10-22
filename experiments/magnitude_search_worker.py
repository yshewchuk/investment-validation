"""Fixed fit-only child: admitted features/training targets, never validation targets."""
import json
import os
import resource
import sys


def emit(value):
    print(json.dumps(value, allow_nan=False, separators=(",", ":")), flush=True)


def main():
    limits = json.loads(sys.argv[2])
    resource.setrlimit(resource.RLIMIT_AS, (limits["memory_bytes"],) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"] + 1))
    os.sched_setaffinity(0, {limits["cpu_id"]})
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

        try:
            predictions, _ = fitted_forecasts(indexed, pd.DataFrame(inputs["training"]),
                request["inner"], request["plan"], request["execution"], progress=progress, strict_failures=True)
        except OpsError as error:
            if error.code != "EXPERIMENT_VARIANT_FAILED":
                raise
            emit({"event": "failed"})
            return
        emit({"event": "succeeded", "forecasts": [{"event_id": event, "horizon_sessions": horizon,
            "predicted_absolute_move_pct": value} for (event, horizon), value in sorted(predictions.items())]})
    except MemoryError:
        emit({"event": "resource_limit"})
    except Exception:
        emit({"event": "unavailable"})


if __name__ == "__main__":
    main()
