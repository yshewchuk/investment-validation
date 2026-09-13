"""Fixed subprocess entrypoint. The launch gate opens only after PID persistence.

No arbitrary modules or executables are accepted. Output is a small protocol
message on an inherited pipe; payloads stay in the assigned staging directory.
"""
from __future__ import annotations

import json
import os
import resource
import sys
from pathlib import Path


def main():
    envelope = json.loads(sys.stdin.buffer.readline())
    os.sched_setaffinity(0, envelope["cpu_ids"])
    root = Path(envelope["staging"])
    os.environ["INVESTING_PLAN_ROOT"] = str(root / "legacy")
    fd = int(envelope["result_fd"])
    try:
        result = dispatch(envelope["worker"], envelope["parameters"], root)
        result.update(schema_version="worker_result.v1.0", job_id=envelope["job_id"],
                      attempt_id=envelope["attempt_id"], fence=envelope["fence"])
        result["self_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except BaseException:
        result = {"schema_version": "worker_result.v1.0", "failure": "WORKER_FAILED"}
    data = json.dumps(result, allow_nan=False).encode()
    os.write(fd, data + b"\n")
    os.close(fd)
    return int("failure" in result)


def dispatch(worker, parameters, root):
    if worker.startswith("legacy_"):
        from engine.v2.ops.legacy_actions import run_action
        values = parameters if isinstance(parameters, dict) else vars(parameters)
        output = run_action(worker, values, root)
        return {"outputs": [{"name": worker, "path": output["path"],
                              "schema": "legacy_action.v1.0"}],
                "completed_ids": [worker], "action": worker,
                "coverage": output}
    if worker == "artifact_check":
        output = root / "receipt.json"
        output.write_text(json.dumps({"checked": parameters["expected_ids"],
                                     "affinity": sorted(os.sched_getaffinity(0)),
                                     "threads": os.environ["OMP_NUM_THREADS"]}))
        return {"outputs": [{"name": "receipt", "path": "receipt.json", "schema": "receipt.v1.0"}],
                "completed_ids": parameters["expected_ids"],
                "no_work": not parameters["expected_ids"],
                "observed": {"affinity": sorted(os.sched_getaffinity(0)),
                             "threads": os.environ["OMP_NUM_THREADS"]}}
    raise ValueError("unsupported worker")


if __name__ == "__main__":
    raise SystemExit(main())
