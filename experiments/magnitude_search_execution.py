"""Synchronous, bounded fixed-child execution; no scheduler or retry authority."""
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from engine.models.no_fit import forbid_fitting as forbid_legacy_fitting
from engine.v2.foundation import canonical_json
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.ops.errors import fail
from engine.v2.ops.executor import THREAD_VARIABLES
from experiments.magnitude_search_records import MAX_BYTES


def validate_execution_limits(limits):
    """Admit an explicit positive, platform-representable resource contract."""
    if (not isinstance(limits, dict) or set(limits) != {"memory_bytes", "cpu_seconds", "wall_seconds"}
            or any(type(value) is not int or not 1 <= value < sys.maxsize for value in limits.values())):
        raise fail("INVALID_EXPERIMENT_SPEC", "search execution requires explicit positive integer resource limits")


def execution_limits(limits):
    """Resolve one allowed Linux CPU after pure resource-contract admission."""
    validate_execution_limits(limits)
    try:
        cpu = min(os.sched_getaffinity(0))
    except (AttributeError, OSError, ValueError):
        raise fail("RESOURCE_UNAVAILABLE", "search execution requires Linux resource containment") from None
    return {**limits, "cpu_id": cpu}


def fit_payload(inputs, inner, plan, execution):
    """Canonical child payload: only the row values the fit and its matrix digests read."""
    keep = ("event_id", "context_id", "horizon_sessions", *execution.get("features", ()))
    features = [{name: row[name] for name in keep} for row in inputs["features"]]
    return canonical_json({"inputs": {"features": features, "training": inputs["training"]},
                           "inner": inner, "plan": {"horizons": plan["horizons"]},
                           "execution": execution}).encode()


def run_fitted_trial(inputs, inner, plan, execution, limits, progress):
    """Run only the installed fit child; commit every progress marker before its ACK."""
    # CPython's Linux affinity API rejects CPU indices at or above INT_MAX.
    if (not isinstance(limits, dict) or set(limits) != {"memory_bytes", "cpu_seconds", "wall_seconds", "cpu_id"}
            or type(limits["cpu_id"]) is not int or not 0 <= limits["cpu_id"] < 2 ** 31 - 1):
        raise fail("INVALID_EXPERIMENT_SPEC", "search execution requires resolved limits and a valid Linux CPU")
    validate_execution_limits({name: value for name, value in limits.items() if name != "cpu_id"})
    forbid_fitting("run_fitted_trial")
    forbid_legacy_fitting("run_fitted_trial")
    payload = fit_payload(inputs, inner, plan, execution)
    if len(payload) > MAX_BYTES:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "search execution input exceeds its byte limit")
    root = Path(__file__).resolve().parents[1]
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(root), "LANG": "C.UTF-8",
           "PYTHONDONTWRITEBYTECODE": "1", **{name: "1" for name in THREAD_VARIABLES}}
    try:
        with tempfile.TemporaryDirectory(prefix="magnitude-fit-") as scratch:
            source = Path(scratch) / "inputs.json"
            source.write_bytes(payload)
            with subprocess.Popen([sys.executable, "-m", "experiments.magnitude_search_worker",
                    str(source), canonical_json(limits)], cwd=root, env=env, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True) as child:
                try:
                    result = _receive(child, limits["wall_seconds"], progress)
                finally:
                    if child.poll() is None:
                        child.kill()
                    child.wait()
            return result
    except OSError:
        raise fail("RESOURCE_UNAVAILABLE", "search execution infrastructure is unavailable") from None


def _receive(child, wall_seconds, progress):
    deadline, buffer, result = time.monotonic() + wall_seconds, bytearray(), None
    with selectors.DefaultSelector() as selector:
        selector.register(child.stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise fail("RESOURCE_LIMIT_EXCEEDED", "search execution exceeded its resource limits")
            if not selector.select(min(remaining, 60.0)):
                continue
            chunk = os.read(child.stdout.fileno(), 65536)
            if not chunk:
                break
            buffer.extend(chunk)
            if len(buffer) > MAX_BYTES:
                raise fail("RESOURCE_LIMIT_EXCEEDED", "search execution output exceeds its byte limit")
            while b"\n" in buffer:
                line, _, buffer = buffer.partition(b"\n")
                try:
                    message = json.loads(line)
                    event = message["event"]
                except (ValueError, TypeError, KeyError, RecursionError):
                    raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search worker protocol differs") from None
                if result is not None:
                    raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search worker protocol differs")
                if event in ("started", "completed") and set(message) == {"event", "fit"}:
                    progress(event, message["fit"])
                    child.stdin.write(b"continue\n")
                    child.stdin.flush()
                elif event in ("failed", "succeeded"):
                    result = message
                elif event == "resource_limit":
                    raise fail("RESOURCE_LIMIT_EXCEEDED", "search execution exceeded its resource limits")
                else:
                    raise fail("RESOURCE_UNAVAILABLE", "search execution infrastructure is unavailable")
        remaining = deadline - time.monotonic()
        try:
            code = child.wait(timeout=max(0, remaining))
        except subprocess.TimeoutExpired:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "search execution exceeded its resource limits") from None
        if time.monotonic() >= deadline:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "search execution exceeded its resource limits")
    if code == -signal.SIGXCPU:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "search execution exceeded its resource limits")
    if code or buffer or result is None:
        raise fail("RESOURCE_UNAVAILABLE", "search execution infrastructure is unavailable")
    return result
