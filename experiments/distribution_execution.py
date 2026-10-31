"""Fixed bounded distribution child; CAS references, never large stdout results."""
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
from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import ArtifactStore, canonical_json, from_document
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.ops.errors import fail
from engine.v2.ops.executor import THREAD_VARIABLES
from engine.v2.research.progress import decision, stage
from experiments.magnitude_search_execution import execution_limits
from experiments.magnitude_search_records import MAX_BYTES

__all__ = ["run_distribution_fit"]


def _receive(child, seconds):
    deadline, payload = time.monotonic() + seconds, bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(child.stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise fail("RESOURCE_LIMIT_EXCEEDED", "distribution child deadline exceeded")
            if not selector.select(min(remaining, 30.0)):
                decision("distribution child", f"waiting; {seconds - remaining:.0f}s elapsed; wall ceiling {seconds}s")
                continue
            chunk = os.read(child.stdout.fileno(), 4097)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > 4096:
                raise fail("RESOURCE_LIMIT_EXCEEDED", "distribution child response exceeds byte limit")
        try:
            code = child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "distribution child deadline exceeded") from None
    if code == -signal.SIGXCPU or time.monotonic() >= deadline:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "distribution child resource limit exceeded")
    if code or not payload:
        raise fail("RESOURCE_UNAVAILABLE", "distribution child did not exit successfully")
    try:
        result = json.loads(payload)
        if type(result) is not dict:
            raise ValueError
    except (ValueError, TypeError, RecursionError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution child response differs") from None
    if set(result) == {"error"} and type(result["error"]) is str and result["error"] in {"EXPERIMENT_VARIANT_FAILED", "INVALID_EXPERIMENT_SPEC",
            "EXPERIMENT_IDENTITY_CONFLICT", "RESOURCE_LIMIT_EXCEEDED", "RESOURCE_UNAVAILABLE"}:
        raise fail(result["error"], "distribution child did not complete")
    return result


def run_distribution_fit(source, recipe, events, horizon, limits):
    """Run installed worker only; callers own admission, unique claim and completion."""
    forbid_legacy_fitting("run_distribution_fit")
    forbid_fitting("run_distribution_fit")
    resolved = execution_limits(limits)
    if type(source) is not dict or set(source) != {"manifest", "inputs_id", "store_root"}:
        raise fail("INVALID_EXPERIMENT_SPEC", "distribution child requires an admitted CAS source")
    payload = canonical_json({**source, "recipe": recipe, "events": events, "horizon": horizon}).encode()
    if len(payload) > MAX_BYTES:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "distribution child header exceeds byte limit")
    root = Path(__file__).resolve().parents[1]
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(root), "LANG": "C.UTF-8",
           "PYTHONDONTWRITEBYTECODE": "1", **{name: "1" for name in THREAD_VARIABLES}}
    with stage("distribution bounded child"), tempfile.TemporaryDirectory(prefix="distribution-fit-") as scratch:
        header = Path(scratch) / "header.json"
        header.write_bytes(payload)
        try:
            with subprocess.Popen([sys.executable, "-m", "experiments.distribution_worker", str(header),
                    canonical_json(resolved)], cwd=root, env=env, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, start_new_session=True) as child:
                try:
                    descriptor = _receive(child, resolved["wall_seconds"])
                finally:
                    if child.poll() is None:
                        child.kill()
                    child.wait()
        except OSError:
            raise fail("RESOURCE_UNAVAILABLE", "distribution child infrastructure unavailable") from None
    try:
        ref = from_document(ArtifactRef, descriptor)
        if ref.schema_ref != "distribution_fit_output.v1" or not 0 < ref.byte_size <= MAX_BYTES:
            raise ValueError
    except (ValueError, TypeError, KeyError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution child result reference differs") from None
    return json.loads(ArtifactStore(source["store_root"]).read_verified(ref))
