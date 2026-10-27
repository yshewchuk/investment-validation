from __future__ import annotations

import json
import resource
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

from engine.v2.ops.errors import fail
from engine.v2.ops.worker import main as worker_main
from engine.v2.serving import projections
from tools.v2_dashboard_project import run_projection


_REQUIRED = frozenset({
    "bundle.tar", "finality.json", "score.json", "preview_input.json",
    "source_provenance.json", "nightly_plan.json",
})


def _log_stage(stage: str, event: str, started: float | None = None) -> None:
    elapsed = "" if started is None else f" elapsed_s={time.monotonic() - started:.3f}"
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print(f"[serving_projection] {stage} {event}{elapsed} rss_bytes={rss_bytes}",
          file=sys.stderr, flush=True)


def _unpack_bundle(source: Path, destination: Path) -> None:
    destination.mkdir()
    try:
        with tarfile.open(source, "r:*") as archive:
            members = archive.getmembers()
            for member in members:
                name = PurePosixPath(member.name)
                if (name.is_absolute() or ".." in name.parts
                        or not (member.isfile() or member.isdir())):
                    raise fail("INTEGRITY_FAILED", "projection bundle contains an unsafe member")
            archive.extractall(destination, members=members)
    except (tarfile.TarError, OSError) as exc:
        raise fail("INTEGRITY_FAILED", "projection bundle cannot be unpacked") from exc


def dispatch(worker, parameters, root, *, envelope=None):
    if worker != "serving_projection":
        raise ValueError("unsupported projection worker")
    envelope = envelope or {}
    bindings = parameters.get("input_bindings") or {}
    missing = sorted(_REQUIRED - set(bindings))
    if missing:
        raise fail("INVALID_REQUEST", "serving projection inputs are incomplete",
                   details={"missing": missing})
    if not parameters.get("snapshot_id"):
        raise fail("INVALID_REQUEST", "serving projection requires a pinned snapshot ID")
    root = Path(root)
    scratch = Path(tempfile.mkdtemp(prefix="serving-projection-", dir="/tmp"))
    try:
        bundle_dir = scratch / "bundle"
        _log_stage("unpack", "start")
        unpack_start = time.monotonic()
        _unpack_bundle(root / "bundle.tar", bundle_dir)
        _log_stage("unpack", "end", unpack_start)
        native_path = root / "native_inputs.json"
        projection_start = time.monotonic()
        _log_stage("projection", "start")
        serving_root = Path(envelope.get("serving_root",
                                         Path(envelope["objects_root"]) / "serving"))
        try:
            code, document = run_projection(
                preview_input=root / "preview_input.json",
                source_provenance=root / "source_provenance.json",
                score_json=root / "score.json",
                bundle_dir=bundle_dir,
                snapshot_id=parameters["snapshot_id"],
                catalog=Path(envelope["catalog_path"]),
                store_root=Path(envelope["objects_root"]),
                serving_root=serving_root,
                requested_as_of=parameters["requested_as_of"],
                resolved_as_of=parameters["resolved_as_of"],
                bundle_format=parameters.get("bundle_format", "legacy"),
                nightly_plan=root / "nightly_plan.json",
                native_inputs=native_path if "native_inputs.json" in bindings else None)
        finally:
            _log_stage("projection", "end", projection_start)
        if code != 0:
            raise fail("PROJECTION_REFUSED", "serving projection findings refused publication")
        projection_binding = document.get("projection_binding")
        if not isinstance(projection_binding, dict):
            raise fail("INTEGRITY_FAILED", "projection result omitted its binding")
        conn = projections.connect(str(serving_root / "serving.sqlite"))
        try:
            if not projections.verify_projection_binding(conn, projection_binding):
                raise fail("INTEGRITY_FAILED", "projection binding failed live-index verification")
        finally:
            conn.close()
        result_path = root / "projection_result.json"
        binding_path = root / "projection_binding.json"
        result_path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
        binding_path.write_text(json.dumps(projection_binding, sort_keys=True,
                                           separators=(",", ":")) + "\n")
        shutil.rmtree(scratch)
        expected = list(parameters["expected_ids"])
        return {
            "outputs": [
                {"name": "projection_result", "path": "projection_result.json",
                 "schema": "serving_projection_result.v1.0"},
                {"name": "projection_binding", "path": "projection_binding.json",
                 "schema": "projection_binding.v1.0"},
            ],
            "completed_ids": expected,
            "no_work": not expected,
        }
    except BaseException:
        # Retain scratch for the supervisor's private diagnostics/reclaim path.
        raise


def main():
    return worker_main(dispatcher=dispatch)


if __name__ == "__main__":
    raise SystemExit(main())
