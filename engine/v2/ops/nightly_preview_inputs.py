"""Produce immutable nightly preview-input and source-lineage documents."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from engine.v2.ops.errors import fail

_INPUTS = ("score.json", "bundle.tar", "finality.json", "model_evidence.json")
MAX_INPUT_BYTES = 512 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024


def _write_atomic(path: Path, payload: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _input_provenance(root, bindings):
    prepared = []
    for name in _INPUTS:
        binding = bindings[name]
        parent, separator, output = binding.partition("#") if isinstance(binding, str) else ("", "", "")
        if not separator or not parent.startswith("job_") or not output:
            raise fail("INVALID_REQUEST", "nightly preview input binding is malformed",
                       details={"name": name})
        path = root / name
        if not path.is_file():
            raise fail("INPUT_CHANGED", "nightly preview input is missing",
                       details={"name": name})
        size = path.stat().st_size
        if size > MAX_INPUT_BYTES:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "nightly preview input exceeds size limit",
                       details={"name": name, "limit_bytes": MAX_INPUT_BYTES,
                                "observed_bytes": size})
        prepared.append((name, parent, output, path, size))
    inputs = {}
    for name, parent, output, path, size in prepared:
        digest = hashlib.sha256()
        streamed = 0
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                if streamed + len(chunk) > MAX_INPUT_BYTES:
                    raise fail("RESOURCE_LIMIT_EXCEEDED",
                               "nightly preview input exceeds size limit",
                               details={"name": name, "limit_bytes": MAX_INPUT_BYTES,
                                        "observed_bytes": streamed + len(chunk)})
                digest.update(chunk)
                streamed += len(chunk)
        if streamed != size:
            raise fail("INPUT_CHANGED", "nightly preview input changed during hashing",
                       details={"name": name, "expected_bytes": size,
                                "observed_bytes": streamed})
        inputs[name] = {"producer_job_id": parent, "output_name": output,
                        "content_hash": "sha256:" + digest.hexdigest()}
    return inputs


def run(parameters, root):
    values = parameters if isinstance(parameters, dict) else vars(parameters)
    snapshot_id = values.get("snapshot_id")
    bindings = values.get("input_bindings") or {}
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise fail("INVALID_REQUEST", "nightly preview inputs require a pinned snapshot_id")
    if not isinstance(bindings, dict) or set(bindings) != set(_INPUTS):
        raise fail("INVALID_REQUEST", "nightly preview inputs require all declared bindings")
    root = Path(root)
    inputs = _input_provenance(root, bindings)
    preview = {"schema_version": "nightly_preview_input.v1.0",
               "snapshot_id": snapshot_id, "inputs": inputs}
    provenance = {
        "schema_version": "nightly_source_provenance.v1.0",
        "snapshot_id": snapshot_id,
        "producing_job_ids": sorted({item["producer_job_id"] for item in inputs.values()}),
        "inputs": {name: {**item, "hash_algorithm": "sha256"}
                   for name, item in inputs.items()},
    }
    for filename, document in (("preview_input.json", preview),
                               ("source_provenance.json", provenance)):
        payload = (json.dumps(document, sort_keys=True, separators=(",", ":"),
                              allow_nan=False) + "\n").encode("utf-8")
        _write_atomic(root / filename, payload)
    return {"outputs": [
        {"name": "preview_input", "path": "preview_input.json",
         "schema": "nightly_preview_input.v1.0"},
        {"name": "source_provenance", "path": "source_provenance.json",
         "schema": "nightly_source_provenance.v1.0"},
    ], "completed_ids": ["nightly_preview_inputs"]}