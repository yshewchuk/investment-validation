"""Complete, sequence-ordered compatibility exports for legacy readers."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from engine.v2.foundation import canonical_json, fsync_directory


def export_generation(conn, root: Path | str, *, generation: str) -> Path:
    """Write all prediction rows through a durable generation and switch CURRENT atomically."""
    if "/" in generation or generation in ("", ".", ".."):
        raise ValueError("unsafe export generation")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / generation
    rows = conn.execute("SELECT kind,payload_json FROM decisions ORDER BY sequence").fetchall()
    grouped = defaultdict(list)
    for row in rows:
        payload = json.loads(row[1])
        date = payload.get("as_of") or payload.get("settled_at") or payload.get("event_date")
        date = str(date)[:10] if date else "unknown"
        bucket = "predictions" if row[0] == "prediction" else "outcomes"
        grouped[(bucket, date)].append(canonical_json(payload).encode("utf-8") + b"\n")
    if destination.exists():
        for (bucket, date), lines in grouped.items():
            path = destination / bucket / (date + ".jsonl")
            if not path.is_file() or path.read_bytes() != b"".join(lines):
                raise ValueError("export generation differs from catalog")
    else:
        destination.mkdir()
        for (bucket, date), lines in grouped.items():
            path = destination / bucket / (date + ".jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"".join(lines))
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        fsync_directory(destination)
    pointer = root / ("CURRENT." + generation)
    pointer.write_text(generation + "\n")
    with pointer.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pointer, root / "CURRENT")
    fsync_directory(root)
    return destination
