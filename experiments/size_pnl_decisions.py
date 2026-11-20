"""EXP-188 S9b: publish the outer stage's decision rows (decisions.parquet) and bind them in the outer receipt.

The rows go to a unique ``.tmp`` in the run directory, fsynced and hashed. ``outer.json`` is written before the final
file is touched, so a changed replay is refused by the receipt and leaves ``decisions.parquet`` as it was. An exact
replay keeps the existing file. A symlinked, non-regular or oversized existing file refuses.
"""
import hashlib
import os
import stat
import tempfile
from pathlib import Path

import pandas as pd

from experiments import size_pnl_tuning as tuning
from experiments.size_pnl_shortvol import Refusal

MAX_DECISIONS_BYTES = 4 << 30
_HASH_BLOCK = 1 << 20
_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
_FRONT = ["model_id", "book", "arm_or_rule", "structure", "event_id", "event_date", "status", "reason"]


def _unique_tmp(out):
    with tempfile.NamedTemporaryFile(dir=out, delete=False, suffix=".tmp") as handle:
        return Path(handle.name)


def _write_decisions(tmp, frames):
    """Write the rows to ``tmp`` with ``_FRONT`` columns first and fsync it. Returns (sha256 of the bytes, rows)."""
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_FRONT)
    frame = frame[_FRONT + [c for c in frame.columns if c not in _FRONT]]
    frame.to_parquet(tmp, index=False)
    digest = hashlib.sha256()
    with open(tmp, "r+b") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""): digest.update(block)
        os.fsync(handle.fileno())
    return digest.hexdigest(), len(frame)


def _file_sha256(path):
    """Streamed sha256 of the regular file at ``path``; refuses a symlink, a non-regular file or an oversized one."""
    bad = Refusal(_CONFLICT, "decisions artifact must be a regular file of at most MAX_DECISIONS_BYTES", {"reason": "decisions_artifact_invalid"})
    try: fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc: raise Refusal(_CONFLICT, f"decisions artifact cannot be opened safely: {exc}", {"reason": "decisions_artifact_invalid"}) from exc
    digest, total = hashlib.sha256(), 0
    with os.fdopen(fd, "rb") as handle:
        st = os.fstat(handle.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_DECISIONS_BYTES: raise bad
        for block in iter(lambda: handle.read(min(_HASH_BLOCK, MAX_DECISIONS_BYTES + 1 - total)), b""):
            total += len(block)
            if total > MAX_DECISIONS_BYTES: raise bad
            digest.update(block)
    return digest.hexdigest()


def publish(out, frames, body):
    """Write ``frames`` to ``out/decisions.parquet`` through a receipted tmp; returns the row count."""
    out = Path(out)
    final, tmp = out / "decisions.parquet", _unique_tmp(out)
    try:
        sha, rows = _write_decisions(tmp, frames)
        try: tuning.write_receipt(out / "receipts" / "outer.json", {**body, "decisions_sha256": sha, "rows": rows})  # before the final file
        except Refusal as refusal:
            if refusal.details.get("reason") != "tuning_receipt_changed": raise
            raise Refusal(_CONFLICT, str(refusal), {**refusal.details, "reason": "outer_receipt_changed"}) from refusal
        if final.exists() and _file_sha256(final) == sha: tmp.unlink()  # exact replay: keep the existing file
        else: os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)
    return rows
