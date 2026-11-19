"""EXP-188 S6: divisor re-tune on the INNER years (2023, 2024).

S6 of guides/size_decision_pnl_design.md (sections 5, 7, R4, R6). For each divisor ``c`` in ``C_GRID`` the seven
short-vol structures are scored over the inner-year quarters with S5's ``score_events``, ``decide`` and ``score``,
and the ``c`` with the largest taken-trade P&L is chosen; ``choose_c`` falls back to 1.0 when the inner sample has
fewer than ``MIN_TRADES`` taken trades. Outer years are refused before any scoring. The S9 driver WILL call
``tune`` and ``write_receipt`` per model (future; no production caller yet). Progress logging: none (pure,
per-model, small loops).
"""
import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path

import pandas as pd

from experiments.size_pnl_shortvol import NAMES, Refusal, decide, score, score_events

C_GRID = (0.75, 0.9, 1.0, 1.1, 1.25, 1.5)
INNER_YEARS = (2023, 2024)
MIN_TRADES = 300
ALPHA = 0.5
_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
_HOLDOUT = "HOLDOUT_ACCESS_DENIED"
MAX_RECEIPT_BYTES = 1 << 20


def choose_c(totals, trades):
    """(c, skip_reason): the argmax of ``totals`` (c -> summed pnl_dollars), or (1.0, reason) below MIN_TRADES.

    Exact ties go to the c nearest 1, then the smaller c. A non-finite or empty totals set is refused.
    """
    if not totals:
        raise ValueError("totals must not be empty")
    if not all(math.isfinite(float(v)) for v in totals.values()):
        raise ValueError(f"totals must be finite, got {totals!r}")
    if trades < MIN_TRADES:
        return 1.0, "fewer_than_300_inner_trades"
    best = max(float(v) for v in totals.values())
    tied = [c for c, v in totals.items() if float(v) == best]
    return float(min(tied, key=lambda c: (abs(c - 1.0), c))), None


def tune(items, model_id, *, alpha=ALPHA, grid=C_GRID):
    """The S6 tuning receipt body for one model. ``items`` is ``[(chunk, events, sims), ...]``, one per quarter."""
    if 1.0 not in grid:
        raise ValueError("grid must include the unity divisor 1.0")
    if not items:
        raise ValueError("items must not be empty")
    for _chunk, events, _sims in items:  # every item is checked before any scoring
        for year in sorted({int(y) for y in pd.to_datetime(events["event_date"]).dt.year}):
            if year not in INNER_YEARS:
                raise Refusal(_HOLDOUT, f"event year {year} is outside the inner tuning years {INNER_YEARS}",
                              {"reason": "outer_year_in_tuning", "year": year})
    totals, trades = {}, {}
    for c in grid:
        total, count = 0.0, 0
        for name in NAMES:
            scored = pd.concat([score_events(chunk, events, sims, name, alpha=alpha, c=c)
                                for chunk, events, sims in items], ignore_index=True)
            decisions = decide(scored)  # the cutoff history spans quarters, recomputed per c
            for chunk, events, _sims in items:
                ids = {str(e) for e in chunk["event_id"]}
                mine = decisions[decisions["event_id"].astype(str).isin(ids)]
                taken = score(chunk, events, mine, name, alpha, c=c)
                total += float(taken["pnl_dollars"].sum())
                count += len(taken)
        totals[float(c)], trades[float(c)] = total, count
    chosen, skip_reason = choose_c(totals, trades[1.0])
    return {"stage": "tuning", "model_id": model_id, "alpha": alpha, "grid": list(grid), "c": chosen,
            "skipped": skip_reason is not None, "skip_reason": skip_reason, "trades": trades[1.0],
            "totals": {str(c): total for c, total in totals.items()}}


def receipt_hash(body):
    """sha256 of the canonical JSON of ``body``; the body itself must not carry ``content_hash``."""
    if "content_hash" in body:
        raise ValueError("a receipt body must not contain content_hash")
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _refuse(reason, path, message):
    return Refusal(_CONFLICT, message, {"reason": reason, "path": str(path)})


def _read_capped(path):
    """Bytes of the regular file at ``path``, opened without following a symlink and capped at MAX_RECEIPT_BYTES.

    O_NONBLOCK keeps a FIFO from blocking the open; the fstat check then refuses it.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"{path} is not a regular file")
        if st.st_size > MAX_RECEIPT_BYTES:
            raise ValueError(f"{path} is larger than {MAX_RECEIPT_BYTES} bytes")
        parts, total = [], 0
        while total <= MAX_RECEIPT_BYTES:  # reads at most MAX + 1 bytes in all
            part = os.read(fd, MAX_RECEIPT_BYTES + 1 - total)
            if not part:
                break
            parts.append(part)
            total += len(part)
        if total > MAX_RECEIPT_BYTES:
            raise ValueError(f"{path} grew past {MAX_RECEIPT_BYTES} bytes")
        return b"".join(parts)
    finally:
        os.close(fd)


def _fsync_dir(directory):
    """Flush a directory so a link created in it is durable."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read(path):
    """The receipt at ``path`` with its content_hash checked against its body; refuses when unreadable."""
    try:
        data = json.loads(_read_capped(path).decode("utf-8"))
    except (OSError, ValueError) as exc:  # UnicodeDecodeError and JSONDecodeError are ValueErrors
        raise _refuse("tuning_receipt_unreadable", path, f"tuning receipt {path} is unreadable") from exc
    if not isinstance(data, dict) or "content_hash" not in data:
        raise _refuse("tuning_receipt_unreadable", path, f"tuning receipt {path} has no content_hash")
    body = {k: v for k, v in data.items() if k != "content_hash"}
    if receipt_hash(body) != data["content_hash"]:
        raise _refuse("tuning_receipt_unreadable", path, f"tuning receipt {path} does not match its hash")
    return data


def _same_or_refuse(path, digest):
    """The existing receipt when it carries ``digest`` (read-only replay); otherwise a conflict."""
    data = _read(path)
    if data["content_hash"] != digest:
        raise _refuse("tuning_receipt_changed", path, f"tuning receipt {path} differs from the new body")
    return data


def write_receipt(path, body):
    """``body`` plus its content_hash, written once. An existing receipt is replayed read-only or refused, never overwritten."""
    path = Path(path)
    digest = receipt_hash(body)
    if path.exists():
        return _same_or_refuse(path, digest)
    full = dict(body, content_hash=digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8", suffix=".tmp")
    try:
        with tmp:
            tmp.write(json.dumps(full, sort_keys=True, indent=2) + "\n")
            tmp.flush()
            os.fsync(tmp.fileno())
        try:
            os.link(tmp.name, path)  # atomic and fails if the target exists
        except FileExistsError:
            return _same_or_refuse(path, digest)
        _fsync_dir(path.parent)
        return full
    finally:
        os.unlink(tmp.name)


def load_receipt(path):
    """The verified receipt at ``path``; refuses when it is missing (the 2025 stage's start gate) or tampered."""
    if not Path(path).exists():
        raise _refuse("tuning_receipt_missing", path, f"tuning receipt {path} is missing")
    return _read(path)
