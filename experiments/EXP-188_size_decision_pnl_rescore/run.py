#!/usr/bin/env python3
"""EXP-188 runner: the S1 pre-registration gate, then the S9 driver for the inner and outer stages."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.v2.ops.errors import OpsError, fail  # noqa: E402
from experiments import size_pnl_prereg  # noqa: E402
from experiments.size_pnl_shortvol import Refusal  # noqa: E402

_INVALID = "INVALID_EXPERIMENT_SPEC"
_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
_INPUT_LIMIT = 1 << 20


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("inner", "outer"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected", required=True,
                        help="JSON with plan_id, registration_id, seal_id, panel_id of the v14 RERUN")
    parser.add_argument("--sources", required=True,
                        help="MODULE:FUNC, called with the parsed args, returns a size_pnl_driver.Sources")
    parser.add_argument("--event-ids", help="optional file, one event id per line (smoke runs)")
    parser.add_argument("--no-ledger", action="store_true", required=True,
                        help="required: every non-recorded run uses --no-ledger")
    return parser


def _read_input(path: str, limit: int = _INPUT_LIMIT) -> bytes:
    """The bytes of ``path``, which must be a regular file (no symlink, no FIFO) of at most ``limit`` bytes."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        st = os.fstat(handle.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size > limit:
            raise fail(_INVALID, f"{path} must be a regular file of at most {limit} bytes",
                       details={"reason": "input_not_regular_or_too_large"})
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise fail(_INVALID, f"{path} grew past {limit} bytes", details={"reason": "input_not_regular_or_too_large"})
    return data


def _read_text(path: str) -> str:
    try:
        return _read_input(path).decode("utf-8")
    except (OSError, ValueError) as exc:  # unreadable, or not UTF-8
        raise fail(_INVALID, f"{path} is unreadable", details={"reason": "input_unreadable"}) from exc


def _read_json(path: str):
    text = _read_text(path)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise fail(_INVALID, f"{path} is not JSON", details={"reason": "input_unreadable"}) from exc


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        # Refuses before anything else, with or without --no-ledger.
        receipt = size_pnl_prereg.require_preregistration()
        print(json.dumps(receipt), file=sys.stderr)
        # The supervisor owns heavy jobs: this run must sit under tools/bounded_run.py --heavy.
        if os.environ.get("BOUNDED_RUN_NESTED") != "1":
            raise fail("RESOURCE_UNAVAILABLE", "run under tools/bounded_run.py --heavy",
                       details={"reason": "not_bounded"})
        args.expected_identities = _read_json(args.expected)
        # Imported only after the gates above, so a missing PLANNED row refuses first.
        from experiments import size_pnl_driver
        module_name, _, func = args.sources.partition(":")
        try:
            factory = getattr(importlib.import_module(module_name), func)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise fail(_INVALID, f"--sources {args.sources} is unreadable",
                       details={"reason": "input_unreadable"}) from exc
        sources = factory(args)
        if sources.expected != args.expected_identities:
            raise fail(_CONFLICT, "--expected differs from the sources' expected identities",
                       details={"reason": "expected_mismatch"})
        event_ids = None
        if args.event_ids is not None:
            event_ids = {line.strip() for line in _read_text(args.event_ids).splitlines() if line.strip()}
        summary = size_pnl_driver.run(sources, args.stage, args.out_dir, event_ids=event_ids,
                                      prereg=lambda: receipt)
    except (OpsError, Refusal) as exc:
        print(json.dumps({"refused": exc.code, "message": str(exc)}))
        return 2
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
