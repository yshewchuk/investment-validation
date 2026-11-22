#!/usr/bin/env python3
"""EXP-188 runner: the S1 pre-registration gate, then the S9b driver (inner and outer stages, optional smoke ids), the SY2 synthetic stage (optional X7 replay), and the S9c report stage (`--stage report`: assembles REPORT.md from the outer receipts; read-only)."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import json
import os
import re
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.ops.errors import OpsError, fail  # noqa: E402
from experiments import size_pnl_prereg, size_pnl_shortvol  # noqa: E402
Refusal = size_pnl_shortvol.Refusal

_INVALID, _CONFLICT, _INPUT_LIMIT = "INVALID_EXPERIMENT_SPEC", "EXPERIMENT_IDENTITY_CONFLICT", 1 << 20


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("inner", "outer", "synthetic", "report"))
    parser.add_argument("--replay", action="store_true", help="synthetic stage only: re-score and compare (X7); other stages ignore it")
    parser.add_argument("--event-ids", default=None, help="optional smoke set: a JSON array of event id strings (validated by the driver)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected", required=True, help="JSON with plan_id, registration_id, seal_id, panel_id of the v14 RERUN")
    parser.add_argument("--sources", required=True, help="MODULE:FUNC, called with the parsed args, returns a size_pnl_driver.Sources")
    parser.add_argument("--sources-config", default=None, help="JSON config read by the --sources factory (design section 13)")
    parser.add_argument("--no-ledger", action="store_true", required=True, help="required: every non-recorded run uses --no-ledger")
    return parser


def _read_input(path: str, limit: int = _INPUT_LIMIT) -> bytes:
    """The bytes of ``path``, which must be a regular file (no symlink, no FIFO) of at most ``limit`` bytes."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        st = os.fstat(handle.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size > limit:
            raise fail(_INVALID, f"{path} must be a regular file of at most {limit} bytes", details={"reason": "input_not_regular_or_too_large"})
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise fail(_INVALID, f"{path} grew past {limit} bytes", details={"reason": "input_not_regular_or_too_large"})
    return data


def _read_json(path: str):
    try:
        return json.loads(_read_input(path).decode("utf-8"))
    except (OSError, ValueError) as exc:  # unreadable, not UTF-8, or not JSON
        raise fail(_INVALID, f"{path} is unreadable", details={"reason": "input_unreadable"}) from exc


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):  # library prints go to stderr; stdout is one JSON document
            receipt = size_pnl_prereg.require_preregistration()  # refuses first, with or without --no-ledger
            print(json.dumps(receipt), file=sys.stderr)
            if os.environ.get("BOUNDED_RUN_NESTED") != "1":  # refuse before reading inputs or importing --sources
                raise fail("RESOURCE_UNAVAILABLE", "run under tools/bounded_run.py --heavy", details={"reason": "not_bounded"})
            args.expected_identities = _read_json(args.expected)
            if not isinstance(args.expected_identities, dict) or not {"plan_id", "registration_id", "seal_id", "panel_id"} <= set(args.expected_identities):
                raise fail(_INVALID, "--expected must be a JSON object with plan_id, registration_id, seal_id, panel_id", details={"reason": "input_unreadable"})
            event_ids = _read_json(args.event_ids) if args.event_ids is not None else None
            if args.event_ids is not None and not (isinstance(event_ids, list) and all(isinstance(e, str) for e in event_ids)): raise fail(_INVALID, "--event-ids must be a JSON array of strings", details={"reason": "event_ids_invalid"})
            if args.stage == "report" and args.event_ids is not None: raise fail(_INVALID, "--event-ids is not supported by the report stage", details={"reason": "event_ids_not_supported"})
            from experiments import size_pnl_assemble, size_pnl_driver, size_pnl_ladder  # after the gates, so a missing PLANNED row refuses first
            module_name, _, func = args.sources.partition(":")
            try:  # no colon gives func "", not an attribute; a bad module raises ImportError
                factory = getattr(importlib.import_module(module_name), func)
                if not callable(factory): raise AttributeError(func)
                inspect.signature(factory).bind(args)  # a signature rejecting the parsed args is a refusal
            except (ValueError, TypeError, ImportError, AttributeError) as exc:
                raise fail(_INVALID, f"--sources {args.sources} is invalid: {(type(exc).__name__ + ': ' + str(exc))[:200]}", details={"reason": "sources_invalid"}) from exc
            made = factory(args)  # outside the try: factory-body errors propagate unchanged
            sources = size_pnl_driver.check_sources(made)
            if sources.expected != args.expected_identities:
                raise fail(_CONFLICT, "--expected differs from the sources' expected identities", details={"reason": "expected_mismatch"})
            if args.stage == "report":
                size_pnl_ladder.check_plan(sources.admitted, sources.plan_frame)  # the shared plan admission the scoring path uses; raises a typed refusal
                plan = sources.plan_frame
                summary = size_pnl_assemble.assemble(args.out_dir, dict(zip(plan["event_id"].astype(str), plan["ticker"])), expected=sources.expected, spec_hash=receipt["spec_hash"], snapshot=sources.snapshot_id,
                                                       plan_binding=size_pnl_driver.plan_binding(plan))
            elif args.stage == "synthetic": summary = size_pnl_driver.run_synthetic(sources, args.out_dir, event_ids=event_ids, replay=args.replay, prereg=lambda: receipt)
            else: summary = size_pnl_driver.run(sources, args.stage, args.out_dir, event_ids=event_ids, prereg=lambda: receipt)
    except (OpsError, Refusal, DataError) as exc:
        details = exc.details if isinstance(exc, Refusal) else exc.problem.details
        print(json.dumps({"refused": exc.code, "message": re.sub(r"(?<![\w.])/[^\s:'\"]+", "<path>", str(exc)), "reason": details.get("reason")}))
        return 2
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
