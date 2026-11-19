#!/usr/bin/env python3
"""EXP-188 runner: the S1 pre-registration gate, then the S9 driver for the inner and outer stages."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.v2.ops.errors import OpsError, fail  # noqa: E402
from experiments import size_pnl_prereg  # noqa: E402


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
        args.expected_identities = json.loads(Path(args.expected).read_text(encoding="utf-8"))
        # Imported only after the gates above, so a missing PLANNED row refuses first.
        from experiments import size_pnl_driver
        module_name, _, func = args.sources.partition(":")
        sources = getattr(importlib.import_module(module_name), func)(args)
        event_ids = None
        if args.event_ids is not None:
            event_ids = {line.strip() for line in Path(args.event_ids).read_text(encoding="utf-8").splitlines()
                         if line.strip()}
        summary = size_pnl_driver.run(sources, args.stage, args.out_dir, event_ids=event_ids,
                                      prereg=lambda: receipt)
    except OpsError as exc:
        print(json.dumps({"refused": exc.code, "message": str(exc)}))
        return 2
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
