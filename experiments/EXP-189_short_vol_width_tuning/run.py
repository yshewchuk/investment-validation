#!/usr/bin/env python3
"""EXP-189 runner: the size-width pre-registration gate, then the width driver (inner and outer stages), or the report stage (`--stage report`: assembles REPORT.md from the outer receipts; reads no sources)."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import json
import logging
import os
import re
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.ops.errors import OpsError, fail  # noqa: E402
from experiments import fair_value_inputs, fair_value_legs, size_pnl_shortvol, size_width_f_select, size_width_prereg  # noqa: E402
Refusal = size_pnl_shortvol.Refusal

_INVALID, _CONFLICT, _INPUT_LIMIT = "INVALID_EXPERIMENT_SPEC", "EXPERIMENT_IDENTITY_CONFLICT", 1 << 20


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("inner", "outer", "report"), help="inner and outer need BOUNDED_RUN_NESTED=1 (tools/bounded_run.py --heavy); report reads no sources, and --expected is compared with the identity bound in the inputs receipts")
    parser.add_argument("--event-ids", default=None, help="optional smoke set: a JSON array of event id strings (validated by the driver); not supported by report")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected", required=True, help="JSON with plan_id, registration_id, seal_id, panel_id")
    parser.add_argument("--sources", required=True, help="MODULE:FUNC, called with the parsed args, returns a size_pnl_driver.Sources (never called by report)")
    parser.add_argument("--sources-config", default=None, help="JSON config read by the --sources factory")
    parser.add_argument("--f-pins", default=None, help="F spec only, inner and outer: JSON with manifest_sha256 and files (the pinned tables)")
    parser.add_argument("--f-quotes", default=None, help="F spec only, inner and outer: MODULE:FUNC, called with the parsed args, returns (read_quotes, snapshot_id)")
    parser.add_argument("--fill-variant", default=None, help="model A spec only, inner and outer: primary (default) or spread050; recorded in the receipts")
    parser.add_argument("--f-max-object-bytes", type=int, default=None, help="F spec only: per-fragment byte cap of the experiments.size_width_f_chains reader (default 256 MiB)")
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


def _factory(spec: str, args, flag: str, reason: str):
    """The result of ``spec`` (MODULE:FUNC) called with the parsed args; a spec that is not a callable the args bind refuses ``reason``."""
    module_name, _, func = spec.partition(":")
    try:  # no colon gives func "", not an attribute; a bad module raises ImportError
        factory = getattr(importlib.import_module(module_name), func)
        if not callable(factory): raise AttributeError(func)
        inspect.signature(factory).bind(args)  # a signature rejecting the parsed args is a refusal
    except (ValueError, TypeError, ImportError, AttributeError) as exc:
        raise fail(_INVALID, f"{flag} {spec} is invalid: {(type(exc).__name__ + ': ' + str(exc))[:200]}", details={"reason": reason}) from exc
    return factory(args)  # outside the try: factory-body errors propagate unchanged


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")  # F load and pricing progress to stderr (the default stream)
    try:
        with contextlib.redirect_stdout(sys.stderr):  # library prints go to stderr; stdout is one JSON document
            receipt = size_width_prereg.require_preregistration()  # refuses first, with or without --no-ledger
            print(json.dumps(receipt), file=sys.stderr)
            if args.stage != "report" and os.environ.get("BOUNDED_RUN_NESTED") != "1":  # refuse before reading inputs or importing --sources
                raise fail("RESOURCE_UNAVAILABLE", "run under tools/bounded_run.py --heavy", details={"reason": "not_bounded"})
            args.expected_identities = _read_json(args.expected)
            if not isinstance(args.expected_identities, dict) or not {"plan_id", "registration_id", "seal_id", "panel_id"} <= set(args.expected_identities):
                raise fail(_INVALID, "--expected must be a JSON object with plan_id, registration_id, seal_id, panel_id", details={"reason": "input_unreadable"})
            event_ids = _read_json(args.event_ids) if args.event_ids is not None else None
            if args.event_ids is not None and not (isinstance(event_ids, list) and all(isinstance(e, str) for e in event_ids)): raise fail(_INVALID, "--event-ids must be a JSON array of strings", details={"reason": "event_ids_invalid"})
            if args.stage == "report" and args.event_ids is not None: raise fail(_INVALID, "--event-ids is not supported by the report stage", details={"reason": "event_ids_not_supported"})
            if args.stage == "report":
                from experiments import size_width_report  # after the gates; no sources are read
                size_width_f_select.check_report(receipt, args.out_dir)  # an F spec's report reads only inputs that bind F
                summary = size_width_report.assemble(args.out_dir, expected=args.expected_identities)
            else:
                from experiments import size_pnl_driver, size_width_driver  # after the gates, so a missing PLANNED row refuses first
                if (args.f_pins is None) != (args.f_quotes is None): raise fail(_INVALID, "--f-pins and --f-quotes must be given together", details={"reason": "f_inputs_incomplete"})
                if args.fill_variant is not None and args.f_pins is None: raise fail(_INVALID, "--fill-variant needs --f-pins and --f-quotes", details={"reason": "f_inputs_incomplete"})
                sources = size_pnl_driver.check_sources(_factory(args.sources, args, "--sources", "sources_invalid"))
                if sources.expected != args.expected_identities:
                    raise fail(_CONFLICT, "--expected differs from the sources' expected identities", details={"reason": "expected_mismatch"})
                pair = _factory(args.f_quotes, args, "--f-quotes", "f_quotes_invalid") if args.f_quotes is not None else None
                if pair is not None and not (isinstance(pair, (tuple, list)) and len(pair) == 2 and callable(pair[0])): raise fail(_INVALID, "--f-quotes must return (read_quotes, snapshot_id)", details={"reason": "f_quotes_invalid"})
                kwargs = {} if pair is None else {"f_inputs": size_width_f_select.FInputs(_read_json(args.f_pins), *pair, "primary" if args.fill_variant is None else args.fill_variant)}
                summary = size_width_driver.run(sources, args.stage, args.out_dir, event_ids=event_ids, prereg=lambda: receipt, **kwargs)
    except (OpsError, Refusal, DataError, fair_value_inputs.FairValueRefusal, fair_value_legs.FairValueRefusal) as exc:
        details = exc.problem.details if isinstance(exc, (OpsError, DataError)) else exc.details
        print(json.dumps({"refused": exc.code, "message": re.sub(r"(?<![\w.])/[^\s:'\"]+", "<path>", str(exc)), "reason": details.get("reason")}))
        return 2
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
