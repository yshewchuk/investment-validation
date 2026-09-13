#!/usr/bin/env python3
"""Fresh-process adapter score runner used by the private canary."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ["INVESTING_PLAN_ROOT"] = str(ROOT)
    from engine.v2.ops.legacy_adapter import legacy_action

    root = args.requests.parent
    params = {"requests_path": args.requests.name, "year_start": 2007,
              "year_end": 2030}
    result = legacy_action("legacy_score_requests", params, root)
    source = root / result["path"]
    args.output.write_bytes(source.read_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
