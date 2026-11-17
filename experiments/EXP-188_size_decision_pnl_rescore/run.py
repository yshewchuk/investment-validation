#!/usr/bin/env python3
"""EXP-188 runner; S1 ships only the pre-registration gate, later slices add the stages."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments import size_pnl_prereg  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-ledger", action="store_true", required=True,
                        help="required: every non-recorded run uses --no-ledger")
    parser.parse_args(argv)
    # Refuses before anything else, with or without --no-ledger.
    receipt = size_pnl_prereg.require_preregistration()
    print(json.dumps(receipt), file=sys.stderr)
    raise NotImplementedError("EXP-188 stages land in later slices")


if __name__ == "__main__":
    main()
