#!/usr/bin/env python3
"""Scripted stand-in for ``tools/bounded_run.py``, for
``tests/test_v2_resource_measurement.py`` only.

The recorder under test launches this file in place of the real bounded run
(never a real heavy job). It ignores its flags like a black box, and is
driven entirely by three environment variables:

* ``V2_RESOURCE_STUB_LINES`` -- JSON list of lines to print to stdout
  (the scripted ``[watchdog]`` readings and/or breach strings);
* ``V2_RESOURCE_STUB_EXIT_CODE`` -- the exit code to return;
* ``V2_RESOURCE_STUB_ARGV_PATH`` -- when set, ``sys.argv`` is written there
  as JSON, so a test can assert on the exact argv the recorder passed
  through (the real pass-through contract lives in the recorder's argv, not
  in its own record).

It is never imported by production code and never launched outside tests.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    argv_path = os.environ.get("V2_RESOURCE_STUB_ARGV_PATH")
    if argv_path:
        Path(argv_path).write_text(json.dumps(sys.argv))
    for line in json.loads(os.environ.get("V2_RESOURCE_STUB_LINES", "[]")):
        print(line, flush=True)
    return int(os.environ.get("V2_RESOURCE_STUB_EXIT_CODE", "0"))


if __name__ == "__main__":
    raise SystemExit(main())
