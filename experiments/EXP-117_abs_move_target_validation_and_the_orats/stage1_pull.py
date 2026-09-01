#!/usr/bin/env python3
"""EXP-117 Stage 1 — fetch Polygon underlying daily bars for the sample.

    python3 stage1_pull.py            # resumable; Tier-1 cache makes re-runs free

One ``v2/aggs/ticker/{T}/range/1/day/{from}/{to}`` call per ticker with
``adjusted=false`` — the full daily life in one call. Pacing, retries, and the
single-process lock come from ``engine.data.throttle`` via the Fetcher.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/root/investing-plan")
sys.path.insert(0, str(ROOT))

from engine.data.fetch import Fetcher  # noqa: E402

HERE = Path(__file__).resolve().parent
FROM = "2006-01-01"
TO = "2026-09-01"
ENDPOINT_TMPL = "v2/aggs/ticker/{ticker}/range/1/day/" + FROM + "/" + TO


def main() -> None:
    sample = json.loads((HERE / "results" / "stage1_sample.json").read_text())
    tickers = sample["tickers"]
    f = Fetcher()
    started = time.time()
    done = cached = failed = 0
    failures = {}
    log_path = HERE / "results" / "stage1_pull_progress.log"
    logf = open(log_path, "a")

    def log(msg: str) -> None:
        line = f"[stage1_pull] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    for i, tk in enumerate(tickers):
        endpoint = ENDPOINT_TMPL.format(ticker=tk)
        params = {"adjusted": "false", "limit": 50000}
        if f.has("polygon", endpoint, params):
            cached += 1
        else:
            try:
                rec = f.fetch("polygon", endpoint, params, note="exp117-stage1")
                status = rec.status
                if status == 200:
                    done += 1
                else:
                    failed += 1
                    failures[tk] = status
                    log(f"HTTP {status} for {tk}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                failures[tk] = str(exc)[:120]
                log(f"ERROR {tk}: {str(exc)[:120]}")
        if (i + 1) % 10 == 0 or i == len(tickers) - 1:
            el = time.time() - started
            rate = (done + failed) / el if el > 0 else 0
            eta = (len(tickers) - i - 1) / rate if rate > 0 else 0
            log(f"{i+1}/{len(tickers)} fetched={done} cached={cached} failed={failed} "
                f"elapsed={el:.0f}s eta={eta:.0f}s")
    log(f"FINISHED fetched={done} cached={cached} failed={failed}")
    if failures:
        (HERE / "results" / "stage1_pull_failures.json").write_text(json.dumps(failures, indent=1))
    logf.close()


if __name__ == "__main__":
    main()
