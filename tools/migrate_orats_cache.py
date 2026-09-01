#!/usr/bin/env python3
"""Import legacy ORATS strike pulls into the engine's Tier-1 cache.

    python3 tools/migrate_orats_cache.py            # dry run, reports only
    python3 tools/migrate_orats_cache.py --apply

Three standalone pullers — ``s2/pull_strikes.py``, ``s3/pull_entry.py`` and
``s5/pull_c2_exit.py`` — spent roughly 19,000 ORATS calls writing gzipped
responses under ``earnings_predictions/data/raw/orats/strikes/``. None of them
goes through :mod:`engine.data.fetch`, so none of it is in the engine's cache,
and none of it was covered by the quota guard. That is how August ran to 875
calls of 20,000 with nothing refusing the next request.

Routing those pullers through the engine is the fix. Doing it without this
migration first would be expensive rather than merely wrong: the engine cache
is content-addressed, every rerouted call would miss, and ~14,000 requests
would be re-spent fetching bytes already on disk. This script makes the paid-for
responses read as cache hits.

**Why one parameter template covers every file.** All three pullers issued the
identical request — ``/hist/strikes`` with ``dte=1,45`` and the same 140-char
``fields`` list. They now share one definition in ``_shared/strike_pull.py``,
which this tool imports, so the key it computes and the key a live pull computes
cannot diverge. The ``_t14`` and ``_c2`` filename suffixes record which strategy
asked, not a different call. The token is added by the adapter at request time
and is not part of the cache key, so it cannot poison the migration.

**The failure mode is cost, not corruption.** A key computed wrongly here does
not produce bad data; it produces a miss, and the engine re-fetches. The check
that matters is therefore whether a rerouted call actually HITS, and that costs
nothing to verify: ``Fetcher.fetch`` on a migrated request must return
``from_cache=True`` without touching the network. Do that before rerouting any
puller, not after.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.data.fetch import Fetcher, cache_key, canonical_params  # noqa: E402

LEGACY_DIR = paths.RAW_ORATS / "strikes"

# The request definition now has exactly one home. This tool used to carry its
# own copy plus a drift check across three puller sources; those pullers now
# share `_shared/strike_pull.py`, so importing from it removes the possibility
# of drift rather than detecting it.
sys.path.insert(0, str(ROOT / "earnings_predictions" / "strategies"))
from _shared.strike_pull import (  # noqa: E402
    DTE,
    ENDPOINT,
    FIELDS,
    build_params,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write; otherwise report only")
    ap.add_argument("--limit", type=int, default=0, help="stop after N files (testing)")
    args = ap.parse_args()

    fetcher = Fetcher()
    files = sorted(LEGACY_DIR.glob("*.json.gz"))
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files):,} legacy files under {LEGACY_DIR.relative_to(ROOT)}")

    stats = {"migrated": 0, "already_cached": 0, "duplicate_key": 0,
             "unreadable": 0, "empty": 0}
    seen: dict[str, str] = {}
    for path in files:
        try:
            with gzip.open(path, "rt") as fh:
                payload = json.load(fh)
            entry_date = payload["entry_date"]
            tickers = payload["tickers"]
            rows = payload["rows"]
        except (OSError, ValueError, KeyError):
            stats["unreadable"] += 1
            continue
        if not tickers:
            stats["empty"] += 1
            continue

        params = build_params(entry_date, tickers)
        key = cache_key("orats", ENDPOINT, params)
        body_path = fetcher.body_path("orats", key)
        meta_path = fetcher.meta_path("orats", key)

        if key in seen:
            # Two strategies asking for the same date+tickers produce the same
            # request and therefore the same entry. Counted, not treated as an
            # error: it is the cache doing its job retroactively.
            stats["duplicate_key"] += 1
            continue
        seen[key] = path.name

        if body_path.exists() and meta_path.exists():
            stats["already_cached"] += 1
            continue
        stats["migrated"] += 1
        if not args.apply:
            continue

        body = json.dumps({"data": rows}, separators=(",", ":")).encode()
        body_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_body = body_path.with_suffix(body_path.suffix + ".tmp")
        with gzip.open(tmp_body, "wb") as fh:
            fh.write(body)
        os.replace(tmp_body, body_path)

        meta = {
            "key": key,
            "source": "orats",
            "endpoint": ENDPOINT.strip("/"),
            "params": json.loads(canonical_params(params)),
            # No url: the legacy response is not being re-attested, and a
            # fabricated one would be a lie in the provenance record.
            "url": None,
            "status": 200,
            "bytes": len(body),
            "elapsed_s": None,
            "quota_remaining": None,
            "fetched_at": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "note": f"migrated from {path.name} by tools/migrate_orats_cache.py",
        }
        tmp_meta = meta_path.with_suffix(".json.tmp")
        tmp_meta.write_text(json.dumps(meta, indent=1, sort_keys=True))
        os.replace(tmp_meta, meta_path)

    print(f"\n{'migrated' if args.apply else 'would migrate':>16}: {stats['migrated']:,}")
    for k in ("already_cached", "duplicate_key", "unreadable", "empty"):
        print(f"{k:>16}: {stats[k]:,}")
    print(f"{'distinct keys':>16}: {len(seen):,}")
    if not args.apply:
        print("\ndry run — nothing written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
