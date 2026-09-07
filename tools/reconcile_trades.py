"""Prune simulated Tier-2 trades to the canonical earnings-event universe."""
from __future__ import annotations

import json

from engine.data import manifest, store
from engine.data.normalize.n_trades import filter_to_canonical_events


def main() -> int:
    print("loading canonical events and Tier-2 trades", flush=True)
    events = store.read_table("earnings_events", columns=["event_id"])
    trades = store.read_table("trades")
    clean, report = filter_to_canonical_events(trades, events)
    print(json.dumps(report, indent=2), flush=True)
    if report["rows_removed"]:
        store.write_table(clean, "trades")
    snapshot = manifest.write_snapshot()
    manifest.write_manifest()
    print(f"snapshot {snapshot}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
