#!/usr/bin/env python3
"""EXP-186 — the TWIN-P pricing-only candidate grid on a pinned v2 snapshot.

Pricing slice only: every (event, step, fill_alpha) candidate row is written
to results/candidates_grid.parquet. Selection, simulation, and the
forecast-dependent geometry columns are a later slice (issue #266).

Run:  python3 experiments/EXP-186_menu7prime_twinp_candidates_v2/run.py \
          --catalog private/ops/catalog.sqlite --store-root private/ops
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from engine import paths  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore, SystemClock  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.research import _replay_run  # noqa: E402
from experiments import lib  # noqa: E402
from experiments.v2_candidate_grid import price_candidate_grid  # noqa: E402

ROOT = paths.ROOT
HERE = ROOT / "experiments" / "EXP-186_menu7prime_twinp_candidates_v2"
RESULTS = HERE / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    # Accepted for CLI convention/compatibility only; this slice writes no ledger row.
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()

    spec = lib.load_spec(HERE / "spec.yaml")
    snapshot_id = spec.get("v2_snapshot_id")
    if not snapshot_id:
        raise SystemExit(
            f"[{spec['id']}] spec.yaml is missing v2_snapshot_id — refusing to resolve "
            "the v2 chains snapshot as \"latest\"; set it explicitly once the pinned "
            "snapshot exists."
        )

    conn = open_catalog(args.catalog, clock=SystemClock())
    try:
        repository = Repository(conn, ArtifactStore(args.store_root))
        snapshot = repository.resolve(snapshot_id)
        events = _replay_run.events_frame(repository, snapshot)
    finally:
        conn.close()
    print(f"[{spec['id']}] {len(events):,} known-session events", flush=True)
    frame = price_candidate_grid(
        "TWIN-P", events, catalog=args.catalog, store_root=args.store_root,
        snapshot_id=snapshot_id,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "candidates_grid.parquet"
    frame.to_parquet(path, index=False)
    print(f"[{spec['id']}] wrote {len(frame):,} rows to {path}", flush=True)


if __name__ == "__main__":
    main()
