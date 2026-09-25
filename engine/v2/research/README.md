# `engine/v2/research`

## Ownership

Implements **research tooling — offline analysis CLIs reading a pinned v2
snapshot** at layer **7.0** of [system rearchitecture](../../../guides/system_rearchitecture.md)
§4.1: a peer of `engine.v2.serving`/`engine.v2.ops` and below
`engine.v2.dashboard`, so nothing in the engine imports it back. It is not a
row of the §4 owner table; it was added by Phase 6 slice 6 (decision UD-4).

Also snapshot-pinned research tooling: the native replay and the trade table it
publishes, built on `engine.v2.data.Repository` over committed snapshots rather
than the legacy mutable store.

Replaces (the store-reaching halves of): `tools/signal_screen.py`,
`tools/fill_quality.py`, and the read path of
`engine/data/pulls/polygon_fills.py`. `tools/log_diagnostics.py` is
deliberately **not** moved: it reads only a CSV transactions log and needs no
snapshot read, so it has no v2 counterpart. It also replaces the replay-over-a
committed snapshot of `engine/replay.py` and `engine/build_trades.py`'s v2
write path.

## Responsibilities

- Pure analysis cores moved verbatim from the legacy research tools.
- One `resolve_pinned`/`resolve` call per run, with the resulting
  `snapshot_id` threaded through every read and into the output.
- Plan and price events against one resolved `SnapshotRef`, at a grid of fill
  alphas, with the same arithmetic the legacy replay uses.
- Publish replay output as the `trades` table through the generic incremental
  write path, pinned to the snapshot that produced it.
- Reconcile the published `trades` table to the canonical event universe,
  tombstoning only the non-canonical simulated rows.
- Move provenance onto every row (`engine.v2.research.replay`) so a v2 row is
  never mistaken for a legacy `engine.replay` row.

## Non-responsibilities

- **Mutate the trades ledger or fetch from a network provider** —
  `engine/build_trades.py` and `engine/data/pulls` (legacy, unchanged) do it
  instead. The Polygon pull's `build_plan`/`execute` network half stays there.
- **Serve live requests or render a dashboard** — `engine.v2.serving` and
  `engine.v2.dashboard` do it instead.
- **Decide a research conclusion or a verdict** — a person, from the written
  report, does that; these tools only produce the measured table.
- Fetch or normalize quotes (`engine.v2.data`).
- Decide whether to trade: selection lives in `engine.v2.scoring`.
- Mutate the legacy tree: `engine/replay.py` and `engine/build_trades.py` keep
  publishing exactly as before.

## Public interface

The names the `tools/v2_*` CLIs import. Those CLIs sit in `tools/`, outside
the `checks/import_layers.py` hook (which only parses `engine*` importers), so
no `engine` package is observed importing this one; everything else here is
internal regardless of underscore convention.
`engine/v2/research/_scan.py` (`read_table`, `resolve_snapshot`, `DEFAULT_SCOPE`)
is internal shared machinery, not an interface. The replay path's split
modules (`_snapshot`, `_pricing`, `_plan`, `_chains`, `_trades_table`,
`_trades_revisions`, `_trades_publish`, `_replay_run`, `_build_run`) are
internal too.

The replay half: `replay.replay`, `replay.replay_one`, `_replay_run.run`,
`_replay_run.events_frame`, `_plan.plan_events`, `_chains.ChainIndex`,
`_trades_table.to_trades_table`, `_build_run.run`, `reconcile_trades.run`,
`_trades_publish.publish`, `build_trades.coverage`, `_pricing.STRUCTURES`.

<!-- public-interface: signal_screen.run, fill_quality.run, polygon_fills.run, replay.replay, replay.replay_one, _replay_run.run, _replay_run.events_frame, _plan.plan_events, _chains.ChainIndex, _trades_table.to_trades_table, _build_run.run, reconcile_trades.run, _trades_publish.publish, build_trades.coverage, _pricing.STRUCTURES -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

_Nothing inside `engine/` imports this package. Its consumers are the CLI
leaves `tools/v2_signal_screen.py`, `tools/v2_fill_quality.py`,
`tools/v2_polygon_fills.py`, `tools/v2_replay.py`,
`tools/v2_build_trades.py` and `tools/v2_reconcile_trades.py`, which the
layering hook does not parse (they are not `engine.*` modules)._

<!-- consumers: none -->

## Usage

Every CLI takes a catalog, its object store root, and either a scope (whose
pinned head is resolved once) or an explicit `--snapshot-id` (to reproduce a
run after the head moves):

```bash
python3 tools/v2_signal_screen.py --catalog data/catalog.sqlite \
    --store-root data/artifacts [--snapshot-id snap_...]
python3 tools/v2_fill_quality.py --catalog data/catalog.sqlite \
    --store-root data/artifacts [--since 2026-07-30] [--csv out.csv]
python3 tools/v2_polygon_fills.py --catalog data/catalog.sqlite \
    --store-root data/artifacts [--min-date 2024-08-19]
```

Each writes a report (parquet/markdown for the first two, JSON for the plan)
that names the snapshot id it read.

The replay half is driven as a library:

```python
from engine.v2.research import _replay_run
result = _replay_run.run(repository, strategies=["STR-THRU"], events=events)
```

Reconciliation is its own library entrypoint and CLI:

```python
from engine.v2.research import reconcile_trades
report = reconcile_trades.run(repository, scope="shadow")
```

## Testing

`tests/test_v2_research_tools.py` (tier 0): a synthetic committed snapshot with
real Parquet fragments, no panel load, no network. Four cases per tool: the
moved function is byte-equal to its legacy counterpart on the same synthetic
frame; the tool's output carries the pinned snapshot id; an explicit
`--snapshot-id` still reads the first snapshot after the head has moved; and a
manifest with deleted fragment membership is refused (`MANIFEST_CORRUPT`)
rather than read.

`tests/test_v2_research_replay.py` and `tests/test_v2_research_build_trades.py`
cover the move-parity requirement against legacy `engine.replay` on synthetic
chains, snapshot pinning, and the refusal paths.
`tests/test_v2_research_reconcile_trades.py` covers the canonical-event prune:
non-canonical rows tombstoned, everything else byte-identical, the pinned
snapshot recorded, and an explicit `--snapshot-id` honoured.
