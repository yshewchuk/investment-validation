# `engine/v2/ops/stores`

## Ownership

Store-tier contracts and acquisition owners under the planned dependency policy
(`checks/ops_dependencies.json`). `refresh_contracts` remains dependency-free;
the [Massive acquisition package](massive/ARCHITECTURE.md) composes stores, provider edges and core runtime guards.

## Responsibilities

- Hold `refresh_contracts.REFRESH_INPUT_DOCUMENT_NAMES`: the staged identity
  document name for each refresh job kind.
- Own supervised Massive acquisition through `massive.acquisition` and pinned daily/action snapshots or parent-bound refresh publication through `massive.snapshot`.

## Non-responsibilities

- **Write or read the staged documents** — `engine.v2.ops.refresh_staging`
  writes them; `engine.v2.ops.calendar_moves_jobs` reads the forward-calendar one.
- **Define refresh parameters or results** — those stay with their owners until
  their own prerequisite slices move them here.

## Public interface

`refresh_contracts` provides `REFRESH_INPUT_DOCUMENT_NAMES`, a mapping from
refresh job kind to staged document file name.

<!-- public-interface: REFRESH_INPUT_DOCUMENT_NAMES -->

## Consumers

`engine.v2.ops.refresh_staging` (writes the documents) and
`engine.v2.ops.calendar_moves_jobs` (forward-calendar callback).

## Usage

```python
from engine.v2.ops.stores.refresh_contracts import REFRESH_INPUT_DOCUMENT_NAMES
```

## Testing

`tests/v2/ops/test_ops_dependencies.py` asserts the import graph, including
function-level imports; `tests/v2/ops/test_v2_ops_refresh_staging.py` exercises
the names through the real staging path.
