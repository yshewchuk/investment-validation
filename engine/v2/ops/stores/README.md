# `engine/v2/ops/stores`

## Ownership

Dependency-free contracts shared by the ops refresh stores, at the `stores`
level of the planned dependency policy (`checks/ops_dependencies.json`). A
module here imports nothing else from `engine.v2.ops`.

## Responsibilities

- Hold `refresh_contracts.REFRESH_INPUT_DOCUMENT_NAMES`: the staged identity
  document name for each refresh job kind.

## Non-responsibilities

- **Write or read the staged documents** — `engine.v2.ops.refresh_staging`
  writes them; `engine.v2.ops.incremental_data` reads the forward-calendar one.
- **Define refresh parameters or results** — those stay with their owners until
  their own prerequisite slices move them here.

## Public interface

`refresh_contracts` provides `REFRESH_INPUT_DOCUMENT_NAMES`, a mapping from
refresh job kind to staged document file name.

<!-- public-interface: REFRESH_INPUT_DOCUMENT_NAMES -->

## Consumers

`engine.v2.ops.refresh_staging` (writes the documents) and
`engine.v2.ops.incremental_data` (forward-calendar callback).

## Usage

```python
from engine.v2.ops.stores.refresh_contracts import REFRESH_INPUT_DOCUMENT_NAMES
```

## Testing

`tests/v2/ops/test_ops_dependencies.py` asserts the import graph, including
function-level imports; `tests/v2/ops/test_v2_ops_refresh_staging.py` exercises
the names through the real staging path.
