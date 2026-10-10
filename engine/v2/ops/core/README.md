# `engine/v2/ops/core`

## Ownership

Dependency-free, immutable contracts of the ops package that sit at the
bottom of its planned dependency policy (`checks/ops_dependencies.json`, level
`core`). A module here imports nothing else from `engine.v2.ops`.

## Responsibilities

- Hold `snapshot_contracts.SNAPSHOT_BINDINGS`: the three artifact names
  (`snapshot_ref.json`, `materialization_request.json`,
  `materialization_manifest.json`) every snapshot-mode job must bind.

## Non-responsibilities

- **Validate or register a job kind** — `engine.v2.ops.stages` owns the kind
  registry and `input_mode_problems`; this package only supplies the constant.
- **Launch or verify a snapshot** — `engine.v2.ops.snapshot_stages` does.

## Public interface

`snapshot_contracts` provides `SNAPSHOT_BINDINGS`, a tuple of binding names in
the order the launch observation records them.

<!-- public-interface: SNAPSHOT_BINDINGS -->

## Consumers

`engine.v2.ops.stages` (snapshot input-mode validation) and
`engine.v2.ops.snapshot_stages` (launch-mode detection and observed-binding
check).

## Usage

```python
from engine.v2.ops.core.snapshot_contracts import SNAPSHOT_BINDINGS
```

## Testing

`tests/v2/ops/test_ops_dependencies.py` asserts the import graph, including
function-level imports; `tests/v2/ops/test_v2_ops_snapshot_stages.py` exercises the
bindings through the real launch path.
