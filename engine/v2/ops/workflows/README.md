# `engine/v2/ops/workflows`

## Ownership

Application services of the ops package (policy level `workflows` in
`checks/ops_dependencies.json`). A module here must not import `cli`, the
runtime modules (`executor`, `supervisor`, `worker`, `nightly_trigger`) or any
peer above it.

## Responsibilities

- `commands`: the plan and submit application services behind `ops plan` and
  `ops submit` (nightly, training, promote, rollback and experiment plans;
  nightly and experiment submission, including the manifest, pre-registration
  and host-fit refusals). Behaviour is unchanged from when they lived in `cli`.

## Non-responsibilities

- **Argparse and output formatting** stay in `engine.v2.ops.cli`.
- **Host resource policy**: callers pass `DEFAULT_POLICY` (`host_policy`), so a
  caller's override reaches the host-fit checks.

## Public interface

`commands` exposes module-private services used across modules by name
(`_plan_command`, `_submit_command`, `_submit_nightly`, `_ticker_list`,
`_read_input_manifest_ref`, `_check_nightly_manifest`, `_snapshot_inputs`,
`_primary_runner_bindings`); the only re-export is `require_valid_population_keys`, imported from `snapshot_planning`. Its public helpers are the
bounded, no-follow regular-file readers `read_bounded_json(path, max_bytes)` and
`read_bounded_bytes(path, max_bytes)`, `require_valid_population_keys(keys)` (defined in `snapshot_planning`, the one `ticker|strategy|YYYY-MM-DD` key check; called by `direct_scope_for`, `_read_expected_population`, `_submit_nightly` and `nightly_trigger`, `INVALID_REQUEST`) and the public constants
`MAX_EXPECTED_POPULATION_BYTES` and `MAX_INPUT_MANIFEST_BYTES`, used by
`ops plan --expected-population`, the `--input-manifest` read
(`read_bounded_bytes`) and by `nightly_trigger` (with
`MAX_EXPECTED_POPULATION_BYTES`).

<!-- public-interface: _plan_command, _submit_command, read_bounded_json, read_bounded_bytes, MAX_EXPECTED_POPULATION_BYTES, require_valid_population_keys, MAX_INPUT_MANIFEST_BYTES -->

## Consumers

`engine.v2.ops.cli` (`ops plan`, `ops submit`, the refresh action, ledger
commands), `engine.v2.ops.nightly_trigger` (in-process plan and submit), and
`engine.v2.ops.backup` (`read_bounded_json` for backup manifest parsing).

## Usage

```python
from engine.v2.ops.workflows import commands

commands._plan_command(args, root, conn, clock)
commands._submit_command(args, root, conn, clock, DEFAULT_POLICY)
```

## Testing

`tests/v2/ops/test_ops_dependencies.py` asserts that `nightly_trigger` no longer
imports `cli`, even lazily; `tests/v2/ops/test_v2_ops_nightly_trigger.py`,
`test_v2_ops_snapshot_stages.py` and `test_v2_ops_cli_manifest_gating.py`
exercise the services through the real plan and submit paths.
