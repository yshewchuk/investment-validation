# `engine/v2/parity`

## Ownership

Not a §4 owner-table row. Added by `spec_ns_c` part c, at layer **—** (6.5) of
§4.1: above every package whose records are compared, below the layer-7
reporting consumers.

Replaces: the numeric field groups and per-dimension comparator that lived
inside `checks/phase4_real.py`, and it is now the home of the record
comparator core (`receipt`, `record_comparator`, `stage_plan`, `tolerance`)
that used to live in `engine/v2/diagnosis`.

### Why the comparator core lives here, and why this layer is legal

The nightly parity report (`engine/v2/ops/native_parity_report.py`) must run
exactly the comparison the Phase 4 checker runs. Production may not import
`checks/`, and `engine/v2/diagnosis` is a sink that no package may import, so
the comparator had to move into a production package. It needs nothing but
`engine.v2.foundation` (`content_hash`), so the layer map pins this package to
`only_imports=(0.5,)`.

Layer 6.5 keeps the §4.1 sink rule's purpose — a comparator never becomes a
dependency of what it compares — structural rather than claimed: every
package whose outputs are compared (`scoring` at 5, `evaluation`, `ledger`,
`research`, `models.training` at 6) sits strictly below it and cannot import
it. Only `ops`/`serving` (7) and the `diagnosis` sink (7.5) can. `diagnosis`
still owns the comparator's public surface: it re-exports these modules and
names unchanged (the same objects, not a copy — the pattern `canonical.py`
already uses for `foundation`), so every `engine.v2.diagnosis...` import in
checks, tools and tests keeps working, and diagnosis stays imported by
nothing. The move changed no logic and no hashed payload.

## Responsibilities

- The record comparator, tolerance policies, stage plan and receipts
  (contracts §15), moved unchanged from `engine/v2/diagnosis`.
- The numeric field groups (`FORECAST_FIELDS`, `SIMULATION_FIELDS`,
  `FINANCIAL_FIELDS`, `GATE_FIELDS`, `ANALOG_FIELDS`) and the dimensions the
  "never ran" rule applies to (`NEVER_RAN_DIMENSIONS`).
- `compare_dimension`: one dimension's expected/actual views in, the
  checker's `{agree, finding_fields, receipt}` out, under `SCORE_RECORD_V1`.

## Non-responsibilities

- **Decide whether a difference is acceptable** — `a person, from the receipt`
  does it instead.
- **Repair the data it found wrong** — `the package that produced it` does it
  instead.
- **Block publication on a mismatch** — `nobody`: the nightly `native_parity`
  stage is optional and only records.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

| Name | What it is |
|---|---|
| `compare_dimension` | The per-dimension comparator, moved verbatim from `checks/phase4_real.py`. |
| `FORECAST_FIELDS`, `SIMULATION_FIELDS`, `FINANCIAL_FIELDS`, `GATE_FIELDS`, `ANALOG_FIELDS` | The checker's numeric field groups, by name. |
| `NEVER_RAN_DIMENSIONS` | The dimensions the "never ran" rule applies to (`simulation`, `verdicts`, `analogs`). |
| `dimensions`, `receipt`, `record_comparator`, `stage_plan`, `tolerance` | The modules; `engine.v2.diagnosis` re-exports the last four under its own names (see its README for the comparator API). |

<!-- public-interface: dimensions, compare_dimension, FORECAST_FIELDS, SIMULATION_FIELDS, FINANCIAL_FIELDS, GATE_FIELDS, ANALOG_FIELDS, NEVER_RAN_DIMENSIONS, receipt, record_comparator, stage_plan, tolerance -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

- engine/v2/ops — `native_parity_report.py` builds the nightly parity report
  from `compare_dimension` and `NEVER_RAN_DIMENSIONS`.
- engine/v2/diagnosis — re-exports the comparator core under its own module
  paths.

`checks/phase4_real.py` imports the same names under the old underscore
aliases, so its call sites and tests keep working unchanged.

<!-- consumers: engine.v2.ops, engine.v2.diagnosis -->

## Usage


```python
from engine.v2.parity.dimensions import compare_dimension

result = compare_dimension(
    {"gate_score": 0.7, "gate_threshold": 0.6, "gate_pass": True},
    {"gate_score": 0.7, "gate_threshold": 0.6, "gate_pass": True},
    "verdicts",
)
assert result["agree"] is True
```

## Testing

Tier 0. The comparator itself is proved by the Phase 4 negative controls
(`tests/test_phase4_planted_defect_control.py`,
`tests/test_phase4_numeric_negative_control_states.py`,
`tests/test_phase4_acceptance_independence.py`) and by the nightly report's
tests (`tests/test_v2_ops_native_shadow_render.py`), all through the real
`compare_records` path: a planted corruption must disagree, and a blind
comparator must be caught. The comparator core keeps its own tests
(`tests/test_diagnosis_comparator.py`), run through the diagnosis re-exports.
