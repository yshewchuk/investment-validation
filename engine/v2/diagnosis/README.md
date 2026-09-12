# `engine/v2/diagnosis`

## Ownership

Implements the **Validation/diagnosis — comparators, tolerance policies, stage plans, ComparisonReceipts** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**—** of §4.1.

Replaces (§4.4): `dashboard/selfcheck.py`, `the parity comparators`.

## Responsibilities

- Comparators that are stage-localized and complete, not first-wins (contracts §15).
- Tolerance policies, declared per field and never global.
- ComparisonReceipts and their tiers.

## Non-responsibilities

- **Decide whether a difference is acceptable** — `a person, from the receipt` does it instead.
- **Repair the data it found wrong** — `the package that produced it` does it instead.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

| Name | What it is |
|---|---|
| `compare_records` | The comparator: two ScoreRecord-shaped mappings in, one `ComparisonReceipt` out. Every independent finding in one pass. |
| `merge_receipts` | Folds per-pair receipts into one corpus-level receipt, with the population that makes a collapse visible. |
| `flatten` | A record to `{field_path: leaf}`, the form findings are named in. |
| `ComparisonReceipt`, `Finding`, `StageHashes`, `Population`, `Envelope` | The §15.2 shapes. |
| `AGREE`, `DIFFER`, `INCOMPARABLE` | The three verdicts. There is no fourth. |
| `StagePlan`, `SCORER_V1`, `load_stage_plan` | The §6.3 stage order, and which stage owns which field. |
| `Tolerance`, `TolerancePolicy`, `EXACT`, `SCORE_RECORD_V1` | Per-field tolerances. There is no global one. |
| `canonical_json`, `content_hash` | RFC 8785 canonical form and `sha256:` identity, per contracts §2.2. Moves to `engine/v2/foundation` when that package is written. |
| `problem` | The shared failure envelope of contracts §2.4. |

<!-- public-interface: compare_records, merge_receipts, flatten, ComparisonReceipt, Finding, StageHashes, Population, Envelope, AGREE, DIFFER, INCOMPARABLE, StagePlan, SCORER_V1, load_stage_plan, Tolerance, TolerancePolicy, EXACT, SCORE_RECORD_V1, canonical_json, content_hash, problem, canonical, receipt, record_comparator, stage_plan, tolerance -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

**None, ever.** `engine/v2/diagnosis` is a sink: it may read every layer's artifacts and no layer may import it, so a comparator can never become a dependency of the thing it compares (§4.1). `checks/import_layers.py` fails any import of this package.

<!-- consumers: none -->

## Usage

```python
from engine.v2.diagnosis import compare_records, load_stage_plan

receipt = compare_records(
    left=frozen_record,            # the tier-0 corpus record
    right=recomputed_record,       # whatever is being proved equal to it
    stage_plan=load_stage_plan("scorer.v1"),
    comparison_kind="serving_replay_parity",
    tier=0,
)
print(receipt.verdict)             # agree | differ | incomparable
for finding in receipt.findings:   # every independent finding, not the first
    print(finding.first_differing_stage, finding.field_path)
```

## Testing

Tier 0. `tests/test_phase0_negative_controls.py` is this package's negative
control and the phase's reason for existing: it seeds the five independent
2026-09-11 causes into one record at once and asserts **one** pass reports
**five** stage-named findings. It also asserts that a record compared against
itself is `agree`, that an empty population is `incomparable` rather than
`agree`, and that removing a field from the digest removes it from the
comparison with no edit to the comparator.
