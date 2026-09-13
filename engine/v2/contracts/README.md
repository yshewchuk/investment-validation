# `engine/v2/contracts`

## Ownership

Implements the **schemas and types only, no logic, no I/O** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**0** of §4.1.

Replaces (§4.4): `the dataclasses currently declared inside score.py`.

## Responsibilities

- Declare every named type in component_contracts.md, with its kind suffix (§2.5) and its schema version.
- Define the shared failure envelope (§2.4) and the reason-code vocabulary (§9.4).

## Non-responsibilities

- **Compute anything** — `every package above it` does it instead.
- **Touch the filesystem, a clock or a network** — `engine/v2/foundation` does it instead.
- **Validate a document** — `engine/v2/foundation` (`from_document`) does it instead, driven by these annotations.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

| Module | Names |
|---|---|
| `jobs` | `JobSpec`, `SubmitRequest`, `JobReceipt`, `CancellationReceipt`, `StageSpec`, `ArtifactRef`, `LegacyFileRef`, `LegacyInputManifest`, `AttemptReceipt`, `OutputCandidate`, `CheckpointCandidate`, `CheckpointReceipt`, `StageResult`; vocabularies `JobState`, `AttemptState`, `ProcessState`, `EffectClass`; the `*_V1` schema versions. |
| `operations` | `Problem`, `FAILURE_CODES`, `QueueReason`, `ProgressEvent`, `ResourceProfile`, `ResourcePolicy`, `LiveWindow`, `CapacitySample`, `ResolvedResources`, `ProcessIdentity`; vocabularies `ProblemCategory`, `ExecutorMode`, `Containment`, `ProgressKind`; the `*_V1` schema versions. |

<!-- public-interface: jobs, operations, JobSpec, SubmitRequest, JobReceipt, CancellationReceipt, StageSpec, ArtifactRef, LegacyFileRef, LegacyInputManifest, AttemptReceipt, OutputCandidate, CheckpointCandidate, CheckpointReceipt, StageResult, JobState, AttemptState, ProcessState, EffectClass, ARTIFACT_REF_V1, ATTEMPT_RECEIPT_V1, CANCELLATION_RECEIPT_V1, CHECKPOINT_RECEIPT_V1, JOB_RECEIPT_V1, JOB_SPEC_V1, LEGACY_INPUT_MANIFEST_V1, STAGE_RESULT_V1, STAGE_SPEC_V1, SUBMIT_REQUEST_V1, Problem, FAILURE_CODES, QueueReason, ProgressEvent, ResourceProfile, ResourcePolicy, LiveWindow, CapacitySample, ResolvedResources, ProcessIdentity, ProblemCategory, ExecutorMode, Containment, ProgressKind, CAPACITY_SAMPLE_V1, PROBLEM_V1, PROGRESS_EVENT_V1, RESOLVED_RESOURCES_V1, RESOURCE_POLICY_V1 -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

- `engine/v2/foundation` — `ArtifactRef`, returned by the artifact store.

- engine/v2/ops — durable job, attempt, resource, checkpoint and progress documents.
<!-- consumers: engine.v2.foundation, engine.v2.ops -->

## Usage

```python
from engine.v2.contracts import JobSpec
from engine.v2.foundation import from_document, to_document

spec = JobSpec(kind="synthetic.echo", implementation_ref="impl", spec_hash=None,
               environment_ref="env", output_namespace="dev", resource_class="io_fetch",
               retry_policy_ref="retry.none", checkpoint_contract_ref="ckpt.none")
assert from_document(JobSpec, to_document(spec)) == spec
```

## Testing

Tier 0, in `tests/test_v2_ops_contracts.py`. The package's own rules are checked
against its source: imports limited to `dataclasses`/`typing`, no function
anywhere (a method is where a hash-on-construction would hide), every type
frozen and keyword-only, one schema family per type, and every contract
round-tripping exactly through the strict decoder.

A negative control here looks like: set `effect_class` to a value outside the
vocabulary and assert the decoder refuses it with `BAD_ENUM` at that path.
