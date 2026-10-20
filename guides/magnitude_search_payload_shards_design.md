# Magnitude-search cohort inputs as content-addressed shards (#991 fix 4)

Status: proposed design; sections 2-5 are the contract the implementation PRs
adopt. `experiments/ARCHITECTURE.md` describes only current behaviour, so it is
unchanged here: split PRs 2-4 each update it (and `experiments/README.md`, which
states the 16 MiB child transport) in the same PR that changes the behaviour. No
code in this PR.

## 1. Problem

A cohort (plan, outer year, inner year) is admitted once (`admit_fold_inputs`) as a
dict: `preflight` (with `planned_inputs`, one entry per participant), `stored_inputs_id`,
`training_inputs` (qualification attestation), `training` rows and `features` rows
(event x horizon). Three places serialize it whole and cap it at
`MAX_BYTES` = 16 MiB:

1. `SearchTrials._start` appends it as the journal record `cohort:<id>`
   (`append_record`, `magnitude_search_records.py`).
2. `run_fitted_trial` builds one canonical payload for the child
   (`magnitude_search_execution.py`) and the worker reads at most `MAX_BYTES + 1`.
3. Every replay/reconcile/outer path re-reads `cohort:<id>` whole and hashes it.

Measured ~2.3 KB per event for the projected fit payload (494 KB / 215 events), so the
cap trips near 7k events; the full universe is ~118k events (~270 MB projected, more
for the journal copy). Raising the cap only moves the wall; it also keeps the whole
cohort in one Python dict in three processes. Fix 4 replaces the blob by shards.

Non-goals: the parent's admission-time memory (fix 5; this format only guarantees it
*can* be written and read streaming), de-duplicating shards across folds, changing
any scientific rule (cutoffs, row order, horizons, features, fit pipeline).

## 2. Shard format and addressing

* A **section** is an ordered row list: `features`, `training`, `planned_inputs`.
  Row order is the admitted order and is never sorted (fit order and matrix evidence
  depend on it). Non-row members (`stored_inputs_id`, `training_inputs`, preflight
  identity fields) stay inline in the manifest; any list-valued member that grows with
  event count is made a section (implementation PR 1 inventories `training_inputs.attrs`
  and the preflight object and states the result in the contract).
* A **shard** is the canonical JSON `{"schema_version": "magnitude_search_shard.v1",
  "section": <name>, "rows": [...]}` published with the existing
  `ArtifactStore.publish_bytes(schema_ref="magnitude_search_shard.v1")`. Address =
  the returned `ArtifactRef` (sha256 content hash, byte size, storage key). The body
  carries no cohort, run or position, so identical bytes have one object.
* **Cut rule (deterministic, versioned):** walk the section in order; close the
  current shard before the row that would take its canonical size over
  `SHARD_TARGET_BYTES` (4 MiB); a shard is never empty; a single row whose own shard
  would exceed `SHARD_MAX_BYTES` (8 MiB) refuses. Same rows give the same shards, so
  retries republish identical objects and the same manifest. Changing the rule is a
  new `schema_version` (new seals), never an in-place change.
* **Manifest:** the journal record `cohort:<id>` becomes
  `{"schema_version": "magnitude_search_cohort.v2", "scalars": {...}, "sections":
  {name: {"row_count": n, "shards": [{"ref": <ArtifactRef doc>, "rows": k,
  "first_key": [...], "last_key": [...]}]}}}`. Keys are the section's primary key
  (`event_id`, or `event_id, horizon_sessions`). `SHARD_MAX_COUNT` (8192) bounds the
  manifest (<= ~4 MB, under the 16 MiB record cap) and the cohort (<= 64 GiB at the 8 MiB hard cap, typically ~32 GiB at the target).
* **`inputs_id` = `content_hash(manifest)`** (Merkle root). The existing
  `inputs:<trial>` record, execution results and terminals keep their field names and
  bind this id; nothing else about their shape changes.

## 3. Reading and verification

Both consumers use one streaming reader, `iter_section(store, manifest, section)`:
for each manifest shard in order, `read_verified(ref)` (size is the allocation
authority, hash checked over the returned bytes), decode, check `schema_version`,
`section`, `rows == len(rows)`, first/last key against the manifest, and yield rows.
At the end `sum(rows) == row_count`. Nothing is trusted before its hash passes; a
manifest ref is admitted only with the shard schema, `0 < byte_size <= SHARD_MAX_BYTES`
and `safe_relative_path(storage_key)` (the journal's existing `_reference` rules).

* **Parent** (`_record`, reconcile, freeze, outer replay): verifies the manifest hash
  against `inputs_id`, then streams shards to recompute `fit_evidence` digests and
  validate targets. Replay reads never hold the whole cohort (the admission-time memo
  is covered by R2). Matrix digests keep today's order: rows are looked up by
  `(event, horizon)` and assembled in the training / evaluation list orders, so the
  canonical matrix documents and hashes are unchanged.
* **Child**: the request file shrinks to a header `{inner, plan, execution, cohort
  manifest, store_root}` (still bounded by `MAX_BYTES`). The worker constructs a
  read-only `ArtifactStore(store_root)` (the constructor creates nothing) and calls
  the same reader, projecting each row (events = training + evaluation, selected
  horizons, `execution.features` columns; today's `_fit_inputs`) before the next shard
  is read. The child still opens no repository, catalog or validation target; it can
  read only objects whose hashes the header names. The stdin/stdout ACK protocol and
  the `MAX_BYTES` bound on a child output line are unchanged.
* **Bounded memory:** transport peak is one shard (<= 8 MiB bytes, a few times that
  decoded) plus the manifest. The fit working set (row index and matrices) is still
  O(rows); it is bounded by the existing `RLIMIT_AS` `memory_bytes` and surfaces as
  `RESOURCE_LIMIT_EXCEEDED`, not by transport. Reducing that working set (columnar
  arrays instead of one dict per row) is fit-pipeline work, outside this design;
  PR 4 (the first with the streaming child) reports the measured child RSS at scale so the user can size `memory_bytes`.

## 4. Failure semantics (R1-R6)

| R | Outcome |
|---|---|
| R1 verified input | Admission order unchanged. Writer cuts rows, publishes shards, then appends the manifest. Reader checks manifest hash vs `inputs_id`, then each shard's size and hash before decode. Hash/size mismatch, wrong schema/section, row-count or key-boundary disagreement, duplicate `(event, horizon)` key, unsafe or malformed ref: `EXPERIMENT_IDENTITY_CONFLICT`. A `cohort:` record in the old v1 shape: `EXPERIMENT_IDENTITY_CONFLICT` (sealed to older closures; no auto-migration). |
| R1 unusable input | A shard object missing, unreadable or `OSError`: retryable `RESOURCE_UNAVAILABLE` through the journal's existing artifact mapping; no association change, no attempt-retry authorization. A row larger than `SHARD_MAX_BYTES` or more than `SHARD_MAX_COUNT` shards: nonretryable `RESOURCE_LIMIT_EXCEEDED` before the manifest is appended; already published shards stay unreferenced. |
| R2 cache | None persisted. The per-instance cohort memo (one entry per cohort, as in the current contract) holds the admitted rows from admission until fix 5 streams admission; from then on it holds only the manifest. Until then parent peak memory is admission-bound, not transport-bound; never a failed admission. Shards are immutable and never invalidated; a cohort's bytes are never rewritten. |
| R3 retry | Publishing a shard is idempotent (`publish_bytes` links or re-verifies an existing object). The immutable `start:` claim is committed before admission, so a crashed `_start` cannot be re-run (unchanged: unfinished claim, selection blocked). Only a first call whose shards were published but whose manifest append failed *within the same call* can be completed by publishing again, because the cohort slot, not the claim, is the commit point; across a crash the orphan shards are inert and the trial stays unfinished. A different trial of the same cohort, whose claim is new, reaches the shard writer, republishes identical shards and appends or replays the same manifest. Child shard reads keep the reader's classification: integrity failures (hash, schema, section, count, key) are `EXPERIMENT_IDENTITY_CONFLICT`, missing/`OSError` is `RESOURCE_UNAVAILABLE`, memory is `RESOURCE_LIMIT_EXCEEDED`; the worker gains a distinct integrity event for this, never a candidate failure. |
| R4 transaction | Shards are published before, and outside, the catalog transaction (no artifact I/O under the lock). The single `INSERT` of the `cohort:<id>` reference is the commit point; the manifest is visible only after every shard it names is durable. |
| R5 partial failure | Shards without a manifest are unreferenced and confer no authority; the store owner may reclaim them (agents never hand-delete under a store). A manifest whose shard later vanishes or corrupts fails closed per R1. No partial cohort is ever readable. |
| R6 idempotency | Same rows give byte-identical shards (deterministic cut) and the identical manifest, so `append_record` replay returns the committed record. A different manifest for the same `cohort:<id>` slot is `EXPERIMENT_IDENTITY_CONFLICT` (input drift is reported exactly as before). |

## 5. What the registration, seal and anchor bind

* Registration binds the snapshot, plan, event IDs and execution spec; it never held
  cohort rows and is unchanged. The anchor (`inputs:<trial>` -> `{cohort, inputs_id}`)
  now binds the manifest hash, which transitively binds every shard hash and size.
* The recorder/outer source closures gain the shard module, so seals change; old
  sealed runs fail closed with no rewriting (the same precedent as the slim-cohort
  change). `v2_magnitude_search_run` seals and `collect` are unchanged in shape.
* **Replay identity:** `inputs_id`, journal IDs and results change for every cohort
  (Merkle id), so a replay compares like with like only within one format version.
  Within it, replay is exact: same snapshot, plan and code give the same shards, manifest
  and id; native replay still launches nothing and opens no repository or labels, only
  the already-published shards.

## 6. Options and recommendation

* **D1 - cohort identity.** (A, recommended) Merkle `inputs_id` over the manifest: cheap
  to verify, trivially streamed, but changes IDs, so existing sealed journals (smoke
  runs) do not replay. (B) Keep `inputs_id = content_hash(full dict)` by streaming
  `iter_canonical_json` fragments across shards: old IDs stay valid for small cohorts,
  but replay re-reads everything to verify the id and the shard cut becomes coupled to
  the canonical encoder's key order. The repo has already accepted A once (slim cohorts).
* **D2 - small cohorts.** (A, recommended) Always shard (one format, one path; a small
  cohort is one shard per section). (B) Inline below a threshold: keeps v1 for the smoke
  scale but doubles the code paths and the tests.
* **D3 - outputs.** Forecasts (`execution-result:` record, child output) and
  `validation:<cohort>` targets also grow with evaluation events; at full scale they
  may approach the cap after the inputs are fixed. Records that grow with the *training*
  population also remain: outer terminals inline `training_inputs` (candidate,
  qualified, read and omitted event lists) and execution receipts repeat event lists per
  horizon. Recommend a measured decision after PR 4 (the first PR with the streaming
  child): if the largest fold's forecasts, validation targets or any such receipt exceed
  ~4 MiB, add the sharded-output PR below (receipt lists included); otherwise leave them
  under the 16 MiB record cap.

## 7. Split list (add, then wire, then remove)

Estimates are added lines: code / tests / docs.

1. **Shard module (add; ~350 / ~250 / ~40, no wiring).** Cut rule, writer,
   manifest builder, verified streaming reader, failure mapping, inventory of
   list-valued cohort members. Tests: determinism, boundary keys, corruption, vanished
   shard, oversize row, shard-count cap. Docs: the new module's section in
   `experiments/ARCHITECTURE.md` (format, cut rule, reader).
2. **Consumers first (~300 / ~200 / ~30).** `_record`, reconcile, freeze, outer,
   `_validation` and `fit_evidence` read cohort rows through one loader that streams
   `iter_section` for a v2 manifest and adapts the existing v1 whole record, with the
   v1 behaviour and ids unchanged (no activation yet). Matrix evidence keeps today's
   order: the loader yields rows keyed by `(event, horizon)` and `fit_evidence` still
   assembles `row_keys` and values in the training / evaluation list orders, hashing
   the same canonical matrix documents. Docs: replay and outer rows.
3. **Activate the format (~300 / ~200 / ~40).** `_start` publishes shards + v2
   manifest; `inputs_id` = manifest hash (including the execution-result binding);
   v1 `cohort:` records refuse; closure/seal update. Docs: journal and cap rows, seal
   note.
4. **Child transport (~250 / ~150 / ~40).** Header request, worker streaming reader and
   projection, parent `_fit_inputs` and the 16 MiB input check removed. Reports
   measured child RSS (supervisor-run scale job; agents run no heavy jobs). Docs:
   execution table and `experiments/README.md` transport line.
5. **Cleanup (remove; net negative code / ~50 / ~20).** Delete dead v1 whole-cohort
   paths, superseded doc lines and this guide (its content now lives in
   `experiments/ARCHITECTURE.md`).
6. **Conditional: sharded outputs** (D3; ~300 / ~200 / ~30), same reader/writer for
   forecasts and validation targets, with chunked child output lines.

Fix 5 (parent memory) reuses PR 1's writer with an iterator source and is independent
of PRs 2-4. A full-universe run needs fixes 1-3 (merged), PRs 2-4, and fix 5 if
measured parent RSS is too high.
