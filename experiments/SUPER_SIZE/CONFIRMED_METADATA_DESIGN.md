# SUPER_SIZE confirmed-event metadata proof checkpoint

Status: proposed design-only refinement of [stage 2](STAGE2_DESIGN.md), before
implementation. This freezes a small metadata-proof boundary, not a panel
context, executable request, scientific amendment or data-read authorization.
The accepted H/HM/HMO dictionary, 33-candidate search, 5/3/1 clocks, grouped
eligibility, causal legacy comparator and all v1 identities remain unchanged.

## Purpose and non-authority

Historical implied means need the complete confirmed-event sequence from the
first known event in pinned metadata, including events before history_start.
Selected development events, H.n_prior, or a caller's first_known assertion
cannot prove that sequence. This slice returns a complete per-ticker metadata
index; a later qualified context selects the strict target/forecast prefix.
It neither returns a numerical contributor set nor calls an event eligible.

The supported source explicitly has **no formal coverage watermark**. Complete
manifest membership establishes only the first confirmed event *within that
dataset*. It cannot establish earliest real-world/source history, point-in-time
calendar knowledge, expected-population coverage or coverage of the proposed
evaluation years. Native coverage/availability receipt references are preserved
as references, not opened or promoted into a stronger attestation. Separate
coverage qualification remains mandatory before sealing or numerical use.

No provider, mutable legacy data, daily/computed-move values, outcomes, stored
aggregates, fitting, registration, ledger or production entrypoint is accessed.
Native resolution may inspect all snapshot tables' catalog metadata, not payloads.
Protected event metadata may retain ordinals; this confers no protected-value
access. The index is not an alternative holdout/grouped-membership classifier.

## Public boundary and native compatibility

Proposed additive API:
`magnitude_confirmed_metadata.build_index(repository, snapshot, *, tickers, limits)`.
Only synthetic native repositories consume the initial implementation. No CLI,
stored artifact reader, caching service or runtime caller is added in this slice.
Use native Repository/DataQuery, strict native document/ref decoding, standard
Python sorting, public event_sequence and foundation canonical hashing. A new
schema framework, registry, custom Parquet reader or legacy adapter is unnecessary.

`snapshot` is an exact native SnapshotRef, admitted through
`snapshot_inputs.admit_snapshot`; no ID-only/latest/path substitute. Use a fresh
request-scoped Repository with `pin_snapshots=True`; the caller owns this fresh
lifetime. The helper checks pins_snapshots is true, not the object's usage history.
Pinning reuses only that call's metadata, never a cross-call proof cache.
Native `resolve_full(snapshot.snapshot_id)` must reproduce the admitted ref.
From it obtain the earnings DatasetManifest, full TableContract and complete
ordered FragmentRecords; verify their native bindings, not caller-rehashed lists.
Do not follow parent manifests to reconstruct membership or enumerate storage.

The only supported earnings TableContractRef is:

```text
schema_version = table_contract_ref.v1.0
contract_id = legacy.earnings_events.v2
definition_hash = sha256:3cdea8601b2ec005dd9f079639998450c63502a41472fe0cdc6b6987ec47aeaf
```

This is the reviewed `table_contract.v2.0`, semantic 2.0.0: primary key
`(event_id,)`, partition `(year,)`, maximum batch 50,000. It is a native catalog
contract despite its legacy compatibility name, not permission to read legacy
files. Compare both the resolved ref and `manifests.table_contract_hash(contract)`
to this exact hash. Same columns/name with another definition, unknown versions
or changed null/time/provenance meanings refuse. Future compatibility requires a
reviewed versioned amendment, not a dynamically widened allowlist.

The initial synthetic compatibility test independently derives this full
definition with default `build_legacy_mapping()` and checks the pinned ref;
runtime must not rebuild all eight legacy mappings on each invocation. The
reviewed definition comes from schemas, annotations and native canonical hashing
at main d8a4fd0e; this is code evidence, not an observed stored-source inventory.

`tickers` is a nonempty built-in list of distinct nonempty built-in strings,
with no surrounding whitespace, NUL or invalid Unicode. No case/symbol rewrite,
iterator consumption, empty-list meaning "all", event-ID selection, history_start,
target date, forecast ceiling, exclusion list or caller first_known is accepted.
Sort ticker names lexicographically for canonical output; input order is immaterial.

## Complete scan and conflict semantics

1. Enumerate **every** year partition from the complete earnings manifest,
   including metadata before history_start and outside development years.
   Partition keys must be canonical four-digit years in 0001..9999. Missing or
   malformed partition metadata refuses; an absent year is not invented.
2. Preflight all partitions before the first object open. Use public
   `query.plan_scan_population` on native records and matching
   `Repository.scan_population_bound`, not estimated matching-row counts.
   Predicates are year equality plus ticker membership; no time interval or
   confirmation predicate can erase an earlier event. Candidate membership must
   exhaust the manifest partitions. Reject inconsistent native planning evidence.
3. Make one earnings `Repository.scan` per manifest partition, with those same
   predicates, `order_by=("event_id",)`, explicit native candidate-row bound and
   capped batch size. The projection is exactly `event_id`, `ticker`,
   `event_date`, `session`, `date_conflict`, `event_cluster_id`, `src_orats`.
   Repository may read its hidden year predicate column; no numerical field is
   projected. Supply max_object_bytes on every scan. Exhaust every stream before
   publishing any result; do not rescan per ticker, event or horizon.
4. Validate all selected metadata before confirmation filtering. event_id is
   an opaque nonempty string, not parsed/rebuilt from ticker/date. Dates must
   represent a nonnull midnight calendar day, serialized YYYY-MM-DD without
   truncating a time or converting zones. session is null/BMO/AMC only;
   src_orats/date_conflict are actual booleans; cluster is null or a nonempty
   string. Do not coerce strings/numbers to booleans or repair malformed rows.
5. Refuse duplicate event_id (even byte-identical repeats), duplicate
   (ticker,event_date), any true date_conflict, or a nonnull cluster associated
   with multiple event IDs/tickers/dates in the requested view. Detect conflicts
   across partitions and batches, including unconfirmed rows. Null clusters are
   permitted by this exact contract and do not assert cluster completeness.
   Do not import computed-move deduplication or prefer one session/source/date.
   This deliberately refuses ambiguous full views rather than silently choosing
   a target-specific prefix; it does not attest unrelated tickers' cluster state.
6. Confirm exactly when src_orats is true and session is BMO or AMC, matching
   the native src_orats-plus-nonnull-session definition after validation. For
   each ticker, sort confirmed rows by (event_date,event_id) and call public
   `event_sequence(dates, warmup=4)` once. Retain zero-based ordinals; its admitted
   flag means only ordinal>=4, never holdout, value or forecast admission. Expose
   that flag as warmup_complete to avoid falsely calling contributors admitted.

Empty ticker requests refuse before repository access. A valid table with no
fragments is a proven empty *dataset view*, not missing provenance. Every requested
ticker receives an entry: `absent_in_snapshot` if it has no selected rows,
`no_confirmed_events` if it has selected rows but none confirmed, otherwise
`confirmed_events`. Empty entries have null first_confirmed_date and events=[];
the latter two states also retain the selected/confirmed counts. None substitutes
for the later known-empty *qualified numerical prefix*. Missing table/ref,
incomplete membership, unsupported metadata or scan failure returns no index.

## Closed output and identity

Return one detached plain-JSON document, with exactly these top-level keys:
`schema_version`, `policy`, `source`, `tickers`, `sequences`, `counts`, `index_id`.
The schema literal is `magnitude_confirmed_metadata.v1`; this does not change
research_magnitude_panel.v1 or any existing dictionary/source closure.

`policy` has exactly `confirmation="src_orats_and_session.v1"`, `warmup_events=4`,
`conflicts="refuse_requested_view.v1"`, and
`coverage="complete_pinned_view_only.v1"`.
`source` has exactly `snapshot`, `table="earnings_events"`, `contract_ref`,
`dataset_manifest`, `fragment_records`, `columns` (the seven ordered names above).
Native nested objects use their existing strict to_document shapes/versions;
fragment_records is the complete native ordered membership, not only hits.
It retains object refs, hashes, recorded bytes/rows and receipt refs without
opening receipt payloads. Manifest coverage/availability refs and knowledge mode
are preserved verbatim; neither reconstructed nor observed means complete/PIT here.

`tickers` is the sorted request. `sequences` is a list in that same order. Each
entry has exactly `ticker`, `status`, `selected_rows`, `confirmed_rows`,
`first_confirmed_date`, `events`. Each event has exactly the seven projected
fields plus `ordinal` and `warmup_complete`; confirmed rows therefore retain
src_orats=true and date_conflict=false, rather than dropping their evidence.
`counts` has exactly `partitions`, `fragments`, `candidate_rows`, `selected_rows`,
`confirmed_rows`. The first three describe the scanned native candidate inventory,
the last two actual exhausted metadata rows, not label/quote contributions.

`index_id=content_hash(document without index_id)` over native canonical JSON.
Ordering, membership and source bindings matter; request order and scan batch
size do not. Execution limits/logs are not semantic source identity and are not
in this output. Changed valid source/ref, ticker membership or metadata changes
the ID. No wall-clock timestamp or caller-supplied digest blesses the output.
A rehashed detached document alone is not authenticated admission: a future
consumer must rebind it through native source admission, under its own authority.
No public validator that promises otherwise belongs in this slice.

## Explicit bounds and failures

`limits` is a built-in dict with exactly the following required built-in positive
integer values below sys.maxsize (booleans, defaults, nulls and coercion refuse):

| Key | Enforced bound |
| --- | --- |
| max_tickers | Request cardinality before repository access. |
| max_partitions | Complete earnings manifest partition count before query expansion. |
| max_fragments | Complete earnings manifest fragment count before query expansion. |
| max_scan_rows | Sum of native candidate row bounds across all partition scans. |
| max_object_bytes | Each recorded object byte_size before scans; raw verified bytes and expanded Parquet bytes through Repository.scan. |
| max_total_object_bytes | Sum of candidate object byte_size per fragment/open, not deduplicated by object ID. |
| max_batch_rows | Effective ceiling min(this value, contract.maximum_batch_rows, positive partition candidate bound); for a zero bound retain a positive batch size. Never increase the caller's cap. |
| max_retained_rows | Cumulative selected rows, including unconfirmed rows, before retaining. |
| max_retained_metadata_bytes | Cumulative canonical UTF-8 bytes of normalized seven-field selected rows, including unconfirmed rows, before retaining. |
| max_proof_bytes | Canonical UTF-8 bytes of the complete returned document, including all source metadata and index_id. |

Check the complete manifest/preflight totals before any metadata payload scan,
not only the next partition. Check retained row/byte growth incrementally and
count proof bytes with foundation canonical streaming before return; no truncated
sequence, dropped ticker, auto-raised cap, silent fallback or automatic retry.
Zero candidate rows use native max_result_rows=0 with positive max_batch_rows.
Caps bound logical scan/retention/output, not Python object overhead, RSS, total
expanded bytes across objects or initial native catalog resolution allocations.
The existing resolver materializes full snapshot metadata before these checks;
hard catalog-transfer/process containment is not claimed and remains a separate
execution boundary. Synthetic cap values are not authorized real-data budgets.
Use existing bounded-cadence progress logs: stage start/end, done/upper-bound,
elapsed/rate/ETA, refusals and stage RSS; no per-event logging or extra scans.

| Failure | Registered nonretryable code unless native code says otherwise |
| --- | --- |
| Malformed request/row/cap shape or pins_snapshots is not true | CONTRACT_MISMATCH |
| Missing/unresolvable/nonexact SnapshotRef | SNAPSHOT_UNRESOLVED via admit_snapshot |
| Absent earnings table | CONTRACT_MISMATCH |
| Unsupported exact contract/ref | UNSUPPORTED_CONTRACT |
| Explicit cap exceeded | RESULT_LIMIT_EXCEEDED |
| Semantic duplicate/date/cluster conflict | IDENTITY_CONFLICT |
| Native integrity/query/object/availability failure | Preserve native DataError code and retryability; no relabeling as empty. |

Native duplicate-primary-key detection can raise MANIFEST_CORRUPT before the
semantic conflict check; preserve it. Do not catch all native errors as generic
CONTRACT_MISMATCH. Every failure is all-or-nothing, with redacted messages and no
values, paths, partial index, artifact/catalog writes or contributor claims.

## Acceptance and small implementation sequence

1. This design-only checkpoint: <=300 added documentation lines, no runtime or
   source reads. It is a proposal for review, not a completed SIZE experiment.
2. Add the metadata index plus synthetic native-repository tests and owning
   research README/ARCHITECTURE contract. Target 220–320 non-test code lines,
   250–400 test lines and <=60 doc additions; architecture growth must obey its
   current ceiling. Split add/admit further rather than exceed repository limits.
3. Separately prove history/grouped-eligibility compatibility, forecast/label
   bounds and source coverage; then design/wire v2 context and numerical reads.
   No forecast/label producer declaration or metadata count replaces those proofs.

Acceptance fixtures cover the exact full contract hash and same-column impostors;
changed snapshot/dataset/fragment/object identity; all manifest years including
pre-history_start/protected metadata; absent/no-confirmed/zero-fragment cases;
duplicates/conflicts across batches/partitions; strict dates/booleans/session;
ordinal warmup and deterministic detached replay; and every cap at/over boundary.
Poison non-earnings payload scans/object reads and provider/writer/fit hooks;
assert zero accesses, allowing native all-table catalog metadata resolution.
Assert one scan per manifest partition and equal indices over batch sizes/request
orders; a late preflight overage must open no objects, and a late streaming
integrity/retention failure must return no partial proof. Inspect source closures
to prove all existing v1 identities unchanged. No fabricated source availability,
historical coverage, protected numerical admission or real-data execution claim.
