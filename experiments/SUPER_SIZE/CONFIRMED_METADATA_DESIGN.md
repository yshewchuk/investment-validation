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
First add `data.confirmed_event_metadata.classify_metadata(events)`, a public facade over
`computed_moves._validate_inputs` and `_event_reasons`, then consume that facade
from research. It validates with empty price sequences and min_scoreable=0 and
returns a tuple of native per-row reasons, without collapse/order/build/price access.
Do not copy their date/bool/session checks or confirmation predicate. The facade
is additive; existing computed_moves and all v1 source-closure files stay unchanged.

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
   The only predicate is year equality: scan all tickers to qualify cluster
   siblings before selecting requested tickers. No time/confirmation filter or
   ticker pruning may hide a sibling. Candidate membership must exhaust the
   manifest partitions. Reject inconsistent native planning evidence.
3. Make one earnings `Repository.scan` per manifest partition, with those same
   predicates, `order_by=("event_id",)`, explicit native candidate-row bound and
   capped batch size. The projection is exactly `event_id`, `ticker`,
   `event_date`, `session`, `date_conflict`, `event_cluster_id`, `src_orats`.
   Repository may read its hidden year predicate column; no numerical field is
   projected. Supply max_object_bytes on every scan. Exhaust every stream before
   publishing any result; do not rescan per ticker, event or horizon.
4. Validate all scanned metadata before ticker/confirmation filtering. event_id is
   an opaque nonempty string, not parsed/rebuilt from ticker/date. Dates must
   represent a nonnull midnight calendar day, serialized YYYY-MM-DD without
   truncating a time or converting zones. session is null/BMO/AMC only;
   src_orats/date_conflict are actual booleans; cluster is null or a nonempty
   string. Do not coerce strings/numbers to booleans or repair malformed rows.
5. Retain the bounded global metadata view until conflict qualification finishes.
   Refuse duplicate event_id, including byte-identical repeats across years.
   For requested tickers refuse duplicate (ticker,event_date), true date_conflict,
   or any nonnull cluster shared with another event ID/ticker/date anywhere in
   the pinned dataset, including unrequested tickers and unconfirmed rows.
   A sibling encountered before its requested event counts equally. Null clusters
   are permitted by this contract and do not assert cluster completeness.
   Do not import computed-move deduplication or prefer one session/source/date.
   This deliberately refuses ambiguous full views rather than silently choosing
   a target-specific prefix. Unrelated clusters confer no qualification claim.
6. Confirm exactly when src_orats is true and session is BMO or AMC, matching
   the facade's native reason=None after validation, not a second predicate. For
   each ticker, sort confirmed rows by (event_date,event_id) and call public
   `event_sequence(dates, warmup=4)` once. Retain zero-based ordinals; its admitted
   flag means only ordinal>=4, never holdout, value or forecast admission. Expose
   that flag as warmup_complete to avoid falsely calling contributors admitted.

Empty ticker requests refuse before repository access. A valid table with no
fragments is a proven empty *dataset view*, not missing provenance. Every requested
ticker receives an entry: `absent_in_snapshot` if it has no selected rows,
`no_confirmed_events` if it has selected rows but none confirmed, otherwise
`confirmed_events`. Only absent_in_snapshot and no_confirmed_events have
first_confirmed_date=null and events=[]; confirmed_events has a nonnull first
date and nonempty events. All three statuses retain selected/confirmed counts.
None substitutes
for the later known-empty *qualified numerical prefix*. Missing table/ref,
incomplete membership, unsupported metadata or scan failure returns no index.

## Closed output and identity

Return one detached plain-JSON document, with exactly these top-level keys:
`schema_version`, `policy`, `source`, `tickers`, `sequences`, `counts`, `index_id`.
The schema literal is `magnitude_confirmed_metadata.v1`; this does not change
research_magnitude_panel.v1 or any existing dictionary/source closure.

`policy` has exactly `confirmation="src_orats_and_session.v1"`, `warmup_events=4`,
`conflicts="global_requested_cluster_qualification.v1"`, and
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
`counts` has exactly `partitions`, `fragments`, `candidate_rows`, `scanned_rows`,
`selected_rows`, `confirmed_rows`. The first three describe global scan inventory;
scanned_rows counts all exhausted metadata rows, while the final two count only
requested-ticker rows. No count denotes label/quote contributions. Output event
sequences contain only requested tickers, never their unrequested cluster siblings.

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
| max_batch_rows | min(this value, contract.maximum_batch_rows, max(1, partition candidate bound)); zero-bound scans therefore use exactly 1. Never increase the caller's cap. |
| max_retained_rows | All globally scanned rows, including unrequested/unconfirmed rows, before retaining for conflict qualification. |
| max_retained_metadata_bytes | Cumulative canonical UTF-8 bytes of all normalized seven-field global rows, including unrequested/unconfirmed rows, before retaining. |
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

STAGE2 section 5 remains the authoritative roadmap; these labels refine its
rows 1, P and 2 (and section-1 parity P1/P2), not a replacement numbering.
This design-only checkpoint adds no runtime or source reads. Each proposed code
PR below targets <=400 combined added code/test/doc lines, stricter than the
repository's separate category caps; re-estimate and split add/wire again if
needed. Estimates are not waivers, completed work or real-data authorization.

| Sub-slice / dependency and acceptance | Added code / tests / docs |
| --- | --- |
| 1.M0: native metadata facade / unchanged v1 closure, existing validator/predicate parity | 30–50 / 70–100 / 10–20 |
| 1.M1: pure global conflict/sequence reducer / 1.M0; cross-ticker siblings and empty states, no admission claim | 100–140 / 150–190 / 20–30 |
| 1.M2: native ref and full-inventory preflight / exact contract, complete membership, all-cap refusal before scans | 100–140 / 150–190 / 20–30 |
| 1.M3: bounded global scan and index composition / 1.M1/1.M2; exhaust once per partition, detached output and poisoned payload tests | 100–140 / 150–190 / 20–30 |
| P.M1: native history/grouped-policy compatibility proof / 1.M3; contaminated histories refuse, no post-aggregate filtering | 100–140 / 150–190 / 20–30 |
| 1.C0: coverage/forecast/label qualification design / 1.M3; freeze available native evidence and source-specific bounds before code | 0 / 0 / 120–160 |
| 1.C1: native coverage-reference admission / 1.C0; authenticate reviewed evidence or refuse, never infer coverage from metadata alone | 90–130 / 150–190 / 20–30 |
| 1.C2: forecast/label-bound admission / 1.C0; prove clocks independently of producer declarations | 100–140 / 150–190 / 20–30 |
| 1.C3: closed v2 context declaration / P.M1,1.C1,1.C2; source-free shape/identity, no admission claim | 100–140 / 150–190 / 20–30 |
| 1.C4: native v2 context composition / 1.C3; bind independently qualified sources and exact history | 100–140 / 150–190 / 20–30 |
| 2.M1: causal numerical query plan / 1.C4 and qualified sources; immutable ceilings and bounded read sets, no values | 90–130 / 140–180 / 20–30 |
| 2.M2: stored historical-quote reader / 2.M1; protected/future poisoning and source anchors | 100–140 / 150–190 / 20–30 |
| 2.M3: quote aggregate and absolute derivation / 2.M2; native math, zero/null semantics and no ordinal restart | 90–130 / 140–180 / 20–30 |
| 2.M4: row/lineage materialization wiring / 2.M3; complete horizon rows, bounded output, no experiment execution | 100–140 / 150–190 / 20–30 |

Only 1.M3 exposes build_index; earlier pieces do not assert authenticated index
admission. Each public-interface slice updates owning README/ARCHITECTURE without
growing an architecture doc already at its ceiling. 1.C0 may block or require a
new reviewed source-proof design; 1.C1/1.C2 estimates do not invent available evidence.
Other STAGE2 work remains separate. No metadata count replaces contributor proof.

Acceptance fixtures cover the exact full contract hash and same-column impostors;
changed snapshot/dataset/fragment/object identity; all manifest years including
pre-history_start/protected metadata; absent/no-confirmed/zero-fragment cases;
duplicates/conflicts across batches/partitions, including requested/unrequested
cluster siblings in either encounter order and unconfirmed siblings;
strict dates/booleans/session with the existing native facade as the oracle;
ordinal warmup and deterministic detached replay; and every cap at/over boundary.
Poison non-earnings payload scans/object reads and provider/writer/fit hooks;
assert zero accesses, allowing native all-table catalog metadata resolution.
Assert one scan per manifest partition and equal indices over batch sizes/request
orders; a late preflight overage must open no objects, and a late streaming
integrity/retention failure must return no partial proof. Inspect source closures
to prove all existing v1 identities unchanged. No fabricated source availability,
historical coverage, protected numerical admission or real-data execution claim.
