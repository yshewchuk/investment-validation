# SUPER_SIZE confirmed-event metadata proof checkpoint

Status: proposed design-only refinement of [stage 2](STAGE2_DESIGN.md), before
implementation. This freezes a small metadata-proof boundary, not a panel
context, executable request or data-read authorization. The October 26 population
amendment below changes the proposed population claim and removes the
source-wide population-coverage qualification dependency, pending design review.
Numerical, holdout and timing policies remain unchanged; this is a design-policy
change, not runtime activation.
The accepted H/HM/HMO dictionary, 33-candidate search, 5/3/1 clocks, grouped
eligibility, causal legacy comparator and all v1 identities remain unchanged.

## Purpose and non-authority

Historical implied means need the complete confirmed-event sequence from the
first known event in pinned metadata, including events before history_start.
Selected development events, H.n_prior, or a caller's first_known assertion
cannot prove that sequence. This slice returns a complete per-ticker metadata
index; a later qualified context selects the strict target/forecast prefix.
It neither returns a numerical contributor set nor calls an event eligible.

### Frozen-dataset population amendment (October 26; design only)

Owner choice: the population is **all eligible confirmed events within the
pinned frozen dataset**, not all events captured by a provider or occurring in
the real world. Eligibility still uses the unchanged native clock, holdout and
grouped policies; confirmation alone is not eligibility. Metadata history,
first_confirmed_date, ordinals and counts are dataset-relative. Preserve the
complete manifest inventory and exhaust every partition, including events before
history_start; selected development rows or caller assertions cannot replace it.

The supported source explicitly has **no formal coverage watermark**. This is
an acknowledged source limitation, not a blocker requiring source-wide coverage
proof for this frozen-population experiment. Complete pinned membership does not
prove provider/real-world completeness, the earliest real-world event,
point-in-time publication knowledge, evaluation-year representativeness or
statistical unbiasedness. Absence is not evidence of a missing real-world event.
Report duplicates/conflicts, observed gaps and coverage patterns within the frozen
dataset; do not invent expected events, fill gaps or silently change eligibility.
Retain strict identity/ref integrity and existing duplicate/conflict refusals.

Native coverage/availability receipt references remain authenticated references,
not authenticated payloads or stronger attestations. No new evidence reader is
required solely to certify population completeness outside the frozen dataset.
Numerical use still requires independently qualified availability, forecast and
label bounds, grouped/holdout admission, and evaluation-year feasibility. The
1.C0 boundary below specifies the remaining work; this amendment neither solves
label provenance nor authorizes a read, fit, seal or runtime-policy change.

No provider, mutable legacy data, daily/computed-move values, outcomes, stored
aggregates, fitting, registration, ledger or production entrypoint is accessed.
Bounded native resolution may inspect all snapshot tables' metadata, not payloads.
Protected event metadata may retain ordinals; this confers no protected-value
access. The index is not an alternative holdout/grouped-membership classifier.

## Public boundary and native compatibility

**Blocking prerequisite (1.B0 plus B1a/B1b/B2/B3a/B3b):** current Repository.resolve/resolve_full
materialize all-table membership before caps; they cannot admit this index.
Before any admit_snapshot/resolve call, a separately reviewed data-owned bounded
read session must cap all-snapshot table count, fragment memberships and catalog
metadata bytes. Bounded SQL count/limit probes precede row transfer. SQL byte/type
checks precede payload transfer; decoded text's UTF-8 size is rechecked before
JSON parsing. Both cumulative byte counters include IDs, contracts and evidence.
Preflight and native identity reconstruction use one consistent read view; no
check-then-unbounded-resolve race. Reuse that admitted view for all later native
lookups/scans. Native manifest/snapshot reconstruction still proves exact identity.
No research-side SQL, caller receipt/boolean or ordinary pinned Repository can
substitute. 1.B0 must freeze/prove the public API and an additive implementation
route preserving existing bytes; index implementation waits for B0 review and all five code slices.

Proposed signatures below are design vocabulary, not installed/exported APIs:
`engine.v2.research.magnitude_confirmed_metadata.build_index(repository, snapshot, *, tickers, limits)`.
Initial implementation is synthetic-only. No CLI, stored artifact reader, cache
or runtime caller is added by this checkpoint.
Use the bounded native session/DataQuery, strict native ref decoding, standard
Python sorting, public event_sequence and foundation canonical hashing. A new
schema framework, registry, custom Parquet reader or legacy adapter is unnecessary.
First add `engine.v2.data.confirmed_event_metadata.classify_metadata(events)` over
`computed_moves._validate_inputs` and `_event_reasons`, then consume that facade
from research. It validates with empty price sequences and min_scoreable=0 and
returns a tuple of native per-row reasons, without collapse/order/build/price access.
The facade requires a pandas DataFrame with unique column labels and required
event_date/session/src_orats; shape failures raise CONTRACT_MISMATCH before native guards.
Do not copy their date/bool/session checks or confirmation predicate. The facade
is additive; existing computed_moves and all v1 source-closure files stay unchanged.
These are internal calls within engine.v2.data, not research private imports
(`checks/package_readmes.py:131–132`); declare only the facade in data's README.

`snapshot` is an exact native SnapshotRef, admitted through
`snapshot_inputs.admit_snapshot` through the required bounded session, never an
ordinary Repository; no ID-only/latest/path substitute. The data-owned factory
owns the request-scoped lifetime and bounds before resolution, not the caller.
Its bounded resolve_full must reproduce the admitted ref without another raw walk.
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

Synthetic compatibility derives this ref with default build_legacy_mapping();
runtime must not rebuild all eight mappings. Its schemas/annotations/native hash
at main d8a4fd0e are code evidence, not an observed stored-source inventory.

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
   the bounded session's scan_population_bound, not matching-row estimates.
   Predicates are year equality and requested-ticker membership. No time or
   confirmation filter may hide an earlier event/same-ticker sibling. Candidate
   membership exhausts the manifest partitions; inconsistent evidence refuses.
3. Make one bounded-session earnings scan per manifest partition, with those same
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
   Pass Arrow timestamp[ns] directly as pandas datetime64[ns], including empty
   frames; no text parsing/timezone conversion/normalization. Format days only
   after native validation; timezone-bearing columns refuse the source type check.
5. Refuse selected duplicate event_id (even identical repeats across years),
   duplicate (ticker,event_date), true date_conflict or nonnull
   (ticker,event_cluster_id) associated with another event ID/date, including
   unconfirmed siblings in either encounter order. Native cluster identity is
   ticker-scoped (engine/v2/data/events.py:132–154; engine/v2/data/event_revisions.py:75–87); producers
   prefix ticker (engine/calendar.py:707,728; engine/v2/ops/forward_calendar_store.py:626). Equal opaque
   cluster strings across tickers are not siblings; do not parse their prefixes.
   Null clusters remain allowed. No global cross-ticker qualification is needed.
   Do not import computed-move deduplication or prefer one session/source/date.
   get_event reports conflict_status; this research index instead refuses an
   ambiguous full view. It has no target clock: refusal, including a future
   sibling, is not a historical-cohort exclusion or an observed-PIT assertion.
6. Confirm exactly when src_orats is true and session is BMO or AMC, matching
   the facade's native reason=None after validation, not a second predicate. For
   each ticker, sort confirmed rows by (event_date,event_id) and call public
   `event_sequence(dates, warmup=4)` once. Retain zero-based ordinals; its admitted
   flag means only ordinal>=4, never holdout, value or forecast admission. Expose
   that flag as warmup_complete to avoid falsely calling contributors admitted.

Empty ticker requests refuse before repository access. Zero fragments establish
only a zero-row pinned view, never provider/real-world coverage. Every requested
ticker receives an entry: `absent_in_snapshot` if it has no selected rows,
`no_confirmed_events` if it has selected rows but none confirmed, otherwise
`confirmed_events`. Only absent_in_snapshot and no_confirmed_events have
first_confirmed_date=null and events=[]; confirmed_events has a nonnull first
date and nonempty events. All three statuses retain selected/confirmed counts.
None substitutes for a known-empty *qualified numerical prefix*. Missing table/ref,
incomplete membership, unsupported metadata or scan failure returns no index.

## Closed output and identity

Return one detached plain-JSON document, with exactly these top-level keys:
`schema_version`, `policy`, `source`, `tickers`, `sequences`, `counts`, `index_id`.
The schema literal is `magnitude_confirmed_metadata.v1`; this does not change
research_magnitude_panel.v1 or any existing dictionary/source closure.

`policy` has exactly `confirmation="src_orats_and_session.v1"`, `warmup_events=4`,
`conflicts="refuse_ticker_scoped_view.v1"`, and
`coverage="complete_pinned_view_only.v1"`.
`source` has exactly `snapshot`, `table="earnings_events"`, `contract_ref`,
`dataset_manifest`, `fragment_records`, `columns` (the seven ordered names above).
Native nested objects use their existing strict to_document shapes/versions;
fragment_records is the complete native ordered membership, not only hits.
It retains object refs, hashes, recorded bytes/rows and receipt refs without
opening receipt payloads. Manifest coverage/availability refs and knowledge mode
are preserved verbatim; neither reconstructed nor observed proves source-wide
completeness or point-in-time publication knowledge here.

`tickers` is the sorted request. `sequences` is a list in that same order. Each
entry has exactly `ticker`, `status`, `selected_rows`, `confirmed_rows`,
`first_confirmed_date`, `events`. Each event has exactly the seven projected
fields plus `ordinal` and `warmup_complete`; confirmed rows therefore retain
src_orats=true and date_conflict=false, rather than dropping their evidence.
`counts` has exactly `partitions`, `fragments`, `candidate_rows`, `selected_rows`,
`confirmed_rows`: the first three describe candidate inventory, the last two
exhausted requested-ticker rows, never label/quote contributions.

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
| max_catalog_tables | All snapshot table memberships, probed before catalog payload transfer. |
| max_catalog_fragments | All referenced dataset fragment memberships (not deduplicated), before materialization. |
| max_catalog_bytes | Each of two cumulative counters: stored-encoding SQL BLOB payload bytes before transfer, and decoded text's UTF-8 bytes before JSON parsing. Not decoded-object size. |
| max_partitions | Earnings partition count, probed before fragment materialization. |
| max_fragments | Earnings fragment membership count, probed before materialization; distinct from the all-table cap. |
| max_scan_rows | Sum of native candidate row bounds across all partition scans. |
| max_object_bytes | Each recorded object byte_size before scans; raw verified bytes and expanded Parquet bytes through Repository.scan. |
| max_total_object_bytes | Sum of candidate object byte_size per fragment/open, not deduplicated by object ID. |
| max_batch_rows | min(this value, contract.maximum_batch_rows, max(1, partition candidate bound)); zero-bound scans therefore use exactly 1. Never increase the caller's cap. |
| max_retained_rows | Selected rows including unconfirmed rows, checked before retaining. |
| max_retained_metadata_bytes | Cumulative canonical UTF-8 bytes of selected seven-field rows, including unconfirmed rows, before retaining. |
| max_proof_bytes | Canonical UTF-8 bytes of the complete returned document, including all source metadata and index_id. |

Check the complete manifest/preflight totals before any metadata payload scan,
not only the next partition. Check retained row/byte growth incrementally and
count proof bytes with foundation canonical streaming before return; no truncated
sequence, dropped ticker, auto-raised cap, silent fallback or automatic retry.
Zero candidate rows use native max_result_rows=0 with positive max_batch_rows.
Catalog caps precede row payload materialization, never follow ordinary resolve.
Caps do not bound SQLite internals, Python overhead, RSS or cumulative expanded
Parquet bytes. Synthetic limits are not authorized real-data resource budgets.
Use existing bounded-cadence progress logs: stage start/end, done/upper-bound,
elapsed/rate/ETA, refusals and stage RSS; no per-event logging or extra scans.

| Failure | Registered nonretryable code unless native code says otherwise |
| --- | --- |
| Malformed non-snapshot request/row/cap shape, missing/noncanonical partition metadata or no supported bounded native session | CONTRACT_MISMATCH |
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

STAGE2 owns scientific scope/dependencies; these labels refine rows 1, P and 2
(parity P1/P2). The estimates below supersede the original coarse estimates
for this named remaining work, not fit within them; unlisted work is additional.
This design-only checkpoint adds no runtime/source reads. Each proposed code
PR below targets <=400 combined added code/test/doc lines, stricter than the
repository's separate category caps; re-estimate and split add/wire again if
needed. Estimates are not waivers, completed work or real-data authorization.

Packaging amendment: existing 1.B1b and P3a foundations may share one PR under
the repository's separate non-test-code and documentation caps instead of the
combined-line target above. Contracts, dependencies and acceptance tests remain
unchanged; this does not enable execution or B2. 1.M1 remains separately reviewed.

| Sub-slice / dependency and acceptance | Added code / tests / docs |
| --- | --- |
| 1.B0: bounded native-session design / contract below; required review before B1a code | 0 / 0 / 190–210 |
| 1.B1a: bounded catalog count/type/byte probes / 1.B0; all-table refusal before payload transfer | 100–140 / 150–190 / 20–30 |
| 1.B1b: bounded catalog transfer/decode / 1.B1a; same read view and independent UTF-8 cap | 100–140 / 150–190 / 20–30 |
| 1.B2: exact bounded identity reconstruction / 1.B1b; native identity parity and poisoned unrelated metadata | 100–140 / 150–190 / 20–30 |
| 1.B3a: closed native metadata session / 1.B2; factory lifetime, detached getters and no SQL fallback | 80–110 / 120–160 / 20–30 |
| 1.B3b: native scanner wiring / 1.B3a; same admitted view, mandatory object cap, iterator closure | 100–140 / 150–190 / 20–30 |
| 1.M0: native metadata facade / unchanged v1 closure, existing validator/predicate parity | 30–50 / 70–100 / 10–20 |
| 1.M1: ticker conflict/sequence reducer / 1.M0; same-ticker siblings and empty states, no admission claim | 100–140 / 150–190 / 20–30 |
| 1.M2: native ref/inventory admission / 1.B3b; exact contract and complete membership with pre-allocation caps | 100–140 / 150–190 / 20–30 |
| 1.M3: bounded ticker scan and index composition / 1.M1/1.M2; exhaust once per partition, detached output and poisoned payload tests | 100–140 / 150–190 / 20–30 |
| P.M1: native history/grouped-policy compatibility proof / 1.M3; contaminated histories refuse, no post-aggregate filtering | 100–140 / 150–190 / 20–30 |
| 1.C0: availability/forecast/label qualification design / 1.M3; separate reference, payload and bound evidence before code | 0 / 0 / 120–160 |
| 1.C1: required availability-evidence admission / 1.C0; only evidence needed for causal availability, not source-wide population coverage | 90–130 / 150–190 / 20–30 |
| 1.C2: forecast/label-bound admission / 1.C0; prove clocks independently of producer declarations | 100–140 / 150–190 / 20–30 |
| 1.C3: closed v2 context declaration / P.M1,1.C1,1.C2; source-free shape/identity, no admission claim | 100–140 / 150–190 / 20–30 |
| 1.C4: native v2 context composition / 1.C3; bind independently qualified sources and exact history | 100–140 / 150–190 / 20–30 |
| 2.M1: causal numerical query plan / 1.C4 and qualified sources; immutable ceilings and bounded read sets, no values | 90–130 / 140–180 / 20–30 |
| 2.M2: stored historical-quote reader / 2.M1; protected/future poisoning and source anchors | 100–140 / 150–190 / 20–30 |
| 2.M3: quote aggregate and absolute derivation / 2.M2; native math, zero/null semantics and no ordinal restart | 90–130 / 140–180 / 20–30 |
| 2.M4: row/lineage materialization wiring / 2.M3; complete horizon rows, bounded output, no experiment execution | 100–140 / 150–190 / 20–30 |

Mapped totals (code/tests/docs): row 1 = 1200–1690/1840–2350/560–750;
P = 100–140/150–190/20–30; row 2 = 380–540/580–740/80–120. The original
one-week target is not reaffirmed. Only 1.M3 exposes authenticated build_index.
Each implementation slice updates its owning data/research README/ARCHITECTURE
under existing ceilings. 1.C0 may block; estimates never invent source evidence.

### Remaining 1.C0 boundary (design required before numerical implementation)

Distinguish three claims: (1) native identity authenticates evidence references;
(2) opening and verifying the referenced payload authenticates its contents;
(3) neither alone independently establishes forecast/label bounds. Do not add a
coverage-watermark reader or separate attestation system for the population
choice above. 1.C1 retains only genuinely required source-availability checks;
C0 must identify the existing native evidence and supported read route for each
such check, without treating absent publication stamps as observed-PIT.

Reuse native `forecast_cutoff`, which already recomputes source-session ceilings
from the pinned calendar. Label intervals need the actual measured source pre/post
anchors, including BMO/AMC and missing-bar selection, not guessed calendar dates,
producer-declared intervals or a receipt hash alone. C0 must establish how those
anchors are authenticated and checked before any dependent protected-value read;
that source-specific route remains unresolved here. Unsupported availability or
unproved label bounds still refuse; no new numerical/clock policy is chosen.

1.C2 retains independent forecast/label checks; 1.C3/1.C4 still depend on P.M1,
the required availability checks and clock/label qualification. Complete frozen
membership removes only the external-population coverage qualification demand.
Explicit bounded resource enforcement remains mandatory. Synthetic fixtures and
catalog caps establish neither real-data capacity nor an authorized run budget.

Acceptance fixtures cover the exact full contract hash and same-column impostors;
changed snapshot/dataset/fragment/object identity; all manifest years including
pre-history_start/protected metadata; absent/no-confirmed/zero-fragment cases;
same-ticker duplicates/conflicts across batches/partitions in either order,
unconfirmed siblings and cross-ticker equal opaque clusters that are not siblings;
strict dates/booleans/session with shared native guards/predicate as the oracle;
ordinal warmup and deterministic detached replay; and every cap at/over boundary.
Poison non-earnings payload scans/object reads and provider/writer/fit hooks;
assert zero accesses; poison oversized unrelated catalog membership/text and
assert refusal before row transfer/decoding, plus no fallback to ordinary resolve.
Assert one scan per manifest partition and equal indices over batch sizes/request
orders; a late preflight overage must open no objects, and a late streaming
integrity/retention failure must return no partial proof. Inspect source closures
to prove all existing v1 identities unchanged. No fabricated source availability,
historical coverage, protected numerical admission or real-data execution claim.

## 1.B0 — bounded native catalog read-session contract (planned only)

This prerequisite freezes the additive route for B1a–B3b; it implements nothing.
The names below are proposed, not installed APIs. No runtime caller, schema,
source read, source-identity amendment or numerical admission is added here.
Data README/ARCHITECTURE interfaces change with their implementation, not now.
Use stdlib sqlite3/contextlib/json and native document/manifest/query/scan code;
an ORM, SQLite copy, custom JSON parser or second Parquet reader is unnecessary.
The repository-specific snapshot membership proof has no generic library substitute.

### Public boundary and lifetime

Proposed data entry point: `bounded_snapshot.open_bounded_snapshot(repository,
snapshot, *, table_name, limits)`, a context manager yielding a `BoundedSnapshot`.
The exact existing `Repository` is only a borrowed connection/store holder, not
admission evidence: ignore its pin flag and both caches; invoke no resolver on it.
The data factory alone opens and admits the bounded view before yielding a session.
Research `build_index` calls this factory with table_name="earnings_events" and
the five catalog/selected-table limits below, then passes the yielded session to
`snapshot_inputs.admit_snapshot`; a caller-created receipt/boolean cannot replace it.
The factory rejects a missing/malformed SnapshotRef with SNAPSHOT_UNRESOLVED.
Non-native holders, empty/non-string table names, unsupported connection
configuration and an already-active transaction refuse CONTRACT_MISMATCH.
An absent selected table refuses CONTRACT_MISMATCH before the factory yields.
Support the native sqlite3.Connection with sqlite3.Row and str text factories;
no supplied callbacks, connection adapters or subclass overrides are admission.
The caller lends exclusive connection use during admission; use cursor-local row
handling, never change the caller's factories/schema/pins or close its connection.

The factory owns one BEGIN/read transaction through probes/transfers/reconstruction,
then ROLLBACK before yielding, including on failure; never roll back a caller's
preexisting transaction. Roll back only a transaction this factory began that
is still active; cleanup never replaces a primary failure. A cleanup-only SQL
failure uses the mapping below and yields no session.
All later reads use the complete detached frozen view;
this preserves that admitted view without holding a SQLite lock during scans.
No later SQL/head lookups, parent traversal, re-resolution or implicit latest.
No nested/concurrent use of a session. Close/cancel started iterators on exit;
an unstarted iterator or further read after exit refuses before object access.
Admission failure exposes no session or partial proof. Reopening performs fresh admission;
no process-global cache, automatic retry, durable write or temporary catalog copy.

The facade exposes only the following native-compatible operations; returned
native dataclasses and every nested mapping/list are deep-detached from its state.
Every method and iterator resume first checks lifetime: use after exit refuses
CONTRACT_MISMATCH before any ID/ref check. The table applies only while active.

| Method | Behavior while active |
| --- | --- |
| resolve(snapshot_id) | Exact admitted ID only; detached SnapshotRef, otherwise SNAPSHOT_NOT_FOUND. |
| resolve_full(snapshot_id) | Same ID rule; detached native ResolvedSnapshot, complete all-table membership. |
| table_contract(snapshot_ref, table_name) | Exact admitted ref and known table; detached full contract. |
| fragment_records(snapshot_ref, table_name) | Same checks; complete native ordered tuple, including empty membership. |
| scan_population_bound(snapshot_id, *, table_name, table_contract_ref, key_filter=(), time_interval=None) | Selected table only; native plan_scan_population candidate-row sum, never result-count estimation. |
| scan(query, *, table_name, max_object_bytes=None) | Selected table only; mandatory native byte-cap validation below and unchanged DataQuery semantics; provisional Arrow batches until exhaustion. |

For an active scan, call objects.validate_max_object_bytes before delegation,
including the rejection-only None default: omitted/None, bool, noninteger or
nonpositive caps raise QUERY_NOT_BOUNDED. Never take Repository.scan's None
path fallback. Preserve other native DataQuery/query/object refusals unchanged.
Provisional means a late stream failure can follow already-yielded batches;
there is no retraction or mutation of those Arrow batches. The consumer must
discard its accumulated result on error/cancellation, and the index publishes
nothing until every required stream exhausts successfully (engine/v2/data/ARCHITECTURE.md:690–691).

On the active facade, unknown tables, nonexact refs passed to table_contract/
fragment_records, or scans outside the selected table refuse
CONTRACT_MISMATCH. There are no exposed latest/head/get_event/get_chain/price
helpers, write methods, raw connection, mutable caches or caller-supplied records.
The facade is request-local, not a serializable provenance token. A missing store
refuses scan with CONTRACT_MISMATCH; metadata-only admission need not open one.

### Pre-transfer admission, including unrelated snapshot tables

`limits` has exactly max_catalog_tables, max_catalog_fragments,
max_catalog_bytes, max_fragments and max_partitions, with the positive built-in
integer/below-sys.maxsize rule above. No defaults, cap raising or truncation.
The latter two cover the selected table; all other caps cover the whole snapshot.
Bounds cover catalog-to-Python payload transfer/retention, not construction of
the caller's already-supplied request, SQLite internal memory/CPU or process RSS.

Before transferring any catalog TEXT/BLOB payload, use parameterized SQL scalar
probes: capped COUNT over LIMIT(cap+1) membership subqueries, typeof fields and
length(CAST(field AS BLOB)); never length(text), SELECT * or payload fetchall.
Count snapshot table memberships and fragment memberships per table binding,
not distinct object/fragment IDs. Probe selected-table distinct partition count
and fragment count too. Empty membership counts zero; every present fragment counts,
even if its row_count is zero, and every distinct present partition counts.
Drive LEFT JOINs from raw memberships: a missing dataset/contract/fragment/object
must refuse, not disappear and make a count or byte total look smaller.
Only bounded fixed-width scalar probe rows cross the boundary before admission;
consume them incrementally. Accumulate integer lengths in Python with early
cap checks, not overflowing SQLite SUM or floating TOTAL. Check every table,
including unrequested tables, before transferring the first payload row.

Use a fixed field projection: snapshot identity/policy/evidence; snapshot-table
bindings; dataset identity/count/mode/parents/evidence/partition hashes; contract
identity plus definition_json; membership ordinal/IDs; fragment identity, object
and contract IDs, counts/bounds/hashes/receipt refs/import hash; ObjectRef fields.
Do not transfer registered_at, commit_receipt_ref or storage_key, which this read
does not consume. Parent IDs and evidence references are values, never followed.
Charge every projected nonnull field's SQL BLOB length (including integer text)
in a fixed schedule: snapshot once, binding/dataset/contract per table, and
membership/fragment/object per fragment membership. Repeated IDs/objects are
charged each occurrence, never discounted by a cache or later deduplication.
Order tables by table_name and each table's membership by its stored ordinal.
SQL types/nullability must match that projection before payload transfer.

After all stored-byte/count probes pass, transfer one admitted row at a time
in the same transaction. Verify types and independently accumulate decoded
TEXT UTF-8 bytes before any JSON parse; this second counter also must fit
max_catalog_bytes. SQL BLOB length uses database encoding, not assumed UTF-8;
UTF-16 and embedded-NUL fixtures must exercise both checks. Invalid stored TEXT
can raise sqlite3.OperationalError during str decoding without a SQLite code;
that and Python decode/re-encode UnicodeError map to MANIFEST_CORRUPT below.
Fixed-width scalars remain count-bounded; no BLOB field is expected.
Use the native JSON duplicate-key hook and strict native document decoder;
JSON syntax/depth/shape failures refuse rather than yielding an empty record.
No re-query through ordinary Repository is allowed after these guards.

### Native identity and additive scanner reuse

Rebuild each FragmentRecord/DatasetManifest/SnapshotRef using the existing
manifests builders/verifiers, not a new hash recipe. Decode the full contracts
and require table_contract_hash to equal both stored and referenced hashes.
Check all stored binding/count/logical-hash/knowledge-mode fields against the
reconstruction, plus ordered membership, contract/table correspondence and
each referenced object. Require complete equality to the supplied SnapshotRef.
Zero-fragment tables still have contracts/manifests; no invented empty source.
This authenticates catalog metadata only: no object bytes, receipt contents,
coverage, observed-PIT guarantee or numerical-contributor evidence is proved.

Add new data modules only; preserve repository.py, documents.py, manifests.py,
objects.py, query.py and every existing v1 source-closure byte. A private data-
package Repository subclass can override resolve and _table_records to use only
the admitted state with pin_snapshots=False, retaining unchanged scan and
scan_population_bound dispatch (repository.py:297–315,849–868). Its SQL route
must be inaccessible/poisoned; never return this subclass itself to research.
Facade resolve_full/table_contract/fragment_records use admitted state, not the
inherited methods that reopen SQL (repository.py:717–756,829–847).
This is same-package native reuse, not a research private import. Each facade
call and iterator resume checks lifetime; only one active scan is permitted.
No silent native fallback when construction or an override is unsupported.

Factory snapshot-reference failures use SNAPSHOT_UNRESOLVED. Active facade
resolve/resolve_full retain native SNAPSHOT_NOT_FOUND for a different ID;
research admit_snapshot translates it to SNAPSHOT_UNRESOLVED (engine/v2/research/snapshot_inputs.py:14–25).
The index's exact earnings-contract allowlist still refuses UNSUPPORTED_CONTRACT;
the generic session authenticates catalog identity, not that research allowlist.

Preserve an existing native DataError first. For new catalog sqlite3.Error,
test a present numeric primary result code (extended code & 255) against the
resource row first; absent codes use the remaining-error row, never message text.
Missing required schema/join partners are malformed catalog, not an absent
snapshot or empty membership. New mappings never flatten scanner DataErrors.

| Failure boundary | Outcome (fixed redacted message; admission returns no partial session) |
| --- | --- |
| Factory invalid holder/table name/configuration/limits or active transaction; facade binding/lifetime guards above | CONTRACT_MISMATCH, nonretryable; excludes every factory snapshot-reference failure below and native DataQuery/scan-cap validation. |
| Factory: missing/malformed SnapshotRef, missing requested snapshot or supplied ref differs | SNAPSHOT_UNRESOLVED, nonretryable. |
| Active scan: omitted/None or invalid max_object_bytes | QUERY_NOT_BOUNDED from objects.validate_max_object_bytes, nonretryable. |
| Any explicit count or either byte counter exceeds cap | RESULT_LIMIT_EXCEEDED, nonretryable; no cap relaxation. |
| Catalog SQLite primary BUSY/LOCKED/IOERR/CANTOPEN/FULL/NOMEM (including extended variants) | RESOURCE_UNAVAILABLE, retryable by registry only; no automatic retry. |
| Catalog: broken join/type/JSON/Unicode/native identity, including DocumentError from stored catalog decoding; SQLite errors excluding the preceding resource codes, including code-less TEXT decode failure | MANIFEST_CORRUPT, nonretryable. |
| Native DataError from identity/query/object checks | Preserve code and retryability; do not blanket-wrap. |

Synthetic acceptance must prove cap equality/one-over, huge unrelated payloads
and IDs rejected before text transfer, repeated memberships charged, missing
join partners, UTF-16/NUL/multibyte text, malformed JSON and identity mutations.
Use native resolve_full as a small-valid-fixture identity/membership oracle,
not an oracle for its unspecified cross-table tuple order;
poison ordinary resolution, SQL after admission, all writer/provider hooks and
nonselected object reads. Verify deep mutation isolation, fresh reopen, failure
cleanup, early/never-started generator closure and exact bound/scan parity.
A concurrent second-connection commit between probes and transfer must not
change the admitted view; no fixture may stand in for private source coverage.
B1a/B1b/B2/B3a/B3b estimates top out at 360 combined lines, within the <=400 ceiling;
re-estimate/split if tests exceed them. Index work waits for all five slices.
