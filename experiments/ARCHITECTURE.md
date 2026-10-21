# Experiments architecture

## Native preregistration

`native_registration.register_native` explicitly registers one resolved arm
before evaluation; `require_native_registration` is its read-only admission
check. Both pin the current scope head once and read canonical event metadata
through the shared holdout population reader, never outcomes or prior results.
The stable registration key is experiment ID plus primary arm. Its immutable
binding includes the full resolved plan, detached spec hash, exact snapshot,
static source closure, numerical environment and declared single-thread policy,
scope, eligible event IDs and versioned holdout context. External input files and non-Python runner paths are refused.
The caller supplies the checkout root used for source fingerprinting.

`read_native_registration` uses the ops bounded verifier with the existing native
16 MiB UTF-8 default evidence/object budget: one SQL snapshot caps evidence transfer
(twice the budget for UTF-16, followed by UTF-8 admission), then validates the registration
reference size before the verified object read. Missing registration remains R1;
malformed, oversized or conflicting catalog/object evidence remains R6. Raw
storage exceptions propagate. This bounds transferred/serialized bytes, not
SQLite internals or decoded-object overhead. It loads v1.0–v1.5 without rewriting
saved bindings; current-source admission still requires a matching closure.
Larger legitimate registrations can supply `max_evidence_bytes`, also exposed as
`--max-registration-bytes` by the gate, prediction, search-plan and serial-search CLIs; magnitude
history binds its temporal plan twice, so 16 MiB is not a universal population bound.
The pure `valid_registration_read_budget` predicate admits only built-in integer
budgets in `[1, 2**62 - 1]`, leaving SQLite headroom for UTF-16. The loader and all
four CLI parsers share it; None never disables this loader's bounds. Later
admission/outcome APIs keep their existing policies. Execution under changed
bound sources requires fresh preregistration; old seals are never migrated.

Registration publishes canonical bytes through `ArtifactStore`, then reserves
the existing primary `hypotheses`/`experiment_runs` records in one catalog
transaction. Re-registering identical content returns the same identity.
The reserved arm is planned; registration does not mark a variant as tried.
Admission verifies the stored object and compares a freshly built binding;
returned documents are copies, not mutable views of registered bytes.
Registration and current-request admission do not fit models, publish reports, write the CSV ledger or emit outcome/refusal receipts.
The supervisor does not call them directly; its dispatched smoke worker reuses `verify_native_registration` for staged binding verification.
Sweep seal reading consumes ops catalog `sql_text_budget`/`decode_bounded_text`, retaining its existing bound and R6 policy. Public registration preparation/reservation and worker I/O/admission/error helpers remain in their owning modules; [README.md](README.md) lists their contracts and callers. Saved binding bytes and declared source roots are not rewritten.

Optional `final_read` metadata creates a versioned native registration that seals recipe, completeness-bar and frozen-model artifact references. Legacy registration bytes remain unchanged when omitted.
`prepare_final_holdout_request` verifies that original registration and derives an immutable request keyed by experiment ID; missing declarations or malformed request metadata refuse `INVALID_EXPERIMENT_SPEC`, and changed catalog/artifact identity refuses `EXPERIMENT_IDENTITY_CONFLICT`.
Preparation reads only registration metadata, performs no repository/head/population/result access or writes, and grants no authority. Foundation owns shared request decoding so ops and research need no peer import; later read admission must verify the referenced evidence and completeness before reading.

Optional `magnitude_training_start` binds v1.2 to a `computed_moves` spec;
years come only from `spec.folds`, validated by pure `magnitude_window`.
The metadata-only planner owns the sealed temporal plan, selected event IDs
and holdout context from one pinned snapshot; caller-supplied plans are refused.
Equivalent explicit/bulk membership gives identical bytes with other inputs fixed;
omission preserves v1.0/v1.1 bytes and identities.

Admission requires the catalog-owned original, even without `expected`, and a
complete fresh binding match. Stored integrity does not establish current
semantic admission: calendar arithmetic and fold reconstruction belong to the planner.
The closed stored schema requires canonical dates, nonempty ordered years,
fixed recipe/policy/availability declarations and snapshot-linked source identities.
Events follow `(event_date, event_id)` order; fold lists retain that order.
Registration itself has no feature/label reads, fitting, aggregate
materialization or final-read authority.

Optional `magnitude_labels_before`, together with `magnitude_training_start`,
creates v1.3 and seals a fixed `magnitude_execution.v1` contract. The explicit
evaluation-label cutoff must be a canonical day strictly after every evaluation
event; it is neither the forecast ceiling nor a training fit boundary.
The contract declares a fold-local arithmetic-mean baseline, constant intercept,
event-pooled RMSE with MAE, omitted skipped/unavailable/late training labels, and
complete finite evaluation labels available strictly before that cutoff.
Quantile calibration is explicitly unimplemented. Seed zero and empty economics
are required; this baseline has no hyperparameter or feature search declaration.
Omitting the new option preserves v1.0/v1.1/v1.2 bytes. Admission binds the original
v1.3 reservation before metadata scans; changing or omitting its execution policy
is an identity conflict. Registration reads metadata only. Only sealed v1.3
opens the intercept baseline; v1.4 below adds stored history. v1.2 remains metadata-only; none permits final reads.

`magnitude_inputs.admit_magnitude_inputs` requires this v1.3 registration with
the explicit training/evaluation clocks, then derives the union of all sealed
fold training candidates and evaluation IDs. It calls the research physical
preflight on exactly that union before any label read, returning the original
registration and detached preflight evidence. Unused selected-window events
are excluded. Admission/metadata/isolation refusals propagate with no writes,
retry or partial return. The magnitude CLI uses this admission before replay.

### Fixed magnitude baseline

`v2_prediction --target absolute_move` requires both magnitude clocks; the
default positive-move target refuses those options before store access.
`magnitude_report` validates the fixed seed-zero specification and sealed v1.3,
preflights every fold participant before labels, then qualifies each training
fold and freezes its arithmetic mean before opening that fold's evaluation labels.
All evaluation labels must be complete, finite and strictly cutoff-available;
empty training or any invalid evaluation refuses the whole assessment.
Each event retains all 5/3/1 full-session forecasts with the same intercept.
Primary RMSE pools equally weighted events, never fold RMSEs; MAE and per-horizon
metrics are also reported. Numerically scaled errors avoid square overflow.
The report binds every opened object, training omission, cutoff and sealed plan.
Schedule/source availability stays unverified reconstruction; quantile calibration,
market/history-feature model fitting, search and final evaluation are not implemented.
The existing native report/outcome/completion, ledger and replay lifecycle applies;
no-ledger runs still persist smoke evidence. Replay admits metadata but does not refit.

| Magnitude registration condition | Outcome |
|---|---|
| Invalid window, non-`computed_moves` source, or combined `final_read` | `INVALID_EXPERIMENT_SPEC` before repository access. |
| Execution cutoff without a magnitude window, malformed cutoff, nonzero seed or nonempty economics | `INVALID_EXPERIMENT_SPEC` before repository access. |
| Execution cutoff on/before any evaluation event | `INVALID_EXPERIMENT_SPEC` after metadata planning but before publication or reservation. |
| Missing admission reservation | `INVALID_EXPERIMENT_SPEC`; no metadata scan or implicit registration. |
| Changed/omitted mode, conflicting expected identity, or malformed/cross-context stored plan | `EXPERIMENT_IDENTITY_CONFLICT`; prior evidence remains unchanged. |
| On admission (`require_native_registration`): mode/request/source/environment/scope/snapshot/month drift or rehashed malformed/permuted metadata | Refusal before population/calendar scans. |
| Planner admission, calendar or empty-fold failure | Shared typed error propagates; no partial registration. |
| v1.2/v1.3 final-request preparation | `INVALID_EXPERIMENT_SPEC`; final declarations remain exclusive to v1.1. |

Existing publication/transaction/retry semantics above apply unchanged.

### Stored-history mean model (v1.4)

`--target absolute_move --magnitude-model history_mean` additionally requires
`--magnitude-feature-snapshot-id` and `--magnitude-history-start` on register and run.
Both magnitude clocks remain mandatory. Omitted model or explicit `pooled_mean`
uses v1.3; positive-move and pooled-mean modes reject history options before store access.
The library selects v1.4 by the same two history keyword arguments; both are required,
with seed zero, empty economics and runner `experiments/v2_prediction.py`.

The closed registration adds `magnitude_history_request` (exact feature snapshot ID,
canonical history start) and `magnitude_history` (binding version, full feature pin,
supported producer context). `magnitude_history_execution.v1` seals two model columns,
the empty-history fallback, evaluation cutoff and equal event/horizon RMSE policy.
The supported context is freshly derived from the original source, calendar, population,
holdout versions, forecast ceilings, history recipe and installed producer fingerprint;
its plan/source must exactly match the outer registration. Caller-supplied contexts fail.
Old wire policies remain unchanged for identical bound inputs; legitimate runner-source
changes still require new preregistration. Consumer wiring does not change producer code.

`magnitude_history_inputs` bridges research metadata admission into registration.
Admission checks mode/request drift before population scans, then derived context,
exact committed feature receipt/contract and complete object isolation before any
feature or outcome payload. Missing/changed feature identity refuses
`EXPERIMENT_IDENTITY_CONFLICT` before attempt/replay; protected isolation refusals retain
`HOLDOUT_ACCESS_DENIED`. There is no implicit build, repair, alternative pin or retry.

`magnitude_history_report` uses the original stored-only reader; it never recomputes history.
Each fold qualifies training labels and freezes their arithmetic mean, then fixes every
evaluation forecast before opening evaluation labels. Positive `n_prior` uses stored
`mean_prior_abs_move`, including zero; only empty history uses the fold mean.
All three forecasts per event are required. RMSE pools event/horizon squared errors,
giving each complete event equal weight; MAE and genuine per-horizon metrics accompany it.
The report distinguishes events from forecasts and binds both snapshots, stored input
digest, every decoded column/object/row, fallback inputs and actual model columns.
The native smoke/recorded/replay lifecycle applies; replay re-admits metadata without
reading feature payloads, labels or computing predictions. No search, quantiles,
holdout evaluation or observed point-in-time claim is added.

| Stored-history execution condition | Outcome |
|---|---|
| Missing/malformed history pair, incompatible runner/mode, unconsumed CLI options | `INVALID_EXPERIMENT_SPEC` (CLI parser errors exit 2 before store access). |
| Changed pin, context, receipt or registered source/environment | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT`; no attempt or ledger slot. |
| Empty qualified training, incomplete evaluation or invalid stored numeric/missingness | Whole assessment refuses; no partial report. Native failure-stage rules apply after attempt. |
| v1.4 final-request preparation | `INVALID_EXPERIMENT_SPEC`; final declarations remain exclusive to v1.1. |

### Fixed fitted-history candidates (v1.5)

`--magnitude-model history_ridge|history_hgb` uses the same two clocks and exact
stored-history pair. Library callers additionally pass `magnitude_fit_model`.
Each registration selects one closed recipe; absent fitting preserves v1.4.
The v1.5 execution binds all eleven stored columns in their original order,
the recipe, separate fold/horizon fits, original percentage-point targets,
nonnegative prediction projection, one numerical thread and equal event/horizon loss.
There are no caller-supplied hyperparameters, target transforms or search.

`models.training.magnitude` owns guarded fitting using maintained sklearn APIs;
the reporter passes recipe data only after ops validates the closed execution policy.
Ridge uses training-only median imputation (all-empty columns retain zero),
all-column missing indicators and StandardScaler before fixed alpha-one Ridge.
HGB uses squared-error loss, fixed seed/iterations and no automatic validation
split or warm start. Training-all-null columns are frozen to zero in train/predict
matrices; other nulls remain native. This avoids sklearn 1.9.0's empty-bin failure
without dropping columns or events and without learning from evaluation values.
Every fold/horizon uses a fresh estimator and only strictly qualified training
events. Empty history is handled by the declared preprocessing, not mean fallback.

The history reporter retains whole-run physical preflight and supported producer
admission. `magnitude_fitted` assembles each horizon's training/prediction matrices
from stored rows, binds their ordered keys/values and qualified labels by digest,
and freezes detached forecasts before that fold's evaluation labels open.
It never builds features. The result adds per-horizon fit evidence and a distinct
schema; existing fixed-history result bytes/policy remain unchanged for identical inputs.
Registration/replay never fit. Native smoke/recorded publication remains unchanged.

| Fitted-history condition | Outcome |
| --- | --- |
| Unknown recipe or missing magnitude/history inputs | `INVALID_EXPERIMENT_SPEC` before repository admission; CLI parser errors exit 2 before store access. |
| Changed recipe, parameter, feature order, preprocessing, source or pin | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT`; no implicit registration or fit. |
| Invalid fit arrays, estimator failure, wrong-shaped/nonfinite predictions | `EXPERIMENT_VARIANT_FAILED`; no partial successful report or baseline fallback. |
| A no-fit guard raises `RuntimeFitForbidden` (v2 or legacy type) during a fit | Propagates unchanged; it is never converted to `EXPERIMENT_VARIANT_FAILED`. |
| Missing/late evaluation labels or empty qualified training | Existing whole-assessment refusal; no row dropping or horizon omission. |
| v1.5 final-request preparation | `INVALID_EXPERIMENT_SPEC`; no final-read authority. |

| Condition | Outcome |
| --- | --- |
| Invalid plan, external inputs, unsupported runner path, or missing/malformed/indirect source | `INVALID_EXPERIMENT_SPEC` before registration. |
| Current scope head cannot be resolved | `SNAPSHOT_UNRESOLVED`; no fallback. |
| Invalid context or explicitly excluded/unknown event | Shared `HOLDOUT_ACCESS_DENIED`; no outcome read. |
| Admission has no native registration | `INVALID_EXPERIMENT_SPEC`; no implicit registration. |
| Loader budget is not a built-in integer in `[1, 2**62 - 1]` | `INVALID_EXPERIMENT_SPEC` before catalog/store access; booleans and None are invalid. |
| Any registration-reading CLI receives an invalid `--max-registration-bytes` | Argument-parser error exits 2 before catalog/store access; noninteger text uses argparse's integer-conversion diagnostic, out-of-range integers use the positive SQLite-safe budget diagnostic. |
| Stored loader evidence exceeds its UTF-8/canonical budget, has malformed text/JSON, or references an oversized/invalid registration object | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT`; SQL transfer and reference-size admission precede decoding/object reads, respectively. No writes. |
| Loader encounters raw `OSError` or `sqlite3.Error` | Original exception propagates unchanged; no rewrite, implicit registration or retry. |
| Existing key has changed code, environment, plan, snapshot, population or context, or corrupt stored evidence | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT`; existing catalog and artifact bytes stay unchanged. |
| Publication or catalog commit fails | No partial registration is admitted; a complete unreferenced artifact may remain. No automatic retry. |

## Sealed magnitude search plans

`magnitude_search_plan.seal_search_plan` / `read_search_plan` compile and verify
`magnitude_search_plan.v1` artifacts from a catalog-owned fitted v1.5 registration.
The standalone `python -m experiments.v2_magnitude_search prepare|verify` CLI has no
production caller. It consumes an existing native catalog/store, anchor spec and request
JSON; verify additionally takes a plan-ref JSON. The public research fold rule supplies
metadata boundaries. Standard-library Cartesian enumeration supplies candidate recipes.
Direct dependencies are `engine.v2.contracts`, `engine.v2.data.errors`, foundation,
ops and research, plus the native-registration experiment adapter. Planning code identity
uses this installed repository root; the supplied anchor must match that installation.
This metadata-only CLI accepts `--no-ledger` as a no-op for runner compatibility;
both flag states publish/verify the same plan and never write a ledger.

| Condition | Enforced outcome |
|---|---|
| Experiment CLI or helper reads one fixed snapshot | Its `Repository` is built with `pin_snapshots=True`: scans reuse table membership cached per instance by immutable ids; output is identical to an unpinned read. |
| Request shape | `magnitude_search_request.v1`: explicit `inner_years`, `space`, `max_candidates`, `max_fits`; Ridge `alpha` and/or HGB `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization` axes. |
| Valid fitted anchor and closed request | Return detached canonical plan bytes binding request, anchor, installed sources/environment, full recipes/hashes, grouped event/horizon metadata, count limits and recorded selection policy. No estimator construction, fitting, labels, features, repository scans or results. |
| Missing/extra fields, invalid or unordered axes/years, or candidate count outside 2..`max_candidates` (itself 2..4096) | Nonretryable `INVALID_EXPERIMENT_SPEC` before search-plan artifact access/expansion; CLI registration metadata may already be read. Ridge alpha and all five HGB axes are explicit; no inferred knobs. |
| Reconstructed outer metadata differs from the anchor | Nonretryable `EXPERIMENT_IDENTITY_CONFLICT`; no publication. |
| Inner-year map omits an outer year or lacks earlier training/evaluation candidates | Nonretryable `INVALID_EXPERIMENT_SPEC`; no fold skipping or publication. |
| Valid nested metadata | Inner events come only from outer training membership; each event remains grouped across horizons. The shared annual rule defines the conservative fit boundary. Recorded label/selection policies are metadata, not evidence of qualified labels or a selected winner. |
| Candidate/inner-fold/horizon fits plus selected outer-fold/horizon refits exceed the explicit fit cap | Nonretryable `INVALID_EXPERIMENT_SPEC` before Cartesian expansion/publication. No time, memory or accuracy guarantee. |
| Catalog ownership, fitted schema, installed anchor sources or environment differ | Nonretryable `EXPERIMENT_IDENTITY_CONFLICT`; no publication. |
| Prepare succeeds or repeats unchanged inputs | Publish/return the same immutable content-addressed plan; no catalog reservation, attempt, ledger or final authority. Existing anchor/producer bytes remain unchanged. |
| Verify receives a reference differing from complete request reconstruction | Nonretryable `EXPERIMENT_IDENTITY_CONFLICT` before that plan object opens. |
| Verified bytes fail integrity | Nonretryable `EXPERIMENT_IDENTITY_CONFLICT`; no repair or automatic retry. |
| Non-integrity foundation `ArtifactError` at plan publication/read, including a missing plan object | `RESOURCE_UNAVAILABLE` with a sanitized `Problem`; no automatic retry or partial success. |
| Opened CLI spec/request/plan-ref descriptor is nonregular or exceeds 1 MiB, or opening refuses a leaf symlink | Nonretryable `INVALID_EXPERIMENT_SPEC` before catalog access; bounded descriptor reads reject growth beyond the cap. |
| CLI typed refusal | Preserve the full `Problem` (including native spec conversion), return exit 2 and close an opened catalog connection. Filesystem/artifact unavailability, including an unopenable socket path, maps to `RESOURCE_UNAVAILABLE`; malformed or excessively nested JSON maps to `INVALID_EXPERIMENT_SPEC`. |
| Unexpected builtin exception from catalog/registration/planning code | Propagate it after closing the catalog; do not misclassify it as malformed user JSON. |

## Recorded magnitude-search selection

`magnitude_search_trials.SearchTrials` exposes the journal API composed by the serial CLI. `magnitude_search_records` owns journal tables, immutability triggers and reference binding over SQLite/foundation; `magnitude_search_errors` tags artifact failures. Direct dependencies include contracts, foundation, ops migrations/catalog/fingerprints/errors, research target reads, `magnitude_search_evidence` and the `magnitude_search_inputs` cohort loader. That shared helper owns native admission through `magnitude_history_inputs.history_binding` and research stored-history/target reads; `magnitude_search_plan.read_search_plan` owns plan admission. Both `engine.models.no_fit` and `engine.v2.models.no_fit` guard native fitting; `magnitude_search_execution` owns its fixed child and consumes `engine.v2.ops.executor.THREAD_VARIABLES`. Public `magnitude_report.magnitude_metrics` owns pooled RMSE/MAE. One `SearchTrials` instance admits each (plan, outer year, inner year) cohort once and shares the admitted inputs across that cohort's trials; `start` returns detached inputs (deep copies of the memoised ones). The in-memory memo holds one entry per cohort (bounded by the number of distinct (outer year, inner year) cohorts in the sealed plan), is never persisted and never holds a failed admission. Drift between instances is detected on re-admission by a fresh instance (see Input drift below); within one instance it is not re-read.

| Condition | Outcome |
|---|---|
| A submission violates multiple boundaries | Schema/plan/seal admission takes precedence over trial identity; missing start takes precedence over argument/content refusal; content refusal takes precedence over missing native-input binding. Later phase outcomes require preceding guards to pass. |
| `seal(clock=...)` | Verify the plan and installed sources before explicitly initializing the search-owned journal through ops migrations; bind one plan per anchor to those recorder sources. |
| `start`, `record` or `freeze` with missing/older, newer or checksum-mismatched journal migrations | `INTEGRITY_FAILED` with the migration reason before plan/evidence/native reads; no implicit migration. |
| `start(outer_year, inner_year, candidate_id)` for an unclaimed planned trial with valid native source/stored-feature bindings, inner participant preflight and training labels strictly before the sealed inner-fit cutoff | One immutable start claim predates native reads. Success returns stored feature rows and qualified training targets bound to the same immutable cohort for every candidate in the fold. |
| Trial outside the sealed plan; `record` before `start` | `INVALID_EXPERIMENT_SPEC` for an unplanned trial; `EXPERIMENT_IDENTITY_CONFLICT` for a missing start; no native reads. |
| An admitted, started trial receives both or neither of `forecasts`/`failure` | `INVALID_EXPERIMENT_SPEC` before native input/label reads or any record publication; a missing start takes precedence as `EXPERIMENT_IDENTITY_CONFLICT`. |
| Complete nonnegative finite real forecasts and integral horizons; complete inner validation labels before the sealed outer-fit cutoff | Immutable JSON float/int forecast evidence predates validation-label reads. Scoring and successful terminals require one immutable validation cohort binding native-derived targets and the full receipt to admitted preflight membership, cutoff and recipe; concurrent candidates must agree exactly. Nonempty future/holdout exclusions in an explicit-ID receipt refuse `EXPERIMENT_IDENTITY_CONFLICT`. |
| Non-JSON request; Boolean, malformed or incomplete forecasts | `INVALID_EXPERIMENT_SPEC` before any validation-label read. |
| Explicit `EXPERIMENT_VARIANT_FAILED` fit failure after input admission; defensive local scoring refusal | Record a candidate terminal. An initial scoring refusal raises its original error; a failed terminal write is chained as its cause, leaving an unfinished claim. No other stage creates automatic candidate refusals. |
| `record(failure=...)` receives a registered non-candidate Problem | Raise its sanitized registered policy without a terminal; retain the unfinished claim. Unregistered/non-Problem values refuse `INVALID_EXPERIMENT_SPEC`. |
| Any shared native-input/label admission failure | Propagate the original exception, including native `DataError` or binding `OpsError`; preserve the unfinished attempt and block selection regardless of error code. Public `start` and `record` refuse restart or forecast resubmission, even for a retryable resource Problem. Native `reconcile` may idempotently reuse the exact committed forecasts only after verifying the durable child result and original admission; it never restarts fitting or replaces forecasts. |
| Input drift between instances | A new `SearchTrials` instance re-admits each cohort and refuses changed inputs with `EXPERIMENT_IDENTITY_CONFLICT` "trial native inputs changed after start" before the first validation read, unless durable validation is reused. Within one instance the memoised admission is reused, so drift is not re-read. |
| Durable journal write; raw SQLite error in migration checks, reference reads or transactions | Publish artifacts before catalog commits; a failed commit may leave an unreferenced object, never admitted evidence. Raw `sqlite3.Error` propagates unchanged, except as the chained cause of a refusal-persistence failure; failed transactions roll back. SQLite failure never becomes a candidate terminal. Records are capped at 16 MiB and cannot be updated, deleted or replaced. |
| `freeze()` with every planned candidate/inner-fold attempt terminal | Exclude a candidate's entire outer score after any failed/refused inner fold; pool all successful event/horizon errors, break exact RMSE ties by candidate ID, and freeze every outer choice plus the latest-outer prospective candidate. Admitted successful folds have nonempty evaluation rows. A pooled-scoring refusal propagates `EXPERIMENT_VARIANT_FAILED` for the whole freeze, with no selection or candidate exclusion. |
| Missing/started-only attempt or outer fold without a successful candidate | Refuse `EXPERIMENT_VARIANT_FAILED`; publish no selection. |
| Exact terminal/freeze replay | Verify submitted-forecast terminals against the separate validation cohort and successful targets/receipt against its exact bytes, without reopening native inputs/labels. Return the original result idempotently; refused replay returns its durable envelope without rerunning scoring or raising the initial error again. |
| Current migrated journal missing its seal; valid submission/evidence without admitted inputs, repeated start, changed submission, another plan for the anchor, source drift or corrupt evidence | Refuse `EXPERIMENT_IDENTITY_CONFLICT`; no retry authorization. Missing migrations instead refuse `INTEGRITY_FAILED` under schema admission above. |
| Unavailable artifact, raw `OSError`, or read-time `UNSAFE_PATH` after stored-path syntax admission | Refuse sanitized retryable `RESOURCE_UNAVAILABLE`, including open-time OS errors and symlink rejection; no journal association changes, symlink traversal or attempt-retry authorization. |
| Invalid stored-path syntax or reference; publication `UNSAFE_PATH` | Refuse `EXPERIMENT_IDENTITY_CONFLICT`; invalid reference syntax is rejected before artifact access. Content/hash mismatch during verified reads remains an identity conflict. |
| Journal payload exceeds 16 MiB | Refuse nonretryable `RESOURCE_LIMIT_EXCEEDED` before publishing that record; incomplete attempts remain unfinished and block selection. |
| Submitted-forecast selection | These submission/selection methods do not fit or authenticate estimator execution; wholesale fabrication of matching evidence is outside their consistency guarantee. Native fitting is separately described under [bounded inner-trial execution](#bounded-inner-trial-execution). Neither path grants outer evaluation, deployment, ledger or final-read authority. |

## Sharded cohort inputs

The rows above describe the journal; this section states the sharded (v2) contract for cohort
inputs. Implemented: the shard module, the loader, the cohort and outer-input writers
(`_start` publishes `cohort:<id>` and the outer writer `outer-inputs:<id>`, each as shards plus a
manifest) and the manifest bindings of `inputs:<trial>`, `execution-result:<trial>` and the outer
result, and the child transport (below). A whole-cohort (v1) record still reads through the
loader's adapter, but only for replay: a new fit always publishes and streams a manifest. Cohort rows (`features`, `training`, preflight
`planned_inputs`, and the same rows in `outer-inputs:<id>`) are ordered sections of
content-addressed shards, not one record or payload. Population-sized receipt lists
(`training_inputs` candidate, qualified, read and omitted event lists, and preflight
`event_ids`) are sections too; only fixed-size scalars stay inline in the manifest.

* A shard is canonical JSON `{schema_version: magnitude_search_shard.v1, section, rows}`
  published with `ArtifactStore.publish_bytes`; its `ArtifactRef` is its address and it
  names no cohort or position. Rows keep the admitted order. The cut is deterministic
  (target 4 MiB; hard maximum 8 MiB; at most 8192 shards in total across all sections of
  one manifest, so a manifest of ~0.5 KB entries stays under ~4 MiB, well below the
  16 MiB record cap), so equal rows give equal shards; a row or count above the
  maxima is `RESOURCE_LIMIT_EXCEEDED`.
* `cohort:<id>` and `outer-inputs:<id>` hold a small manifest (inline scalar members plus,
  per section, `row_count` and ordered `{ref, rows, first_key, last_key}`).
  `magnitude_search_shards` requires inline values to be JSON scalars (str, int, float,
  bool, None); a list or dict value is refused (`INVALID_EXPERIMENT_SPEC` at build,
  `EXPERIMENT_IDENTITY_CONFLICT` at verify). `inputs_id`
  is `content_hash(manifest)`. A whole-cohort record is read through the loader's v1
  adapter, bound by its content hash; none is rewritten.
* One streaming reader serves parent and child: each shard is `read_verified` (size is
  the allocation bound) and checked for schema, section, row count and boundary keys;
  totals equal `row_count`. Each section has one key comparator (the row's sort key):
  within a shard keys strictly increase, and across shards the previous `last_key` is
  strictly below the next `first_key`, so out-of-order or overlapping shards are refused.
  An empty section has `row_count` 0, an empty shard list and no keys; any shard listed
  for it, or a nonzero count with an empty list, is a count disagreement. Parent and
  child apply the same rule. Matrix evidence keeps today's training / evaluation list
  order. The child transport: the child gets a header file (inner fold, horizons, execution spec,
  manifest, expected `inputs_id`, store root; limits stay an argument) and no input rows. It calls
  `load_fit_inputs`, which checks the manifest hash against the header, streams only the
  `training` and `features` sections through the reader above, keeps the training and
  evaluation events at the selected horizons, and opens no repository, catalog or validation
  target. A hash, shape or shard failure inside the child is an infrastructure failure
  (`RESOURCE_UNAVAILABLE`), since the parent just published and bound that manifest.
  Transport memory is the projected fit rows plus one shard; there is no input byte cap, only
  the child's `RLIMIT_AS`. The child output cap stays 16 MiB.
* Shards are published, outside the catalog transaction, before the manifest append,
  which is the commit point.
* The manifest hash is a Merkle `inputs_id`, and every cohort is sharded, including empty and one-row cohorts.
* `magnitude_search_shards` implements the cut rule, writer, manifest hash and verified reader
  and lists the list-valued members (`LIST_MEMBERS`) as data. A member without a strictly
  increasing sort key carries a `problem` note and no key function; the writer takes a
  caller-supplied key and does not consult the inventory; keyless members are written as
  `{position, row}` rows below.

* `magnitude_search_inputs.load_inputs(store, record, expected_id)` is the one loader every
  consumer (`_record`, reconcile, freeze, outer replay, `_validation`) reads
  through. A record with the manifest schema is verified against `expected_id`, its sections
  are streamed shard by shard and placed at their dotted paths beside the inline members, which
  are flat scalars keyed by dotted path (`preflight.count`); any other
  record is a v1 whole cohort whose id is its content hash. Both return the same inputs shape.
  A section without a sort key stores rows `{position, row}` (position 0, 1, ...; the loader
  checks it is consecutive and unwraps `row`), so every section has a strictly increasing key.
  A manifest must carry exactly the `LIST_MEMBERS` sections. Non-retryable loader conflicts
  surface as each caller's own evidence-chain refusal; retryable errors propagate unchanged.

| Condition | Outcome |
|---|---|
| Shard missing or `OSError` | Retryable `RESOURCE_UNAVAILABLE`; no association change or candidate failure. |
| Hash/size mismatch, wrong schema or section, count or boundary-key disagreement, duplicate key, invalid or unsafe ref syntax, manifest hash differing from `inputs_id` (parent or child), a manifest `inline` value that is a list or dict | `EXPERIMENT_IDENTITY_CONFLICT`; no rows used. |
| Crash after some shards, before the manifest | Orphan shards confer no authority and are reclaimable by the store owner; the immutable claim stays unfinished. |
| Manifest committed, later shard lost or corrupted | Fails closed as above on every read. |
| Same slot, different manifest | `EXPERIMENT_IDENTITY_CONFLICT`. |
| Same slot, same manifest (retry after an ambiguous append) | No-op success; shards are content-addressed, so republishing is harmless, and no cache entry depends on the attempt. |
| A row above 8 MiB or more than 8192 shards | Nonretryable `RESOURCE_LIMIT_EXCEEDED` before the manifest append; published shards stay unreferenced. A stored manifest or section listing more than 8192 shards is `EXPERIMENT_IDENTITY_CONFLICT`. |
| Writer given a row whose sort key is missing, not a tuple of strings/integers, or not strictly above the previous key, or a row that is not canonical JSON or uses the reserved `__nonfinite__` key, or a manifest builder given a malformed manifest or section | `INVALID_EXPERIMENT_SPEC` before that shard is published. |
| Read-time `UNSAFE_PATH` (for example a symlink) on a syntactically valid shard ref | Retryable `RESOURCE_UNAVAILABLE`, like a missing shard; invalid ref syntax stays a conflict. |
| Loader given a manifest with a missing or extra inventory section, an inline member colliding with a section path, or a keyless-section row that is not exactly `{position, row}` with consecutive positions from 0; or a record that is not a dict | `EXPERIMENT_IDENTITY_CONFLICT`; no rows used. |
| Whole-cohort record whose content hash differs from the expected `inputs_id` | `EXPERIMENT_IDENTITY_CONFLICT`; no rewrite or migration. |

Registration, snapshot, plan and execution-spec bindings are unchanged. `inputs:<trial>`,
`execution-result:<trial>` and the outer result bind the manifest hash, so every shard. Outer
replay takes the expected id from the stored terminal and the loader verifies the manifest against
it before reading any shard. Source closures include the shard module, so seals change. Journals
written before activation (smoke and test runs) bound the whole-cohort hash and stop replaying;
this is accepted. A `cohort:` or `outer-inputs:` slot already holding a different record is
`EXPERIMENT_IDENTITY_CONFLICT`. Replay identity holds within one format version.

## Stored research-history entrypoint

`python -m experiments.magnitude_history_store build|read` requires an existing catalog file.
Other inputs are an existing artifact-store directory, exact source snapshot, holdout month, evaluation years,
training start, history start (on/before every forecast ceiling) and optional repeatable `--event-id`.
`read` also requires the exact feature snapshot.
The entrypoint fingerprints its installed source closure and numerical environment;
callers cannot submit features, a context document or an alternative producer.
Research owns the native contract, protected build and stored-only reader; ops
supplies existing bootstrap/fingerprinting, with no peer-layer import added.
The CLI uses `engine.v2.data` Repository/errors and foundation ArtifactStore/errors/clock.
Build returns the source/context/output identities; read returns stored rows.
Both actions run shared catalog bootstrap/migrations; stored-only describes feature consumption.
Before bootstrap or ArtifactStore construction, a missing/non-directory store root refuses `UNSAFE_PATH`.
Native object admission then checks source contents; the CLI never initializes a missing root.
Both are explicit local operations, never implicit provider downloads or model
runs. Data/ops failures retain code/message; artifact failures retain only their
code. Raw local OS/SQLite failures produce `INPUT_CHANGED` without exception text.
Handled runtime failures are stdout JSON and exit 2; argument refusals use argparse stderr.
No partial successful output or ledger entry is returned.

## Native outcome publication

`native_outcomes.publish_native_outcome` records one admitted success or typed
refusal; `replay_native_outcome` completes its interrupted publication without
evaluation. Both verify the supplied native registration. Callers must also
perform current-request admission before replay or fitting. A success requires
an actual attempt; registration alone never counts as an attempted variant.
Immutable report and outcome objects use `ArtifactStore`; existing catalog evidence
owns separate smoke and recorded outcome slots, globally binding the
registration, result, original row date and hashed ledger destination. Snapshot
and holdout provenance derive from the verified registration, never duplicate fields. No new
catalog table or supervisor coordinator is introduced. File/CSV effects stay outside
short catalog transactions; completion follows durable append and never authorizes promotion.
`export_native_report` creates `REPORT.md` at the registration's fixed store-local
smoke/recorded address only after verified completion; existing bytes never change.
The export has an independent inode: a private copy is flushed and synced before
create-only installation, so editing the export cannot alter stored report bytes.
An existing export sharing the stored object's inode is an identity conflict.
Report-bearing publication, replay and export preflight the fixed destination
before new result artifacts, intent reservation, ledger append or completion.
Existing non-directory ancestors and other destination conflicts are R6 before
effects; preflight and CSV/export are not one atomic transaction.
Recorded destinations must be nonempty and bind the exact normalized path used
for append. Saved outcomes require the complete canonical schema, object-valued
failure details and a nonblank UTF-8 success report before any ledger effect.
Saved completion payloads must match canonical expected metadata, preserving JSON
scalar types, before any append or completion publication.
Outcome, report and completion references must match the exact identity rederived
from verified bytes; malformed metadata refuses before ledger or catalog effects.

| Condition | Outcome |
| --- | --- |
| R1 invalid specification, R2 unresolved snapshot, or R6 identity conflict | `publish_native_refusal` publishes private evidence only; no result or ledger row. |
| Empty, directory or symlinked recorded ledger destination | R1 before outcome reservation or ledger access; no default-ledger fallback or conflict evidence. |
| Publish, replay or export called within an active caller transaction | R1 before path access or artifact/CSV effects; leave the caller transaction unchanged. |
| Publish, replay or export with metadata-only magnitude v1.2 | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT` with private refusal evidence only, before outcome/report reads or catalog/ledger/export effects. Validated v1.0/v1.1/v1.3/v1.4/v1.5 are accepted; v1.2 remains registration-only. |
| Admitted R3 look-ahead or R5 holdout denial | Immutable refusal and `refused` row, unless smoke; no current-attempt report. |
| Admitted R4 variant failure | Immutable refusal and `failed` row, unless smoke; no current-attempt report. |
| Same canonical registration/result/destination repeated, including tuple/list-equivalent refusal details | Reuse the first outcome and date; reconcile one terminal CSV row per experiment/variant identity without refitting. |
| Changed result, status, binding or recording destination; corrupt, incomplete or semantically invalid saved evidence | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT` with private conflict evidence before ledger effects; preserve prior report, catalog slot and CSV bytes. |
| Publication/append/commit interruption | Keep complete published objects and any committed intent/row; no rollback or automatic retry. Explicit replay reconciles the same intent. |
| `no_ledger=True` | Do not resolve, read, create, lock or append any ledger path. The smoke slot cannot consume the recorded slot. |

Pre-admission receipts cannot invent admission; rejected-request ledger identities belong to the runner.

## Reconstructed gate evaluation

`python -m experiments.v2_gate register|run` evaluates one preregistered arm.
It composes native registration, `gate_report` and native outcome/replay/export;
the separate supervisor pricing/evaluation kinds share the fixed child entrypoint.
The fixed recipe requires this runner, seed zero, ordered unique year folds,
option-chain prices, and exactly `strategy` and finite `fill` economics.
Other declarations refuse before repository access. Each arm independently
resolves its immutable plan and reprices its own training and evaluation rows.
The registration binds recipe source, environment, snapshot, population and
holdouts; the report also binds actual history, cutoffs and replay inputs.

This is exploratory reconstructed-history research.
Calendar sessions come from the exact snapshot without projected extensions.
The feature ceiling is the completed session strictly before the earlier of
the strategy's decision and entry. Protected `load_event_history` supplies
`n_prior` and `mean_prior_abs_move` from the calendar's first session onward.
Empty history alone maps the missing mean to zero, retaining its zero count.
History/source dates are never converted into claimed observation timestamps;
schedule-as-known and source publication remain unverified.

Folds partition by gate-decision year. Training uses earlier decision years
whose exits precede both the year start and earliest evaluation feature ceiling.
Each fold fits a fresh depth-one sklearn DecisionTreeClassifier on profitable
training trades, requiring both classes. The shared training-score median
rule freezes its threshold before evaluation quotes/outcomes are read.
Per-event test histories may incorporate earlier completed events, but test
labels never fit the model or threshold. Selection uses score >= threshold.
Every requested event must have a complete native calendar plan; each consumed
pricing set must be complete and finite, otherwise the whole arm refuses.

Reports contain selected and all-candidate mark-based mean P&L (no commissions), row counts,
fold memberships, scores, cash flows, feature ceilings and complete provenance.
They make no observed-PIT, fill-validation, holdout, promotion or winner claim.
One arm counts as one attempt only when evaluation starts. Recorded success
uses its own report and ledger identity; smoke never accesses a ledger path.
Replay re-admits current inputs then reconciles existing evidence without fitting.
Changed inputs under the same arm identity refuse R6, preserving prior evidence.

| Condition | Outcome |
| --- | --- |
| Explicit excluded/unknown event or final-read registration | `HOLDOUT_ACCESS_DENIED` before features, pricing or metrics; no partial report. |
| Missing prior session, empty fold, incomplete/nonfinite inputs or fitting failure | `EXPERIMENT_VARIANT_FAILED`; existing native failure publication applies. |
| Admission, publication or replay failure | Existing native lifecycle's typed refusal and interruption semantics apply. |

## Native gate-variant pricing

The `native_gate_pricing` supervisor kind launches only the fixed unlayered
`native_gate_entrypoint` child. After the ops stdin/PID launch gate it accepts exactly
`native_gate_pricing` and `native_gate_evaluation`, importing their fixed callbacks.
For pricing, both entrypoint and adapter belong to the verified registration/source snapshot.
The adapter reads staged `spec.json` and
`native_registration.json`, verifies existing preregistration, and checks
the requested identity, resolved plan, source closure and single-thread
environment before pricing. Required `economic_params.strategy` selects the
native strategy independently of arm IDs; the fixed worker requires it before
registration reads or effects, and declared strategies must be supported. Snapshot,
event IDs and holdout context come only from registration.
It never re-resolves a scope head or accepts a supplied trades frame.
Submission uses the existing `SubmitRequest` API and artifact input bindings;
no production entrypoint submits this kind. The worker's catalog connection is read-only.
Each staged document itself must be a nonsymlink regular file of at most 16 MiB;
pathname replacement cannot redirect admitted bytes.
The worker installs one `native_gate_consumption.v1.0` receipt with the immutable plan,
stage inputs and mark-based cash flows; the supervisor publishes it only after worker success.
Installation is create-only and never assigns partial bytes; unsupported hard links
refuse `INPUT_CHANGED`. Existing destinations are never opened, overwritten or unlinked.
This receipt is not an evaluation report or promotion evidence.

| Native worker condition | Outcome |
| --- | --- |
| Recording requested or missing smoke opt-in | Submission `INVALID_REQUEST`; worker defense `INVALID_EXPERIMENT_SPEC` before catalog access. |
| Non-mapping or ill-typed worker arguments, including non-string purpose; unsupported recipe or malformed staging, including missing/null required spec fields or excessive JSON nesting; missing staged bytes | `INVALID_EXPERIMENT_SPEC` before catalog access for malformed arguments and before pricing for malformed staging; `INPUT_CHANGED` for missing bytes. |
| Changed staged spec, registration, expected identity, source (including missing, unreadable or malformed Python) or numerical environment | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT` before pricing; preserve registration and prior evidence. |
| Unresolved pin or non-sweep purpose | `SNAPSHOT_UNRESOLVED` or `HOLDOUT_ACCESS_DENIED`; no pricing receipt or partial report. |
| Incomplete/nonfinite pricing or caught pinned-data failure | `EXPERIMENT_VARIANT_FAILED` for pricing/validation failures; `DataError` uses the shared canonical ops mapping below, preserving ops policy for existing codes and data category/retryability for data-only codes. Underlying codes/types stay in private `data_failure_code`; no success receipt. The replay/catalog regressions define the catch boundary. |
| Unexpected programming failure | The existing worker classifier remains authoritative: unexpected replay exceptions such as `RuntimeError`, `IndexError` or `AttributeError` yield generic `WORKER_FAILED`, private type/location diagnostics and no successful output. This admission-only boundary does not reconcile full-sweep R4 outcomes. |
| Retry/cache/transaction | One attempt; cross-job cache reuse is disabled so admission is checked again. No worker catalog transaction or CSV access. |
| Existing receipt destination or receipt I/O failure | Non-retryable `INPUT_CHANGED`; preserve existing destinations and the original failure. Private partial bytes or a complete assigned receipt may remain, never a partial assigned receipt or successful output. |
| Successful smoke | Operations job/output artifact rows only; no reusable checkpoint is stored. Native registration/outcome and experiment ledger rows stay unchanged. |

This pricing-only path does not fit gates, record outcomes or orchestrate sweeps.

## Sealed gate sweep submission

`python -m experiments.v2_gate_sweep register|submit --no-ledger` accepts a bounded regular JSON roster, a catalog migrated through version 13, its store, explicit `--code-root` and a holdout month; read-only schema admission requires current checksums (`INTEGRITY_FAILED` otherwise), without migration. CLI success exits 0 with `{"roster": <hash>}` for registration or `{"state": "submitted_uncollected", "job_ids": [...]}` for submission; typed refusal exits 2 with only `{"refused": <code>}`. Untranslated foundation `ArtifactError` retains its code in CLI refusals; typed sweep/evaluation boundaries preserve registered codes, otherwise use `INTEGRITY_FAILED` with the original code in private `data_failure_code`. Existing data/registration translations remain authoritative. The closed `native_gate_sweep_request.v1.0` envelope contains `variants` and an ordered, two-member `primary_comparison`; other arms are exploratory. Arm order and explicit event-ID order are canonicalized. Every arm has one immutable `v2_gate.py` plan; experiment, hypothesis, strategy, folds, seed and price source must agree. Only supported fill economics may differ; other declarations refuse `INVALID_EXPERIMENT_SPEC` before population reads or reservation. No prior experiment output or shared prepriced frame is an input. `register_sweep` binds every arm to one verified scope-head `SnapshotRef` and eligible population; the roster seals requests, registrations (each with its environment), coordinator source and implementation ref. A shared SQL reader bounds stored text before Python materialization; the decoder caps JSON at 16 MiB UTF-8: malformed, non-object or oversized rows give `EXPERIMENT_IDENTITY_CONFLICT`; oversized new seals give `INVALID_EXPERIMENT_SPEC` before insertion. Missing seals give `INVALID_EXPERIMENT_SPEC` on submit; register may create one. Existing roster-arm/spec reservations are checked in the seal transaction before insertion. Every reservation call after sealing, including replay, must match the roster before artifact publication and again in its committing transaction; either writer ordering preserves roster-arm bindings. Earlier non-roster registrations remain independent and runnable, outside sweep submission/accounting. Catalogs without a roster table retain single-arm admission.

SQLite transactions and existing canonical JSON/content-hash helpers own identity and atomicity; no new serialization or scheduling framework is used. `submit_sweep` requires matching catalog evidence before staging; `submit_graph` inserts children atomically and replays their job IDs. The regular operations service owns execution; submission never starts a supervisor. Single-arm `v2_gate.py` bindings retain their runner-only source roots. The sealed worker implementation manifest separately binds the evaluation entrypoint and is verified by the launch gate; existing registration bytes are never rewritten. The roster is preregistration, and job success is durable child evidence only. Neither registration nor submission creates scientific outcomes, CSV rows, reports, aggregate attempted counts or a completed sweep index. A failed child cannot imply scientific completion. This interface has no scientific collection, recording, reconciliation or incomplete-run accounting authority. Non-sweep purpose refuses `HOLDOUT_ACCESS_DENIED` before catalog/input access; missing smoke opt-in refuses before effects. No final-read authority is accepted.

| Condition | Outcome |
| --- | --- |
| Register replay or interrupted arm reservation | Verify the saved pin and resume missing reservations; never resolve a replacement head. |
| Submit with partial registration | Refuse `EXPERIMENT_IDENTITY_CONFLICT` before staging/jobs; only register resumes missing reservations. |
| Changed request, source bytes, environment, binding or existing roster arm | `EXPERIMENT_IDENTITY_CONFLICT`; preserve sealed evidence and create no roster for a pre-existing roster-arm/spec conflict. Earlier non-roster arms do not prevent sealing. Both Python APIs translate filesystem/SQLite failures during preparation, sealing, reservation, staging or graph submission to `INPUT_CHANGED`, with private exception type; existing typed errors retain their policy. |
| Unresolved initial or saved pin | `SNAPSHOT_UNRESOLVED`, without fallback. |
| `code_root` that is not path-like | `INVALID_EXPERIMENT_SPEC`, never a leaked `TypeError`. |
| Changed explicit IDs | Saved-pin metadata admission gives R5 for excluded/unknown IDs; changed eligible membership gives R6. |
| Child admission | Exact staged identity, runner-source closure and single-thread environment are required before inputs; malformed staged `ValueError`/`TypeError`/`KeyError`/`RecursionError` gives `INVALID_EXPERIMENT_SPEC`. |
| Child success | Independent prior-session history/fitting/pricing; one create-only `native_gate_result.v1.0` supervisor-committed artifact. |
| Child failure | No successful output or sibling suppression; one attempt, no reusable checkpoint or cache. |
| Data refusal (pricing and evaluation) | Codes registered by ops retain ops category/retryability, including non-retryable `INPUT_CHANGED`. Data-only codes map to a canonical ops code with the data registry's category/retryability (for example `OBJECT_CORRUPT` → `INTEGRITY_FAILED`); the original code stays in private `data_failure_code`. |
| Evaluation input refusal | Registered `ArtifactError` codes retain ops semantics; foundation-only codes give `INTEGRITY_FAILED`. Caught evaluation validation failures give `EXPERIMENT_VARIANT_FAILED`, with private `data_failure_code`. Raw filesystem/SQLite failures give `INPUT_CHANGED` in admission or evaluation. Unexpected programming errors keep generic worker classification. |
| Evaluation receipt | Serialization `ValueError`/`TypeError`/`RecursionError` gives `EXPERIMENT_VARIANT_FAILED` with private exception type; receipt I/O remains `INPUT_CHANGED`. Neither publishes successful output. |

`v2_gate_variant.price_variant(repository, snapshot, spec, *, strategy,
as_of_month, event_ids=None)` resolves one immutable single-arm plan and
freshly prices its eligible canonical events from that snapshot's option
chains and calendar. It accepts a finite `fill` and optional `strategy`
economics declaration, which must match the explicit strategy argument.
The resolver permits that key only for `native-gate-replay`, the fixed native worker and `v2_gate.py`.
The admitted worker requires that declaration; manual fill-only calls retain
their explicit strategy input. Replay consumes the resolved fill. Returned
stage inputs bind the strategy, alpha, snapshot, event population and holdout
context. No prepriced trade frame or previous experiment output is accepted.
The source population uses `research.experiment_population.load_population`
with `purpose="sweep"`; raw quotes are read only after exclusion succeeds.
This is a pricing prerequisite, with no gate fitting or sweep lifecycle.

`python -m experiments.v2_gate_variant --no-ledger` is the smoke entrypoint.
It pins the current scope head once, opens the existing catalog read-only,
and prints execution evidence. It writes no report, artifact or ledger row.
This manual CLI is separate from the admitted smoke worker; neither authorizes promotion.

| Condition | Outcome |
| --- | --- |
| Unknown runner/strategy, external input files, invalid fill, or other economics | `INVALID_EXPERIMENT_SPEC` before population or quote reads. |
| Missing or invalid pin/context, or an explicit excluded event | Preserve the shared population reader's typed refusal; no pricing result or report. |
| Replay cannot price every eligible event | `EXPERIMENT_VARIANT_FAILED`; no partial result is returned. Native data integrity refusals propagate. |
| Repeated call or failure | No cache, internal retry, transaction or publication; recompute from the same pin and never retain a partial result. |
| CLI omits `--no-ledger` | Refuse before opening the catalog; recorded execution is unavailable. |

## Scientific gate sweep collection

| Condition | Outcome |
| --- | --- |
| Collection or observer requested | `native_sweep_collection.collect_sweep` returns immediately after sealed `submit_sweep` admission/exact job replay and collection. `v2_gate_collect.run_sweep` waits for submitted jobs to terminate or reach its positive finite timeout, then collects. Neither starts/stops workers or takes over supervisor ownership; timeout leaves jobs pending. No new scheduler; workers retain no recording authority. |
| CLI purpose, mode and context | Both actions require an existing migrated catalog and seal, explicit source/holdout context, and exactly one smoke opt-in or ledger destination. Non-sweep purpose refuses `HOLDOUT_ACCESS_DENIED`; missing/conflicting mode refuses `INVALID_EXPERIMENT_SPEC`, both before CLI catalog/input access. `main(argv)` prints sorted index/refusal JSON: exit 0 for complete success, 2 for incomplete results or handled refusals; unexpected programming errors propagate. |
| Collection/CLI dependency boundary | The collector consumes `v2_gate_sweep.submit_sweep`, native registration/outcomes, `gate_report.render_gate_report`, ops catalog transactions/bounded-text admission, registered artifacts, spec parsing, fingerprints, bounded native evidence readers/writers, job reads/`MAX_JOB_FAILURE_BYTES`/errors, contracts and foundation. `native_outcomes` consumes `ops.native_registration.read_native_evidence` and `native_evidence_json` for bounded publication/replay. The CLI consumes staged-input reading, ops catalog/job reads/errors, data `Repository`/errors and foundation artifacts; its only execution edge is to the collector. |
| Immutable collection context and replay | The first canonical arm owns immutable `native_sweep_{mode}_intent` and `native_sweep_{mode}_index` references without replacing other registration evidence. The intent binds roster hash, installed collector-source closure, mode and normalized ledger address; its hash is the collection run ID. Drift under an existing mode refuses R6 with immutable-context recovery. Smoke/recorded identities are separate but use the same staged jobs. Exact replay compares immutable index/result bytes without fitting, job retry, duplicate recording or evidence replacement. Complete content-addressed objects may remain unreferenced after interrupted reservation. |
| Unverifiable collector implementation | R6 (a declared `.py` file over `source_closure`'s 8 MiB per-file size bound is an unverifiable source and refuses `EXPERIMENT_IDENTITY_CONFLICT`); recognized propagated source-read I/O interruptions instead yield retryable `RESOURCE_UNAVAILABLE`. Before submission, admission requires the installed on-disk closure to match its import-time snapshot and both caller coordinator entries to match installed entries; a failed baseline or known installed drift refuses before further source reads. The collector's installed-root exception in root architecture §5 permits source comparison only: no catalog/artifact/network effects or bytecode-attestation claim, and no caller-data path uses that root. Other worker sources remain governed by the seal. |
| Terminal output admissibility | Success requires the job's exact terminal attempt number/state/fence, its sole named canonical `native_gate_result` and exact registered identity/provenance, including the reference derived from verified bytes; outputs from other attempts cannot substitute. Cancellation/recovery may advance the job fence. After exact-attempt admission, a succeeded attempt’s sole `native_gate_result` output naming an absent artifact registration refuses `INTEGRITY_FAILED`. |
| Generic worker failure or missing result | A dispatched child ending in generic `WORKER_FAILED` or with neither a result nor a typed refusal receives immutable coordinator R4 (`EXPERIMENT_VARIANT_FAILED`) in either mode, distinct from the original ops retry policy. In recorded mode its permanent CSV row is keyed by `(id, spec_hash)` with `stage=failed`, preventing a later `ran` row under that identity. The original ops problem/diagnostic reference remains in the index; worker records are not rewritten. |
| Registered refusal | Apart from the explicit generic-worker R4 accounting above, preserve code, category and retryability. Native-supported R4/R5/feature refusals use existing outcomes; identity and typed infrastructure refusals remain indexed without fabricated scientific outcomes. |
| Recorded success | Render the arm's own result, compare/publish the existing native outcome, reconcile its replay-keyed ledger row and export its report. A conflicting earlier single-arm outcome refuses R6. |
| Smoke | Publish only mode-labelled report/index artifacts and sweep-specific metadata. Never call native outcome/replay/export APIs, write native smoke/outcome slots, or touch a ledger. |
| Interrupted publication | Collector/observer storage interruptions are retryable `RESOURCE_UNAVAILABLE`; catalog constraint violations and malformed stored collection evidence refuse R6. Collection opts into R6 for queue/resource/progress receipt corruption, including initial submission replay; malformed stored failure JSON retains `INTEGRITY_FAILED`. Unexpected programming errors propagate, and delegated APIs retain their documented refusal policies. Changed sealed context remains R6. Intent and per-arm native intents resume ledger/report reconciliation; no final index exists until every terminal arm's applicable publication finishes. |
| Invalid collection input or active caller transaction | `INVALID_EXPERIMENT_SPEC` for rejected mode, path or roster admission, or a non-idle catalog connection; other input-admission filesystem I/O remains retryable `RESOURCE_UNAVAILABLE`. Ledger admission rejects non-directory or looping parents and overlong paths, permits absent destinations/parents, and requires any existing leaf to be regular without following symlinks; this preflight does not prevent later replacement races. |
| Pending, attempted or failed arms | Each admitted/dispatched variant counts once, regardless of success, refusal or failure; queued/cancelled jobs without an attempt count zero. Pending states, including `retry_wait`, have no native recording or frozen index and cannot complete a run. Once all children are terminal, the index is immutable with `complete=false` if any arm failed/refused, retaining successful siblings. |
| Native evidence byte admission | Collection's 16 MiB UTF-8 budget bounds native registration/outcome/completion/result JSON, registered-reference metadata and canonical evidence writes, including intent/index. Oversized or malformed native evidence refuses R6 before artifact reads. Registered-reference metadata admission instead retains `INTEGRITY_FAILED` for oversized, non-text or invalid-encoding values; admitted metadata that cannot be decoded as a reference refuses R6. Existing intent/index references must match the reference derived from bounded expected bytes; verified reads remain no-follow and bounded by registered size. `max_evidence_bytes` covers submission, registration verification and native outcome/replay/export; default None preserves existing callers. Persistence remains owned by SQLite transactions, foundation identity/artifact verification and native publication; filesystem work stays outside catalog write transactions. |
| Existing report comparison | Collection accepts only an exact byte match through a bounded, no-follow regular-file read; oversize, growth or differing bytes refuse R6. This comparison bound follows expected report size, not a report-wide JSON ceiling; callers that do not opt in retain existing limits. |
| Stored job/attempt failures and authority | Collection explicitly opts into a 1 MiB UTF-8 ceiling for job-failure reads through initial submission replay and observation; its attempt-failure reads use the same ceiling. Oversized failures refuse R6 before decoding, without truncation. Wrong-type, invalid-encoding or malformed job failures retain `INTEGRITY_FAILED`; malformed attempt failures refuse R6. The shared default is 8 MiB, with explicit larger finite budgets for historical recovery. Future oversized supervisor failures preserve the full original `problem.v1.0` artifact and persist a bounded summary/ref. SQLite internals and other receipt fields are outside this failure-field bound. Collection grants no final-read, promotion or winner authority; reconstructed-history/holdout limitations remain unchanged. |

## Evaluation recording and promotion receipts

| Condition | Outcome |
| --- | --- |
| `record_evaluation_result` has no `run_dir` or its results are not a `Mapping` | Raise `LedgerError` before appending a `ran` row. |
| `publish_receipt=False` | Append the `ran` row; skip artifact validation, finalization, and receipt publication. |
| Receipt publication is enabled and the artifact matches the run and spec | Append the row, finalize the checklist and `recording_mode`, then publish a receipt for the exact final bytes. |
| Validation or receipt publication fails after the append | Keep the `ran` row; the row alone cannot authorize promotion. |
| Promotion consumes an artifact | Require matching artifact, requested spec, receipt, run ID, digest, and qualifying ledger row; otherwise refuse with `PROMOTION_LEDGER_RECEIPT_MISSING`. |
| An `EvalResult` is recorded | Use its own `run_dir` for its metrics artifact, including for grid arms. |

## Durable experiment CSV publication

`lib.ledger_append` and `ledger_ensure` serialize cooperating writers through
one stable `<ledger>.append.lock`, including first-file creation. Publication
preserves the exact previous byte prefix and exposes only complete CSV records.
Success guarantees file and directory durability, including interrupted parent
creation; unrelated ancestors need only traversal access. Existing access-mode
bits are retained; a newly created ledger is private.
This lock is distinct from the ops refusal identity lock, which may enclose an
append; the append helper never acquires that outer lock. Direct file editors
do not participate in this concurrency guarantee. No catalog transaction is
opened, and callers remain responsible for their outcome/receipt reconciliation.

| Condition | Outcome |
| --- | --- |
| Missing required row/header fields, invalid replay key or directory-creation marker, malformed CSV, exceeded CSV field limit, unterminated existing record, or symlinked destination | `LedgerError`; no CSV replacement. |
| `unique_by` omitted | Append every supplied row, preserving existing legacy duplicate-row behavior. |
| Optional `unique_by` names a previously stored key | An identical complete row is a no-op; contradictory or duplicate stored rows refuse with `LedgerError`. The return value counts newly appended rows. |
| Temporary write or file fsync fails before replacement | Prior CSV remains; a killed process may leave an unreferenced temporary file. No automatic retry. |
| Parent creation or its sync fails | No CSV publication. Persistent directory-creation markers let a retry finish only the required parent syncs. |
| Directory fsync fails after replacement | New complete CSV exists, durability is uncertain; no rollback. Exact keyed replay can complete directory sync without duplicating rows. |
| Existing CSV changes before replacement outside the shared lock | Refuse the observed conflict without restoring stale bytes. Concurrent uncooperative edits are unsupported. |
| Repeated `ledger_ensure`, or identical keyed replay | Existing CSV bytes remain unchanged. Neither operation emits a recording or promotion receipt. |

## Native prediction prior baseline

`prediction_report.validate_prediction_spec` admits only the Python runner
`experiments/v2_prediction.py`, `computed_moves`, empty economics and ordered,
unique four-digit year folds, with an integer seed in [0, 2**32) bound as run
identity. The prior estimator is deterministic and does not consume
randomness, so `prediction_result` supports only seed `0` and refuses other
valid seeds before computed-move outcome reads rather than implying they
change the computation. Registration may still bind another valid seed as
identity.
`prediction_result(repository, registration)`
reads only registered eligible targets from the exact registered snapshot/month.
Each fold fits `DummyClassifier(strategy="prior")` through the shared fold helper
on earlier-year labels available strictly before its January 1; constant
intercept features make no market-feature claim. Fitting limits numerical
thread pools to one within a restoring context. Empty train/test or single-class
training refuses `EXPERIMENT_VARIANT_FAILED`; unsupported declarations refuse
`INVALID_EXPERIMENT_SPEC` before target reads. No final-read option exists.
The deterministic JSON result includes OOF/per-fold Brier and clipped log-loss,
ten fixed reliability bins with weighted ECE, and score/event-ID rank deciles.
Empty bins use null rates. Random and rolling holdouts are separately excluded.
`render_prediction_report(..., spec=..., no_ledger=...)` binds the supplied
hypothesis by spec hash and renders these metrics, provenance and limitations.
These functions have no cache, retry, writes or partial-result return; the caller
owns admission/publication/recording. No return, P&L or promotion claim follows.

`python -m experiments.v2_prediction register|run` consumes a JSON spec and an
existing catalog/store and explicit holdout month. It fingerprints its executing checkout. Register is
metadata-only. `read_native_registration` verifies the stored identity; run performs current
admission before replay or evaluation; missing registration is R1. R3/R4/R5
refusals use that admitted identity and never export a current-attempt report.
R1/R2/R6 preserve private receipts only. Untyped evaluation/rendering errors use
retryable `WORKER_FAILED`, never terminal outcomes or CSV rows. `--no-ledger`
isolates smoke recording; otherwise `--ledger` is required. Completed `REPORT.md` is exported through the
shared immutable outcome lifecycle; retries reuse the original result/date.
Stored spec/source/snapshot/scope/environment identity is checked before population
admission, so a changed request cannot consume an original variant's refusal slot.
A current holdout denial preserves R5 and its private receipt even when an earlier
immutable completion prevents a second terminal CSV row; prior reports stay unchanged.

## Bounded inner-trial execution

`SearchTrials.execute(outer_year, inner_year, candidate_id, limits=...)` is a public
native caller of the existing recorder and `magnitude_fitted.fitted_forecasts`.
It consumes one declared candidate and inner fold; it does not run outer evaluation,
reclaim starts, change the selection policy, write a ledger or grant final-read authority.
The source seal explicitly includes the fixed child and its transitive fitting sources.

| Condition | Outcome |
|---|---|
| Explicit positive integer `memory_bytes`, `cpu_seconds`, `wall_seconds` | Bind these limits to the immutable trial execution start. Linux child affinity uses the lowest currently allowed CPU; numerical thread variables are one. This selects a core without reserving it from other host work. |
| Concurrent `SearchTrials.execute` calls on one host | Serialize trials per host or size `wall_seconds` for peak concurrency (#845). |
| Missing/extra limit keys, Boolean/noninteger/nonpositive or platform-unrepresentable values | `INVALID_EXPERIMENT_SPEC` before a claim or subprocess launch. |
| Direct `run_fitted_trial` call | Requires exactly those three resource keys plus an integer `cpu_id` in `[0, 2**31 - 1)`, excluding Boolean values. Revalidates all limits before input access, scratch creation or launch; malformed resolved limits refuse `INVALID_EXPERIMENT_SPEC`. CPU availability is still enforced by child setup, not reserved by this pure check. |
| Parent Linux affinity query unavailable or empty | `RESOURCE_UNAVAILABLE` before a claim or subprocess launch. Child limit/affinity setup failure instead preserves the post-claim unfinished attempt and refuses `RESOURCE_UNAVAILABLE`. |
| A buffered child-output record exceeds 16 MiB, or the child exhausts `RLIMIT_AS` loading its rows | `RESOURCE_LIMIT_EXCEEDED`; preserve the unfinished claim, publish no candidate failure and block freeze. The input is not serialized and has no byte cap. |
| Started trial with admitted cohort | A fixed fit-only subprocess streams stored features and qualified training targets from the manifest, never a repository, catalog or validation targets. Every horizon uses the existing fresh-fit pipeline and declared candidate recipe. |
| Child reaches a horizon fit | Parent independently reconstructs its matrix/event evidence and commits a non-replayable fit admission before acknowledging. Completion is separately acknowledged only after its immutable evidence commits. Counts distinguish admitted fit attempts from completed fits; an interrupted admission does not prove the estimator was entered. |
| Child returns a checked result after exit and reap | Commit immutable `execution-result:<trial_id>` with the exact native result, original attempt/limits and admitted input hash before execution/forecast/scoring publication. Successful fits publish complete forecasts before validation labels open; original scoring policy applies. Public `record` cannot finish an unfinished native attempt. |
| Candidate scoring refuses after successful native fits | Preserve the successful execution receipt and durable `refused` terminal. Initial scoring propagates `EXPERIMENT_VARIANT_FAILED`; replay returns the intact receipt without refitting or rescoring, and freeze excludes that candidate while retaining successful peers. |
| Estimator-local `EXPERIMENT_VARIANT_FAILED` | Record the attempted/completed prefix and candidate failure. Other candidates and folds remain required by `freeze`. |
| Memory, CPU or elapsed-time limit; launch/import/pipe/protocol/abnormal-exit failure | Limits refuse `RESOURCE_LIMIT_EXCEEDED`; invalid JSON, missing events, messages after a result or evidence/completion mismatches refuse `EXPERIMENT_IDENTITY_CONFLICT`; unsupported events/progress envelopes, setup, I/O, missing results and other infrastructure failures refuse `RESOURCE_UNAVAILABLE`. Retain the unfinished claim and block freeze; never convert these failures into candidate exclusion or retry, even for retryable Problems. |
| Resource enforcement | Child `RLIMIT_AS` bounds virtual address space and `RLIMIT_CPU` bounds process CPU time; the parent bounds child elapsed lifetime, including waits for journal acknowledgements. Parent I/O is not preempted; an expired child deadline is checked when it returns and after child exit, before accepting completion. These are per-process fit caps, not shared-host reservations or a cap on parent admission/journal I/O. Only fixed single-threaded native estimators run. |
| Native completion replay/freeze | Revalidate the durable child result, every admitted/completed horizon, matrix digest, complete forecast digest and ordinary terminal chain; launch nothing and reopen no native labels/features. Changed limits, missing/corrupt evidence and caller-submitted terminals refuse native replay. |
| Interrupted execution or publication without a committed child result | Immutable claims cannot restart. Reconciliation refuses missing/incomplete result evidence, including a failure before the result association commits; unreferenced artifact bytes confer no authority. Selection remains blocked. |
| `SearchTrials.reconcile(outer_year, inner_year, candidate_id, limits=...)` | Revalidate the sealed source/plan, exact limits, admitted cohort, durable result and complete fit-marker prefix; idempotently finish only missing execution/forecast/validation/terminal writes. Conflicting existing evidence refuses. No child launch, fitting, claim deletion/reset, forecast replacement or candidate-count reduction; sealed retries remain zero. |
| Reconciliation needs validation evidence | Reuse independently checked durable shared validation without reopening labels/features; otherwise revalidate current native inputs and publish forecasts before native validation-label reads. Shared-input/resource errors propagate without candidate exclusion. Only estimator-local failure or candidate scoring refusal creates a candidate terminal. |
| Reconciliation after terminal publication | Verify and return the original receipt without writes or native reads, under both no-fit guards. A refused terminal retains its successful fit receipt and does not rescore. |
| Fitting guard active | Both parent no-fit guards refuse before a new claim or subprocess launch; completed replay remains read-only. |

The child uses standard-library subprocess, selectors, resource limits and JSON; no
optimizer, alternate estimator pipeline or general-purpose scheduler is introduced.
Execution uses ops thread-environment constants and existing foundation/journal publication.
Reconciliation reuses those immutable SQLite/artifact primitives, not a new queue or retry engine.
Historical journals sealed to older recorder sources are not automatically recoverable;
missing native result/forecast bytes cannot be reconstructed or replaced by a new fit.
Scratch inputs are removed when the call exits. Completed evidence is consistency-checked,
not cryptographically authenticated against wholesale journal fabrication. Frozen selections
retain their existing submitted-forecast label and confer no additional execution authority.

## Exploratory RUNUP exit comparison

`runup_exit_report.compare_runup_exits` and `python -m experiments.v2_runup_exits`
compare a declared target/stop roster with the existing STR-RUNUP fixed-day control.
They use the merged native population, calendar, entry selector and exit walker;
there is no scheduler, model fitting, threshold search or production caller.
The closed request requires explicit unique event IDs, one fill alpha, explicit
`quote_price_times_quantity` or `entry_debit_fraction` threshold units and one to
eight named target/stop arms. Fractions are signed ratios of each position's positive
net opening debit at the declared alpha, including spread but excluding fees,
commissions and contract multipliers. No per-event thresholds, clipping or conversion
from percentage points is performed. Fractional reports additionally expose each
arm's effective quote-unit thresholds and entry-debit basis; P&L retains quote units.
The CLI requires an exact snapshot ID and holdout as-of month; its JSON input
must be a regular file bounded to 1 MiB. Admission precedes any quote read.
`--no-ledger` is an accepted no-op; both CLI modes are equally read-only.

The existing D−14 entry and last-pre-print exit use only observed snapshot sessions.
All admitted events must plan successfully. Contracts are selected once from each
entry chain using native STR-RUNUP (long ATM call/put, first expiry at least 30 DTE).
Selected DTE must agree with expiry minus entry day; ATM ties use lower strikes.
A fixed-day walk validates every held leg on every session through the common
14-session horizon before any target/stop arm runs. Missing/ambiguous/unusable
marks, expiry overrun or incomplete calendar refuse the entire cohort, including
when an arm would have exited before the missing mark. No survivor subset returns.

Every arm retains those exact contracts, quantities, entry dates and fill alpha.
Daily crossing, equality and conservative ambiguous-stop behavior belong to
`experiment_exits.walk_exit`; ambiguous stops use alpha zero at exit. Monetary
outputs are quote price × quantity: spread is modeled, commissions/fees and the option
contract multiplier are absent. These are daily mark-based hypothetical results,
not intraday execution, stop-price fills, realized returns or an optimized strategy.

The JSON report contains paired per-event outcomes and deltas, all-arm counts,
entry contracts, snapshot/holdout pins, full recipes, source closure and environment
identity. Its canonical binding hash identifies reproducible inputs; a separate
report hash covers the complete output. This is exploratory evidence, not a sealed
preregistration, recorded hypothesis, immutable run or promotion authority.
The CLI opens the catalog read-only and prints once; neither helper nor CLI writes
reports, artifacts, ledger rows or orders. No refusal contains partial metrics.
There is no retry, fallback source, projected calendar or partial-success mode.

| Condition | Refusal and context |
| --- | --- |
| Malformed/duplicate-key, nonregular, oversized or invalid request | `INVALID_EXPERIMENT_SPEC` before repository access; no stage/counts. |
| Holdout exclusion or invalid admission context | `HOLDOUT_ACCESS_DENIED`, population stage/counts. |
| Entry selection, common coverage, fractional debit/conversion or pairing fails | `EXPERIMENT_VARIANT_FAILED`, current stage/counts; zero compared. |
| Repository integrity or unresolved snapshot | Native code; stage/counts only after comparison admission starts. |
| CLI filesystem or SQLite failure | `INPUT_CHANGED`, no stage/counts. |

## Selected-candidate outer evaluation

`magnitude_search_outer.execute_outer_fold` is a selected-fold runner. It requires an existing
frozen selection and completed native execution/terminal evidence for every inner
trial before public `freeze`/`execute` replay. Missing or submitted-only evidence
never launches inner work; permitted candidate failures retain sealed exclusions.
Exact inner-RMSE ties remain candidate-ID ordered. A distinct immutable `outer-*`
namespace binds the plan, selection, recipe, limits and installed outer/child sources.
Only that fold's winner receives bounded horizon refits. Every fit marker commits
before its child ACK; complete forecasts and successful accounting commit before
outer targets open. Training is strictly prior to the outer fit boundary.

`magnitude_search_evidence` shares native input admission, matrix identities,
forecast coverage and target-receipt validation across inner and outer callers.
SearchTrials wrappers preserve their existing error boundaries and Real/Integral
submission semantics; outer wrappers require JSON-native numeric types. Target
validation reconstructs the protected receipt from admitted physical preflight.
Static imports include these helpers in recorder and outer source bindings.
Completed replay verifies the complete evidence chain without native payloads or fitting.

Fresh inner/outer admission streams complete validation, retaining only fold participants'
primary keys and registered numeric features. Full stored `inputs_id`, matrices and fits are preserved.
Slim cohort documents have their own honest content hashes: their journal/result IDs
are not the old full-row hashes. Changed producer/recorder closures require fresh build contexts and seals; old sealed
runs fail closed, and no historical receipt or artifact is rewritten. The child
projects qualified training/evaluation rows and selected horizons/columns while streaming
shards. Child output and the journal retain their 16 MiB limits; the input cap is removed,
so a large fold is bounded by the child's memory limit. No scale is inferred
from event counts; scientific selection and cutoff rules are unchanged.

| Condition | Enforced outcome |
|---|---|
| Invalid outer year or limits | `INVALID_EXPERIMENT_SPEC`; no outer claim or payload access. Limit admission precedes journal/plan checks. |
| Missing/older/newer or checksum-mismatched journal migrations | `INTEGRITY_FAILED` before plan/evidence/native reads; no implicit migration. |
| Missing frozen selection, incomplete/submitted-only inner evidence, changed source/selection/input/result binding | `EXPERIMENT_IDENTITY_CONFLICT`; no missing inner execution or replacement. Prior journal associations remain unchanged. |
| Either no-fit guard on a new outer call | Its original `RuntimeFitForbidden` propagates before affinity resolution/claim; completed replay never invokes either guard. |
| Invalid child progress, completion or forecast coverage/types | `EXPERIMENT_IDENTITY_CONFLICT`; retain committed fit evidence, open no outer targets, publish no terminal. |
| Selected estimator fails after one or more horizon starts | `EXPERIMENT_VARIANT_FAILED`; persist exact attempted/completed prefix and failed execution, with no forecasts or successful terminal. |
| Shared native-input/target admission or resource failure, including late/missing outer targets | Preserve the owning typed refusal; retain the unfinished claim and any durable forecasts, with no partial successful result. No retry or replacement is authorized. |
| Repeated unfinished outer claim | `EXPERIMENT_IDENTITY_CONFLICT` at create-only start; no refit, replacement or publication recovery. |
| Artifact publication/read failure, 16 MiB journal cap, or raw SQLite failure | Existing journal policy applies: integrity conflicts, resource unavailability, `RESOURCE_LIMIT_EXCEEDED`, or unchanged raw SQLite error. Keep committed evidence/unreferenced complete objects; no retry or rollback of earlier records. |
| Invalid replay evidence or target receipt | `EXPERIMENT_IDENTITY_CONFLICT`; no native payloads, fitting or new associations. |
| Missing any outer terminal during collection | `EXPERIMENT_IDENTITY_CONFLICT` before any outer replay; no writes or missing-work launch. |
| Metric computation refuses | `EXPERIMENT_VARIANT_FAILED` propagates; no successful terminal/report or candidate substitution. |

`collect_outer_results` inventories all outer terminals/limits, admits native inner evidence once per call,
then verifies every fold's source/evidence bindings through a replay-only internal path. It returns a deterministic
read-only aggregate; no admission is cached across calls. Fold/horizon metrics retain complete forecast-target pairs;
primary RMSE pools event/horizon errors rather than fold RMSEs. Equal complete
horizon counts preserve equal event weights. The prospective recipe is always the
latest outer fold's frozen INNER winner. Journals own source/fit/input evidence;
returned result dictionaries carry their IDs and selected summaries, not all journal
payloads. Collection does not persist a report artifact, export or ledger row.
No holdout evaluation, final read, release or deployment authority is added.
Research remains reconstructed, not observed point-in-time. Direct dependencies are
the search plan/journal/trial/execution/evidence/inputs (cohort loader) modules, contracts, foundation,
ops migrations/fingerprints/errors, both no-fit guards, `engine.v2.research.magnitude_inputs.load_magnitude_targets`
for outer label reads and `magnitude_report.magnitude_metrics` for fold/aggregate scoring.
The evidence helper separately consumes native history admission and research stored-history/target APIs; no new scheduler, parser, estimator implementation or third-party dependency is needed.

## Serial search orchestration

`v2_magnitude_search_run` composes public plan, journal, trial and outer APIs. Separate `seal`, `run` and read-only `collect` commands retain explicit ownership; run never migrates or seals implicitly. A distinct driver seal binds plan/ref and full driver source closure; atomic absence guards prevent blessing started library journals. Run and CLI collection
require that exact binding, without changing the scientific report hash. The complete roster is checked for unfinished claims and premature selections before new work. Durable inner results reconcile first; completed outer results replay next, before any fresh serial fitting. Scientific scoring refusals continue only after public reconciliation verifies the
durable refused terminal; shared/resource failures stop. Freeze precedes all selected outer evaluation, with zero fit retries and unchanged winner policy.

The CLI reuses the bounded staged-input reader, strict duplicate-key decoder, typed spec/reference admission and shared canonical JSON stream. Output is limited by a configured UTF-8 byte budget; an oversized report is discarded without truncation and only the refusal document is written to stdout. Collection preserves the existing outer report identity and uses a read-only
SQLite connection. No persisted report framework, scheduler, ledger, final-read or release authority is introduced.

Each invocation revalidates bindings/evidence; there is no cross-call admission cache. The CLI calls `seal_search_run`/`run_search`; focused research tests also call these helpers. Direct engine dependencies are contracts, data, foundation and ops.

| Condition | Enforced outcome |
|---|---|
| Invalid CLI flags/resource arguments, registration-read budget or embedded-NUL paths | Parser exit 2 with stderr and no stdout or storage. All actions share the loader's pure budget validator; `--max-registration-bytes` defaults to 16 MiB and affects only initial registration loading. Direct `run_search` limit admission instead refuses `INVALID_EXPERIMENT_SPEC` before journal access. |
| Missing metadata, catalog or store; raw artifact, filesystem or SQLite failure | CLI emits sanitized `RESOURCE_UNAVAILABLE`, exit 2; absent inputs are not created. |
| Malformed/duplicate-key JSON, nonobject metadata, invalid typed reference, or nonregular/symlink/oversized metadata file | `INVALID_EXPERIMENT_SPEC` before dispatch; no partial result. Metadata reads are capped at 16 MiB per file. |
| Run/collect with missing or incompatible journal migrations | Driver admission refuses `INTEGRITY_FAILED` before search-plan/evidence access; no implicit migration. |
| Missing/different driver seal or changed installed source/plan binding | `EXPERIMENT_IDENTITY_CONFLICT`; no fitting or binding replacement. An unbound started journal cannot acquire a driver seal. |
| First seal, exact seal replay or concurrent phase publication | Seal uses the existing journal migrations and immutable append; phase absence is checked in the association transaction. Exact replay is read-only; a lost absence race refuses without a driver association. |
| Unfinished/orphan inner or outer claim, or selection before fresh inner work | `EXPERIMENT_IDENTITY_CONFLICT` before fresh fits; no claim reset or fit retry. |
| Durable inner result or completed outer terminal | Verify/reconcile all claimed inner results and replay completed outer folds before fresh fits; finish only missing inner publication, never refit or replace evidence. |
| Candidate scoring refusal; shared/resource failure; no eligible candidate | Continue only after public reconciliation verifies the durable refused terminal; other failures stop. No eligible candidate refuses `EXPERIMENT_VARIANT_FAILED` before new outer work. |
| Journal publication/transaction fails | Existing journal policy applies: failed transactions roll back their association, earlier evidence and unreferenced complete objects remain; no whole-run rollback or automatic retry. |
| CLI typed `OpsError`/`DataError`; unexpected builtin failure | Preserve the complete typed `Problem` in one refusal document, exit 2; unexpected builtin errors propagate. An opened connection closes in either case. |
| Output exceeds the byte budget or output staging fails with an artifact, filesystem or SQLite error | Report bytes are discarded; only the refusal document reaches stdout, emitting `RESOURCE_LIMIT_EXCEEDED` or sanitized `RESOURCE_UNAVAILABLE`. Other staging errors are unexpected builtin failures and propagate. Prior journal commits remain. |
| Stdout write or flush failure | The original stream error propagates with no retry and no second document; downstream may have received a prefix, and stdout remains caller-owned. |
