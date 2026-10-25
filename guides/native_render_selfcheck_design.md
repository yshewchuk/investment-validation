# Native board producer and publication selfcheck (#879)

Status: design, 2026-10-25. Contract level only; no code in this PR.
Replaces `legacy_render`, `legacy_selfcheck`, `legacy_model_evidence` and the
`legacy_bundle` row source. Authority: [Phase 7 cutover](rearchitecture_phase7_cutover.md),
[remaining work](cutover_remaining_work.md). Related: #717, #740, #681, #505, #818.

## 1. Standing decisions this design obeys

- Clean cutover: no board that dispatches between legacy and native per row or per night.
  New job kinds are added unused, wired in one PR, then the legacy ones are deleted.
- Legacy is removed at cutover. Pure functions move; outward-reaching ones are rewritten.
- Parity bar: per-field numeric tolerance before qualification; decisions may drift.
  Nothing here changes that bar or fixes #681. Qualification stays a separate gate.
- P6 bar: runtime reachability. Every new symbol below has a named production caller
  (section 7), and each code PR must show it, not just that the symbol resolves.
- Models-evidence page is deferred past cutover (cutover parity bar, Q2).

## 2. What exists today

| Piece | Where | Reads | Produces |
|---|---|---|---|
| `legacy_render` | `ops/legacy_adapter.py::_action_render` | legacy score doc, `model_evidence.json`, ledger generation, a legacy `Scorer`/`FeatureContext` | `bundle.tar` (`board`, `tickers/*`, `meta`, `health`, `flags`, `strategies`, `book`, `models`), `render.json` |
| `legacy_selfcheck` | `_action_selfcheck` | the bundle, a legacy `Scorer` | `selfcheck.json` (sampled re-score digest and field diff) |
| `legacy_model_evidence` | `_action_model_evidence` | legacy panel, trades, registry | `model_evidence.json`, with a degraded/stale flag |
| `legacy_bundle` row source | `serving/legacy_bundle.py`, `projections.build_candidate(row_sources=("legacy_bundle",))` | bundle `board`/`tickers/*` | rows the bridge compares with the score document |
| Native row source | `serving/native_shadow_render.py`, `native_render.native_display_row` | native `ScoreRequest` and `NativeScoreInputs` pairs | the same row shape, row source `native_shadow` |

Graph today (`ops/nightly.py`): `export, model_evidence -> projection(render) -> selfcheck -> publication`;
`native_score_batch` is only a tick-loop sidecar, so nightly never builds a native projection (#717).

Who reads what. The React client reads only `/api/v1/releases`, `/events`, score detail and
`/operations` and the native-parity routes (`/native_parity*`); all come from the serving index and the publisher, not from the bundle. The bundle
is read by (a) `load_legacy_bundle` for rows, (b) the operations listener's legacy shell
(`/legacy/`, `/board`, `/book`, `/models`, `/flags`, `/health.json` mirror), (c)
`tools/v2_dashboard_verified_input.py`. So the artifact the board needs is the release; the other
bundle members only feed legacy pages.

## 3. Native producer of each artifact

| Artifact (legacy name) | Native producer | Status |
|---|---|---|
| Board rows and per-event detail (`board`, `tickers/*`) | Serving release built by a new `serving_projection` job kind from persisted native score-batch records and refusals, snapshot-pinned | new (slice 3) |
| Row-level flags (`flags`, panel lag, calendar conflict, execution clock) | Per-row `flags` already on the display row, plus `operations_status.json` session fields; no separate flags document | exists; legacy document dropped |
| Health (`health.json`) | `operations_health.v1.1` (`ops/health.py`) and the ledger calibration `ledger_health.v1` file | exists |
| Model identity and release (`meta` model block, `strategies`) | `/models/release` view over the deployed model release (`models.deployment`) | exists |
| Book (`book`) | Ledger portfolio (`engine/v2/ledger/portfolio.py`); no bundle copy | exists; see question Q3 |
| Per-input model evidence (`models`, `model_evidence.json`) | None at cutover. `model_evidence_ref` stays `None`, which the release contract already allows | deferred (Q2) |
| Publication selfcheck (`selfcheck.json`) | New `native_selfcheck` job kind, section 4 | new (slices 1-2) |

Row source. The release's `row_sources` becomes exactly `("native",)`. The `native_shadow` name
says "shadow" after the pointer is no longer shadow, so it is renamed in the wiring PR (Q1).
It is part of the release-id derivation, so releases are rebuilt, not migrated, which the serving
doc already requires for any row-source change. `legacy_bundle` is removed from the accepted set
in the removal PR, not before.

Projection input rule. `serving_projection` reads the persisted batch records
(`native_score_batch_records.v2.0`) and projects them with `native_display_row`; it never calls
`score_one`. Re-scoring belongs to the selfcheck only. If a display field needs stage
observations the batch does not persist (the analog fields behind #681 are the known case), the
batch writer persists them in slice 3 and the schema version is bumped. No fallback fills a
missing field from any other source.

## 4. Native selfcheck (`native_selfcheck`)

Purpose. The legacy check re-scored sampled bundle rows through `engine.score.score` to prove
"the bundle still equals what the engine produces now". The native equivalent keeps that
question: does the projected release equal what the pinned code, snapshot and model release
produce now, and has no row been altered after scoring?

Inputs (all bound job inputs, none discovered; the batch records carry a per-record input hash
and `source_record_hash`, which slice 3 adds if absent): the `serving_projection` output (release rows and
`projection_binding`), the native score-batch records, the pinned snapshot and the deployed model
release identity.

Procedure.
1. Draw a sample of `N` projected rows (default 20, the legacy default) with a seed that is a
   content hash of `release_id` and the plan identity. Never a clock or `random` default seed.
   The sample is first the forced rows (one per strategy present, plus DYN-SV chooser rows up to
   a cap of 10), then random rows up to `N`. Forced rows count toward `N`; if they alone exceed
   `N`, the sample is the forced rows (size `max(N, forced)`), recorded in the output.
2. For each sampled row, rebuild its `NativeScoreInputs` from the pinned snapshot (not from the
   batch's persisted copy) and compare the input hash with the batch's recorded input hash.
3. Re-run `score_one` and compare (a) the record digest with the stored `source_record_hash`
   and (b) every displayed field at display precision with the release row. The precision constant
   (today `BUNDLE_PRECISION` = 6, defined in the legacy renderer) moves to a serving-contract
   constant in slice 1; the selfcheck must not import the legacy renderer.
4. A chooser row is verified against the structure it names: re-score `chosen_strategy`, restore
   the chooser name, require the digest to agree; `detail` is compared by shape. Same rule as
   today, now on native records.
5. Write `native_selfcheck.v1.0`: `ok`, `n_checked`, `n_release_rows`, `seed`, `sample_keys`,
   `mismatches` (scrubbed: field, stored and fresh value, no paths or exception text),
   `snapshot_id`, `release_id`. Elapsed time goes to stderr only, so the artifact is byte-stable. `ok` is false on any mismatch or any unbuildable
   input in the sample.

What it does not claim: it does not prove the model is good, the population is complete (the
projection's own findings do that) or that parity holds. It proves reproducibility and absence
of post-scoring edits for the sample.

Publication gate. The manifest `gates` key stays `selfcheck`. `publication._gate_is_bound`
binds it to a `native_selfcheck.v1.0` artifact whose `release_id` equals the staged release and
whose `snapshot_id` equals the projection's pinned snapshot and whose `ok` is true. A `legacy_selfcheck` artifact no longer binds after the wiring PR.

Cost. Work is bounded by `N` rows, not the universe: it loads only the snapshot slices for the
sampled tickers. This is the property #505 needed, so the interim 3 GiB `projection` profile and
the whole-table reads of `legacy_nightly_read_plan.py` are deleted rather than resized. The new
job kinds get profiles measured on a subset run by the owner of the wiring PR, with the
unmeasured full-universe fit stated as unknown in that PR, as the ops doc does today.

## 5. Failure semantics

| Concern | `serving_projection` | `native_selfcheck` |
|---|---|---|
| Missing input (batch records, snapshot, release identity not bound) | `VALIDATION_FAILED`, no release row, no index write | `VALIDATION_FAILED`; no `selfcheck.json`; publication cannot bind the gate |
| Score batch has refusals | Refused keys are kept as typed refusal rows; they are not dropped or rescored. Planned-population bridge findings still decide `PROJECTION_REFUSED` | Refused keys are not sampled |
| Findings not ok | `PROJECTION_REFUSED`, findings artifact kept, no release (unchanged serving rule) | n/a |
| Mismatch | n/a | Artifact is written with `ok: false`; publication refuses with `PUBLICATION_REFUSED`, `failed_gates: ["selfcheck"]`; the previous `CURRENT` release stays |
| Cannot run (OOM, snapshot unreadable, lease lost) | Job fails with the supervisor's typed failure; retry follows the existing attempt rules | Same. A failed check is never read as passed and there is no "unknown but publish" path |
| Retry | Unchanged supervisor retry. No in-job retry | Same. Seed is derived, so a retry draws the same sample |
| Idempotency | `release_id` is content-derived and index writes are `INSERT OR IGNORE`: a rerun on the same inputs writes nothing new | Output is a pure function of its bound inputs: byte-identical on rerun |
| Partial write | Objects before the index transaction as today; a crash leaves only unreferenced objects | Single atomic artifact publication through the existing worker commit |
| Transaction | Existing short `BEGIN IMMEDIATE` index transaction | None needed |
| Cache | None. Releases are immutable | None. Never reused across releases because it binds `release_id` |
| Degraded evidence | n/a | There is no degraded mode. The legacy "stale evidence beats a dark board" rule applied to model evidence, which is out of scope here |
| Logging | Progress to stderr: stage start/end, rows done/total, RSS at boundaries, per AGENTS.md | Same, plus sample size and seed |

## 6. Graph after the cutover

```
export ──────────────► score(native_score_batch, now a required stage)
                          └─► serving_projection ─► native_selfcheck ─┐
engineering_gate ─────────────────────────────────────────────────────┴► publication
```

- `model_evidence` is removed from the DAG. `projection` no longer waits for it.
- `native_score_batch` becomes a required stage that `serving_projection` binds as a hard
  input. The tick-loop shadow sidecar is deleted with the wiring PR (it is the only caller that
  today reaches the batch). This ties to #740; legacy decisions, settlement and the legacy
  `legacy_score` job are replaced by that work and are not redesigned here. This design
  states one dependency: the wiring PR cannot merge before the native batch is a required,
  snapshot-pinned stage with the full expected population.
- `prior_selfcheck_ref` and the ledger-generation staging in render are deleted: nothing native
  consumes them (health comes from `operations_health`).
- `ABSENT_STAGES` and `render_flags` in `render_inputs.py` move or die: the execution-clock
  and stale-session logic that is pure (`resolve_effective_session`) already lives in
  `session_resolution.py`; the rest has no native reader and is deleted.

## 7. Reachability and what is removed

Each added symbol and its production caller:

| Symbol | Production caller |
|---|---|
| `native_selfcheck` job kind | `nightly.GRAPH` and the supervisor job-kind registry; `publication` gate binding |
| `serving_projection` job kind | same graph; `publication` stages its binding |
| Row source `native` | `projections.build_candidate`, called by the job worker |

Timing: slices 1-4 add and wire the shadow path only; legacy authority stays until the P7-5
switch. Removal slices (5-7) merge only after that switch, and the rollback window in the
[Phase 7 plan](rearchitecture_phase7_cutover.md) decides whether deletion waits for it.
The removal PRs delete, with their tests and docs: the actions `legacy_render`,
`legacy_selfcheck`, `legacy_model_evidence` and their `ACTION_NAMES`/profile/effects-graph/read-plan
entries (`legacy_nightly_read_plan.py`); `render_inputs` legacy parts; `legacy_bundle.py`;
`LegacyBundleError`; the `bundle_format` and `legacy` options of the projection tool; the legacy
branch of `tools/v2_dashboard_verified_input.py`; the operations listener's `/legacy/` bundle
hosting and the shell views fed by bundle members (`/book`, `/flags`, `/models`, `/explorer`, `/`
shell) once Q3 is answered. Deleting the unbounded `load_legacy_bundle` reads closes the legacy
half of #818 and the legacy listener half of #799.

`LegacyScoreBridge`, `LEGACY_DISPLAY_MAPPING_V1` and `bridge.py` are contract names that outlive
the legacy producer. They are renamed in a dedicated PR after the wiring PR (release ids already
change at wiring, so the rename adds no second rebuild).

Standard-tool check: tarfile, hashing and JSON schemas used here are the existing ones;
no hand-rolled standard format is added. The sample draw uses a seeded `random.Random`
over a sorted key list, not a new PRNG.

## 8. Questions for the user

**Q1. Row-source name after cutover.** The release's `row_sources` string.
(a) Keep `native_shadow` until the pointer switch (P7-5). Release ids stay stable until then.
(b) Rename to `native` in the wiring PR. One rebuild, and the name stops saying "shadow".
Recommendation: (b), because releases are rebuilt at wiring anyway.

**Q2. Per-input model evidence page.** The legacy `models` page shows per-feature statistics.
(a) Retire the job kind and page at wiring; `model_evidence_ref` stays `None`; the
`/models/release` view carries model identity. This matches the standing "deferred post-cutover".
(b) Keep `legacy_model_evidence` alive until a native producer exists. This breaks "legacy removed
at cutover" and keeps the memory-heavy legacy panel read in the nightly.
Recommendation: (a). A native producer from frozen-release fit artifacts is a post-cutover issue.

**Q3. Legacy shell pages `/book`, `/flags`, `/explorer`, `/models`, `/legacy/`.**
(a) Retire them in the removal PR; the React app plus `/operations` and `/models/release` are the
cutover surface. (b) Keep them until React has a book and strategies screen.
Recommendation: (a), but please confirm the React app does not owe a book or strategies screen
before cutover; if it does, that screen reads the ledger portfolio directly, not a bundle copy.

**Q4. Selfcheck strength.** (a) As in section 4: rebuild inputs from the pinned snapshot for the
sample, then re-score. (b) Re-score from the batch's persisted inputs only: cheaper, but it
cannot see an input corrupted at batch time. Recommendation: (a); the extra cost is the sample's
snapshot slices only.

## 9. Documents to update in the code PRs

`engine/v2/serving/ARCHITECTURE.md` (row sources, legacy bundle exception, dependencies),
`engine/v2/ops/ARCHITECTURE.md` (graph, job kinds, `legacy_render` memory paragraph),
`engine/v2/dashboard/ARCHITECTURE.md` (legacy-companion sentence), root `ARCHITECTURE.md`
(consumer/legacy lists), `checks/legacy_adapters.json` count, `engine/v2/serving/README.md`, `engine/v2/ops/README.md`.

## 10. Slice size estimates (non-test code / test / doc lines)

| # | Slice | Code | Test | Doc |
|---|---|---|---|---|
| 1 | Pure `native_selfcheck`, schema, precision constant | 250 | 250 | 20 |
| 2 | Job kind, worker, gate binding, profile | 300 | 200 | 40 |
| 3 | `serving_projection` job, batch hash fields | 380 | 250 | 40 |
| 4 | Wire graph, `native` row source, required batch stage | 300 | 250 | 80 |
| 5 | Remove legacy actions, render_inputs, read plan | 60 | 80 | 40 |
| 6 | Remove bundle loader, tool options, shell pages | 60 | 80 | 40 |
| 7 | Rename bridge names | 120 | 120 | 40 |

Removal slices are mostly deleted lines, which do not count toward size. Slice 3 is nearest the
limit; if the batch change grows, it splits into batch fields, then projection.
