# `engine/v2/scoring` — architecture

Layer **5.0** in the root `/ARCHITECTURE.md` layer table (§2). Replaces
`score.py` (split by stage), `entry_rules.py`, `replay.py` and
`trailing_cutoff` from `pnl_sim.py`. See `engine/v2/scoring/README.md` for
the exhaustive, checker-enforced Public interface / Consumers lists
(`checks/package_readmes.py` fails an import of a name absent from that
list); this doc gives the structural picture the README does not.

## Purpose

The scoring application: the §6.3 stage pipeline that turns answer-free
source material into one immutable `ScoreRecord` — forecasts, geometry,
pricing, analog/simulation summaries, gate and DYN-SV chooser decisions,
financial diagnostics — plus the identity/replay/frozen-batch machinery
around it. It never reads a future outcome (that is `engine/v2/evaluation`)
and never fits a model (that is `engine/v2/models/training`); every model,
residual pool, payoff calibration and analog population it reads is frozen
data, verified by content hash before use.

As of this PR the package also owns the one production reader of a live
deployment's frozen release catalog (`release_bindings.py`, below) — the
piece `guides/rearchitecture_phase5_runbook.md` names as missing: "no
production path resolves states from `phase5_release.json` yet: that is the
Phase 6 cutover." Nothing calls it yet; a later PR (cutover wiring PR-3)
wires it into the per-night `SourceBundle` assembler.

## Primary contracts and public interfaces

Full list: `engine/v2/scoring/README.md` § Public interface. The load-bearing
entrypoints:

- `application.py` — the shared kernel: `score_one(request, NativeScoreInputs) -> ScoreRecord`,
  `score_many`/`score_batch`/`score_event` (batch variants over the same
  one-request kernel), `score_frozen(request, inference, release,
  inference_request, fields) -> ScoreRecord` (verified frozen inference),
  `replay(score_id, records, legacy_fields) -> (ScoreRecord, ReplayReceipt)`.
- `source_inputs.py` — `SourceBundle` (answer-free source dataclass) and
  `build_native_score_inputs(bundle) -> NativeScoreInputs`, the boundary
  every acceptance/parity caller starts from.
- `frozen_executor.py` — `FrozenStageExecutor`/`FrozenRecipeExecutor`,
  running one frozen-model binding through `engine.v2.models.loader.
  FrozenInference` and raising `FrozenStageRefusal(code, detail,
  reason_codes=...)` (code is always `"MODEL_NOT_READY"` for an unresolved
  or hash-mismatched binding) — the refusal shape every other module in
  this package that resolves a frozen artifact reuses.
- `frozen_batch.py` / `frozen_inputs.py` — the Phase 6 production frozen
  batch boundary (`score_frozen_batch`) and inference-input builder
  (`build_inference_requests`, `validate_answer_free`).
- `release_bindings.py` (new) — `resolve_release_binding(release_root) ->
  ScoringReleaseBinding`: the production reader of a live deployment's model
  identity, model artifact refs, and analog/payoff/recalibration artifacts.
  See its own section below.
  `ScoringReleaseBinding` (frozen dataclass, every mapping field a read-only
  `MappingProxyType` set in `__post_init__` — the `FrozenStageResult`
  convention `frozen_executor.py` already uses) fields: `release_id: str`;
  `model_identity: Mapping[str, ModelIdentity]`, **keyed `"{role}:{strategy_id}"`**
  (never by `role` alone — a release binds `role="gate"` twice, once per
  strategy, e.g. `("gate", "STR-THRU")` and `("gate", "STR-RUNUP")`, so
  `role` is not a unique key; two bindings that resolve to the same
  `"{role}:{strategy_id}"` key is itself a release defect and refuses
  `ModelNotReady` naming that key, not a silently-overwritten dict entry);
  `model_artifact_refs: Mapping[str, str]`, **keyed by `binding_id`**
  (every release binding's `binding_id` is unique by construction, unlike
  `role`). `ModelIdentity` carries `role`, `strategy_id`, `decision_clock_id`,
  `model_id`, `binding_id`, `adapter`, `feature_order`, `output_names`, and
  `artifact_hash` (`engine.v2.foundation.content_hash` over the binding's own
  sorted, verified member content hashes — never re-derived from unverified
  bytes; `model_artifact_refs[binding_id]` carries this same value).
  `payoff_artifacts`, `recalibration_artifacts`, `analog_artifacts` are each
  `Mapping[str, tuple[...]]` **keyed by the loaded artifact's own `.strategy`
  field** (never by the manifest row's declared `strategies` list — the
  artifact's own field is what a per-request lookup can trust), holding
  every hash-verified, causally-keyed object of that family the release
  stages (a family can hold several per strategy: e.g. one
  `payoff_line:STR-THRU` object per calibration fold). Picking the one
  object matching a request's own `(strategy, alpha, cutoff)` — over each
  artifact's own `.key` property — and mapping a resolved binding to the
  specific forecast-output name(s) `SourceBundle.model_artifact_refs`
  expects (e.g. `"driver_prediction"`) are both the per-request assembler's
  job (cutover PR-2/3), not this module's.
- `identity.py`, `financial.py`, `chooser_inputs.py`, `native_*.py` —
  content-addressed request/record identity, financial diagnostics, and the
  native (non-legacy) arithmetic for the analog stage, the DYN-SV chooser,
  the entry-rule gates, the payoff-calibration/model layer and frozen
  residual-state reading. Each is pure arithmetic over declared inputs; none
  performs I/O.
- `compatibility.py` — `score_legacy_request`, the one legacy adapter this
  package owns (see Dependencies).

## Inputs

- A `ScoreRequest` (`engine.v2.contracts`) and either a hand-built
  `NativeScoreInputs` or a `SourceBundle` run through
  `build_native_score_inputs`. `SourceBundle` carries raw context, raw
  quotes, a feature vector and missing mask, model identity/artifact refs,
  and recipes (never calculated forecasts, selected legs, prices, or
  decisions — enforced by `source_inputs.py`'s `_ANSWER_FIELDS` check).
- Frozen release state read through `engine.v2.models`: a `ModelRelease`'s
  bindings (`FrozenInference`/`FrozenStageExecutor`), payoff-calibration
  artifacts (`payoff_artifact.PayoffArtifactLoader`), recalibration maps
  (`recalibration_artifact.RecalibrationArtifactLoader`), and the other
  frozen non-model states (`frozen_state.FrozenStateLoader`) — driver/paired
  residual pools, the admissible-depth table, the board analog matcher, the
  DYN-SV chooser's analog pool, the entry-rule gate's trailing cutoff.
  `release_bindings.py` is now the one production caller that resolves
  *which* release and *which* artifacts from a live deployment pointer,
  rather than a test/acceptance harness supplying them directly.
- The legacy `Scorer`/`ScoreRequest`/`FillModel` (`compatibility.py` only).

## Outputs

One immutable `ScoreRecord` per request (`engine.v2.contracts`), with every
nested mapping/sequence field deep-frozen (`frozen_record.py`) before it
leaves this package. `identity.py` derives the record's content-addressed
`score_id`/`request_hash` from the immutable payload, excluding operational
timestamps, so a replay of the same inputs reproduces the same id.
`release_bindings.py` additionally returns a `ScoringReleaseBinding` — an
in-process, immutable snapshot of one release's resolved catalog; it is not
persisted anywhere and carries no operational envelope of its own.

## Dependencies

May import layers 0-4 (`checks/layer_map.py`, `only_imports` unset, strictly
less than 5.0): `engine.v2.contracts` (0.0), `engine.v2.foundation` (0.5),
`engine.v2.features` (2.0, `default_feature_registry` in `application.py`),
`engine.v2.models` (3.0), `engine.v2.registry` (3.0), and the `engine.v2.domain.*`
packages (4.0/4.5). It must never import `checks` or `tests`
(`checks/import_layers.py`'s `verification-imported` rule) — every module in
this package, including `release_bindings.py`, ports logic it needs from a
`checks/` module rather than importing it.

One declared legacy adapter (`checks/legacy_adapters.json`, package
`engine.v2.scoring`, module `compatibility.py`): `engine.score.Scorer`,
`engine.score.ScoreRequest`, `engine.fills.FillModel`, all confined to
`score_legacy_request`, removal targeted at "phase-5 scoring extraction."

Callers (checked against the import graph, README § Consumers):
`engine.v2.models.training` (layer 6, `native_payoff`'s pure fitting math
only), `engine.v2.ops` (`score_one`/`NativeScoreInputs`/`StageReceipt` for
the read-only `ops rescore` CLI), `engine.v2.serving.native_render` (layer
7, display-only analog row ids). `checks/phase4_frozen_bridge.py` and
`tools/capture_tier0_corpus.py` call `frozen_inputs.py` as compatibility
callers. `release_bindings.py` has no caller yet — a later cutover-wiring PR
wires it into the per-night `SourceBundle` assembler.

## External systems and libraries

No network. Local filesystem only: the deployment content-addressed store
and `phase5_release.json` (via `engine.v2.models`, read-only from this
package), and fold files elsewhere in the release
pipeline (not read directly by this package). `numpy`/`scipy.stats.norm`
(`stages.py`, `native_*.py` — deterministic arithmetic, no fitting) and
`pandas` (`compatibility.py`, legacy request construction only).

## Failure semantics

Package-wide invariant (root doc §5): a missing or unusable input produces
an explicit typed refusal, never a silent default — `FrozenStageRefusal`
(`code="MODEL_NOT_READY"`, `reason_codes`, `missing_features`) is this
package's one refusal shape for an unresolved or hash-mismatched frozen
artifact, reused by every stage that reads one (`frozen_executor.py`,
`source_inputs.py`'s frozen-recipe/residual/analog/payoff/recalibration
readers). No stage caches beyond the process (`FrozenStateLoader`/
`PayoffArtifactLoader`/`RecalibrationArtifactLoader` each hold an in-memory,
per-instance, content-hash-keyed cache only — never written to disk), no
stage retries, no stage performs a partial write (scoring is read-only
except `frozen_record.py`'s in-memory freeze), and every stage is
idempotent: the same `ScoreRequest` + `NativeScoreInputs` always produces
the same `ScoreRecord` (`identity.py`'s content-addressed `score_id`).

### `release_bindings.py` (the 4c R1–R6 template)

- **R1, missing input.** `resolve_release_binding(release_root)` validates
  its one argument (a non-empty path) before any I/O. `deployment.
  current_pointer()` returns `None`, rather than raising, when nothing has
  ever been promoted at `release_root`; this module turns that into a typed
  refusal, `NoCurrentRelease` (distinct from the per-member refusal below —
  there is no release to resolve members *against* yet). A malformed or
  nonexistent `release_root` resolves to the same refusal (`current_pointer`
  treats a missing `DEPLOYED` file and a missing directory identically), not
  a crash.
- **R1 (continued), a member missing or corrupt.** Once a release is
  resolved, every model binding's member objects and every declared
  payoff/recalibration/analog state object is hash-verified and typed-loaded
  before this module trusts it. A missing file, a status other than
  `STAGED`, a payload whose sha256 disagrees with its declared
  `content_hash`, or a document a typed loader rejects (`PayoffArtifactError`/
  `RecalibrationArtifactError`/`FrozenStateError`) all raise `ModelNotReady`
  (`code="MODEL_NOT_READY"`, matching `engine.v2.models.contracts.
  MODEL_NOT_READY` and the `FrozenStageRefusal` convention above), naming
  the exact `member_id` (e.g. `"model:gate:STR-THRU"`,
  `"payoff_line:STR-THRU"`, `"board_analog_matcher"`) and, where useful, the
  specific object name inside it. The `phase5_release.json` catalog itself
  is verified the same way before any of its rows are trusted: it must
  parse as a JSON object whose own declared `manifest_hash` matches a fresh
  `content_hash` of its other fields, and its `release_id` must equal the
  live pointer's `release_id` — the layout assumes one catalog file
  describes the release root's *current* release, an assumption a rollback
  to an older staged release would break — a `phase5_release.json` that
  fails either check raises `ModelNotReady("phase5_release.json", ...)`
  before any state-family row is read.
- **A hash mismatch never falls back.** There is no branch anywhere in this
  module that substitutes a different object, an older cached value, or a
  default when a hash disagrees — `ModelNotReady` is the only outcome. This
  is a property of the control flow (one verify-then-load path per member,
  no alternate source), proven by a test that a mismatched object never
  yields a resolved binding of any kind.
- **R2, cache.** None beyond the single call's own artifact loaders (each
  constructed fresh inside `resolve_release_binding`, discarded when it
  returns). Two calls against the same `release_root` re-read and
  re-verify every byte from disk; nothing is memoized across calls or
  persisted to disk by this module.
- **R3, retry.** None. A refusal is not retried internally; a caller that
  wants to retry (e.g. after promoting a new release) calls
  `resolve_release_binding` again, which re-resolves the pointer from
  scratch.
- **R4, transaction.** Not applicable — read-only, single-pass, no
  multi-step state to roll back. Resolution either completes and returns
  one immutable `ScoringReleaseBinding`, or raises before returning anything.
- **R5, partial write.** None possible: this module performs no writes.
  Every function it calls into (`deployment.current_pointer`/
  `resolve_release`, the artifact loaders' `.load`) is documented read-only
  in its own module; `release_bindings.py` adds none of its own.
- **R6, idempotency.** Resolving the same `release_root` while its pointer
  is unchanged always returns an equal `ScoringReleaseBinding` (frozen dataclasses,
  structural equality); resolving it again after a promotion or rollback
  reflects the new pointer, never a stale one, because nothing is cached
  between calls.

## Invariants

Root doc §5 invariants this package is responsible for: "Native vs. legacy
provenance" (`frozen_inputs.py`'s `validate_answer_free`), "Missing input →
typed refusal, never a silent default" (`FrozenStageRefusal`/
`ModelNotReady`, above), "No parity-only or legacy-emulation modes" (one
scoring code path; `checks/phase4_real.py`/`native_parity_report.py` do the
comparing, this package never branches on "are we being compared"), and
"Failure semantics are stated, not implied" (this doc's section above).
Package-specific: `release_bindings.py` never imports `engine.score`/
`engine.fills` or any other legacy module — it is pure v2, unlike
`compatibility.py`.

## Diagrams

```mermaid
flowchart LR
    SB["SourceBundle\n(answer-free)"] --> BNI["build_native_score_inputs"]
    BNI --> NSI["NativeScoreInputs"]
    NSI --> SO["application.score_one"]
    SO --> SR["ScoreRecord\n(deep-frozen)"]

    DP["deployment.current_pointer(release_root)"] -->|None| NCR["NoCurrentRelease"]
    DP -->|PointerState| RR["deployment.resolve_release\n+ phase5_release.json catalog"]
    RR -->|member missing/corrupt/\nhash mismatch| MNR["ModelNotReady(member_id)"]
    RR -->|every member verified| RB["ScoringReleaseBinding\n(model identity, artifact refs,\nanalog/payoff/recalibration artifacts)"]
    RB -.->|no caller yet;\ncutover PR-3 wires this in| SB
```
