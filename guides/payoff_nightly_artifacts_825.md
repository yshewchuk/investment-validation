# Nightly exact-cutoff payoff artifacts (#825, follow-up to #192)

Design PR, contract level. Refs #825, #192, #681, #849. Closes nothing by itself.

## 1. Decision and scope

User decision 2026-10-17: **retain only**. The nightly release adds the payoff
artifacts at the session cutoff; every earlier night's objects stay in the
release (carried, never replaced). A row's cutoff is `min(as_of, entry_date)`
(`native_score_batch._row_cutoff`). On its entry night that equals the session
cutoff, so the entry-night object exists; on later nights the same object is
read again. A row with no object under its exact key refuses `MODEL_NOT_READY`,
counted and documented. There is no latest-cutoff fallback, no nightly backfill
and no per-row release resolution. A stored entry-time record is #849, not here.

Out of scope: rows entered before slice 2 ships or on a missed night (they stay
refused until they expire; the standard rebuild, not nightly code, closes any
gap); driver pools and the chooser (monthly retrain, #805); the paired and
trailing pools (session cutoff only, already #192 slice 4).

## 2. What the nightly advances

Selection is exact-key `(strategy, round(alpha, 4), cutoff)`
(`engine/v2/models/payoff_artifact.py`, `payoff_artifact_key`). Alpha is the
shadow fill alpha 0.5, as `native_score_batch._SHADOW_FILL_ALPHA`.

| Member | Advances nightly | Why |
|---|---|---|
| `payoff_line:STR-THRU` | yes | read by the scorer (`CONSUMED`, `tools/phase5_calibration_keys.py`) |
| `payoff_surface:STR-RUNUP` | yes | the exact-key surface `native_runup_inputs._surface` requires (#782) |
| `recalibration_map:STR-THRU` | yes | read by the scorer at the same key |
| `payoff_line:STR-RUNUP`, `recalibration_map:STR-RUNUP` | no, carried | staged by the catalog but not read by the scorer; they keep their prior objects (supervisor default, flagged to the user) |

The board analog pool already advances at the session cutoff (#826), so the
analog block of #681 needs no new member, only the payoff members above.

## 3. Cutoff semantics

- One cutoff per night: the session's, as in #826. The new objects are stamped
  with it. Object names carry the cutoff so keys are unique.
- Retention uses the existing carry rule: a prior object whose cutoff is not
  after the new one is kept (`_check_carried`, `checks/phase5_release_assembly.py`).
  Payoff member rows are copied from the prior catalog today (`_derive_catalog`);
  slice 1 changes only those three rows, by appending the new object to the
  prior objects.
- Assembly gains one rule per advancing payoff member: the new object's key
  must not equal a retained object's key with different content. Same key and
  same hash is idempotent (a rerun); different content refuses `ID_CONFLICT`.
- `pool_cutoff` still advances strictly. No object older than the session is
  ever added; that is why there is no backfill.

## 4. Slices (each under the size limit)

1. **Stage the calibration members.** `phase5_state_stage` and assembly accept
   the three payoff members as advancing, bound like the pool state files;
   contract text goes into `engine/v2/models/ARCHITECTURE.md` §2.2 in this slice.
   Depends on #826 (the §2.2/§2.3 text and the sidecar).
2. **Train and wire.** The pool sidecar's phase 1 adds three `training` jobs
   (`mode=recipe`, `alpha=0.5`, `cutoffs=(session,)`) for the three members; the
   stage job binds their outputs; ops doc updated. Release id and keys unchanged
   in shape (the phase list grows, so keys include the member count).
3. **Refusal accounting.** A row whose exact-key payoff or analog object is
   absent refuses `MODEL_NOT_READY` with a distinct fixed detail
   (`no payoff object for the row cutoff`); the batch summary counts those
   refusals by strategy and cutoff, so parity reports separate "expired entry"
   rows from real defects. Scoring doc updated.
4. **Close-out.** Update #681/#782 with the measured post-deploy result and
   record the count of rows refused for a missing entry-night object.

## 5. Failure semantics (R1-R6)

| Rule | Behaviour |
|---|---|
| R1 verified input | Training outputs are plain `training/*.json` from succeeded jobs bound to the session's manifest; the stage job re-classifies by schema and member and verifies hashes; a payoff object's own cutoff must equal the session cutoff, key and alpha must match the request |
| R2 cache | None. Prior catalog, objects and training outputs are re-read each call; scoring selects from the loaded release with no memo across batches |
| R3 retry | Training jobs and the stage job are one attempt each (as #826); a failed phase stalls the cycle for operator recovery (#154). A failed payoff job blocks the stage, so a night with no advance keeps the prior release deployed |
| R4 transaction | New objects go to the shared content-addressed store; the staged manifest and catalog exist only under the new release id; promotion stays the guarded `models_promote` job. Payoff and pool members advance in one candidate or not at all |
| R5 partial failure | Orphan objects or a staged candidate may remain and a retry completes or refuses; the deployed release is unchanged. A night with no new payoff object leaves rows at their prior cutoffs refused or served by retained objects exactly as before |
| R6 idempotency | Same inputs give the same catalog bytes; a rerun with the same key and hash is a no-op; same key with a different hash, or the same release id with different content, refuses `ID_CONFLICT` |

## 6. Resource cost

Payoff and recalibration fits are least-squares over trade rows (cheap next to
the board analog build). Each is one `training` job running `build_dataset` for
its recipe, so the dominant cost is dataset construction, three times a night.
Heavy-job rules apply: jobs run one at a time under the supervisor's ops slot;
this PR and its owner never run them. The measured cost of the three jobs is a
slice 2 deliverable, reported before any further nightly work.

## 7. Open items for slice authors

- Confirm `recalibration_map:STR-THRU` is keyed by the same `(strategy, alpha,
  cutoff)` (`recalibration_artifact_key`) and that its recipe takes `cutoffs`
  the way the payoff recipes do (`FoldScheme("request_cutoff")`); if not, slice 1
  stops and reports.
- Whether the train-time dataset can be shared across the three jobs; only
  worth doing if slice 2 measures it as the cost.
