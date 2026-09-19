# Rearchitecture Phase 3B + 4 — remaining work

Status: rewritten 2026-09-18 from branch `codex/phase4-remediation-track-a` at
`9bf3a1c` (= `origin/main`); refreshed 2026-09-19 against `origin/main`
`6ba53e5` (each claim re-checked against code, git log and the real
`data/models/tier4` files; gates were NOT re-run for the refresh). The original
assessment was recorded at `129a368`; everything below reflects what was
actually merged and verified since, not that snapshot. Phase 5 progress is
recorded in the [Phase 5 guide](rearchitecture_phase5_models.md), not here.
Authority remains the [delivery plan](rearchitecture_delivery_plan.md),
the [3B data plan](rearchitecture_phase3_incremental_data.md) and the
[Phase 4 scoring plan](rearchitecture_phase4_scoring.md). Where this file
conflicts with those, they win and this file is wrong.

## Current standing

| Subject | Verdict at `9bf3a1c` | Source |
|---|---|---|
| Phase 3B | **CLOSED** — all eight items done | `guides/rearchitecture_phase3b_closeout.md` |
| 3B gate on retained evidence | PASS, 0 findings, 8/8 subjects | `checks/rearchitecture_phase3b.py --evidence <run>/evidence.json --artifact-root <run>` |
| Phase 4 foundation gate | FAIL, 5 findings | `checks/rearchitecture_phase4_gate.py` |
| Phase 4 completion review | BLOCKED, 3 blockers (P4-B01, B08, B09) | `checks/phase4_completion_review.py` |
| Phase 4 native audit | FAIL (P4N-002, P4N-003) | `checks/phase4_native_audit.py` |
| Saved-release parity | expected 20, compared **0**, all 20 `input_trace: missing` | `evidence.json::saved_release_comparison` |
| Test suite (phase4) | 137 passed, **1 failed** (red by design) | `pytest tests/test_phase4_*.py` |
| Phase 5 | P5-1..P5-6 merged (`9a54951`..`6ba53e5`); Phase 5 gate red by design until every release member lands | [Phase 5 guide](rearchitecture_phase5_models.md) |

The verdicts above are the `9bf3a1c` readings. They were not re-run at
`6ba53e5`. Nine of the ten Phase 4 findings trace to the missing capture
(below), which has still not produced a traced corpus, so they are expected
to stand; re-run the gates before quoting a new count.

The single remaining red is
`tests/test_phase4_completion_review.py::test_native_full_release_gate_is_phase4_completion`.
It asserts the Phase 4 completion criterion and stays red until the corpus
carries traces and Phase 5 lands. It is deliberately NOT rewritten to expect
failure — doing so would enshrine incompleteness in the suite.

## Closed since the original assessment

Phase 3B: R3B-1, R3B-2, R3B-4, R3B-5, R3B-8 (honest closeout slice, bounded
scope recorded as an accepted limitation rather than claiming
`production_acceptance`); R3B-6 (`STALE_EXPECTATION` registered); R3B-7
(finality private helpers routed through the legacy monkeypatch seam — the
native path had been silently falling through to a real ORATS read); R3B-3
(the nightly now submits `incremental_refresh`; legacy stays the default and is
pinned byte-identical by a regression test, because cutover is Phase 7).

Phase 4: R4-1 (strict trace capture reads per-role vectors and raises a typed
error), R4-3 (nine previously uncompared record fields, each with its own
failing-pair test), R4-4 (checkpoint bundle contract, both writer and reader,
validator finally wired in), R4-6 (bucket analog recipe expressible through
`SourceBundle`), R4-7 (analog stage no longer skips silently), R4-8 (straddle
expiry resolved before strike), R4-11 (acceptance path cannot construct
answer-bearing inputs), R4-12 (the two mis-named controls now mean what they
say). Cross-cutting: RX-1, RX-2, RX-3. RX-4 is closed — `main` and the branch
are pushed and identical.

Since the 2026-09-18 rewrite: R4-9 (native actionable flags, `451b330`, with
the advisory-vs-refusal taxonomy unified in `e05a009`), R4-17 (STR-RUNUP
native model layer, `86069bd`), R4-5 (analog independence control, this
refresh), and the Tier-4 cache blocker in the capture lane (below).

Two findings surfaced only because a fix made them visible, and both are worth
remembering:
* R4-7 proved native scoring had **never** computed analogs through
  `build_native_score_inputs`; two tests had been passing vacuously on the
  silence.
* R4-12's honest control flipped `full_saved_release_compared` from a
  champion-artifact hash check to the real comparison result, which is `False`.
  The gate finding count did not improve; its honesty did.

## The capture blocker — the critical path

Nine of the ten outstanding Phase 4 gate/audit findings collapse to one cause:
`population_expected=20, compared=0`, every case `input_trace: missing`, all ten
runtime stages uncovered. Only P4-B01 (Phase 5) is independent.

Two blockers in the capture lane were found and FIXED (merged at `e324251`):
1. The `STR-THRU`/`STR-RUNUP` allowlist in `canonical_v2_request` was
   **vestigial** — it predated R4-6 and the legacy `Phase4TraceCollector`
   already captured generically. Now derived as
   `STRUCTURES - DISABLED_STRATEGIES`.
2. `attach_strict_probe` was **all-or-nothing**: a real 48-candidate run wrote
   zero pairs because two rows were unsupported. Rows now probe independently
   and a row that cannot trace records a typed `trace_disposition: "gap"`.

A third blocker, the Tier-4 serving caches, is **RESOLVED**. Three capture
attempts had been SIGKILLed at the identical point:

    [train] implied_t1: 100,548 (event, decision day) rows

because the legacy `Scorer` trained `implied_t1` in-process when no usable
serving cache existed for the needed folds. The user authorised writing
derived caches into `data/models/tier4` (decision 4 below), and
`tools/prepare_phase4_tier4_caches.py` gained `--model` for every Tier-4
producer (`7973781`, merged `241c238`). Verified 2026-09-19 against the real
files: all 4 models (`size_v1_4`, `opf_implied_t1_gbm`,
`runup_move_d14_v1_gbm`, `iv_crush_v1_gbm`) × the capture's 14 folds
(202207, 202208, 202310, 202311, 202401, 202402, 202412, 202501, 202504,
202505, 202512, 202601, 202608, 202609) exist at the current snapshot
`a67873b4eb95` and all 56 embed `pool_pred`/`pool_res`. Twelve other files
at that snapshot, outside the 14-fold plan (for example 202211, 202301,
202610), are still old-format; they're not on the capture's path.

**The capture is still blocked, now on memory.** The forward-pass fixes
landed: the forward-corpus runaway (`54bc8f5`: slice one residual population,
stop swallowing `MemoryError`), the analog-evidence spike and its shared,
read-only row cache (`a929c1c`, `fd34bcd`, `25faaa8`), the residual
population re-copy (`53d2f8a`), and the per-candidate analog recipe
projection (`549d5f0`). The forward pass now completes. The run bursts at the
**boundary pass**, which is where it is killed now (reported from the
2026-09-18/19 capture attempts; not re-measured for this refresh, since it is
a heavy run). Until the boundary pass fits, the corpus cannot carry traces,
and Phase 4 cannot close.

## Phase 4 remaining

| ID | Item | Status |
|---|---|---|
| R4-2 | Record-by-record parity over the complete saved release | Blocked on the capture |
| R4-5 | Analog parity is unexercised | **Closed 2026-09-19** (commit `50d7a47`, based on `6ba53e5`). `_numerical_independence_control`'s bundle (now `_numerical_independence_source`) declares the legacy bucket recipe over a synthetic population. The native record's `exp_pnl_analog`/`win_analog`/`ci_low`/`ci_high`/`n_analogs` must agree, under the saved-release exact policy (`_compare_dimension`), with `engine.analogs.AnalogMatcher` on the same synthetic frame. Four planted defects must fail that comparison: a perturbed source row, a changed `min_analogs`, a changed bootstrap seed, and a row tampered after the hash binding. Poisoned analog outputs in the analogs block must not survive. New completion controls `native_analogs_independently_recomputed` and `analog_corruption_rejected` are in `final_controls`. Tests: `tests/test_phase4_acceptance_independence.py`. Real-corpus analog parity still waits on the capture (R4-2) |
| R4-9 | Actionable flags have no native derivation | **Closed** (`451b330`; commits `70e1cfa`, `3acd013`). `engine/v2/scoring/stages.py` derives `PROJECTED_CALENDAR`, `STALE_QUOTE`, `WIDE_MARKET`, `EXTRAPOLATED` and `OUT_OF_DOMAIN` from source-owned inputs, never from the legacy record's flags. Annotation-only flags don't refuse; the advisory-vs-refusal taxonomy was unified in `e05a009` |
| R4-10 | Planned-exit simulation not expressible through `SourceBundle` | Open. `_RESIDUAL_RECIPE_FIELDS` permits none of `pre_iv30`, `dte_exit`, `event_date`, `residuals` |
| R4-13 | Frozen Phase 5 inference not integrated | Open, progressed through Phase 5. P5-1..P5-6 are merged: inventory (`9a54951`), the no-fit guard and acceptance audit (`58b7ee6`, `3dea05e`), training recipes (`926cca9`), frozen payoff, recalibration and residual artifacts (`30e8563`, `3dea05e`, `52ef989`, `f2b4d88`), staged releases with an atomic pointer (`8c90b90`), and the P5-6 gate plus Phase 4 replay against the staged release (`33bec3d`, `6ba53e5`). Still emits P4-B01: `checks/phase4_real.py` sets `phase5_inference_integrated` true only when every final control passes, which needs the capture. Stays a Phase 4 blocker (decision 2) |
| R4-14 | Factory / refusal / DYN-SV parity | Partly closed: factory parity is 18/18 across all 11 strategies with every negative control rejecting. `chooser_corpus_present` still false |
| R4-15 | Planted-defect control incomplete | Blocked on the capture — a defect cannot be planted in a comparison that never runs |
| R4-16 | Forecast/gate recipes are inline linear only | Open |
| R4-17 | STR-RUNUP has no native model layer — parity regression, not a scope boundary | **Closed 2026-09-18** (commit `75167f0`, based on `02ec7a1`). `engine/v2/scoring/native_payoff.py` gained `fit_runup_payoff_surface`/`runup_exit_value_per_spot`/`simulate_runup_model_returns`/`scale_runup_move`/`runup_payoff_design`, pure re-derivations of `engine/payoff.py`'s `RunupPayoffSurface`/`fit_runup_payoff`/`simulate_runup_returns`, bit-parity tested against the legacy functions directly (incl. residual-capping-above-5000 and a reordered-draws negative control). `stages.py`'s `_PAYOFF_DRIVER_STRATEGIES` now includes `STR-RUNUP`, dispatched from `_execute_model` to a new `_execute_runup_model`/`_runup_model_inputs`/`_runup_fit_and_pool`, so `NO_PAYOFF_MAP` fires for STR-RUNUP only where legacy fires it — a fed row now scores instead of refusing `NO_SCORE`. `source_inputs.py`'s `SourceBundle` gained `runup_move_residual_rows` (the second driver's own held-out pool; `model_residual_rows` now serves the first driver, `implied_t1`, for this strategy). Judgement call, mine: the native/local forecast path does not apply `scale_runup_move` before the model stage runs (unlike the frozen-inference path's own `application._runup_frozen_output`), so the model stage reads `driver_prediction`/`runup_move_prediction` at the model's native D14 scale and applies the scale itself once, using the answer-free `days_before_print` fact — documented in `_execute_runup_model`'s docstring. Tests in `tests/test_v2_scoring_native_payoff.py` (23 in the file, 12 new for STR-RUNUP). |
| R4-18 | STR-THRU recalibration parity gap | Open. Found in the P5-2 acceptance audit (Phase 5 guide, "P5-2 acceptance audit and recalibration artifact"). Legacy `Scorer` pushes `win_model` through `Scorer.recalibration(strategy, alpha, evidence_cutoff)` whenever `recalibration_pairs.parquet` supports a map (`engine/score.py`, `recalibration`). Native STR-THRU never applied recalibration before P5-4. It now applies a frozen map only when a bundle declares one (`stages.py`: undeclared bundles keep `win_model` raw), and no Phase 4 capture declares one. So on real rows where legacy recalibrated, native `win_model` will disagree with the saved record (`win_model` is in `_SIMULATION_FIELDS`). Closing it means declaring the frozen `recalibration_artifact` in the capture's source bundles for every (strategy, alpha, cutoff) legacy recalibrated, not loosening the comparison |
| — | R4-4 coverage | The bundle covers 8 axes of 51 and 1 strategy of 11. `diagnostic_checkpoint_bundle_valid` is vacuously True on every real corpus today: it proves a DECLARED bundle verifies, not that one was verified |
| — | `_native()` residual | `checks/phase4_real.py::_native()` still builds inputs via `from_legacy_fields` for side controls (`:734`, `:735`, `:740`, `:774`, `:871` at `50d7a47`). The main comparison is unaffected: it scores verified strict traces through `application.score_frozen` (`_native_parity`, `:1813`/`:1820`), and the independence control builds its inputs with `build_native_score_inputs` |

## Decisions

1. **`LAYER_DISAGREE` as advisory** — DECIDED by the user 2026-09-18: it is
   advisory (warning-only), never a refusal. `7d31d36` added
   `_ADVISORY_FLAGS` excluding it from comparison (`checks/phase4_real.py`),
   and `engine/v2/scoring/stages.py` records it among the annotation-only
   flags with the decision cited. Row status still follows legacy
   `ScoreResult.scored` (a number present), not merely "no refusing flag"
   (`b8b1d92`).
2. **Phase 5 as a Phase 4 blocker** — DECIDED, reconfirmed by the user
   2026-09-18: it stays a Phase 4 sign-off blocker (R4-13 / P4-B01).
   Redefining it as a documented handoff is a scope change only the delivery
   plan and system design §12 can make.
3. **3B scale** — DECIDED: the bounded run is recorded as an accepted
   limitation in the 3B acceptance report, which states the exercised fraction
   explicitly rather than claiming production acceptance.
4. **Writing to `data/models/tier4`** — DECIDED by the user 2026-09-18:
   derived Tier-4 serving caches are written for all models (first
   `implied_t1`, then extended to `size_v1_4`, `runup_move_d14_v1_gbm` and
   `iv_crush_v1_gbm`: an in-place atomic upgrade that adds
   `pool_pred`/`pool_res` and leaves the fitted model unchanged). This is a
   narrow exception to the read-only `data/` rule, not a general one. Done:
   see the capture section above.
5. **`reports/phase3b_acceptance/report.json`** — RESOLVED 2026-09-18. The
   hand-authored file was moved out of `reports/` rather than deleted (kept
   in a session scratchpad); `reports/phase3b_acceptance/` now holds only the
   generated `report.md`, which is the 3B acceptance record.
6. **Switching `fixtures/tier0/CURRENT` to the traced capture** — DECIDED by
   the user 2026-09-18: the switch happens after the consumer checks. Write
   the full traced capture as a new version first, run every `CURRENT`
   consumer against it (`checks/tier0_corpus.py`, `checks/replay_identity.py`,
   `checks/rearchitecture_phase3_{publish,current_switch,preview}.py`,
   `tools/baseline_export.py`), then flip the pointer and keep the old
   version. `CURRENT` still names `20260912T233551Z`.
