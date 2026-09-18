# Rearchitecture Phase 3B + 4 — remaining work

Status: rewritten 2026-09-18 from branch `codex/phase4-remediation-track-a` at
`9bf3a1c` (= `origin/main`). The original assessment was recorded at `129a368`;
everything below reflects what was actually merged and verified since, not that
snapshot. Authority remains the [delivery plan](rearchitecture_delivery_plan.md),
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

**A third blocker is open and needs an operator decision.** Three capture
attempts were SIGKILLed at the identical point:

    [train] implied_t1: 100,548 (event, decision day) rows

The legacy `Scorer` trains `implied_t1` in-process because no usable serving
cache exists for the needed folds. `tools/prepare_phase4_tier4_caches.py`
exists for exactly this, but:
* it only **upgrades** existing caches — its dry run reports 6 old and 7
  missing, and each missing fold is "left untouched because no old cache
  exists";
* it writes to `tier4.SERVING_DIR` = `data/models/tier4`, which is under the
  **read-only real root** `data/`, and it exposes no `--cache-dir` flag.

So the capture cannot proceed without either authorising a write into
`data/models/tier4`, or bounding the Scorer's training footprint another way.
Until then the corpus cannot carry traces, and Phase 4 cannot close.

## Phase 4 remaining

| ID | Item | Status |
|---|---|---|
| R4-2 | Record-by-record parity over the complete saved release | Blocked on the capture |
| R4-5 | Analog parity is unexercised | Open. R4-6 made the bucket recipe expressible, which is the precondition. No control currently exercises analog independence — `_numerical_independence_control` legitimately asserts forecast/simulation/gate only |
| R4-9 | Actionable flags have no native derivation | Open. `STALE_QUOTE`, `WIDE_MARKET`, `PROJECTED_CALENDAR`, `EXTRAPOLATED`, `OUT_OF_DOMAIN` have zero occurrences in `engine/v2/` |
| R4-10 | Planned-exit simulation not expressible through `SourceBundle` | Open. `_RESIDUAL_RECIPE_FIELDS` permits none of `pre_iv30`, `dte_exit`, `event_date`, `residuals` |
| R4-13 | Frozen Phase 5 inference not integrated | Open. Emits P4-B01; decided to REMAIN a Phase 4 blocker rather than be redefined as a handoff |
| R4-14 | Factory / refusal / DYN-SV parity | Partly closed: factory parity is 18/18 across all 11 strategies with every negative control rejecting. `chooser_corpus_present` still false |
| R4-15 | Planted-defect control incomplete | Blocked on the capture — a defect cannot be planted in a comparison that never runs |
| R4-16 | Forecast/gate recipes are inline linear only | Open |
| R4-17 | STR-RUNUP has no native model layer — parity regression, not a scope boundary | Open. Legacy computes `exp_pnl_model`/`win_model` for STR-RUNUP via `_score_runup_model` (`engine/score.py:2220`) and `RunupPayoffSurface`/`fit_runup_payoff`/`simulate_runup_returns` (`engine/payoff.py:207`, `:366`, `:469`) — a two-driver (implied move + moneyness) surface, distinct from STR-THRU's single-driver payoff line. `engine/v2/scoring/stages.py`'s `_PAYOFF_DRIVER_STRATEGIES = frozenset({"STR-THRU"})` (`:201`) has no STR-RUNUP entry, so a native STR-RUNUP row silently gets no model number. Since `b8b1d92` tightened `scored` to require a real `exp_pnl_model` or `exp_pnl_analog` number, a STR-RUNUP row with no analog number now refuses as `NO_SCORE` where legacy scores it — a parity regression, not a not-applicable case. Not ported in the `native-model-layer` pass (STR-THRU only); needs its own `RunupPayoffSurface` port |
| — | R4-4 coverage | The bundle covers 8 axes of 51 and 1 strategy of 11. `diagnostic_checkpoint_bundle_valid` is vacuously True on every real corpus today: it proves a DECLARED bundle verifies, not that one was verified |
| — | `_native()` residual | `checks/phase4_real.py::_native()` still builds inputs via `from_legacy_fields` for side controls (`:495`, `:496`, `:501`, `:535`, `:632`). The main comparison is unaffected — it uses `build_native_score_inputs` at `:243` |

## Decisions

1. **`LAYER_DISAGREE` as advisory** — still undecided. `7d31d36` added
   `_ADVISORY_FLAGS` excluding it from comparison. The native side has no
   layer-disagreement concept, so this masks a behavioural difference rather
   than a cosmetic one. It needs a recorded disposition or a revert; a code
   comment is not a disposition.
2. **Phase 5 as a Phase 4 blocker** — DECIDED: it stays a blocker. Redefining
   it as a documented handoff is a scope change only the delivery plan and
   system design §12 can make.
3. **3B scale** — DECIDED: the bounded run is recorded as an accepted
   limitation in the 3B acceptance report, which states the exercised fraction
   explicitly rather than claiming production acceptance.
4. **Writing to `data/models/tier4`** — OPEN, blocks the capture. See above.
5. **`reports/phase3b_acceptance/report.json`** — OPEN. Superseded by the
   generated `.md` but still claims `production_acceptance: true`. `reports/`
   is gitignored, so deletion is unrecoverable.
