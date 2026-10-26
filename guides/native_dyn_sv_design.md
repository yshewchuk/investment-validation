# Native DYN-SV: design (issue #1157 step 6)

Contract-level design only. No code in this PR. Steps 1-2 are merged (#1164, #1209);
step 3 (#1424, gate-less `_StrategySpec` + TWIN-P/TWIN-P5) is in review; steps 4-5
register the other five menu members. This note says what DYN-SV adds on top of
them, so that its slices can be built one at a time after they land.

## 1. What legacy does

- `engine/score.py` `DYNAMIC_MENU` (line 5105): TWIN-P, TWIN-P5, CND-PS, BFLY-P,
  BFLY-P5, RAMP7, CTR5. `DYNAMIC_STRATEGY = "DYN-SV"` (line 5110).
- `Scorer.score` (line 2187) scores every menu member as an ordinary strategy and,
  after the gate, calls `_score_chooser` (line 4262), which fills `chooser_score`
  from the frozen champion `dyn_sv_chooser_v1_1` over a 67-feature frame
  (`_chooser_frame`, ~line 4440).
- `score_calendar` (line 5080) appends `dynamic_short_vol(frame)` (line 5161): one
  extra row per (ticker, event_date) that relabels the winning member's own row as
  `DYN-SV` and adds `chosen_strategy`, `chosen_margin`, `menu_size`, `detail`.
- Ranking: highest `chooser_score` among members with a finite `exp_pnl_sim`. If no
  member has a chooser score, rank on `exp_pnl_sim`. A mixed event ranks on the
  scores that exist. Nothing finite: no DYN-SV row ("no opinion").
- The winner keeps its own entry-rule/gate verdict; the chooser does not re-gate.

DYN-SV therefore has no legs, gate or size forecast of its own. It is a selection
over seven independently scored member rows.

## 2. What native already has

| Piece | Where |
|---|---|
| Menu constant, DYN-SV registry entry (`chooser` role `dyn_sv_chooser_v1_1`) | `engine/v2/registry/strategies.py:15,89-141` |
| One `DYN-SV` `BoardRequest` per event, last in the universe | `ops/native_board_universe.py:216-251` |
| Chooser feature derivation (50 computed columns) | `scoring/native_chooser_features.py`, `native_chooser.py` |
| Chooser block from a bundle (`chooser_recipe`, fold pools, analog pool, admissible table) | `scoring/chooser_inputs.py` (`chooser_block`, line ~85) |
| Frozen state: `admissible_table:dyn_sv`, `chooser_analog_pool` | `scoring/release_bindings.py:75-76,164-165` (resolved, not consumed by the nightly) |
| Menu accounting: one outcome per member, record or typed refusal, `INVALID_CHOOSER_MENU` / `CHOOSER_MENU_INCOMPLETE` | `scoring/dynamic_outcomes.py` |
| Selection over scored records (`_choose_dynamic`: chooser_score else exp_pnl_sim, margin, tie, `NO_CHOOSER_CANDIDATE`) | `scoring/application.py:1456-1505`, used only by `score_event` |
| Geometry, size tables, entry-rule gate for the members | steps 1-5 |

## 3. What is missing

1. **Chooser declaration on member rows.** `release_bindings.py` states "chooser
   state selection is not wired into nightly". `native_score_batch._assemble_one_event`
   declares no `chooser_recipe` and no chooser state, so member records carry no
   `chooser_score`. Needs: selecting the analog pool / admissible table (and the
   Tier-4 fold pools the chooser producers read) for the event's cutoff, and a
   chooser field on the strategy spec.
2. **A selection post-pass.** `score_batch` pairs request to record one to one by
   position (`_native_score_batch_documents`). DYN-SV is a fan-in over seven other
   rows of the same event, so it cannot be a registry entry or a per-row pair.
3. **A public, record-level selector.** `_choose_dynamic` is private to
   `application.py` and returns a re-labelled `ScoreRecord`; the batch worker needs
   it (and the menu accounting) over records and refusals it already holds.
4. **Documents.** `records.json` / `refusals.json` carry no DYN-SV selection fields
   (`chooser_candidates`, `chooser_selection`) in their schema and the parity pairing
   does not know them.

Depends on #1424: yes, for the spec shape (gate-less member, chooser field on the
same dataclass) and, more strongly, on steps 4-5: a menu with members unregistered
would be ranked over a partial menu, which legacy never does. DYN-SV is therefore
NOT a first slice that can ride on #1424 alone.

## 4. Design

**Members are scored once; DYN-SV selects.** The nightly already requests all seven
members as board rows. DYN-SV reuses those exact records instead of re-scoring them
(no second compute, no divergent copies). The post-pass runs in the worker after
`score_batch`, per event, over `records_by_key` and the member refusals.

**Failure semantics (R1-R6).**

- R1 inputs: the seven member outcomes of the event, as `DynamicCandidateOutcome`s
  (record summary or the member's own typed refusal), under one
  `DynamicMenuIdentity` built from the event context. No other input.
- R2 completeness: `validate_dynamic_menu` must pass (all seven accounted, any may
  be refused). A violation is a worker bug and aborts the batch, never a row refusal.
- R3 partial menu: members that refused are excluded and listed in
  `chooser_candidates` with their code; the rest are ranked (legacy behaviour).
  Ranking key is `chooser_score` if any eligible member has one, else `exp_pnl_sim`;
  the chosen key is recorded in the selection.
- R4 nothing eligible: DYN-SV gets a typed refusal `NO_CHOOSER_CANDIDATE` carrying the
  per-member codes. Legacy emits no row; native emits a refusal so the parity report
  can classify it. This is a documented difference in representation, not behaviour.
- R5 ties: broken by `DYNAMIC_MENU` order (the existing stable sort); the selection
  says `status: tie`. Slice 2 confirms legacy's frame order is the same, with a test.
- R6 determinism and no state: pure function of the seven outcomes; DYN-SV never
  mutates a member record, never writes, never retries. The winner's gate verdict is
  copied verbatim.

**No emulation.** No legacy-only ranking mode. The `exp_pnl_sim` fallback exists
because the champion contract defines it, not to match a report.

**Standard tool check.** Selection is a max over seven floats; no library applies.

## 5. Split list (one PR open at a time, after #1424 and steps 4-5)

1. **Chooser state on member rows** (wire): chooser field in the strategy spec,
   release-binding selector for chooser state by cutoff, `chooser_recipe` declaration
   in `_assemble_one_event`. Typed `MODEL_NOT_READY` when state is missing. ~150
   code, ~40 doc. Member rows gain `chooser_score`; no DYN-SV row yet.
2. **Selection post-pass** (add + wire): public record-level selector extracted from
   `_choose_dynamic`, worker fan-in using `validate_dynamic_menu`, DYN-SV record or
   `NO_CHOOSER_CANDIDATE` refusal in the output documents. ~200 code, ~50 doc.
3. **Parity pairing and report** (wire): pair DYN-SV by key, surface
   `chosen_strategy` and margin in the comparison. ~80 code, ~20 doc.

Each slice is behaviour-neutral on the board until the next lands; slice 2 is what
removes the DYN-SV `UNSUPPORTED_STRATEGY` refusals.

## 6. Open questions (for slice 1/2 authors, not blockers for this note)

- Whether the Tier-4 fold pools the chooser producers need are already staged by the
  nightly or need their own staging step (measure in slice 1).
- The issue counted 37 DYN-SV refusals in the 2026-10-21 report; the current figure
  is about 130 per session. Re-measure from the next parity report before slice 2
  sets its expected effect.
