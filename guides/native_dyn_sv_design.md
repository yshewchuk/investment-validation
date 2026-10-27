# Native DYN-SV: design (issue #1157 step 6)

Contract-level design only. No code in this PR. Steps 1-2 are merged (#1164, #1209);
step 3 (#1424, gate-less `_StrategySpec` + TWIN-P/TWIN-P5) was in review when this
was written (check its status before slice 1); the next steps register the other
five menu members. This note says what DYN-SV adds on top of them, so that its slices
can be built one at a time after they land.

**Acceptance.** This note is accepted when it (a) cites legacy behaviour with file:line,
(b) says which native pieces exist and what is missing, (c) states the failure
semantics of the new post-pass, and (d) gives a split list with estimates.
**Out of scope:** any code, the member-registration steps, changing legacy, and
measuring refusal counts (a later parity report does that).

## 1. What legacy does

- `engine/score.py` `DYNAMIC_MENU` (line 5105): TWIN-P, TWIN-P5, CND-PS, BFLY-P,
  BFLY-P5, RAMP7, CTR5. `DYNAMIC_STRATEGY = "DYN-SV"` (line 5110).
- `Scorer.score` (def line 1784; chooser call at 2187) scores every menu member as an
  ordinary strategy and, after the gate, calls `_score_chooser` (line 4261), which fills `chooser_score`
  from the frozen champion `dyn_sv_chooser_v1_1` over a 67-feature frame
  (`_chooser_frame`, ~line 4440).
- `score_calendar` (def line 4954; append at 5080) appends `dynamic_short_vol(frame)` (line 5161): one
  extra row per (ticker, event_date) that relabels the winning member's own row as
  `DYN-SV` and adds `chosen_strategy`, `chosen_margin`, `menu_size`, `detail`.
- Ranking runs over ATM rows only (`strike_offset` null; ladder rows never compete):
  highest `chooser_score` among members with a non-null `exp_pnl_sim` (native
  `_choose_dynamic` requires it to be finite). This is a known divergence: legacy
  keeps a `+inf` simulation, native excludes it. Native follows the finite rule; slice
  2 counts how many real rows differ before claiming parity. If no
  member has a chooser score, rank on `exp_pnl_sim`. A mixed event ranks on the
  scores that exist. Nothing finite: no DYN-SV row ("no opinion").
- The winner keeps its own entry-rule/gate verdict; the chooser does not re-gate.

DYN-SV therefore has no legs, gate or size forecast of its own. It is a selection
over seven independently scored member rows.

## 2. What native already has

| Piece | Where |
|---|---|
| Menu constant, DYN-SV registry entry (`chooser` role `dyn_sv_chooser_v1_1`) | `engine/v2/registry/strategies.py:15,89-141` |
| One `DYN-SV` `BoardRequest` per event, last in the universe | `engine/v2/ops/native_board_universe.py:216-251` |
| Chooser feature derivation (50 computed columns) | `engine/v2/scoring/native_chooser_features.py`, `native_chooser.py` |
| Chooser block from a bundle (`chooser_recipe`, fold pools, analog pool, admissible table) | `engine/v2/scoring/chooser_inputs.py` (`chooser_block`, line ~85) |
| Frozen state: `admissible_table:dyn_sv`, `chooser_analog_pool` | `engine/v2/scoring/release_bindings.py:75-76,164-165` (resolved, not consumed by the nightly) |
| Menu accounting: one outcome per member, record or typed refusal, `INVALID_CHOOSER_MENU` / `CHOOSER_MENU_INCOMPLETE` | `engine/v2/scoring/dynamic_outcomes.py` |
| Selection over scored records (`_choose_dynamic`: chooser_score else exp_pnl_sim, margin, tie, `NO_CHOOSER_CANDIDATE`) | `engine/v2/scoring/application.py:1439`; called by `score_event` and, as a private import, `tools/phase4_targeted_replay.py:661` (via `phase4_real.application`) and `checks/phase4_real.py` (`_replayed_chooser`, `_chooser_controls`) |
| Geometry, size tables, entry-rule gate for the members | steps 1-5 |

## 3. What is missing

1. **Chooser declaration on member rows.** `release_bindings.py` states "chooser
   state selection is not wired into nightly". `native_score_batch._assemble_one_event`
   declares no `chooser_recipe` and no chooser state, so member records carry no
   `chooser_score`. Needs: selecting the analog pool / admissible table (and the
   Tier-4 fold pools the chooser producers read) for the event's cutoff, and a
   chooser field on the batch `_StrategySpec` (the registry `StrategySpec` already
   names the `chooser` role).
2. **A selection post-pass.** `score_batch` pairs request to record one to one by
   position (`_native_score_batch_documents`). DYN-SV is a fan-in over seven other
   rows of the same event, so it cannot be a registry entry or a per-row pair.
3. **A public, record-level selector.** `_choose_dynamic` is private to
   `application.py` (slice 2 migrates the private imports in `tools/phase4_targeted_replay.py` and
   `checks/phase4_real.py`) and returns a re-labelled `ScoreRecord`; the batch worker needs
   it (and the menu accounting) over records and refusals it already holds.
4. **Output and pairing.** `ScoreRecord` already declares `chooser_candidates` and
   `chooser_selection`, and `to_document` serialises them, so slice 2 populates
   existing fields. The parity pairing does not read the selection.

Depends on #1424: yes, for the spec shape (gate-less member, chooser field on the
same dataclass) and, more strongly, on steps 4-5: a menu with members unregistered
would be ranked over a partial menu, which legacy never does. DYN-SV is therefore
NOT a first slice that can ride on #1424 alone.

## 4. Design

**Members are scored once; DYN-SV selects.** The board universe's supported set
includes the seven members (import-time assert in `native_board_universe.py`); the batch worker registers all seven (#1157 steps 3-5); only DYN-SV itself is still refused as `UNSUPPORTED_STRATEGY`. DYN-SV reuses those exact records instead of re-scoring them
(no second compute, no divergent copies). The post-pass runs in the worker after
`score_batch`, per event, over the live `ScoreRecord` objects and the member refusals,
keyed by `BoardRequest`. It runs before `to_document` serialises them
(`records_by_key` is serialised dictionaries and is not an input).

**Failure semantics (R1-R7).** The R labels are this note's own.

- R1 inputs: the seven member results of the event: each member's `ScoreRecord`
  (it carries `exp_pnl_sim`, `chooser_score`, status) or its typed refusal, plus
  `DynamicCandidateOutcome`s built from them (accounting only), under one
  `DynamicMenuIdentity` built from the event context. No other input.
- R2 completeness: `validate_dynamic_menu` must pass (all seven accounted, any may
  be refused). A violation refuses only that event's DYN-SV row with the
  menu code (`INVALID_CHOOSER_MENU` / `CHOOSER_MENU_INCOMPLETE`) and never sinks the
  batch; member rows are untouched.
- R3 partial menu: members that refused are excluded and listed in
  `chooser_candidates` with their code; the rest are ranked (legacy behaviour).
  Ranking key is `chooser_score` if any eligible member has one, else `exp_pnl_sim`;
  the chosen key is recorded in the selection.
- R4 nothing eligible: DYN-SV gets a typed refusal `NO_CHOOSER_CANDIDATE` carrying the
  per-member codes. Legacy emits no row; native emits a refusal so the parity report
  can classify it. This is a documented difference in representation, not behaviour.
  Serialization (slice 2 writes it and lands the schema bump with every reader of the
  tag; slice 3 uses it): `NativeScoreBatchRowRefusal` keeps its single `code` and
  `detail` unchanged. It gains an optional `member_codes`, a list of
  `{strategy, code}` objects in `DYNAMIC_MENU` order, one per member including those
  that were not refused (`code: null`). `as_document` emits it only when set. The keyed
  `native_score_batch_refusals` tag is bumped for the added optional field
  (`guides/component_contracts.md` §2.3), so documents are not byte-identical across
  the bump. The tag check is exact in `nightly._native_score_batch_document_schema_ok`
  and in `run_native_parity_worker`; slice 2 updates both to accept the new tag, else
  new batches are rejected between merges. Metadata path: `native_score_batch.py`
  builds keyed refusals directly from `code`/`detail` (it calls `as_document` only for
  `INVALID_KEY_FIELD`), so slice 2 routes DYN-SV refusals through `as_document`; and
  `native_parity_report.py` reduces keyed refusals to code strings before
  `apply_native_refusals`, so slice 2 widens that input to carry `member_codes` per
  key. Slice 3 reads it to classify a DYN-SV refusal (all members refused vs. no
  finite score) and carries it into the report entry.
- R5 ties: broken by `DYNAMIC_MENU` order (stable sort); the selection says
  `status: tie`. Legacy is stable only on the `chooser_score` branch; its
  `exp_pnl_sim` fallback (`score.py:5223`) uses default quicksort, so fallback ties
  are unspecified there. Legacy's stable order is its frame (scoring) order, while
  native's is `DYNAMIC_MENU` order; the board universe's request order
  (`native_board_universe.py:50`) may differ from both. `component_contracts.md`
  §9.4 requires preserving the actual baseline tie behaviour, so `DYNAMIC_MENU`
  order is only the default: slice 2 measures the baseline tie order on real events
  and pins native to it, and a mismatch is a slice 2 finding, not a design choice.
- R7 identity and refusal-only events: `validate_dynamic_menu` and
  `DynamicMenuIdentity` apply only to an event whose calendar row passed validation
  (valid `event_id`, calendar, snapshot and release context the worker holds, never
  taken from a member record). An event refused before any calendar row exists
  (`CALENDAR_ROW_INVALID`, `PRICE_HISTORY_NOT_AVAILABLE`, `PANEL_HISTORY_NOT_AVAILABLE`
  carry only a `BoardRequest`) has no valid identity: skip menu validation and refuse
  DYN-SV with `NO_CHOOSER_CANDIDATE`, carrying the per-member codes, keyed by the
  DYN-SV `BoardRequest` alone. All members refused with a valid calendar row follows
  R4 under a normal identity.
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
   code, ~150 test, ~60 doc (includes the feature list and its order test). Member rows gain `chooser_score` and may now refuse
   `MODEL_NOT_READY`; no DYN-SV row yet. Slice 1 also owns the feature schema: it
   records the complete ordered chooser input list (every primitive input, its
   position, and the function or artifact that supplies it) in
   `native_chooser_features.py` and the scoring `ARCHITECTURE.md`, and adds a test
   that this list equals the frozen `dyn_sv_chooser_v1_1` feature order. The 50
   computed columns come from these groups (`native_chooser_features.py`):
   `ANALOG_COLUMNS` (5, kNN analog pool), `BREAKEVEN_COLUMNS` (5) and `SHAPE_COLUMNS`
   (8) from leg geometry, `SIZE_BAND_COLUMNS` (6, sizing forecast and served-fold
   band) and `PRODUCER_COLUMNS` (7, the two other Tier-4 producers), plus the
   remaining geometry, spread, depth and edge columns. Slice 1 pins the exact
   order; this note does not duplicate it.
2. **Selection post-pass** (add + wire): public record-level selector extracted from
   `_choose_dynamic`, worker fan-in using `validate_dynamic_menu`, DYN-SV record or
   `NO_CHOOSER_CANDIDATE` refusal in the output documents, `member_codes` field, schema
   tag bump with both readers. ~220 code, ~200 test, ~60 doc.
3. **Parity pairing and report** (wire): pair DYN-SV by key, surface
   `chosen_strategy` and margin in the comparison. ~80 code, ~70 test, ~20 doc.

Slice 1 changes member rows (new field, possible refusals); slices 2-3 only add
rows. Slice 2 is what
removes the DYN-SV `UNSUPPORTED_STRATEGY` refusals.

## 6. Open questions (for slice 1/2 authors, not blockers for this note)

- Whether the Tier-4 fold pools the chooser producers need are already staged by the
  nightly or need their own staging step (measure in slice 1).
- The issue counted 37 DYN-SV refusals in the 2026-10-21 report; the current figure
  is about 130 per session. Re-measure from the next parity report before slice 2
  sets its expected effect.
