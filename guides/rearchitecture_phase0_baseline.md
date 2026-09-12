# Rearchitecture Phase 0 — Baseline

**Objective:** make every later phase checkable. Phase 0 writes no production
logic and changes no strategy, model, deployment or number. It produces the
instruments the migration is measured with — the frozen compatibility package,
a corpus of what the current engine does, a comparison contract that can name
what moved, the enforced layer map, and the empty `engine/v2/` skeleton those
layers describe.

**Exit gate (from §12):** every strategy and critical refusal reproducible;
five seeded defects yield five stage-named findings in one tier-0 pass; the
layer check runs green over the v2 skeleton with an empty adapter ledger.

---

## 1. Why this phase exists

§12 builds the replacement alongside the legacy store and deletes legacy whole
at phase 8. That choice rests entirely on parity: the new path is trusted
because it produces numbers identical to the old one. Parity needs an oracle,
and the oracle has to exist before the first replacement package is written.

It also has to be *fast*. On 2026-09-11 a single red self-check signal — ten of
twenty rows — carried five independent causes, and each correct fix left the
signal red because more causes remained behind it. Two properties fix that
class, and both are phase-0 deliverables: a corpus that answers in seconds, and
a comparison result that reports every independent finding at once rather than
the first.

Nothing here is a refactor. If a phase-0 change alters a board number, the
change is wrong.

---

## 2. Design references

Bare section marks such as §3.2 refer to
[system rearchitecture](system_rearchitecture.md); references to the other
three documents are named. The four together are the design, and where this
guide disagrees with any of them the design wins and this guide is wrong. Read
the row for a step before starting it.

| Step | System rearchitecture | Component contracts | Other |
|---|---|---|---|
| 1. v2 skeleton, layer map | §4.1 two trees, §4.2 import direction, §4.5 READMEs | §2.5 type kinds | — |
| 2. ComparisonReceipt | §11 validation table and its tiers | §15 comparators and diagnosis; §2.4 failure envelope | — |
| 3. Baseline package | §3.1 inventory, §3.2 package contents, §5.5 knowledge modes | §2.1 common types, §2.3 evolution rules, §7 registration | [generation and simulation](structure_generation_and_simulation.md) §6.1 named legacy adapters |
| 4. Tier-0 corpus | §3.2 fixture rules, §6.3 execution order | §2.2 canonical content vs envelope, §9.2 ScoreRequest, §9.3 ScoreRecord, §9.4 verdict and reason codes, §9.5 identity and replay | [generation and simulation](structure_generation_and_simulation.md) §7 legacy golden corpus; [data model](rearchitecture_data_model.md) §1 identity rules |
| 5. Negative controls | §11 named regression cases | §15.1 stage localization | [data model](rearchitecture_data_model.md) §4 cross-entity checks |
| 6. Budgets, READMEs | §4.3 budgets, §4.6 ratchet, §4.7 publication refusal | §17 review order | — |

Two of these are easy to skip and expensive to skip. **Contracts §2.2** defines
canonical JSON, the payload/envelope split and content hashing — every capture
rule in step 4 derives from it. **Contracts §9.5** requires that replay be
proved across a fresh process, reordered inputs, batch versus single and a
serialized round trip; running twice in one process is explicitly insufficient,
and that sentence is the acceptance criterion for the whole corpus.

---

## 3. Build order

Strictly ordered; each step is usable before the next starts.

```
1. v2 skeleton + layer map + checks/import_layers.py   (nothing to break yet)
2. ComparisonReceipt + the staged comparator            (needs no corpus)
3. Baseline package: environment lock, definitions,     (says what to capture)
   artifacts, worked examples
4. Tier-0 corpus capture                                (needs 2 and 3)
5. Negative controls                                    (proves 2 and 4 work)
6. checks/code_budgets.py + package READMEs             (locks 1 in)
```

Step 1 first because it is the only step that cannot fail a comparison — there
is no code in v2 yet. Step 2 before step 4 because a corpus without a
comparator is a directory of JSON nobody reads. Step 3 before step 4 because
capturing a corpus in an unpinned environment produces fixtures that cannot
distinguish a code change from a library change.

---

## 4. Step 1 — the v2 skeleton and the layer map

Create the tree of §4.1 as empty packages, each with `__init__.py` and the
`README.md` of §4.5:

```
engine/v2/
  contracts/ foundation/ data/ features/ models/ registry/
  domain/{generation,scenarios,valuation,simulation}/
  scoring/ evaluation/ ledger/ models/training/
  serving/ ops/ dashboard/ diagnosis/
```

`checks/import_layers.py` holds the layer numbers as a literal map and enforces
the three rules of §4.2:

1. inside `engine/v2/**`, imports point down only; `engine/v2/diagnosis` is
   imported by nothing;
2. every `engine/v2/** -> engine/*` dependency is a declared entry in
   `checks/legacy_adapters.json`, confined to one adapter module per package;
3. no `engine/* -> engine/v2/**` import exists.

Parse with stdlib `ast` over staged blobs, following the `read_staged_blob`
pattern in `checks/repo_hygiene.py`. Do not import the modules to inspect them:
importing `engine.score` loads a panel.

`checks/legacy_adapters.json` starts as `{"count": 0, "adapters": []}`. It is
the migration's progress meter per §4.6 — it may only shrink, and phase 8
begins when it reaches zero.

**Acceptance.** The checker fails a planted upward import inside v2, fails a
planted undeclared legacy import, fails a planted `engine/* -> engine/v2/*`
import, and passes the real tree. Runs in under two seconds.

---

## 5. Step 2 — ComparisonReceipt and the staged comparator

Implement the minimum of [contracts §15](component_contracts.md#15-comparators---diagnosis)
that the corpus needs: `ComparisonReceipt`, `Finding`, and one comparator over
two `ScoreRecord`-shaped mappings. Full schema fidelity is not required in
phase 0; the two behavioural properties are:

- **Stage-localized.** Each finding names the first stage whose inputs agreed
  and outputs did not, plus the field path. Stage plan for the current scorer,
  following §6.3: `resolve_context`, `features`, `forecast`, `geometry`,
  `pricing`, `analogs`, `simulation`, `gate`, `chooser`, `serialization`.
  In phase 0 that localization is *observed* from record fields along the
  declared stage graph: it names where two records first disagree, not which
  internal computation diverged. Execution-level per-stage hashes are
  rearchitecture phase 1, and fine-grained scorer stages arrive with phase 4
  extraction; a receipt must not claim more than it observed.
- **Complete, not first-wins.** One pass reports every independent finding.
  Stopping at the first difference is what turned five causes into five nights.

Derive the compared field set from the digest's own field list, never a
hand-maintained constant — `28cf8b1` already made that fix for the explainer
and it must not be reintroduced. `verdict: incomparable` when the population is
empty or an input is missing; an empty comparison never reports agreement.
Failures use the shared envelope of contracts §2.4.

This code lands in `engine/v2/diagnosis/`, which nothing imports, so it is
subject to the v2 budgets from its first line and cannot become a dependency of
what it compares.

**Acceptance.** Two records differing in three unrelated fields produce three
findings, not one. A record compared against itself produces `agree`. An empty
input produces `incomparable`. Removing a field from the digest removes it from
the comparison with no edit to the comparator.

---

## 6. Step 3 — the baseline compatibility package

§3.2 requires five things, of which the corpus is one. The other four say what
the corpus is a corpus *of*, and without them a fixture that disagrees cannot
be attributed. Export all of it as immutable, dated artifacts.

### 6.1 Environment lock

The repository currently has no `requirements.txt` and no `pyproject.toml`, so
this is real work and it comes first. Without it a fixture cannot distinguish a
code change from a library change — the same gap that makes the Tier-4 serving
cache key incomplete, since `_serving_path` carries a panel hash but neither
seed nor runtime.

Record: Python version, the exact installed version of every imported
third-party package, the source commit, and the platform. Pin them. The lock is
part of the baseline's identity, not metadata about it.

### 6.2 Resolved definitions

Export the *resolved* values, not references to the source that computes them.
A definition that has to be re-derived from code is not frozen.

- Every structure in `engine.structures.STRUCTURES` — factory parameters,
  ordered strike and expiry selectors, quantities, anchors, reference legs,
  forecast width divisors, collision rules.
- Entry, exit and decision offsets per strategy.
- Gate definitions: fitted gates with their thresholds, and the arithmetic
  rules in `entry_rules.py` with their exact constants — the trailing
  six-month top-20% bar, the 25% relative-spread ceiling, the $10B market-cap
  floor. §3.1 is explicit that these differ from the learned-gate domain floor
  and must not be consolidated.
- DYN-SV: the ordered seven-member menu, ranking rule, tie behaviour,
  partial and missing-score behaviour, fallback resolver.
- Validation status per structure: promoted, tracked, or disabled with its
  refusal code.

`engine/models/structures.json` and `engine/models/registry.json` already hold
part of this; the export resolves them rather than pointing at them.

### 6.3 Artifacts and state

- All champion artifacts with their fingerprints. Nine registered models across
  six roles, seven of them champions: `size_v1_4`, `opf_implied_t1_gbm`,
  `runup_move_d14_v1_gbm`, `iv_crush_v1_gbm`, `gate_midfill_str_runup`,
  `gate_midfill_str_thru_forecast_analog`, `dyn_sv_chooser_v1_1`. Non-champions
  (`size_v1_3`, `gate_midfill_str_thru`) are exported too — §3.1 requires
  historical definitions preserved.
- Feature recipes and preprocessing, with the upstream model dependencies the
  registry's `ROLE_TIER` graph already records.
- Fold artifacts, residual pools, calibration and payoff state. Per §7 these
  are currently fitted in-process; export the state each fixture consumed so a
  replay does not refit.
- The named legacy adapters of
  [generation and simulation](structure_generation_and_simulation.md) §6.1 —
  `legacy.structure_selectors.v1`, `legacy.paired_move_crush.v1`,
  `legacy.put_exit_bs.v1`, `legacy.pnl_return.v1`, `legacy.payoff_map.v1`,
  `legacy.runup_surface.v1` — whose content is defined by this export rather
  than by their names.

### 6.4 Conventions and worked examples

- Calendar version, quote and fill conventions, data snapshot references, and
  the exact score requests and results behind each example.
- Existing report, book, prediction and settlement examples with provenance.
  Entry-cost estimate at decision time and actual later repricing are
  **separate fields**, never reconciled into one.
- Each export records its knowledge mode per §5.5. Almost all of it is
  `attested_stable` or `reconstructed`; nothing captured today is `observed`.

**Acceptance.** A second export from the same tree state — commit plus
working-tree diff — and snapshot is byte-identical, and
`tools/baseline_export.py --verify` records that in a receipt the gate reads.
Every strategy, model role and legacy adapter named in §3.1 appears. The lock
reproduces the environment on a clean checkout.

---

## 7. Step 4 — the tier-0 corpus

Frozen `(request, record)` pairs captured through the real public entry points,
per §3.2: *"Fixtures must come through real public entry points. Do not invent
a column such as `event_id` in a fixture if the current serving row does not
contain it."*

### 7.1 Coverage

| Axis | Required |
|---|---|
| Strategies | All 11 in `engine.structures.STRUCTURES` — `STR-THRU`, `STR-RUNUP`, `CAL-P`, `CND-P`, `CND-PS`, `TWIN-P`, `TWIN-P5`, `BFLY-P`, `BFLY-P5`, `RAMP7`, `CTR5` — plus `DYN-SV` |
| Model roles | All 6: `size`, `implied_t1`, `runup_move`, `iv_crush`, `gate`, `chooser` |
| Refusal codes | `UNVALIDATED_STRUCTURE`, `OUT_OF_DOMAIN`, `NO_CHAIN`, `BAD_QUOTE`, `COARSE_LADDER`, `NO_FORECAST` — at least one row each. `BAD_QUOTE_COST_PCT` is not a seventh code: it is the 30% threshold behind the single `BAD_QUOTE` flag (`engine/score.py` emits `BAD_QUOTE` only on that bar), exported as a constant in the baseline package |
| Sessions | BMO and AMC; a year boundary; a month boundary |
| Geometry | Pinned `structure_params` beside the selector-resolved pair it was pinned from; a computed `width_moneyness`, and a requested strike that is itself a listed strike; a coarse ladder; an exact-mirror requirement. Every geometry axis but the coarse-ladder refusal needs a priced row |
| DYN-SV | Full menu; a partial menu; a missing chooser score falling back to the resolver — each chosen over a board-shaped frame (one row per structure per event), frozen in order. **A tie is not a required fixture** (decision 2026-09-12): no genuine tie between two structures exists in the store or the prediction ledger, and a fixture may not invent one. The tie rule — input-row order, on both ranking paths — is frozen in `definitions/dyn_sv.json` and tested on the real `dynamic_short_vol` with tied rows in both orders. The test calls legacy code, so v2 needs its own test of whatever tie rule it declares |
| Disabled | `CAL-P` and `CND-P` refusing in production and replaying under research |

`CAL-P` and `CND-P` must appear as *refusals*. A fixture that scores them is a
fixture of a strategy this program does not run.

### 7.2 Capture rules

- **Full precision.** Serialize replay inputs unrounded. `b33036c` and
  `6b9d5cf` are exactly this defect: `json_safe` rounded `structure_params` to
  six places and `_write_pair` re-rounded after the exemption. A corpus written
  through the board's display path would freeze the bug as the baseline.
- **Deterministic payload, separate envelope** (contracts §2.2). Wall-clock
  time, worker id and duration live outside the hashed payload, so a replay
  reproduces the payload without reproducing the elapsed time.
- **No network, no panel load, no fitting** on replay. Pin the fold artifact
  from §6.3; a corpus that fits is not a tier-0 corpus.
- **Private.** Fixtures carry real quotes and go to the private mirror, never
  the public repo — convention 10. Extend `checks/repo_hygiene.py` to cover the
  fixture directory *before* the first capture, not after.

### 7.3 Comparison rules

Per §3.2, exact for IDs, selected contracts, integer quantities, gate verdicts,
flags, null masks and frozen canonical records. Declared **per-field**
tolerances for independently recomputed floats. Never a global tolerance, and
never widened to make a test pass — a widened tolerance is a finding.

Null masks and refusal reasons are compared, not just non-null values. Five of
the six defects this corpus exists to catch were invisible in the non-null
values alone.

**Acceptance** (contracts §9.5, split by tier). Tier 0 cannot re-score — the
legacy scorer loads a panel — so the §9.5 cases divide:

- **Tier 0**, under ten seconds, network disabled: the corpus is intact,
  addressable, covering and round-trip-stable, its pinned fixtures agree with
  their sources, the seeded controls of §8 behave over the real corpus, and
  the check is deterministic across batch/single and a fresh process.
- **Tier 1**, `tools/replay_tier1.py`: every declared pair — every strategy,
  DYN-SV included, and every refusal code — re-scored through the production
  entry points in a **fresh process**, written to a real file and read back,
  against hash-verified frozen dependencies. The receipt binds to the content
  hash of the code, the frozen dependencies, the baseline and the store
  snapshot; a partial, skipped or stale receipt does not count.
- **Rearchitecture phase 1** (acceptance O30): the engine-level matrix of
  **reordered inputs**, **batch versus single** and restart. A tier-0 case
  that reorders file loading cannot fail and is not evidence of it.

---

## 8. Step 5 — negative controls

A check that has never failed is not known to work. Seed each of the five
2026-09-11 causes as a controlled corruption and assert the comparator names
it, in the right stage:

| Seeded corruption | Must be reported as |
|---|---|
| Forecast suppressed when `structure_params` are replayed (`e845f3e`) | `forecast`, forecast block null |
| A field removed from the explainer's compared set (`28cf8b1`) | Impossible by construction — the set derives from the digest |
| Analog bootstrap reseeded by row order (`b9aa1fd`) | `analogs`, `ci_low`/`ci_high` only |
| A replay input rounded to six places (`b33036c`) | `serialization`, `structure_params.*` |
| Rounding reapplied after the exemption (`6b9d5cf`) | `serialization`, and the written file disagreeing with its digest |

All five seeded at once must produce five findings in **one** pass. That single
assertion is the phase's reason for existing: it is the difference between five
nights and one.

Seed them twice. At **tier 0**, over the real frozen corpus (contracts §15.3):
each seedable cause planted into its own distinct pair, one pass, each control
producing exactly its specified findings with every other pair clean — distinct
pairs make the one pass an ablation as well. At **tier 1**, through real engine
stages: `tools/replay_tier1.py --seed-defects` patches the legacy forecast and
analog stages in its own process and the serialization path, so the controls
exercise the code the defects lived in rather than a record edited to look like
their output. The `6b9d5cf` control is an integrity check — the digest stored
beside a written artifact against the bytes written — not a record field that
any other corruption would also move. `28cf8b1` stays by construction, and every
run checks it: each pair's compared population must equal its own leaves.

Add the §11 controls too — corrupt a timestamp, a feature builder, a geometry,
a model hash and a dataset membership, and prove the corresponding check fails.
[Data model](rearchitecture_data_model.md) §4 lists the cross-entity
relationships a restore must be able to walk; a control that breaks one of them
must be caught rather than silently replayed.

---

## 9. Step 6 — budgets and READMEs

`checks/code_budgets.py`, stdlib `ast`, staged blobs, enforcing §4.3 over
`engine/v2/**` only: complexity 15, 80 lines per function, 600 per module,
fan-out 8 for non-orchestrators, **zero exemptions**. Legacy `engine/` is
exempt wholesale per §4.6.

Every v2 package carries the §4.5 README with its seven sections. Two are
machine-checked against the import graph: a claimed consumer that does not
import, or an omitted one that does, fails; so does importing a name absent
from the declared public interface.

Wire both checks plus `import_layers` into the versioned hook at
`checks/hooks/pre-commit`, install it, and have the nightly re-run them over
`HEAD` — including whether the hook is installed. A failure refuses publication
per §4.7 while ingestion, scoring, settlement and backup advance normally.
The nightly half of that is phase 1's to wire (design §12), because it edits
legacy `engine/dashboard/nightly.py`; phase 0 delivers the gate's `--json`
output the nightly will call.

---

## 10. What phase 0 must not do

- No production logic in `engine/v2/`. The skeleton is empty packages, the
  comparator, and the checks.
- No edit to legacy `engine/` beyond extending `repo_hygiene.py` for the
  fixture path. In particular no legacy module gains a v2 import — §4.2 rule 3.
- No defect fixed because the corpus revealed it. Record it, freeze current
  behaviour as the baseline, raise it as a separate decision. §3.2: *"A known
  defect requires a separately recorded correction and new version, not a
  silent update to the baseline fixture."*
- No strategy, threshold, champion, fill convention or clock touched.

---

## 11. Known failure modes

- **Capturing through the display path.** The board rounds to six places. A
  corpus taken from `board.json` freezes rounded inputs and every later parity
  check inherits the error. Capture from the engine result.
- **Capturing before the lock.** A fixture taken in an unpinned environment
  cannot later distinguish a code change from a library upgrade.
- **A corpus that fits or fetches.** It stops being seconds, stops running on
  every edit, and becomes a tier-2 check nobody waits for.
- **Tolerances chosen to make the first run pass.** Set them per field from the
  arithmetic that produces the number, before running.
- **An empty population passing.** A fixture whose universe collapsed compares
  zero rows and reports agreement. `incomparable` exists for this.
- **Fixtures in the public repo.** They contain licensed quotes. Extend the
  hygiene scan before the first capture.
- **A green suite proving nothing.** 1,567 tests passed over the five defects
  above, and the determinism test passed the same frame twice. The negative
  controls of §8, not the pass count, are the evidence this phase worked.

---

## 12. Definition of done

Per convention 6: exit criteria met, acceptance tests green, and a generated
report documenting the evidence.

1. The baseline package exports reproducibly, covering all 12 strategies, 6
   model roles, 9 registered models and 6 legacy adapters, with the environment
   locked.
2. All 12 strategies and all 6 refusal codes reproduce from frozen requests in
   a fresh process through the production entry points, written and read back
   (tier 1, bound to the current code and frozen dependencies), and the corpus
   passes tier 0 under ten seconds with the network disabled. Reordered and
   batched engine-level parity is phase 1's O30.
3. The seeded defects produce their stage-named findings in one pass, over the
   real corpus at tier 0 and through real engine stages at tier 1.
4. `import_layers`, `code_budgets` and the README check pass over the v2
   skeleton; `legacy_adapters.json` reads `{"count": 0}`.
5. The hook is installed and versioned, and its state — with every check
   above — is machine-readable in `checks/rearchitecture_phase0_gate.py
   --json`. Wiring the nightly to report it is rearchitecture phase 1 (design
   §12): it edits legacy `engine/dashboard/nightly.py`, which §10 forbids here,
   and where this guide and the design disagree the design wins (§2).
6. The board's numbers are unchanged — the corpus captured them and nothing
   else moved.
