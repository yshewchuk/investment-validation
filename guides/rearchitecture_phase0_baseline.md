# Rearchitecture Phase 0 — Baseline

**Objective:** make every later phase checkable. Phase 0 writes no production
logic and changes no strategy, model, deployment or number. It produces the
instruments the migration is measured with — a frozen corpus of what the
current engine does, a comparison contract that can name what moved, the
enforced layer map, and the empty `engine/v2/` skeleton those layers describe.

Design reference: [system rearchitecture](system_rearchitecture.md) §3, §4.1-4.7,
§11, §12; [component contracts](component_contracts.md) §15. Where this guide
and the design doc disagree, the design doc wins and this guide is wrong.

**Exit gate (from §12):** every strategy and critical refusal reproducible;
five seeded defects yield five stage-named findings in one tier-0 pass; the
layer check runs green over the v2 skeleton with an empty adapter ledger.

---

## 1. Why this phase exists

§12 builds the replacement alongside the legacy store and deletes legacy whole
at phase 8. That choice rests entirely on parity: the new path is trusted
because it produces numbers identical to the old one. Parity needs an oracle,
and the oracle has to exist before the first replacement package is written.

It also has to be *fast*. On 2026-09-11 a single red self-check signal —
ten of twenty rows — carried five independent causes, and each correct fix left
the signal red because more causes remained behind it. Two properties fix that
class, and both are phase-0 deliverables: a corpus that answers in seconds, and
a comparison result that reports every independent finding at once instead of
the first.

Nothing here is a refactor. If a phase-0 change alters a board number, the
change is wrong.

---

## 2. Build order

Strictly ordered; each step is usable before the next starts.

```
1. v2 skeleton + layer map + checks/import_layers.py   (nothing to break yet)
2. ComparisonReceipt + the staged comparator            (needs no corpus)
3. Tier-0 corpus capture                                (needs 2 to be useful)
4. Negative controls                                    (proves 2 and 3 work)
5. checks/code_budgets.py + package READMEs             (locks 1 in)
```

Step 1 first because it is the only step that cannot fail a comparison — there
is no code in v2 yet. Step 2 before step 3 because a corpus without a
comparator is a directory of JSON nobody reads.

---

## 3. Step 1 — the v2 skeleton and the layer map

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

## 4. Step 2 — ComparisonReceipt and the staged comparator

Implement the minimum of [contracts §15](component_contracts.md#15-comparators---diagnosis)
that the corpus needs: `ComparisonReceipt`, `Finding`, and one comparator over
two `ScoreRecord`-shaped mappings. Full schema fidelity is not required in
phase 0; the two behavioural properties are:

- **Stage-localized.** Each finding names the first stage whose inputs agreed
  and outputs did not, plus the field path. Stage plan for the current scorer,
  following §6.3: `resolve_context`, `features`, `forecast`, `geometry`,
  `pricing`, `analogs`, `simulation`, `gate`, `chooser`, `serialization`.
- **Complete, not first-wins.** One pass reports every independent finding.
  Stopping at the first difference is what turned five causes into five nights.

Derive the compared field set from the digest's own field list, never a
hand-maintained constant — `28cf8b1` already made that fix for the explainer
and it must not be reintroduced. `verdict: incomparable` when the population is
empty or an input is missing; an empty comparison never reports agreement.

This code lands in `engine/v2/diagnosis/`, which nothing imports, so it is
subject to the v2 budgets from its first line and cannot become a dependency of
what it compares.

**Acceptance.** Two records differing in three unrelated fields produce three
findings, not one. A record compared against itself produces `agree`. An empty
input produces `incomparable`. Removing a field from the digest removes it from
the comparison with no edit to the comparator.

---

## 5. Step 3 — the tier-0 corpus

The corpus is frozen `(request, record)` pairs captured through the real public
entry points, per §3.2: *"Fixtures must come through real public entry points.
Do not invent a column such as `event_id` in a fixture if the current serving
row does not contain it."*

### 5.1 Coverage

| Axis | Required |
|---|---|
| Strategies | All 11 in `engine.structures.STRUCTURES` — `STR-THRU`, `STR-RUNUP`, `CAL-P`, `CND-P`, `CND-PS`, `TWIN-P`, `TWIN-P5`, `BFLY-P`, `BFLY-P5`, `RAMP7`, `CTR5` — plus `DYN-SV` |
| Model roles | All 6: `size`, `implied_t1`, `runup_move`, `iv_crush`, `gate`, `chooser`; both champion and non-champion entries resolvable |
| Refusal codes | `UNVALIDATED_STRUCTURE`, `OUT_OF_DOMAIN`, `NO_CHAIN`, `BAD_QUOTE`, `BAD_QUOTE_COST_PCT`, `COARSE_LADDER`, `NO_FORECAST` — at least one row each |
| Sessions | BMO and AMC; a year boundary; a month boundary |
| Geometry | Pinned `structure_params` and selector-resolved; a computed `width_moneyness` and a round listed strike; a coarse ladder; an exact-mirror requirement |
| DYN-SV | Full menu; a partial menu; a tie; a missing chooser score falling back to the resolver |
| Disabled | `CAL-P` and `CND-P` refusing in production and replaying under research |

`CAL-P` and `CND-P` must appear as *refusals*. A fixture that scores them is a
fixture of a strategy this program does not run.

### 5.2 Capture rules

- **Full precision.** Serialize replay inputs unrounded. `b33036c` and `6b9d5cf`
  are exactly this defect: `json_safe` rounded `structure_params` to six places
  and `_write_pair` re-rounded after the exemption. A corpus written through
  the board's display path would freeze the bug as the baseline.
- **Deterministic payload, separate envelope** (contracts §2.2). Wall-clock
  time, worker id and duration live outside the hashed payload so a replay
  reproduces the payload without reproducing the elapsed time.
- **No network, no panel load, no fitting** on replay. If a fixture needs a
  fold model, pin the artifact; a corpus that fits is not a tier-0 corpus.
- **Private.** Fixtures carry real quotes and go to the private mirror, never
  the public repo — convention 10. `checks/repo_hygiene.py` must be extended to
  cover the fixture directory before the first capture, not after.

### 5.3 Comparison rules

Per §3.2, exact for IDs, selected contracts, integer quantities, gate verdicts,
flags, null masks and frozen canonical records. Declared **per-field**
tolerances for independently recomputed floats. Never a global tolerance, and
never widened to make a test pass — a widened tolerance is a finding.

Null masks and refusal reasons are compared, not just non-null values. Five of
the six defects this corpus exists to catch were invisible in the non-null
values alone.

**Acceptance.** Every strategy and every refusal code reproduces from its
frozen request in a fresh process. Total runtime under ten seconds. The corpus
runs with the network disabled.

---

## 6. Step 4 — negative controls

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

Add the §11 controls too — corrupt a timestamp, a feature builder, a geometry,
a model hash and a dataset membership, and prove the corresponding check fails.

---

## 7. Step 5 — budgets and READMEs

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

---

## 8. What phase 0 must not do

- No production logic in `engine/v2/`. The skeleton is empty packages, the
  comparator, and the checks.
- No edit to legacy `engine/` beyond extending `repo_hygiene.py` for the
  fixture path. In particular no legacy module gains a v2 import — §4.2 rule 3.
- No defect fixed because the corpus revealed it. Record it, freeze current
  behaviour as the baseline, and raise it as a separate decision. §3.2: *"A
  known defect requires a separately recorded correction and new version, not a
  silent update to the baseline fixture."*
- No strategy, threshold, champion, fill convention or clock touched.

---

## 9. Known failure modes

- **Capturing through the display path.** The board rounds to six places. A
  corpus taken from `board.json` freezes rounded inputs and every later parity
  check inherits the error. Capture from the engine result.
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
  controls of §6, not the pass count, are the evidence this phase worked.

---

## 10. Definition of done

Per convention 6: exit criteria met, acceptance tests green, and a generated
report documenting the evidence.

1. All 12 strategies and all 7 refusal codes reproduce from frozen requests in
   a fresh process, network disabled, under ten seconds.
2. Five seeded defects produce five stage-named findings in one pass.
3. `import_layers`, `code_budgets` and the README check pass over the v2
   skeleton; `legacy_adapters.json` reads `{"count": 0}`.
4. The hook is installed and versioned; the nightly reports its state.
5. The board's numbers are unchanged — the corpus captured them and nothing
   else moved.
