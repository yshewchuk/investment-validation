# Twin pricing module: chain index and Resolution split (design)

Design PR; no code. Slices (a)–(d) are in section 9. Contract lines: `engine/SCORING.md` section 9. Decision: user 2026-11-24, after the binding finding in section 0.

**Problem (supervisor measurement 2026-11-24).** On synthetic 300-event put chains with 5 short-vol structures, `engine.structures.price_structure` costs 12.8 ms per call against 1.8 ms for the old experiment resolver. About 54% is `ExpirySelector.select` re-run for every leg (pandas `drop_duplicates`, sort, filter), about 12% is `StrikeSelector.select` (pandas filter, grid rebuild per leg), the rest is per-leg boolean masks over the whole chain plus fill pricing; building the `ChainSnapshot` is 4%. It is repeated work on immutable chains: each structure has 2–7 legs.

**Hard rules.** Outputs are bit-identical to `price_structure`, including the SAME exception types and messages (`LadderTooCoarse` too). No running experiment is affected. No cross-structure vectorising. No real store, `data/` or `/root/rebuild-logs` reads; no heavy jobs; CI is the evidence.

## 0. Binding finding and the twin decision

Computed with `engine.v2.ops.fingerprints.source_closure` (static, no data read): `engine/structures.py` and `engine/fills.py` are in the closure of the v14 SIZE plan entries (`v2_magnitude_search`, `v2_prediction`, `magnitude_search_outer`, `size_search_binding`), the EXP-188 runner, `size_pnl_*` and `size_width_*`; they are not in `source_closure(size_panel.py)`. Any byte change, or a new module they import, breaks the sealed plan and runner manifests (`plan_registration_mismatch`, `INPUT_CHANGED`).

**Decision: a TWIN module.** New `engine/chain_index.py`:
- It IMPORTS, unchanged, `Structure`, `LegSpec`, the builders, `ChainSnapshot`, `ResolvedLeg`, `StructurePrice`, the error classes and `FillModel`. It adds no import to, and edits no byte of, `engine/structures.py` or `engine/fills.py`.
- It DUPLICATES only expiry selection, strike selection and the `price_structure` loop (about 350 lines today), in two forms: a plain pandas path (a resolve-only copy of the legacy loop, the fallback of section 1) and the indexed path.
- Nothing in the v14 closure imports it. A CI test asserts `engine/chain_index.py` is not in `source_closure(root, ["experiments/size_search.py", "experiments/magnitude_search_worker.py"])`, and that the two bound files' content hashes equal `origin/main`'s.
- A third copy already exists: `engine/v2/research/_pricing.py` (a verbatim port used by native replay). It is out of scope and unchanged; slice (a2) adds a drift test comparing it with the twin on the battery's cases.
- Adoption: EXP-189's adapter uses the twin (a separate experiment slice). The board keeps `price_structure` for now. `experiments/` is not touched by this work.
- Retirement: section 8.

## 1. The index

`IndexedChain` (not `ChainIndex`, which `engine/v2/research/_chains.py` and `engine/score.py` already export) is built lazily on the first indexed call for a snapshot and holds copies of what the legacy path re-derives:

- **Columns, original row order:** `expiry` as int64 ns via `astype("datetime64[ns]")` (exact for s/ms/us/ns units; used for comparisons) plus the original-dtype values (returned `Timestamp`s keep the column's unit), `strike` float64, `right`, `bid`, `ask`, `dte`, and `delta` if the column exists.
- **Expiries:** the distinct expiries, ascending, each with its single `dte` (the legacy `drop_duplicates().sort_values("expiry")` frame as arrays).
- **Groups:** per `(expiry, right)` and per `(expiry, ALL)` (the all-rights fallback, section 3), built on first touch: row positions in original order, so strike, bid, ask, dte and delta are original-order slices, plus the cached grid `np.sort(np.unique(strike))`.
- **Memo of expiry-rule results:** key `(kind, target_dte, min_dte, max_dte, expiry)`, value a `Timestamp` or the stored `(exception class, message)`. A hit re-raises a fresh exception of the same class and message. Event date and session are per snapshot, so not in the key; an unhashable `expiry` bypasses the memo.

**Why original order.** Legacy tie rules are row-order rules (`idxmin`, `hit.iloc[0]`); sorting changes them. Groups are small (about 100 rows), so the same numpy expressions on a group are exact and cheap.

**Lifetime.** The index lives and dies with its snapshot. No global LRU and no sharing. `ChainSnapshot` is not edited, so the index lives in a module dict `{id(snapshot): (weakref(snapshot), index)}` with a `weakref.finalize` that removes the entry at collection; a lookup that finds a dead or different referent rebuilds. A lock guards the build; the index is never pickled.

**Immutability and invalidation.** `ChainSnapshot.rows` must be treated as immutable once a snapshot has been priced through the twin. Pandas offers no cheap enforcement, so it is layered:
1. The index copies its arrays; a later in-place edit leaves it at its build-time view, so mutation is a contract violation, not a refresh.
2. The twin slice audits every `ChainSnapshot(` construction site that will use it (EXP-189's adapter first) for post-build `rows` mutation and records the result in its PR.
3. **Verify mode:** `ENGINE_CHAIN_INDEX_VERIFY=1`, and unconditionally the battery and the regression tool, store a CRC32 fingerprint of every column the index reads at build time and re-check it on every indexed call. A mismatch raises `ChainSnapshotMutated(RuntimeError)`, deliberately NOT a `StructureError`: callers catch that and write `NO_CHAIN`, hiding the bug.
4. Default mode adds no per-call O(N) check. There is no TTL or invalidate API: a changed chain is a new snapshot.

**Eligibility, then fallback.** The indexed path is used only when the snapshot is in the proven shape; otherwise the plain pandas path (the twin's own copy of the legacy loop) runs. The decision is made before any leg runs, so paths never mix inside a call. Ineligible: empty rows; duplicate `rows.index` labels (legacy `loc` then returns a Series and raises `TypeError`); an expiry outside the ns range (1678–2262); `strike` not float64 or int64, or non-finite; `expiry` not tz-naive datetime64, or any NaT; `dte` with NaN, or an expiry with several distinct `dte` (legacy quicksort tie order is unspecified); non-numeric `bid`, `ask` or `delta`; spot not a finite float; for `first_post_event`, an `event_date` that is not a tz-naive `pd.Timestamp`; for a pin or `fixed` selector, a non-tz-naive expiry.

## 2. Resolution and how `pin=` and `closing` map onto it

```python
@dataclass(frozen=True)
class ResolvedContract:   # one leg located on a chain, quote already validated
    name; right; side; qty        # side = the structure's OPENING side
    expiry: pd.Timestamp; strike: float   # the selected strike, not the matched listed one
    dte: int; bid: float; ask: float

@dataclass(frozen=True)
class Resolution:
    structure: str; ticker: str; obs_date; event_date; spot: float
    legs: tuple[ResolvedContract, ...]    # structure leg order
```

- `resolve_structure(structure, snapshot, *, pin=None) -> Resolution`: spot, then per leg expiry, strike, contract lookup, quote validation; after the last leg, the collision check. It takes no fill, because nothing before pricing depends on `alpha`.
- `price_resolution(resolution, snapshot, fill, *, closing=False) -> StructurePrice`: per leg the identical calls `fill.price`, `fill.cash_flow`, `FillModel.is_wide` on the stored bid/ask, with `side` flipped through the opposite map when `closing`. It requires `snapshot` to be the object the Resolution was resolved on (weak reference) and raises `ResolutionMismatch(RuntimeError)` otherwise; it never re-reads quotes. The type is deliberately not `ValueError`/`StructureError`: `engine/score.py` writes `NO_CHAIN` on those and `engine/replay.py` turns them into skip reasons, either of which would hide a caller bug.
- `price_structure_indexed(structure, snapshot, fill, *, pin=None, closing=False)` is `price_resolution(resolve_structure(...), ...)`, with `price_structure`'s signature; it is the function the battery compares with `engine.structures.price_structure`.
- **`closing` is a pricing-time argument.** A Resolution is direction-free: legs carry opening sides.
- **`pin=`** keeps its meaning: a pinned leg takes `expiry` and `strike` from the pin by name, skips both selectors, and the strike enters the cross-leg `resolved` dict so later unpinned legs can reference it. Entries naming no leg are ignored; a duplicate name keeps the last. `pin` accepts `Sequence[ResolvedLeg]` (no identity check, as today) and also a `Resolution` or its `.legs`. A `Resolution` pin is identity-checked: a different `ticker` or `structure` raises `ResolutionMismatch` before any leg runs. Bare `.legs` carry no identity and keep the legacy rule.
- **Exit reuse.** The entry `Resolution` replaces `pin=entry_price.legs`: `exit_res = resolve_structure(structure, exit_snapshot, pin=entry_res)`, then `price_resolution(exit_res, exit_snapshot, fill, closing=True)`. The pinned path still checks the expiry exists on the exit chain and the contract is listed.

## 3. Exact-equivalence rules

The reference is `engine.structures.price_structure` itself, with `ExpirySelector.select` and `StrikeSelector.select`; the original stays in the tree until retirement.

**Expiry rules.** Same inclusive filters, in order: `fixed` first (absence message, bypasses min/max), then `min_dte`/`max_dte` (`no expiry survives {selector!r}`), then the kind. `first_post_event`: `str(session).upper() == "AMC"` requires expiry `>` event date, otherwise `>=`, both messages unchanged with `event_date.date()`. `first_dte_at_least`: first sorted expiry with `dte >= target`. `nearest_dte`: sort by `(abs(dte - target), -dte)`, ties go long. Returned values are `pd.Timestamp(column value)` (column unit) except `fixed`, which returns `pd.Timestamp(self.expiry)` as before.

**Strike rules, by kind.**
- `atm`/`moneyness`: `gap = np.abs(strike_orig - target)`, `strike_orig[gap.argmin()]`: first row at the minimum, equal to `idxmin`. Ties between two listed strikes go to whichever is listed first in row order, NOT the lower strike.
- `delta`: `|delta|` gap in row order, first minimum, NaN skipped (`nanargmin`); the "needs a delta column" check (missing or all-NaN) runs on the group the selector uses AFTER the all-rights fallback, so a calls-only chain with finite call deltas selects a call strike and fails later at the contract lookup.
- `bracket`, `grid_step`, `offset_from`: the same numpy expressions on the cached ascending grid. `offset_from` first filters the grid to strikes strictly above (`moneyness > 0`) or below the anchor, then takes `argmin(|grid - (anchor + moneyness * spot)|)`; its error is "no listed strike {above|below} {anchor:.4f} at this expiry". Ties for `offset_from` go to the lower grid index, which differs from `atm`; do not unify. `bracket` takes the nearest listed strike at or below (above) spot and uses `isclose` only when `steps` is set; `grid_step` anchors use the lowest `isclose` hit.
- `same_as`, `mirror`, `fixed`: the returned value is the TARGET (`resolved[ref]`, the mirror, `self.strike`), never the matched listed strike; contract lookup then runs `isclose` again.
- Right restriction: the same-right group is used when non-empty, else the all-rights group at the expiry, as the legacy `rows = same_right` fallback.

**`np.isclose` semantics.** Every strike match is `np.isclose(listed, target)` with the defaults `rtol=1e-5`, `atol=1e-8`, evaluated as `abs(listed - target) <= atol + rtol * abs(target)`. The tolerance scales with the SECOND argument (the target): asymmetric, widening with the strike. The index calls `np.isclose` itself, on the group array in the same argument order, never a hand-written inequality or `searchsorted` window, so rounding and NaN handling are inherited. Consequences kept on purpose: (a) a target within tolerance of several listed strikes matches all; contract lookup takes the FIRST in row order (`hit.iloc[0]`), `grid_step`/`bracket` anchors the LOWEST grid index (`hit[0]`); (b) a NaN target matches nothing; (c) duplicate rows for one contract resolve by row order.

**Contract lookup** (after expiry and strike): rows of the `(expiry, right)` group with `isclose(strike, resolved)`; empty raises the unchanged message; else the first row gives `dte`, `bid`, `ask` as today.

## 4. Failure semantics (R1–R6 where they fit)

| Rule | Statement |
| --- | --- |
| **R1 missing input** | A snapshot missing a required column raises in `ChainSnapshot.__post_init__`, before any index exists. No spot raises `StructureError("chain snapshot has no spot price")`, first. Nothing is partly resolved. |
| **R3 determinism** | For an eligible snapshot the output equals `price_structure`'s bit for bit, whatever the call order or count. Memoised expiry results are pure in `(selector, event date, session, chain)`; a cached error re-raises a NEW exception of the same class and message. |
| **R5 identity** | `Resolution` records ticker, `obs_date`, `event_date`, spot and contracts; `price_resolution` refuses any snapshot but its own. Bound files stay byte-identical. |
| **R6 causality** | Nothing reads beyond the chain given. Resolution ignores the entry quotes a pin carries, so none reaches the exit price. |

**Failure order, unchanged.** Per leg, in structure order, the first failure wins: (1) pinned leg: pinned expiry absent from the chain (`StructureError`); else the expiry rule (`StructureError`, order of section 3); (2) the strike rule (`StructureError`, messages per kind, including "no chain rows at the selected expiry"); (3) no listed contract (`StructureError: {leg}: no {right} at strike {strike} expiry {date} on {obs}`); (4) the quote check, `ValueError` from `FillModel.mid` for NaN, negative bid, negative ask, crossed, in that order. **After all legs**, `LadderTooCoarse` (a `StructureError` subclass) with legs grouped by `(right, strike, expiry)`, groups sorted as now and rendered with `:g`, including zero-quantity reference legs. Because (4) runs inside `resolve_structure` per leg, a later leg's `StructureError` cannot pre-empt an earlier leg's `ValueError`, as in legacy; `price_resolution` cannot fail on data. A bad `alpha` fails at `FillModel` construction.

## 5. Equivalence battery and closure test

`tests/v2/research/test_chain_index_equivalence.py`. The subject is `price_structure_indexed`; the reference is `engine.structures.price_structure` called on a fresh, identical snapshot (no copied oracle: the original stays). Each case asserts the two `StructurePrice` objects equal field by field (`Timestamp` values and `.unit`, floats exact, which `to_dict()` strings would hide) and `to_dict()` equal, or the same exception TYPE and `str`:
- Every key of `STRUCTURES`, default parameters, on AMC, BMO and `None` sessions; width and `steps` sweeps for the five short-vol builders, including zero-quantity reference legs.
- `pin=`: entry onto a post-print chain, `closing=True`, a pinned expiry absent on the exit chain, an unlisted pinned strike, a pin naming no leg, a duplicate pin name, a `Resolution` pin with another ticker or structure (`ResolutionMismatch`, tested on the twin only: `price_structure` takes no `Resolution` pin).
- Coarse ladders: the KEN and ABM geometries of `guides/coarse_ladder_collision.md`, with the exact `LadderTooCoarse` text.
- Every refusal in section 4, each expiry-rule and strike-selector error, a calls-only chain for a put leg (all-rights fallback), crossed, negative and NaN quotes, a `ValueError` on leg 1 with a `StructureError` on leg 2, missing spot, missing column.
- `isclose` edges: a target just inside and just outside tolerance, two listed strikes inside one tolerance, duplicate rows for one contract, equidistant strikes in both row orders, `nearest_dte` ties in both orders.
- Ineligible shapes (section 1) match `price_structure` via the plain path; a seeded property test of 200 random chains per structure.
- Call-order independence, verify mode (an in-place `rows` edit raises `ChainSnapshotMutated`), `price_resolution` refusing another snapshot (`ResolutionMismatch`, twin only), and a planted-defect control (a deliberately wrong tie rule must fail the battery).
- **Closure test** (`tests/v2/integration/test_chain_index_closure.py`): `engine/chain_index.py` is absent from `source_closure(root, [...])` for each bound root: `experiments/size_search.py`, `magnitude_search_worker.py`, `size_search_binding.py`, `v2_magnitude_search.py`, `size_pnl_shortvol.py`, `size_width_sweep.py` and the EXP-188 `run.py`; and the content hashes of `engine/structures.py` and `engine/fills.py` equal `origin/main`'s.

## 6. Real-data regression tool

`tools/px_ledger_regression.py` re-prices the board's recorded structures from the prediction ledger and requires exact agreement. **The supervisor runs it after the heavy run; it never runs in CI on real data.** It opens the ledger read-only, reads only the paths given on the command line and writes nothing in the repo, so `--no-ledger` does not apply.

For each prediction row with `strategy`, `structure_params`, a quote date and recorded legs and `entry_cost`, it rebuilds the structure from `STRUCTURES`, loads that date's chain through the board's loader, and prices with BOTH `price_structure` and `price_structure_indexed` at the recorded fill. **Pass**: every row identical (all `StructurePrice` fields compared as in section 5, or the same exception type and message). Otherwise it exits non-zero and prints up to 20 mismatching row ids. Drift versus the ledger's recorded contract and cost is reported only. Progress goes to stderr (stage start/end, done/total, rate, ETA, RSS); stdout is one JSON summary. `--max-bytes` and `--max-rows` (200,000) refuse before decoding the ledger; `--limit` and `--sample-seed` bound a run; skipped rows are counted by reason; verify mode is on.

## 7. Benchmark script

`tools/bench_price_structure.py`, no data read: seeded synthetic chains (300 events, put ladders at $0.5/$1/$5/$10 spacings, 3–6 expiries, mixed AMC/BMO), the five short-vol structures (`CND-PS`, `BFLY-P`, `BFLY-P5`, `RAMP7`, `CTR5`), `price_structure` versus `price_structure_indexed`. It reports median and p95 per call for **cold** (fresh snapshot: build plus price) and **warm** (one snapshot across the five structures), and asserts identical outputs. Targets, not CI-gated: warm at most 3 ms per call (baseline 12.8 ms), cold not slower than `price_structure`. Verify mode is off, with a verify-on row reported separately.

## 8. Retirement, with a pre-stated trigger

**Trigger:** the v14-bound experiments have ended: EXP-188, EXP-189 and the EXP-188 feedback rerun. Then, in one PR: `price_structure` becomes a thin wrapper over the twin, the original selection code in `engine/structures.py` is deleted, and a frozen copy of the original moves into the test tree as the oracle that the battery compares against. The twin's imports from `engine.structures` then become circular, so the wrapper imports the twin lazily inside `price_structure`. The entry is `TD-13` in `guides/rearchitecture_tech_debt.md`. Until then both implementations stay and the battery compares them directly.

## 9. Split list

Estimates (code / test / doc lines):

| Slice | Content | Estimate | Binding |
| --- | --- | --- | --- |
| **(a)** | `engine/chain_index.py`: index, eligibility, plain path, memo, verify mode, `Resolution`, `resolve_structure`, `price_resolution`, `price_structure_indexed`; the battery and closure test. Built by pxidx1 as two PRs: (a1) plain path, index, expiry rules, eligibility, memo, verify mode, closure test: 280 / 250 / 5; (a2) strike rules, contract lookup, `Resolution` and the split: 250 / 250 / 10. | 530 / 500 / 15 | Neutral |
| **(b)** | Real-data ledger regression tool, and the benchmark script as its own PR. Supervisor runs the tool after the heavy run. | 130 / 70 / 10 and 120 / 50 / 5 | Neutral |
| **(c)** | EXP-189 adapter adopts the twin (`experiments/`, separate experiment slice; the board keeps `price_structure`). | 60 / 80 / 5 (its Spec confirms) | Experiment-owned |
| **(d)** | Retirement, later, on the section 8 trigger. | about 40 / 30 / 5 (net deletion) | Edits bound file, after the experiments end |

## 10. Standard tools

numpy and pandas only; no library covers the legacy tie rules, and `np.isclose` is called, not re-implemented.
