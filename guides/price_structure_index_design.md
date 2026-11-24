# `price_structure` chain index and Resolution split (design)

Design PR; no code. Slices S1–S5 (section 9) add code. Contract lines: `engine/SCORING.md` section 9.

**Problem (supervisor measurement 2026-11-24).** On synthetic 300-event put chains with 5 short-vol structures, `engine.structures.price_structure` costs 12.8 ms per call against 1.8 ms for the old experiment resolver. About 54% is `ExpirySelector.select` re-run for every leg (pandas `drop_duplicates`, sort, filter), about 12% is `StrikeSelector.select` (pandas filter, grid rebuild per leg), the rest is per-leg boolean masks over the whole chain plus fill pricing; building the `ChainSnapshot` is 4%. It is repeated work on one immutable chain: `engine/replay.py` quotes every structure on one decision snapshot, and each structure has 2–7 legs.

**Hard rules (user).** Outputs are bit-identical: same strikes, prices, cash flows, flags, and the SAME exception types and messages, `LadderTooCoarse` included. No experiment changes: `experiments/` is not touched, and no running experiment may be affected. No cross-structure vectorising. Never read the real store, `data/` or `/root/rebuild-logs`; no heavy jobs; CI is the evidence for code slices.

## 0. Binding check (done before any code, per the brief)

Computed with `engine.v2.ops.fingerprints.source_closure` (static, no data read):

| Entry | Closure includes `engine/structures.py` and `engine/fills.py` |
| --- | --- |
| `experiments/size_panel.py` (the closure EXP-188/189 forbid editing) | **No** (150 files) |
| `experiments/EXP-188_*/run.py`, `size_pnl_driver/_shortvol/_straddle/_synthetic/_sources/_receipts.py`, `size_width_sweep.py`, `size_width_tuning.py` | **Yes** |
| `experiments/v2_magnitude_search.py`, `v2_prediction.py`, `v2_magnitude_search_run.py`, `magnitude_search_outer.py`, `size_search_binding.py` (the v14 SIZE plan closure: `magnitude_search_plan` defaults to the first two) | **Yes** |
| `engine/v2/ops/experiments.py` `capability_manifest` (hashes the runner closure) | Yes, for every runner that imports them |

So any byte change to `engine/structures.py` or `engine/fills.py`, or a new module they import, changes the closure the v14 sealed plan and the registered-runner manifests are bound to (`size_pnl_sources._read_plan` refuses a plan whose implementation differs from the registration's; the memory note "installed code == v14 closure" says the same). **This design PR changes no `.py` file, so nothing is rebound by it.** Per the brief, code slices stop at this finding until the supervisor decides. The split in section 9 is built so that S1–S4 are closure-neutral and only one small, optional "wire" slice (S5) edits a bound file.

**Closure-neutral rule for S1–S4.** `engine/structures.py` and `engine/fills.py` stay byte-identical. All new logic is in a NEW module `engine/chain_index.py` that nothing bound imports. It reads `ChainSnapshot`, `Structure`, `ResolvedLeg`, `StructurePrice` and the error classes from `engine.structures`. `engine/fills.py` is never edited: quote validation reuses the public `FillModel.mid`. A CI test (S2) asserts the git diff of the two bound files is empty for S1–S4.

## 1. The index

`ChainIndex` is built lazily on the first indexed call for a snapshot and holds copies of what the legacy path re-derives:

- **Columns, original row order:** `expiry` as int64 ns (comparisons) plus the original-dtype values (returned `Timestamp`s keep the column's unit), `strike` float64, `right`, `bid`, `ask`, `dte`, and `delta` if the column exists.
- **Expiries:** the distinct expiries, ascending, each with its single `dte` (the legacy `drop_duplicates().sort_values("expiry")` frame as arrays).
- **Groups:** per `(expiry, right)` and per `(expiry, ALL)` (the all-rights fallback, section 3), built on first touch: the positions of its rows in original order, so the strike, bid, ask, dte and delta arrays are original-order slices, plus the cached grid `np.sort(np.unique(strike))`.
- **Memo of expiry-rule results:** key `(kind, target_dte, min_dte, max_dte, expiry)`, value a `Timestamp` or the stored `(exception class, message)`. A miss runs the rule on the expiry arrays; a hit re-raises a fresh exception of the same class and message. Event date and session are per snapshot, so they are not in the key. A selector whose `expiry` field is unhashable bypasses the memo.

**Why original order, not sorted.** The legacy tie rules are row-order rules (`idxmin` first minimum, `hit.iloc[0]` first `isclose` row); sorting changes both on ties and duplicate quotes. Groups are small (about 100 rows), so the same numpy expressions on a group are exact and cheap; no `searchsorted`.

**Lifetime.** The index lives and dies with its snapshot. No global LRU and no sharing across snapshots. Until S5 `ChainSnapshot` is not edited, so the index lives in a module dict `{id(snapshot): (weakref(snapshot), index)}` with a `weakref.finalize` that removes the entry at collection; a lookup that finds a dead or different referent rebuilds. (A weak-keyed dict cannot be used: the frozen dataclass hashes its `DataFrame` field.) The build is guarded by a lock; a racing second build would be identical, and the first stored wins. The index is never pickled: it is not an attribute of the snapshot.

**Immutability and invalidation.** `ChainSnapshot.rows` must be treated as immutable once a snapshot has been priced. Pandas offers no cheap way to enforce this for a mutable `DataFrame`, so enforcement is layered:
1. The index copies its arrays, so it never aliases `rows`; a later in-place edit leaves the index at its build-time view, which is why mutation is a contract violation rather than a refresh.
2. S2 audits every `ChainSnapshot(` construction site and records in its PR that none mutates `rows` after building it; sites that reuse one frame across snapshots are listed.
3. **Verify mode:** `ENGINE_CHAIN_INDEX_VERIFY=1`, and unconditionally the battery, the regression tool and the benchmark, store a CRC32 fingerprint of every column the index reads at build time and re-check it on every indexed call. A mismatch raises `ChainSnapshotMutated(RuntimeError)`. It is deliberately NOT a `StructureError`: callers catch `StructureError` and write `NO_CHAIN`, which would hide the bug.
4. Default mode adds no per-call O(N) check (about 0.1 ms at 5,000 rows, 5% of the target budget).

There is no TTL and no explicit invalidate API: a changed chain is a new snapshot.

**Eligibility, then fallback.** The index is used only when the snapshot is in the proven shape. Ineligible snapshots run the legacy code (S5: kept as `_price_structure_legacy`; before S5, the indexed entry point calls `engine.structures.price_structure`). The decision is made before any leg runs, so paths never mix inside a call. Ineligible: empty rows; `strike` not float64 or int64, or non-finite; `expiry` not tz-naive datetime64, or any NaT; `dte` with NaN, or an expiry with several distinct `dte` (legacy quicksort tie order is unspecified); non-numeric `bid`, `ask` or `delta`; spot not a finite float; for `first_post_event`, an `event_date` that is not a tz-naive `pd.Timestamp`; for a pin or `fixed` selector, a non-tz-naive expiry. Validated ORATS chains are eligible; the fallback means exactness never depends on that.

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

- `resolve_structure(structure, snapshot, *, pin=None) -> Resolution` runs the single pass over legs: spot, then per leg expiry, strike, contract lookup, quote validation; after the last leg, the collision check. It takes no fill, because nothing before pricing depends on `alpha`.
- `price_resolution(resolution, snapshot, fill, *, closing=False) -> StructurePrice` does only the arithmetic: per leg the identical calls `fill.price`, `fill.cash_flow`, `FillModel.is_wide` on the stored bid/ask, with `side` flipped through `_OPPOSITE` when `closing`. It does not resolve. It checks that `resolution.ticker`, `obs_date`, `event_date` equal the snapshot's and raises `ValueError` otherwise; it does not re-read quotes, so passing a different chain is a caller error, not a repricing.
- `price_structure(structure, snapshot, fill, *, pin=None, closing=False)` is `price_resolution(resolve_structure(structure, snapshot, pin=pin), snapshot, fill, closing=closing)`: same signature, same behaviour.
- **`closing` is a pricing-time argument.** A Resolution is direction-free: its legs carry opening sides, as legacy `ResolvedLeg` entries carry them before the flip.
- **`pin=`** keeps its meaning: a pinned leg takes `expiry` and `strike` from the pin by name, skips both selectors, and the strike enters the cross-leg `resolved` dict so later unpinned legs can still reference it. Pin entries naming no leg are ignored; a duplicate pin name keeps the last, as the legacy dict does. `pin` accepts `Sequence[ResolvedLeg]` (unchanged) and also a `Resolution` or its `.legs`, because the code reads only `.name`, `.expiry`, `.strike`.
- **Exit reuse.** The entry `Resolution` replaces `pin=entry_price.legs`: `exit_res = resolve_structure(structure, exit_snapshot, pin=entry_res)`, then `price_resolution(exit_res, exit_snapshot, fill, closing=True)`. The pinned path therefore still checks the entry's expiry exists on the exit chain and that the contract is listed there. A resolution never prices a chain it was not resolved on without `resolve_structure(pin=...)`, so "closing without pinning is nearly always a bug" stays enforceable.

## 3. Exact-equivalence rules

The reference is the current `price_structure`, `ExpirySelector.select` and `StrikeSelector.select`, frozen as the oracle (section 6).

**Expiry rules.** Same inclusive filters, in order: `fixed` first (absence message, bypasses min/max), then `min_dte`/`max_dte` (`no expiry survives {selector!r}`), then the kind. `first_post_event`: `str(session).upper() == "AMC"` requires expiry `>` event date, otherwise `>=`, both messages unchanged and using `event_date.date()`. `first_dte_at_least`: first sorted expiry with `dte >= target`. `nearest_dte`: sort by `(abs(dte - target), -dte)`, so ties go long. Returned values are `pd.Timestamp(column value)` (column unit) except `fixed`, which returns `pd.Timestamp(self.expiry)` as before.

**Strike rules, by kind.**
- `atm`/`moneyness`: `gap = np.abs(strike_orig - target)`, `strike_orig[gap.argmin()]`: first row at the minimum, equal to `idxmin`. Ties between two listed strikes go to whichever is listed first in the row order, NOT the lower strike.
- `delta`: `|delta|` gap in row order, first minimum, NaN skipped (`nanargmin`); the "needs a delta column" check runs on the same-right group, as now.
- `bracket`, `grid_step`, `offset_from`: the same numpy expressions on the cached ascending grid. For `offset_from` and `bracket` ties go to the lower grid index, which differs from `atm`; do not unify.
- `same_as`, `mirror`, `fixed`: the returned value is the TARGET (`resolved[ref]`, the mirror, `self.strike`), never the matched listed strike. The contract lookup then runs `isclose` again.
- Right restriction: the same-right group is used when it is non-empty, else the all-rights group at the expiry, exactly as the legacy `rows = same_right` fallback.

**`np.isclose` semantics.** Every strike match is `np.isclose(listed, target)` with the defaults `rtol=1e-5`, `atol=1e-8`, evaluated as `abs(listed - target) <= atol + rtol * abs(target)`. The tolerance scales with the SECOND argument (the target), so it is asymmetric and widens with the strike. The index calls `np.isclose` itself, on the group array in the same argument order, never a hand-written inequality or a `searchsorted` window, so rounding and NaN handling are inherited rather than re-derived. Consequences kept on purpose: (a) a target within tolerance of several listed strikes matches all of them; contract lookup takes the FIRST in row order (`hit.iloc[0]`), `grid_step`/`bracket` anchors take the LOWEST grid index (`hit[0]`); (b) a NaN target matches nothing; (c) duplicate rows for one contract are resolved by row order.

**Contract lookup** (after expiry and strike): rows of the `(expiry, right)` group with `isclose(strike, resolved)`; empty raises the unchanged message; else the first row gives `dte` (`int(...)`), `bid`, `ask` (`float(...)`). The recorded `strike` on the leg is `float(resolved strike)`, as today.

## 4. Failure semantics (R1–R6 where they fit)

R3 (determinism) and R5 (identity) are the load-bearing ones; the others are mostly "no new state".

| Rule | Statement |
| --- | --- |
| **R1 missing input** | A snapshot missing a required column raises `StructureError("chain snapshot missing columns: ...")` in `ChainSnapshot.__post_init__`, before any index exists. No spot raises `StructureError("chain snapshot has no spot price")`, first. Nothing is partly resolved. |
| **R2 resources** | Bounded by the chain: copies of seven columns plus group arrays, freed with the snapshot. |
| **R3 determinism** | For an eligible snapshot the output equals the oracle bit for bit, whatever the call order or count. Memoised expiry results are pure in `(selector, event date, session, chain)`; a cached error re-raises a NEW exception of the same class and message. |
| **R4 persistence** | None: no file, ledger or receipt. |
| **R5 identity** | `Resolution` records ticker, `obs_date`, `event_date`, spot and contracts; `price_resolution` refuses a snapshot whose identity differs. Bound files stay byte-identical until S5. |
| **R6 causality** | Nothing reads beyond the chain given. A pin carries only expiry and strike, so no entry quote reaches the exit price. |

**Failure order, unchanged.** Per leg, in structure order, the first failure wins: (1) pinned leg: pinned expiry absent from the chain (`StructureError`); else the expiry rule (`StructureError`, in the order in section 3); (2) the strike rule (`StructureError`, messages per kind, including "no chain rows at the selected expiry"); (3) no listed contract (`StructureError: {leg}: no {right} at strike {strike} expiry {date} on {obs}`); (4) the quote check, `ValueError` from `FillModel.mid` for NaN, negative bid, negative ask, crossed, in that order. **After all legs**, `LadderTooCoarse` (a `StructureError` subclass) with legs grouped by `(right, strike, expiry)`, groups sorted as now and rendered with `:g`, including zero-quantity reference legs. Because (4) runs inside `resolve_structure` per leg, a later leg's `StructureError` still cannot pre-empt an earlier leg's `ValueError`, which is the legacy order; `price_resolution` itself cannot fail on data. The `ValueError` for a bad `alpha` is raised by `FillModel` at construction, before either function.

## 6. Equivalence battery

`tests/v2/research/test_price_structure_equivalence.py`, with the frozen oracle in `tests/v2/research/price_structure_oracle_support.py`: a verbatim copy of today's `price_structure` loop, both `select` methods and the collision check, so later engine edits cannot move the reference. A test checks the oracle against `engine.structures` while they are the same code and records its source hash. The battery is parametrised on a subject callable (S1: the current function; S2 onward: `engine.chain_index`).

Each case asserts equal `StructurePrice.to_dict()` and field types, or the same exception TYPE and `str`:
- Every key of `STRUCTURES`, default parameters, on AMC, BMO and `None` sessions; width and `steps` sweeps for the five short-vol builders, including zero-quantity reference legs.
- `pin=`: entry onto a post-print chain, `closing=True`, a pinned expiry absent on the exit chain, an unlisted pinned strike, a pin naming no leg, a duplicate pin name.
- Coarse ladders: the KEN and ABM geometries in `guides/coarse_ladder_collision.md`, $5 and $10 ladders, the exact `LadderTooCoarse` text.
- Every refusal in section 4, each expiry-rule and strike-selector error, a calls-only chain for a put leg (all-rights fallback), crossed, negative and NaN quotes, a `ValueError` on leg 1 with a `StructureError` on leg 2, missing spot, missing column.
- `isclose` edges: a target just inside and just outside tolerance, two listed strikes inside one tolerance, duplicate rows for one contract, equidistant strikes in both row orders.
- Ineligible shapes (NaN or float32 strike, tz-aware expiry, two `dte` for one expiry, empty chain) match the oracle via the fallback.
- Seeded property test: 200 random chains per structure; results equal or both refuse identically.
- Call-order independence (shuffled structure orders on one snapshot, repeat calls) and verify mode (an in-place `rows` edit raises `ChainSnapshotMutated`).
- S2 to S4: the diff of `engine/structures.py` and `engine/fills.py` against `origin/main` is empty.

## 7. Real-data regression tool

`tools/px_ledger_regression.py` re-prices the board's recorded structures from the prediction ledger and requires exact agreement. **The supervisor runs it after the heavy run; it never runs in CI on real data.** It opens the ledger read-only, reads only the paths given on the command line and writes nothing in the repo, so `--no-ledger` does not apply.

For each prediction row with `strategy`, `structure_params`, a quote date and recorded legs and `entry_cost`, it rebuilds the structure from `STRUCTURES`, loads that date's chain through the board's loader, and prices with BOTH the oracle and the indexed path at the recorded fill. **Pass**: every row identical (all `StructurePrice.to_dict()` fields, or the same exception type and message). Otherwise it exits non-zero and prints up to 20 mismatching row ids. An informational table compares each row with the ledger's recorded contract and cost; drift versus the ledger is reported and, by the pass condition, identical for both paths. Progress goes to stderr (stage start/end, done/total, rate, ETA, RSS); stdout is one JSON summary. `--limit` and `--sample-seed` allow a bounded first pass; verify mode is on.

## 8. Benchmark script

`tools/bench_price_structure.py`, no data read: seeded synthetic chains (300 events, put ladders at $0.5/$1/$5/$10 spacings, 3–6 expiries, mixed AMC/BMO), the five short-vol structures (`CND-PS`, `BFLY-P`, `BFLY-P5`, `RAMP7`, `CTR5`), oracle versus indexed. It reports median, p95 and total per call for **cold** (fresh snapshot per call: build plus price) and **warm** (one snapshot across the five structures), the speed-up, and asserts identical outputs on every call. Targets, reported not CI-gated: warm at most 3 ms per call (baseline 12.8 ms; old resolver 1.8 ms), cold not slower than legacy. If cold is slower, the build trims eager work, since single-structure callers such as `engine/score.py` pay it every time.

## 9. Slice split list (about 150 code lines each; test code excluded)

| Slice | Content | Binding |
| --- | --- | --- |
| **S1** | Oracle support module and the battery harness, run against the current code. Test code only. | Neutral |
| **S2** | `engine/chain_index.py`: `ChainIndex`, eligibility, weak registry, verify mode, expiry memo; `price_structure_indexed` using the index for the expiry rule only. Expiry and refusal battery rows. | Neutral (diff-empty test) |
| **S3** | Strike and contract-lookup groups; whole `price_structure_indexed` on the index; remaining battery rows. | Neutral |
| **S3b** | `tools/bench_price_structure.py` and `tools/px_ledger_regression.py` (need the indexed path); benchmark numbers in the PR. | Neutral |
| **S4** | `Resolution`, `ResolvedContract`, `resolve_structure`, `price_resolution`, `pin=Resolution`, all inside `chain_index.py`; `price_structure_indexed` becomes the composition. | Neutral |
| **S5 (optional, gated)** | Edit bound `engine/structures.py` so `price_structure` delegates to the indexed path, with the legacy body kept as the fallback. Needs a supervisor decision on the closure binding of the v14 plan and runner manifests first. | **Edits bound file** |

EXP-188/189 adapters adopting the split is a separate later piece of work in `experiments/` and needs the same binding decision (its adapters sit inside the bound closures regardless). It can call `engine.chain_index` directly, so S5 is not on its path.

## 10. Standard-tool check

numpy and pandas only. No library covers exact legacy tie rules, and `np.isclose` is called, not re-implemented. No standard format is hand-rolled.

## 11. Open point for the supervisor

Section 0: whether S5 (the bound-file edit) is wanted at all, or callers adopt `engine.chain_index` directly and `price_structure` stays as the oracle.
