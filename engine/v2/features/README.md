# `engine/v2/features`

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for this component's contracts,
inputs/outputs, dependencies and invariants.

## Ownership

Implements the **Feature engine — registered transforms and their causal dependencies** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**2** of §4.1.

Replaces (§4.4): `features.py`, `data/features/panel.py`, `data/features/tier4.py`.

## Responsibilities

- Registered feature recipes, their units and their missing-value policy.
- Causal dependency declaration for every transform.
- Tier-4 columns materialized from frozen feature-model artifacts.

## Non-responsibilities

- **Select an implicit latest dataset** — `engine/v2/data` does it instead.
- **Silently change a missing-value policy** — `a new recipe version` does it instead.
- **Import model training — a feature depends on a frozen artifact, never on the code that fits one** — `engine/v2/models/training` does it instead.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

`recipes` provides `FeatureRegistry` and `default_feature_registry`; `context`
provides `FeatureContextPlanner` and causal `FeatureFrame` construction.

`runup_math.add_runup_features(frame, prices_by_ticker, as_of_column)` computes
shared streak and price-history inputs from caller-selected event/price frames.

`daily_state_inputs.scan_daily_state_inputs(repository, snapshot, *, ticker,
history_start, decision_session)` reads one ticker's bounded `daily_market` rows
from a pinned snapshot and returns `daily_state_inputs.DailyStateInputs`. The
result is session-only: it is not a complete forward panel or a qualified
board, and no production raw-row assembler calls it yet.

`panel_row_inputs.scan_panel_row(repository, snapshot, key, *,
decision_session, history_start)` composes `daily_state_inputs`/`panel_math`/
`regime`/`runup_math` into one `BoardRequest` key's full-superset
`panel_row_inputs.PanelRowInputs` (`panel_row`, `panel_anchor`). No production
raw-row producer calls it yet (`engine/v2/ops/ARCHITECTURE.md` "Cutover
PR-6").

<!-- public-interface: recipes, FeatureRegistry, default_feature_registry, runup_math, add_runup_features, regime.add_regime_features, daily_state_inputs.DailyStateInputs, daily_state_inputs.scan_daily_state_inputs, panel_row_inputs.PanelRowInputs, panel_row_inputs.scan_panel_row -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

`engine/v2/scoring` resolves feature scopes and recipe identities before scoring.

<!-- consumers: engine.v2.scoring -->

## Usage

The package includes pure regime calculation logic over explicit inputs.
No production path calls the regime helper yet; source reads and forward
panel assembly remain separate integration work.

`panel_math.advance_history(last_row)` advances a caller-selected realized
panel row into next-event history aggregates. It is pure arithmetic; callers
retain ownership of event selection and observation cutoffs. No production
panel-row builder calls it yet.

## Testing

`regime.add_regime_features` accepts explicit event and market frames, returning
regime values with their actual observation dates. It has no production caller.
Its real captured-source parity test is marked `needs_corpus`; the payload stays
private and can be selected with `V2_REGIME_CORPUS_CSV`; its sibling
`manifest.json` records the source and verifies the captured file hash.

Tier 0 (`component_contracts.md` §15.3): seconds, from frozen fixtures, no
panel load, no network, no fitting. Fixtures live in the private
`fixtures/tier0/` corpus (`checks/tier0_corpus.py`), never in this repo — they
carry licensed quotes.

A negative control here looks like: corrupt one field of a frozen record, run
the comparator, and assert it names **this package's stage** and that field
path — not that "a row is red". A check that has never failed is not known to
work.
