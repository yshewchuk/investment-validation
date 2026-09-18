# `engine/v2/models/training`

## Ownership

Implements the **Model training — dataset and model recipes, folds, fitting, evidence, release candidates** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**6** of §4.1.

Replaces (§4.4): `models/training/, rewritten above scoring rather than moved`.

## Responsibilities

- Dataset and model recipes, folds, fitting and residual construction.
- Evidence and release candidates; atomic promotion.

## Non-responsibilities

- **Run inside a score request** — `engine/v2/models` does it instead.
- **Be imported by a feature or a scorer** — `engine/v2/models` does it instead.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

P5-4 (`payoff.py`) adds the payoff-calibration artifact builders:
`build_payoff_line_artifact`, `build_payoff_surface_artifact`. Each fits via
`engine.v2.scoring.native_payoff`'s unchanged math (layer 5, strictly below
this package's layer 6) and wraps the result with `engine.v2.models`'s
(layer 3) `make_payoff_line_artifact`/`make_payoff_surface_artifact`, so the
returned artifact is bit-identical to the corresponding inline fit on the
same rows and cutoff.

<!-- public-interface: build_payoff_line_artifact, build_payoff_surface_artifact -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

_Nothing yet — no package imports this one. The first importer is added here in the same commit._

<!-- consumers: none -->

## Usage

    from engine.v2.models.training.payoff import build_payoff_line_artifact

    artifact = build_payoff_line_artifact(
        rows, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        before="2026-09-16",
    )
    # artifact is None when fewer than min_trades rows survive the causal
    # (exit_date < before) filter -- the same NO_PAYOFF_MAP condition the
    # inline fit refuses on today.

## Testing

Tier 0 (`component_contracts.md` §15.3): seconds, from frozen fixtures, no
panel load, no network, no fitting. Fixtures live in the private
`fixtures/tier0/` corpus (`checks/tier0_corpus.py`), never in this repo — they
carry licensed quotes.

A negative control here looks like: corrupt one field of a frozen record, run
the comparator, and assert it names **this package's stage** and that field
path — not that "a row is red". A check that has never failed is not known to
work.
