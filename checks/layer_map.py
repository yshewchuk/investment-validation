#!/usr/bin/env python3
"""The declared layer map for ``engine/v2/`` — one literal, read by three checks.

`system_rearchitecture.md` §4.1 states the layering as a table. A table in a
document constrains nothing, so it is transcribed here once and read by
:mod:`checks.import_layers`, :mod:`checks.package_readmes` and
:mod:`checks.code_budgets`. Three copies of the same table is how two of them
end up disagreeing about which layer ``domain/simulation`` is on.

Layer numbers are floats rather than integers because §4.1 splits two of the
integer layers and the splits are load-bearing:

* ``contracts`` (0.0) and ``foundation`` (0.5) are both "layer 0" in the
  document, but foundation imports contracts, so they cannot be peers here;
* ``domain/simulation`` is 4b (4.5) and the other three domain packages are 4a
  (4.0), because §4.1 makes "the scenario builder must not price a position"
  structural by putting valuation *below* simulation rather than beside it.

The rule the numbers encode is **strictly less than**: a package may import a
package on a lower layer, never on its own. That is what makes the three 4a
peers unable to import each other without a separate hand-written rule, and it
matches the document's "May import 0-3" phrasing exactly.

One open question is recorded rather than silently resolved. §4.1's table gives
``features`` "May import 0-1", while the prose two paragraphs below says
"features may import inference but never training". Those disagree: ``models``
is layer 3, and it may import ``features``, so honouring the prose as a direct
import edge would create a 2 <-> 3 cycle. The table is what is enforced here —
it is strictly stronger and never contradicts "never training" — and the
question of how Tier-4 materialization reaches a frozen artifact is left to the
phase that writes it.

Stdlib only, and it imports nothing from ``engine`` — the checks that read it
run as a pre-commit hook in a bare clone where nothing else is installed.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "V2_ROOT",
    "LEGACY_ROOT",
    "Package",
    "PACKAGES",
    "CONTAINERS",
    "package_of",
    "layer_of",
    "README_SECTIONS",
]

V2_ROOT = "engine.v2"
LEGACY_ROOT = "engine"


@dataclass(frozen=True)
class Package:
    """One row of the §4.1 table."""

    #: Dotted module path, e.g. ``engine.v2.domain.simulation``.
    dotted: str
    #: Layer number. Lower may be imported by higher; equal may not.
    layer: float
    #: The §4.1 label, as written in the document ("0", "4a", "-").
    label: str
    #: Which row of the §4 owner table this package implements.
    owner: str
    #: Legacy modules it replaces, from §4.4.
    replaces: tuple[str, ...] = ()
    #: When set, the ONLY layers this package may import. §4.1 gives
    #: ``dashboard`` "7 only", which the strictly-less-than rule alone would
    #: not enforce.
    only_imports: tuple[float, ...] | None = None
    #: True for ``diagnosis``: it may read every layer, and no layer may
    #: import it. A comparator must never become a dependency of what it
    #: compares.
    sink: bool = False
    #: Orchestrators are exempt from the fan-out budget and deliberately not
    #: from length or complexity (§4.3).
    orchestrator: bool = False
    #: Responsibilities / non-responsibilities, from the §4 owner table. Held
    #: here so the README generator and the README check read one source.
    responsibilities: tuple[str, ...] = ()
    #: ``(what it must not do, the package that does it instead)``.
    non_responsibilities: tuple[tuple[str, str], ...] = ()

    @property
    def path(self) -> str:
        """Repo-relative directory, e.g. ``engine/v2/domain/simulation``."""
        return self.dotted.replace(".", "/")


#: Directories that exist only to hold subpackages. They carry an
#: ``__init__.py`` so the subpackages import, and nothing else: no layer, no
#: README, and nothing may import them directly.
CONTAINERS: tuple[str, ...] = (
    "engine.v2",
    "engine.v2.domain",
)


PACKAGES: tuple[Package, ...] = (
    Package(
        dotted="engine.v2.contracts",
        layer=0.0,
        label="0",
        owner="— schemas and types only, no logic, no I/O",
        replaces=("the dataclasses currently declared inside score.py",),
        responsibilities=(
            "Declare every named type in component_contracts.md, with its kind "
            "suffix (§2.5) and its schema version.",
            "Define the shared failure envelope (§2.4) and the reason-code "
            "vocabulary (§9.4).",
        ),
        non_responsibilities=(
            ("Compute anything", "every package above it"),
            ("Touch the filesystem, a clock or a network", "engine/v2/foundation"),
        ),
    ),
    Package(
        dotted="engine.v2.foundation",
        layer=0.5,
        label="0",
        owner="paths, env, canonical JSON, session/calendar arithmetic, causality primitives",
        replaces=("paths.py", "env.py", "jsonio.py", "audit.py",
                  "session arithmetic from calendar.py"),
        responsibilities=(
            "Canonical JSON (RFC 8785) and content hashing, per contracts §2.2.",
            "Path and environment resolution.",
            "Session arithmetic: BMO/AMC anchoring, trading-day offsets.",
            "Causality primitives — the cutoff comparison every feature respects.",
        ),
        non_responsibilities=(
            ("Fetch a calendar", "engine/v2/data"),
            ("Decide whether a difference is acceptable", "engine/v2/diagnosis"),
        ),
    ),
    Package(
        dotted="engine.v2.data",
        layer=1.0,
        label="1",
        owner="Ingestion — fetch receipts, normalization, coverage, finality, source revisions",
        replaces=("data/sources/", "data/normalize/", "data/pulls/", "store.py",
                  "fetch.py", "throttle.py", "finality.py", "rebuild.py",
                  "calendar sourcing from calendar.py"),
        responsibilities=(
            "Fetch receipts and raw object retention.",
            "Normalization into versioned datasets with declared contracts.",
            "Coverage watermarks, finality and source revisions.",
            "Atomic snapshot commit, per §5.5.",
        ),
        non_responsibilities=(
            ("Compute a trading verdict", "engine/v2/scoring"),
            ("Change a champion", "engine/v2/models/training"),
        ),
    ),
    Package(
        dotted="engine.v2.features",
        layer=2.0,
        label="2",
        owner="Feature engine — registered transforms and their causal dependencies",
        replaces=("features.py", "data/features/panel.py", "data/features/tier4.py"),
        responsibilities=(
            "Registered feature recipes, their units and their missing-value policy.",
            "Causal dependency declaration for every transform.",
            "Tier-4 columns materialized from frozen feature-model artifacts.",
        ),
        non_responsibilities=(
            ("Select an implicit latest dataset", "engine/v2/data"),
            ("Silently change a missing-value policy", "a new recipe version"),
            ("Import model training — a feature depends on a frozen artifact, "
             "never on the code that fits one", "engine/v2/models/training"),
        ),
    ),
    Package(
        dotted="engine.v2.models",
        layer=3.0,
        label="3",
        owner="Model inference — registry, artifact loading, inference adapters",
        replaces=("models/registry.py", "artifact loading and inference adapters"),
        responsibilities=(
            "The model registry and its champion resolution.",
            "Artifact loading with fingerprint verification.",
            "Inference adapters; residual and calibration state as frozen data.",
        ),
        non_responsibilities=(
            ("Fit anything", "engine/v2/models/training"),
            ("Reach into a training recipe", "engine/v2/models/training"),
        ),
    ),
    Package(
        dotted="engine.v2.registry",
        layer=3.0,
        label="3",
        owner="Strategy registry — StrategySpec and DeploymentSpec",
        replaces=("structure_registry.py", "the StrategySpec/DeploymentSpec store"),
        responsibilities=(
            "Versioned StrategySpec and DeploymentSpec registration (contracts §7).",
            "Validation status per structure: promoted, tracked, or disabled "
            "with its refusal code.",
        ),
        non_responsibilities=(
            ("Score a strategy", "engine/v2/scoring"),
            ("Mutate itself during a score request", "engine/v2/scoring"),
        ),
    ),
    Package(
        dotted="engine.v2.domain.generation",
        layer=4.0,
        label="4a",
        owner="Structure generator — template resolution, finite placement search, completeness receipts",
        replaces=("structures.py", "forecast_sizing.py", "fills.py"),
        responsibilities=(
            "Resolve a structure template against a listed strike ladder and "
            "expiry set.",
            "Finite placement search with a validity and completeness receipt.",
            "Forecast-sized geometry, recording the forecast even when the "
            "shape is pinned.",
        ),
        non_responsibilities=(
            ("Rank candidates by PnL", "engine/v2/domain/simulation"),
            ("Change strategy selection rules", "engine/v2/scoring"),
        ),
    ),
    Package(
        dotted="engine.v2.domain.scenarios",
        layer=4.0,
        label="4a",
        owner="Scenario builder — causal outcome populations, weights, mappings and RNG",
        replaces=("analogs.py", "ResidualPool from pnl_sim.py"),
        responsibilities=(
            "Historical and synthetic outcome populations, with their weights.",
            "The deterministic RNG policy, including the legacy seed recipe "
            "preserved through migration (contracts §2.2).",
            "Analog population membership and its recorded size.",
        ),
        non_responsibilities=(
            ("Choose contracts", "engine/v2/domain/generation"),
            ("Price a position", "engine/v2/domain/valuation"),
        ),
    ),
    Package(
        dotted="engine.v2.domain.valuation",
        layer=4.0,
        label="4a",
        owner="Position valuator — frozen-position revaluation under time and parameter shocks",
        replaces=("payoff.py", "black_scholes_put from pnl_sim.py"),
        responsibilities=(
            "Revalue a frozen position at a shocked spot, vol and time.",
            "Terminal payoff and modeled value at the planned exit, kept "
            "distinct (§6.4).",
        ),
        non_responsibilities=(
            ("Select a winning strategy", "engine/v2/scoring"),
            ("Call a model mark an executable fill", "engine/v2/evaluation"),
        ),
    ),
    Package(
        dotted="engine.v2.domain.simulation",
        layer=4.5,
        label="4b",
        owner="PnL simulator and accounting — cash flows and PnL distributions",
        replaces=("expected_pnl from pnl_sim.py",),
        responsibilities=(
            "Draw a PnL distribution from a scenario set and a valuation policy.",
            "Cash-flow accounting with explicit debit/credit conventions and "
            "multipliers.",
        ),
        non_responsibilities=(
            ("Select a winning strategy", "engine/v2/scoring"),
            ("Build its own scenarios", "engine/v2/domain/scenarios"),
        ),
    ),
    Package(
        dotted="engine.v2.scoring",
        layer=5.0,
        label="5",
        owner="Scoring application — forecasts, shape, pricing, gate/chooser decisions, diagnostics",
        replaces=("score.py split by the stages in §6.3", "entry_rules.py",
                  "replay.py", "trailing_cutoff from pnl_sim.py"),
        responsibilities=(
            "The §6.3 execution order, one module per stage.",
            "Gate and chooser decisions, including DYN-SV menu resolution.",
            "Financial diagnostics and a validated immutable ScoreRecord.",
        ),
        non_responsibilities=(
            ("Read future outcomes", "engine/v2/evaluation"),
            ("Mutate a strategy or model registry", "engine/v2/registry"),
            ("Fit a model during a score request", "engine/v2/models/training"),
        ),
        orchestrator=True,
    ),
    Package(
        dotted="engine.v2.evaluation",
        layer=6.0,
        label="6",
        owner="Evaluation/portfolio — realized outcomes, capital accounting, report generation",
        replaces=("evaluate.py", "report.py", "build_trades.py", "calibrate.py",
                  "recalibrate.py"),
        responsibilities=(
            "Realized outcomes against actual traded or quoted evidence.",
            "Capital accounting and report generation.",
        ),
        non_responsibilities=(
            ("Recreate the selection logic used to choose trades", "engine/v2/scoring"),
            ("Use fitted vendor marks as a realized PnL source",
             "engine/v2/ledger, from settlement evidence"),
        ),
    ),
    Package(
        dotted="engine.v2.ledger",
        layer=6.0,
        label="6",
        owner="Prediction and position ledger — append-only facts",
        replaces=("ledger.py", "ledger_settlement.py", "portfolio.py"),
        responsibilities=(
            "Append-only prediction commits and position lifecycle events "
            "(contracts §12).",
            "Cash and position reconciliation.",
        ),
        non_responsibilities=(
            ("Rewrite a committed record", "a correcting append"),
            ("Decide a trading verdict", "engine/v2/scoring"),
        ),
    ),
    Package(
        dotted="engine.v2.models.training",
        layer=6.0,
        label="6",
        owner="Model training — dataset and model recipes, folds, fitting, evidence, release candidates",
        replaces=("models/training/, rewritten above scoring rather than moved",),
        responsibilities=(
            "Dataset and model recipes, folds, fitting and residual construction.",
            "Evidence and release candidates; atomic promotion.",
        ),
        non_responsibilities=(
            ("Run inside a score request", "engine/v2/models"),
            ("Be imported by a feature or a scorer", "engine/v2/models"),
        ),
    ),
    Package(
        dotted="engine.v2.serving",
        layer=7.0,
        label="7",
        owner="API/projection layer — filter, paginate, authorize, serialize computed records",
        replaces=("the data half of dashboard/render.py", "dashboard/earnings_app.py"),
        responsibilities=(
            "Bounded, paginated reads over saved score records.",
            "The financial display values §6.4 moves out of rendering.",
            "Immutable release publication, one release per read.",
        ),
        non_responsibilities=(
            ("Fit a model", "engine/v2/models/training"),
            ("Simulate PnL", "engine/v2/domain/simulation"),
            ("Fetch vendor data in a GET request", "engine/v2/data"),
        ),
    ),
    Package(
        dotted="engine.v2.ops",
        layer=7.0,
        label="7",
        owner="Supervisor/catalog — transactions, leases, dependencies, capacity, retry history",
        replaces=("new supervisor and catalog",
                  "dashboard/nightly.py becomes a job graph",
                  "tools/bounded_run.py becomes an executor adapter"),
        responsibilities=(
            "Durable job submission, leases, retry history and dependencies.",
            "Resource admission and per-job CPU placement.",
            "The nightly job graph and its release boundary.",
        ),
        non_responsibilities=(
            ("Decide research conclusions", "engine/v2/evaluation"),
            ("Compute a score", "engine/v2/scoring"),
        ),
        orchestrator=True,
    ),
    Package(
        dotted="engine.v2.dashboard",
        layer=8.0,
        label="8",
        owner="UI — navigation, formatting, tables, charts, loading/error states",
        replaces=("the formatting half of dashboard/render.py", "dashboard/static/"),
        only_imports=(7.0,),
        responsibilities=(
            "Navigation, formatting, tables, charts, loading and error states.",
        ),
        non_responsibilities=(
            ("Compute gates, financial ratios or return estimates", "engine/v2/serving"),
            ("Do portfolio accounting", "engine/v2/evaluation"),
        ),
    ),
    Package(
        dotted="engine.v2.diagnosis",
        layer=7.5,
        label="—",
        owner="Validation/diagnosis — comparators, tolerance policies, stage plans, ComparisonReceipts",
        replaces=("dashboard/selfcheck.py", "the parity comparators"),
        sink=True,
        responsibilities=(
            "Comparators that are stage-localized and complete, not first-wins "
            "(contracts §15).",
            "Tolerance policies, declared per field and never global.",
            "ComparisonReceipts and their tiers.",
        ),
        non_responsibilities=(
            ("Decide whether a difference is acceptable", "a person, from the receipt"),
            ("Repair the data it found wrong", "the package that produced it"),
        ),
    ),
)


#: The §4.5 headings, in order. A missing one fails the README check.
README_SECTIONS: tuple[str, ...] = (
    "Ownership",
    "Responsibilities",
    "Non-responsibilities",
    "Public interface",
    "Consumers",
    "Usage",
    "Testing",
)

_BY_DOTTED = {pkg.dotted: pkg for pkg in PACKAGES}


def package_of(module: str) -> Package | None:
    """The package a dotted module belongs to, by longest declared prefix.

    Longest-prefix, not first match: ``engine.v2.models.training.gate`` belongs
    to ``engine.v2.models.training`` (layer 6), not to ``engine.v2.models``
    (layer 3), and getting that backwards would let training be imported by a
    scorer.
    """
    best: Package | None = None
    for dotted, pkg in _BY_DOTTED.items():
        if module == dotted or module.startswith(dotted + "."):
            if best is None or len(dotted) > len(best.dotted):
                best = pkg
    return best


def layer_of(module: str) -> float | None:
    pkg = package_of(module)
    return None if pkg is None else pkg.layer
