"""The model registry: one manifest, one champion per role, no silent drift.

``registry.json`` is a list of entries describing every model this program has
trained. An entry records what the model is, what it eats, when it was trained,
what it scored, whether it is the current champion, and which experiment
justified promoting it.

Four integrity rules are enforced at load, each of them a failure that has
happened to somebody:

1. **Exactly one champion per (strategy, role).** Two champions is not a
   tie-break to resolve at call time; it is an unresolved promotion.
2. **The artifact hash matches.** A model retrained in place while the manifest
   still quotes its old metrics is the most expensive kind of stale: every
   report downstream cites numbers the file cannot produce. The manifest stores
   the artifact's sha256 and the loader refuses a mismatch.
3. **The feature list matches the artifact.** A model whose stored feature order
   differs from the manifest's would be scored on permuted inputs and would
   return plausible nonsense.
4. **Every feature is live-servable.** A model trained on a feature that only
   exists for realized events backtests perfectly and has nothing to read on the
   morning it matters — see :data:`engine.features.LIVE_UNAVAILABLE`.

Registry edits are Phase 2's ``promote.py`` job. Phase 1 writes it once, at
training time, through :func:`register`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from engine import paths

__all__ = [
    "ROLES",
    "MODEL_TIERS",
    "ROLE_TIER",
    "TIER4_COLUMNS",
    "RegistryError",
    "ModelArtifact",
    "RegistryEntry",
    "Registry",
    "load_registry",
    "champion",
    "register",
    "REGISTRY_PATH",
    "ARTIFACT_DIR",
]

#: What a model is *for*. The scoring engine asks for a role, never a filename.
#:
#: ``size``        predicted |earnings move|, in percent of spot
#: ``implied_t1``  predicted quoted implied move at the last pre-print close
#: ``gate``        predicted per-trade return at mid fills — the selection signal
ROLES = ("size", "implied_t1", "gate")

#: Strategy scope. ``"*"`` means the model is strategy-agnostic (the size model
#: predicts a property of the *event*, not of any structure traded around it).
ANY_STRATEGY = "*"

#: Where a model sits in the dependency graph.
#:
#: ``feature``   its output can be materialised as a Tier-4 column and read by
#:               other models as an input
#: ``decision``  it consumes features to make a call, and nothing reads it back
#:
#: The registry is the only place that can answer "what breaks if I re-promote
#: the size model", and it could not answer it while it recorded *what* each
#: model eats but never *which models feed which*.
MODEL_TIERS = ("feature", "decision")

#: The tier a role occupies by default. Not a guess — it is what the roles
#: already mean, so an entry written before these fields existed lands on the
#: right value without anyone editing it.
ROLE_TIER = {"size": "feature", "implied_t1": "feature", "gate": "decision"}

#: The Tier-4 vocabulary: the columns a feature model may declare it produces
#: and a decision model may declare it consumes. It lives here rather than in
#: ``engine.data.features.tier4`` so the registry can validate the dependency
#: graph without importing the layer it describes; ``tests/test_tier4.py`` holds
#: the two in agreement.
TIER4_COLUMNS = (
    "pred_abs_move",
    "pred_abs_move_p10",
    "pred_abs_move_p90",
    "pred_abs_move_sd",
)

REGISTRY_PATH = paths.ENGINE / "models" / "registry.json"

#: Artifacts are data, not code: regenerable at a fixed seed from
#: ``engine/models/training/``, too large for the repo, and blocked from it by
#: both the allowlist ``.gitignore`` and the hygiene check.
ARTIFACT_DIR = paths.DATA / "models"


class RegistryError(RuntimeError):
    """The registry, or an artifact it points at, failed an integrity check."""


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------


@dataclass
class ModelArtifact:
    """A trained model plus everything needed to use and audit it.

    ``residuals`` is the point of this class. The scoring engine does not push a
    point prediction through a payoff and call the result an expectation — it
    pushes a *distribution*, and the distribution comes from the model's own
    held-out errors rather than from an assumed normal. Earnings-move residuals
    are right-skewed and fat-tailed; a normal approximation would understate
    both the upside of a long-vol structure and the tail that matters for a
    short leg. Storing the empirical residuals with the model is what makes the
    honest version possible at scoring time.
    """

    model: Any
    role: str
    features: tuple[str, ...]
    residuals: np.ndarray
    target: str
    train_years: tuple[int, ...] = ()
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    seed: int = 0
    created: str = ""
    notes: str = ""
    #: Optional per-bucket residual pools, keyed by where the PREDICTION falls.
    #: ``{"edges": ndarray, "pools": [ndarray, ...], "min_pool": int,
    #: "kind": "prediction_decile"}``. When absent — which is every artifact
    #: saved before EXP-115 — ``residual_draws`` uses the flat pool and behaves
    #: exactly as it always has.
    #:
    #: The flat pool stays authoritative and is never replaced: a bucket thinner
    #: than ``min_pool`` falls back to it, so conditioning can only ever refine
    #: the estimate, never leave a sparse region of the prediction range with a
    #: pool too thin to be a distribution.
    residual_buckets: dict | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; known: {ROLES}")
        self.features = tuple(self.features)
        self.residuals = np.asarray(self.residuals, dtype=float)
        self.residuals = self.residuals[np.isfinite(self.residuals)]
        if self.residual_buckets:
            b = dict(self.residual_buckets)
            b["edges"] = np.asarray(b["edges"], dtype=float)
            b["pools"] = [np.asarray(pool, dtype=float) for pool in b["pools"]]
            if len(b["pools"]) != len(b["edges"]) - 1:
                raise ValueError(
                    f"{self.role}: {len(b['pools'])} pools for "
                    f"{len(b['edges'])} edges; expected len(edges) - 1"
                )
            self.residual_buckets = b
        if not self.created:
            self.created = date.today().isoformat()

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float).ravel()

    def residual_pool(self, prediction: float | None = None) -> tuple[np.ndarray, str]:
        """The residual pool to draw from, and a label saying which one it is.

        Returns the flat pool unless this artifact carries buckets AND a
        prediction was supplied AND the matching bucket is thick enough. The
        label is returned rather than inferred by the caller because "which pool
        did this interval come from" is the first question to ask of a width
        that looks wrong, and a silent fallback is indistinguishable from a
        conditioning that did nothing.
        """
        if self.residuals.size == 0:
            raise RegistryError(f"{self.role}: artifact carries no residuals")
        # getattr, not attribute access: joblib pickles the instance dict, so an
        # artifact saved before this field existed unpickles WITHOUT it and
        # __post_init__ does not run to supply the default.
        buckets = getattr(self, "residual_buckets", None)
        if not buckets or prediction is None or not np.isfinite(prediction):
            return self.residuals, "flat"
        edges = buckets["edges"]
        index = int(np.clip(np.searchsorted(edges, prediction, side="right") - 1,
                            0, len(buckets["pools"]) - 1))
        pool = buckets["pools"][index]
        if pool.size < int(buckets.get("min_pool", 0)):
            return self.residuals, f"flat (bucket {index} thin: {pool.size})"
        return pool, f"bucket {index}"

    def residual_draws(self, n: int, rng: np.random.Generator,
                       prediction: float | None = None) -> np.ndarray:
        """``n`` bootstrap draws from the held-out residual distribution.

        ``prediction`` is optional and ignored unless the artifact carries
        buckets, so every existing caller keeps its exact behaviour.
        """
        pool, _ = self.residual_pool(prediction)
        return rng.choice(pool, size=n, replace=True)

    def save(self, path: Path) -> str:
        import joblib

        path = paths.assert_writable(Path(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return artifact_sha256(path)


def bucket_residuals(
    predictions, residuals, *, deciles: int = 10, min_pool: int = 250
) -> dict | None:
    """Group held-out residuals by the decile of the prediction they came from.

    The pairing is the whole point and it is only available here, at training
    time: ``residuals`` and the predictions that produced them come out of the
    same walk-forward frame. By the time an artifact is loaded for scoring the
    predictions are gone, which is why the incumbent could only ever offer one
    pool for every event.

    Edges come from the training predictions alone, so a bucket is a statement
    about where a prediction sits in the distribution the model was fitted on —
    not about the row being scored. The outer edges are opened to +/-inf so a
    live prediction beyond anything seen in training still lands somewhere
    rather than falling off the end.

    Returns ``None`` when the sample cannot support the split, so the caller
    stores nothing and the artifact keeps its flat-pool behaviour.
    """
    pred = np.asarray(predictions, dtype=float)
    res = np.asarray(residuals, dtype=float)
    if pred.shape != res.shape:
        raise ValueError(f"predictions {pred.shape} and residuals {res.shape} differ")
    ok = np.isfinite(pred) & np.isfinite(res)
    pred, res = pred[ok], res[ok]
    if pred.size < deciles * min_pool:
        return None

    edges = np.unique(np.quantile(pred, np.linspace(0, 1, deciles + 1)))
    if edges.size < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    index = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, edges.size - 2)
    pools = [res[index == i] for i in range(edges.size - 1)]
    return {
        "kind": "prediction_decile",
        "edges": edges,
        "pools": pools,
        "min_pool": int(min_pool),
        "n": int(pred.size),
        "thin": [i for i, pool in enumerate(pools) if pool.size < min_pool],
    }


def artifact_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_artifact(path: Path) -> ModelArtifact:
    import joblib

    obj = joblib.load(path)
    if not isinstance(obj, ModelArtifact):
        raise RegistryError(f"{path} does not hold a ModelArtifact ({type(obj).__name__})")
    return obj


# --------------------------------------------------------------------------
# entries
# --------------------------------------------------------------------------


@dataclass
class RegistryEntry:
    id: str
    role: str
    strategy: str
    artifact: str
    artifact_sha256: str
    features: list[str]
    target: str
    train_window: str
    train_years: list[int] = field(default_factory=list)
    eval: dict = field(default_factory=dict)
    champion: bool = False
    promoted: str = ""
    evidence: str = ""
    seed: int = 0
    notes: str = ""
    #: Selection threshold, for gate models: the score at or above which a
    #: candidate passes. Stored with the model because it is part of the
    #: decision rule, not a dashboard preference.
    threshold: float | None = None
    #: Position in the dependency graph — see :data:`MODEL_TIERS`. Left empty,
    #: it is filled from :data:`ROLE_TIER`.
    tier: str = ""
    #: The Tier-4 column this model materialises, for feature models. ``None``
    #: means the model is a feature model whose output is not (yet) written to
    #: Tier 4 — trained and evaluated, but nothing downstream reads it as data.
    produces: str | None = None
    #: Tier-4 columns this model reads, for decision models. This is the half of
    #: the graph that says which forecasts a promotion would disturb.
    consumes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise RegistryError(f"{self.id}: unknown role {self.role!r}")
        if not self.features:
            raise RegistryError(f"{self.id}: empty feature list")
        self.tier = self.tier or ROLE_TIER.get(self.role, "decision")
        if self.tier not in MODEL_TIERS:
            raise RegistryError(f"{self.id}: unknown tier {self.tier!r}; known: {MODEL_TIERS}")
        if self.tier == "feature":
            if self.consumes:
                raise RegistryError(
                    f"{self.id}: a feature model may not declare `consumes` — Tier 4 is "
                    "one layer deep on purpose, and a forecast built from another "
                    "forecast would need a fold order this build cannot express"
                )
            if self.produces is not None and self.produces not in TIER4_COLUMNS:
                raise RegistryError(
                    f"{self.id}: produces {self.produces!r}, which is not a Tier-4 "
                    f"column; known: {TIER4_COLUMNS}"
                )
        else:
            if self.produces is not None:
                raise RegistryError(
                    f"{self.id}: a decision model produces no Tier-4 column "
                    f"(got {self.produces!r})"
                )
            unknown = sorted(set(self.consumes) - set(TIER4_COLUMNS))
            if unknown:
                raise RegistryError(
                    f"{self.id}: consumes {unknown}, which are not Tier-4 columns; "
                    f"known: {TIER4_COLUMNS}"
                )

    @property
    def key(self) -> tuple[str, str]:
        return (self.strategy, self.role)

    @property
    def path(self) -> Path:
        p = Path(self.artifact)
        return p if p.is_absolute() else paths.ROOT / p

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------


@dataclass
class Registry:
    entries: list[RegistryEntry]
    path: Path = REGISTRY_PATH

    def __post_init__(self) -> None:
        self._validate_uniqueness()

    def _validate_uniqueness(self) -> None:
        ids = [e.id for e in self.entries]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise RegistryError(f"duplicate registry id(s): {dupes}")

        champions: dict[tuple[str, str], list[str]] = {}
        for entry in self.entries:
            if entry.champion:
                champions.setdefault(entry.key, []).append(entry.id)
        contested = {k: v for k, v in champions.items() if len(v) > 1}
        if contested:
            detail = "; ".join(f"{k}: {v}" for k, v in sorted(contested.items()))
            raise RegistryError(f"more than one champion for a (strategy, role) — {detail}")

    # -- lookup ------------------------------------------------------------

    def get(self, model_id: str) -> RegistryEntry:
        for entry in self.entries:
            if entry.id == model_id:
                return entry
        raise RegistryError(f"no registry entry {model_id!r}")

    # -- the dependency graph ---------------------------------------------

    def producers(self, column: str, *, champions_only: bool = True) -> list[RegistryEntry]:
        """Entries that materialise ``column`` into Tier 4."""
        return [
            e
            for e in self.entries
            if e.produces == column and (e.champion or not champions_only)
        ]

    def consumers(self, column: str, *, champions_only: bool = True) -> list[RegistryEntry]:
        """Entries that read ``column`` out of Tier 4.

        The answer to "what breaks if I re-promote the size model": every one of
        these was fit against forecasts a promotion would replace.
        """
        return [
            e
            for e in self.entries
            if column in e.consumes and (e.champion or not champions_only)
        ]

    def tier4_graph(self) -> dict[str, dict[str, list[str]]]:
        """``{column: {"produced_by": [...], "consumed_by": [...]}}`` for champions."""
        return {
            column: {
                "produced_by": [e.id for e in self.producers(column)],
                "consumed_by": [e.id for e in self.consumers(column)],
            }
            for column in TIER4_COLUMNS
        }

    def champion(self, role: str, strategy: str = ANY_STRATEGY) -> RegistryEntry:
        """The champion for a role, preferring a strategy-specific entry.

        A strategy-specific champion beats the wildcard, so a per-strategy gate
        can be promoted later without touching the shared one.
        """
        if role not in ROLES:
            raise RegistryError(f"unknown role {role!r}; known: {ROLES}")
        specific = [e for e in self.entries if e.champion and e.key == (strategy, role)]
        if specific:
            return specific[0]
        wildcard = [e for e in self.entries if e.champion and e.key == (ANY_STRATEGY, role)]
        if wildcard:
            return wildcard[0]
        raise RegistryError(
            f"no champion for role {role!r} (strategy {strategy!r}) — "
            f"train one with `python3 -m engine.models.training.train_all`"
        )

    def has_champion(self, role: str, strategy: str = ANY_STRATEGY) -> bool:
        try:
            self.champion(role, strategy)
            return True
        except RegistryError:
            return False

    # -- artifact loading --------------------------------------------------

    def load(self, entry: RegistryEntry | str, *, verify: bool = True) -> ModelArtifact:
        """Load an entry's artifact, refusing anything the manifest misdescribes."""
        if isinstance(entry, str):
            entry = self.get(entry)
        path = entry.path
        if not path.exists():
            raise RegistryError(
                f"{entry.id}: artifact {path} is missing. Artifacts are not "
                f"committed; rebuild with `python3 -m engine.models.training.train_all`"
            )
        if verify:
            actual = artifact_sha256(path)
            if actual != entry.artifact_sha256:
                raise RegistryError(
                    f"{entry.id}: artifact hash mismatch — the file at {path} is not "
                    f"the model the registry describes (recorded {entry.artifact_sha256[:12]}…, "
                    f"found {actual[:12]}…). Its recorded metrics {entry.eval} do not "
                    f"describe it. Retrain and re-register rather than editing the hash."
                )
        artifact = load_artifact(path)
        if verify and list(artifact.features) != list(entry.features):
            raise RegistryError(
                f"{entry.id}: feature list disagrees with the artifact.\n"
                f"  registry: {list(entry.features)}\n"
                f"  artifact: {list(artifact.features)}\n"
                "Scoring on a permuted feature vector produces plausible nonsense."
            )
        if verify and artifact.role != entry.role:
            raise RegistryError(
                f"{entry.id}: artifact role {artifact.role!r} != registry role {entry.role!r}"
            )
        return artifact

    def load_champion(
        self, role: str, strategy: str = ANY_STRATEGY, *, verify: bool = True
    ) -> tuple[RegistryEntry, ModelArtifact]:
        entry = self.champion(role, strategy)
        return entry, self.load(entry, verify=verify)

    # -- validation --------------------------------------------------------

    def validate(self, *, check_artifacts: bool = True) -> list[str]:
        """Full integrity sweep. Returns a list of problems; empty means clean."""
        from engine.features import LIVE_UNAVAILABLE, QUARANTINED_FEATURES

        problems: list[str] = []
        for entry in self.entries:
            unservable = sorted(set(entry.features) & set(LIVE_UNAVAILABLE))
            if unservable:
                problems.append(
                    f"{entry.id}: features {unservable} do not exist for an upcoming "
                    "event — the model could never be served live"
                )
            # A feature that CAN be served and is WRONG. `or_exern_z252` leaks
            # the future on 507 stored panel rows; no entry has ever listed it,
            # and this is what keeps that true.
            # TODO(2026-Q4): drop with the column.
            quarantined = sorted(set(entry.features) & set(QUARANTINED_FEATURES))
            if quarantined:
                problems.append(
                    f"{entry.id}: features {quarantined} are quarantined — their "
                    "stored values are known to be computed wrong"
                )
            if entry.role == "gate" and entry.champion and entry.threshold is None:
                problems.append(f"{entry.id}: champion gate carries no threshold")

        # A Tier-4 column with two champion producers has no defined value, and
        # a consumer of a column nobody builds would read NULL forever.
        for column in TIER4_COLUMNS:
            producers = self.producers(column)
            if len(producers) > 1:
                problems.append(
                    f"Tier-4 column {column!r} is produced by more than one champion: "
                    f"{[e.id for e in producers]}"
                )
            consumers = self.consumers(column)
            if consumers and not producers:
                problems.append(
                    f"Tier-4 column {column!r} is consumed by {[e.id for e in consumers]} "
                    "but no champion produces it"
                )

        if not check_artifacts:
            return problems
        for entry in self.entries:
            try:
                self.load(entry)
            except RegistryError as exc:
                problems.append(str(exc))
        return problems

    # -- persistence -------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": 1,
                "doc": (
                    "Model manifest. Champions are looked up by (strategy, role); "
                    "nothing imports an artifact path directly. Edited only by "
                    "training/registration code, never by hand."
                ),
                "models": [e.as_dict() for e in sorted(self.entries, key=lambda e: e.id)],
            },
            indent=1,
            sort_keys=False,
        )

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n")
        return path


def load_registry(path: Path | None = None, *, missing_ok: bool = True) -> Registry:
    path = Path(path or REGISTRY_PATH)
    if not path.exists():
        if missing_ok:
            return Registry(entries=[], path=path)
        raise RegistryError(f"no registry at {path}")
    doc = json.loads(path.read_text())
    entries = [RegistryEntry(**row) for row in doc.get("models", [])]
    return Registry(entries=entries, path=path)


def champion(role: str, strategy: str = ANY_STRATEGY) -> tuple[RegistryEntry, ModelArtifact]:
    """Convenience: load the current champion for a role."""
    return load_registry().load_champion(role, strategy)


def register(
    entry: RegistryEntry,
    *,
    path: Path | None = None,
    demote_others: bool = True,
) -> Registry:
    """Add or replace an entry, keeping the one-champion-per-role invariant.

    ``demote_others`` is what makes promotion atomic: registering a champion
    demotes the incumbent for that (strategy, role) in the same write, so the
    registry never passes through a two-champion state on disk.
    """
    registry = load_registry(path)
    kept = [e for e in registry.entries if e.id != entry.id]
    if entry.champion and demote_others:
        for other in kept:
            if other.key == entry.key and other.champion:
                other.champion = False
    kept.append(entry)
    updated = Registry(entries=kept, path=Path(path or REGISTRY_PATH))
    updated.save()
    return updated
