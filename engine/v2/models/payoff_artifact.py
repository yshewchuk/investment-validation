"""Versioned, content-hashed payoff-calibration artifacts (P5-4).

``engine/v2/scoring/native_payoff.py`` fits the payoff-calibration line
(``fit_payoff_line``, ``np.polyfit``) and the STR-RUNUP two-driver surface
(``fit_runup_payoff_surface``, ``np.linalg.lstsq``) inline, at scoring time,
from raw source rows -- the last inline fit standing after P5-2 rigged every
legacy fitting/cache-write path to fail. This module is the frozen artifact
TYPE that fit belongs in instead: an immutable, content-hashed record of the
line or surface, its capped residuals and its provenance, plus a verified
read-only loader.

Layering (``checks/layer_map.py``): this package is layer 3
(``engine.v2.models``), strictly below ``engine.v2.scoring`` (layer 5) and
``engine.v2.models.training`` (layer 6) -- so it must not import
``native_payoff``'s fitting math, and does not. The BUILDER that calls
``native_payoff.fit_payoff_line``/``fit_runup_payoff_surface`` and wraps the
result into one of the frozen types below lives at
``engine/v2/models/training/payoff.py`` (layer 6, "the layer allowed to
fit" -- see this package's own README non-responsibilities), which may
import both this module (3) and ``native_payoff`` (5) because both are
strictly lower than training's own layer (6). Scoring (5) may import this
module (3) to READ a verified artifact; it must never import training (6).

Keyed by ``(strategy, alpha, cutoff)`` -- the same identity legacy's own
``Scorer.payoff``/``.runup_payoff`` cache keys on (``engine/score.py:1311``,
``round(alpha, 4)``), so an artifact produced here can be looked up with
exactly the values a scoring request already carries (fill alpha, decision
evidence cutoff).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from engine.v2.foundation.canonical import (
    CONTENT_HASH_PREFIX,
    canonical_json,
    content_hash,
    untag_nonfinite,
)

__all__ = [
    "PAYOFF_LINE_ARTIFACT_V1",
    "PAYOFF_SURFACE_ARTIFACT_V1",
    "PayoffArtifactError",
    "PayoffArtifactKey",
    "PayoffArtifactRef",
    "PayoffLineArtifact",
    "PayoffSurfaceArtifact",
    "PayoffArtifactLoader",
    "payoff_artifact_key",
    "make_payoff_line_artifact",
    "make_payoff_surface_artifact",
    "serialize_payoff_artifact",
]

PAYOFF_LINE_ARTIFACT_V1 = "payoff_line_artifact.v1.0"
PAYOFF_SURFACE_ARTIFACT_V1 = "payoff_surface_artifact.v1.0"

#: ``(strategy, alpha rounded to 4dp, cutoff as an ISO date or None)``.
PayoffArtifactKey = tuple[str, float, "str | None"]


def _cutoff_str(cutoff: Any) -> str | None:
    return None if cutoff is None else str(cutoff)[:10]


def payoff_artifact_key(strategy: str, alpha: float, cutoff: Any) -> PayoffArtifactKey:
    """``(strategy, alpha, fold/cutoff)`` -- the identity one payoff artifact binds to.

    Mirrors legacy's own cache key (``engine/score.py:1311``,
    ``round(float(alpha), 4)``) so a request built from a slightly
    differently-typed ``alpha`` still resolves the same artifact. ``cutoff``
    normalizes to its ISO date, or ``None`` -- "fit through the end", the
    same meaning ``before=None`` has in :mod:`native_payoff`.
    """
    return (str(strategy), round(float(alpha), 4), _cutoff_str(cutoff))


def _residual_tuple(residuals: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in residuals)


def _payload_common(
    *, strategy: str, alpha: float, cutoff: Any, window: tuple[Any, Any] | None,
    n: int, resid_sd: float, r: float | None, residuals: Sequence[float],
) -> dict[str, Any]:
    window_start, window_end = window if window is not None else (None, None)
    return {
        "strategy": str(strategy),
        "alpha": round(float(alpha), 4),
        "cutoff": _cutoff_str(cutoff),
        "window_start": _cutoff_str(window_start),
        "window_end": _cutoff_str(window_end),
        "n": int(n),
        "resid_sd": float(resid_sd),
        "r": None if r is None else float(r),
        "residuals": list(_residual_tuple(residuals)),
    }


@dataclass(frozen=True, kw_only=True)
class PayoffLineArtifact:
    """``exit_value / spot ~= intercept + slope * driver``, frozen (P5-4).

    Same math ``native_payoff.fit_payoff_line`` returns; this is that
    result plus provenance and a content hash, immutable and safe to cache
    across requests.
    """

    schema_version: str = PAYOFF_LINE_ARTIFACT_V1
    strategy: str
    driver: str
    alpha: float
    cutoff: str | None
    window_start: str | None
    window_end: str | None
    n: int
    intercept: float
    slope: float
    resid_sd: float
    r: float | None
    residuals: tuple[float, ...]
    content_hash: str

    @property
    def key(self) -> PayoffArtifactKey:
        return payoff_artifact_key(self.strategy, self.alpha, self.cutoff)

    def _payload(self) -> dict[str, Any]:
        payload = _payload_common(
            strategy=self.strategy, alpha=self.alpha, cutoff=self.cutoff,
            window=(self.window_start, self.window_end), n=self.n,
            resid_sd=self.resid_sd, r=self.r, residuals=self.residuals,
        )
        payload.update({
            "schema_version": self.schema_version,
            "driver": self.driver,
            "intercept": float(self.intercept),
            "slope": float(self.slope),
        })
        return payload


@dataclass(frozen=True, kw_only=True)
class PayoffSurfaceArtifact:
    """STR-RUNUP's two-driver exit-value surface, frozen (P5-4).

    Same math ``native_payoff.fit_runup_payoff_surface`` returns (the
    ``runup_payoff_design`` least-squares coefficients), plus provenance and
    a content hash.
    """

    schema_version: str = PAYOFF_SURFACE_ARTIFACT_V1
    #: Kept explicit (today always "STR-RUNUP") rather than hard-coded, so a
    #: second surface-calibrated strategy is a data change, not a code one.
    strategy: str
    alpha: float
    cutoff: str | None
    window_start: str | None
    window_end: str | None
    n: int
    coefficients: tuple[float, ...]
    resid_sd: float
    r: float | None
    residuals: tuple[float, ...]
    content_hash: str

    @property
    def key(self) -> PayoffArtifactKey:
        return payoff_artifact_key(self.strategy, self.alpha, self.cutoff)

    def _payload(self) -> dict[str, Any]:
        payload = _payload_common(
            strategy=self.strategy, alpha=self.alpha, cutoff=self.cutoff,
            window=(self.window_start, self.window_end), n=self.n,
            resid_sd=self.resid_sd, r=self.r, residuals=self.residuals,
        )
        payload.update({
            "schema_version": self.schema_version,
            "coefficients": [float(value) for value in self.coefficients],
        })
        return payload


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return content_hash(payload)


def serialize_payoff_artifact(artifact: "PayoffLineArtifact | PayoffSurfaceArtifact") -> bytes:
    """The canonical bytes one artifact should be written to disk as.

    Writing exactly these bytes is what makes :class:`PayoffArtifactLoader`'s
    two checks -- the raw-byte hash against ``PayoffArtifactRef.content_hash``
    and the reconstructed artifact's own ``content_hash`` -- agree by
    construction: ``content_hash`` is defined as sha256 over the canonical
    JSON of the payload (``engine.v2.foundation.canonical.content_hash``), so
    hashing these exact bytes reproduces it. A file written with a plain
    ``json.dumps`` (different key order, different float formatting) would
    still load -- ``json.loads`` does not care about byte-level form -- but
    its raw-byte hash would not equal the artifact's ``content_hash``, so a
    caller minting its own ``PayoffArtifactRef`` should use ``content_hash``
    as the ref and these bytes as the file, not re-derive either separately.
    """
    return canonical_json(artifact._payload()).encode("utf-8")


def make_payoff_line_artifact(
    fit: Mapping[str, Any],
    *,
    strategy: str,
    driver: str,
    alpha: float,
    cutoff: Any = None,
    window: tuple[Any, Any] | None = None,
) -> PayoffLineArtifact:
    """Wrap ``native_payoff.fit_payoff_line``'s return value as a frozen artifact.

    Pure: takes the already-fitted line and adds provenance/hash, so calling
    it does not fit anything and does not need the no-fit guard. Callers that
    DO fit (``engine/v2/models/training/payoff.py``) are the only intended
    callers besides tests.
    """
    payload = _payload_common(
        strategy=strategy, alpha=alpha, cutoff=cutoff, window=window,
        n=fit["n"], resid_sd=fit["resid_sd"], r=fit.get("r"),
        residuals=fit["residuals"],
    )
    payload.update({
        "schema_version": PAYOFF_LINE_ARTIFACT_V1,
        "driver": str(driver),
        "intercept": float(fit["intercept"]),
        "slope": float(fit["slope"]),
    })
    return PayoffLineArtifact(
        strategy=payload["strategy"], driver=payload["driver"],
        alpha=payload["alpha"], cutoff=payload["cutoff"],
        window_start=payload["window_start"], window_end=payload["window_end"],
        n=payload["n"], intercept=payload["intercept"], slope=payload["slope"],
        resid_sd=payload["resid_sd"], r=payload["r"],
        residuals=_residual_tuple(payload["residuals"]),
        content_hash=_hash_payload(payload),
    )


def make_payoff_surface_artifact(
    fit: Mapping[str, Any],
    *,
    strategy: str = "STR-RUNUP",
    alpha: float,
    cutoff: Any = None,
    window: tuple[Any, Any] | None = None,
) -> PayoffSurfaceArtifact:
    """Wrap ``native_payoff.fit_runup_payoff_surface``'s result as a frozen artifact."""
    payload = _payload_common(
        strategy=strategy, alpha=alpha, cutoff=cutoff, window=window,
        n=fit["n"], resid_sd=fit["resid_sd"], r=fit.get("r"),
        residuals=fit["residuals"],
    )
    payload.update({
        "schema_version": PAYOFF_SURFACE_ARTIFACT_V1,
        "coefficients": [float(value) for value in fit["coefficients"]],
    })
    return PayoffSurfaceArtifact(
        strategy=payload["strategy"], alpha=payload["alpha"], cutoff=payload["cutoff"],
        window_start=payload["window_start"], window_end=payload["window_end"],
        n=payload["n"], coefficients=tuple(payload["coefficients"]),
        resid_sd=payload["resid_sd"], r=payload["r"],
        residuals=_residual_tuple(payload["residuals"]),
        content_hash=_hash_payload(payload),
    )


class PayoffArtifactError(ValueError):
    """A payoff artifact could not be verified or loaded."""


@dataclass(frozen=True, kw_only=True)
class PayoffArtifactRef:
    """Where one payoff artifact's serialized JSON lives, and what it must hash to."""

    path: str
    content_hash: str
    schema_version: str = "payoff_artifact_ref.v1.0"


def _artifact_from_document(document: Mapping[str, Any]) -> PayoffLineArtifact | PayoffSurfaceArtifact:
    # canonical_json tags a non-finite float (a degenerate resid_sd from a
    # tiny/perfect fit) as {"__nonfinite__": ...} rather than emitting a bare
    # NaN/Infinity (RFC 8785 has no such literal); undo that right after
    # json.loads so every downstream field is a real float again.
    document = untag_nonfinite(document)
    schema = document.get("schema_version")
    if schema == PAYOFF_LINE_ARTIFACT_V1:
        payload = _payload_common(
            strategy=document["strategy"], alpha=document["alpha"],
            cutoff=document.get("cutoff"),
            window=(document.get("window_start"), document.get("window_end")),
            n=document["n"], resid_sd=document["resid_sd"], r=document.get("r"),
            residuals=document["residuals"],
        )
        payload.update({
            "schema_version": schema,
            "driver": document["driver"],
            "intercept": float(document["intercept"]),
            "slope": float(document["slope"]),
        })
        digest = _hash_payload(payload)
        return PayoffLineArtifact(
            strategy=payload["strategy"], driver=payload["driver"],
            alpha=payload["alpha"], cutoff=payload["cutoff"],
            window_start=payload["window_start"], window_end=payload["window_end"],
            n=payload["n"], intercept=payload["intercept"], slope=payload["slope"],
            resid_sd=payload["resid_sd"], r=payload["r"],
            residuals=_residual_tuple(payload["residuals"]), content_hash=digest,
        )
    if schema == PAYOFF_SURFACE_ARTIFACT_V1:
        payload = _payload_common(
            strategy=document["strategy"], alpha=document["alpha"],
            cutoff=document.get("cutoff"),
            window=(document.get("window_start"), document.get("window_end")),
            n=document["n"], resid_sd=document["resid_sd"], r=document.get("r"),
            residuals=document["residuals"],
        )
        payload.update({
            "schema_version": schema,
            "coefficients": [float(value) for value in document["coefficients"]],
        })
        digest = _hash_payload(payload)
        return PayoffSurfaceArtifact(
            strategy=payload["strategy"], alpha=payload["alpha"], cutoff=payload["cutoff"],
            window_start=payload["window_start"], window_end=payload["window_end"],
            n=payload["n"], coefficients=tuple(payload["coefficients"]),
            resid_sd=payload["resid_sd"], r=payload["r"],
            residuals=_residual_tuple(payload["residuals"]), content_hash=digest,
        )
    raise PayoffArtifactError(f"unsupported payoff artifact schema_version: {schema!r}")


class PayoffArtifactLoader:
    """Verified, read-only loading of payoff-calibration artifacts from disk.

    Mirrors ``engine.v2.models.loader.FrozenInference``'s verification shape:
    every load re-hashes the raw bytes against the reference before trusting
    them, a path is refused if it would resolve outside the artifact root,
    and results are cached by content hash so a warm read never re-parses.
    """

    def __init__(self, artifact_root: Path | str) -> None:
        self._root = Path(artifact_root).resolve()
        self._cache: dict[str, PayoffLineArtifact | PayoffSurfaceArtifact] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def load(self, ref: PayoffArtifactRef) -> PayoffLineArtifact | PayoffSurfaceArtifact:
        cached = self._cache.get(ref.content_hash)
        if cached is not None:
            return cached
        path = (self._root / ref.path).resolve()
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise PayoffArtifactError("artifact path escapes artifact root") from exc
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise PayoffArtifactError(f"missing payoff artifact: {ref.path}") from exc
        digest = CONTENT_HASH_PREFIX + hashlib.sha256(raw).hexdigest()
        if digest != ref.content_hash:
            raise PayoffArtifactError(f"payoff artifact hash mismatch: {ref.path}")
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PayoffArtifactError("payoff artifact is not valid JSON") from exc
        if not isinstance(document, Mapping):
            raise PayoffArtifactError("payoff artifact document must be a JSON object")
        artifact = _artifact_from_document(document)
        if artifact.content_hash != ref.content_hash:
            # Defense in depth: the bytes matched ``ref`` above, but a writer
            # bug (rounding, a field the payload builder forgot) could still
            # produce a document whose OWN recomputed identity disagrees with
            # what it was published under.
            raise PayoffArtifactError(
                f"payoff artifact content disagrees with its own hash: {ref.path}"
            )
        self._cache[ref.content_hash] = artifact
        return artifact
