"""The entry-rule gate's trailing ``pnl_sim`` cutoff as a frozen artifact (P5-4).

Seven board strategies (legacy ``engine/entry_rules.py`` ``ENTRY_RULES``)
admit a trade only when its simulated expectation clears a bar:
``exp_pnl_sim >= pnl_cutoff``. Legacy recomputes the bar on every score
(``engine/pnl_sim.py`` ``trailing_cutoff``): the top ``quantile`` of the
stored ``exp_pnl_sim`` history over the ``window_months`` calendar months
before the event's month, or no bar when that window holds fewer than
``min_window`` values. The history file is unversioned, so no release could
pin the bar a score used. This record freezes it:

* the causal key ``(history_id, month)``: ``month`` is the first day of the
  event's month, which is the window's exclusive end (legacy
  ``Timestamp(as_of).to_period("M").to_timestamp()``);
* the recipe constants (window, quantile, minimum window);
* the bar itself, or ``None`` when legacy had none (a thin window or no
  history); ``None`` is served, never replaced by a default, so the rule
  stays undetermined exactly as legacy's does;
* its :class:`~engine.v2.models.lineage.Lineage` and a content hash over all
  of it (``sha256(serialize_frozen_state(cutoff)) == cutoff.content_hash``).

Layer 3: this wraps a computed bar. The window arithmetic lives in the
layer-6 builder (``engine/v2/models/training/trailing_cutoff.py``); scoring
only reads the bar (``engine/v2/scoring/native_entry_rule.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Mapping

from engine.v2.foundation.canonical import content_hash
from engine.v2.models.lineage import Lineage, lineage_from_document

__all__ = [
    "PNL_HISTORY_ID",
    "TRAILING_CUTOFF_ARTIFACT_V1",
    "TRAILING_MIN_WINDOW",
    "TRAILING_QUANTILE",
    "TRAILING_WINDOW_MONTHS",
    "TrailingCutoffArtifact",
    "TrailingCutoffError",
    "cutoff_month",
    "make_trailing_cutoff_artifact",
    "trailing_cutoff_from_document",
    "trailing_cutoff_key",
]

TRAILING_CUTOFF_ARTIFACT_V1 = "trailing_pnl_cutoff_artifact.v1.0"
#: The stored history the bar is computed over (legacy ``pnl_sim.HISTORY_PATH``).
PNL_HISTORY_ID = "features.pnl_sim_history"
#: engine/pnl_sim.py ``WINDOW_MONTHS``/``QUANTILE`` and ``trailing_cutoff``'s
#: ``min_window`` default (copied; the artifact test pins them equal).
TRAILING_WINDOW_MONTHS = 6
TRAILING_QUANTILE = 0.20
TRAILING_MIN_WINDOW = 100

TrailingCutoffKey = tuple[str, str]


class TrailingCutoffError(ValueError):
    """The trailing cutoff artifact is malformed or cannot be verified."""


def cutoff_month(as_of: Any) -> str:
    """The first day of ``as_of``'s month, ``YYYY-MM-01``."""
    text = str(as_of)[:10]
    if len(text) < 7 or text[4] != "-":
        raise TrailingCutoffError(f"unparsable as_of for a trailing cutoff: {as_of!r}")
    return text[:7] + "-01"


def trailing_cutoff_key(history_id: str, month: Any) -> TrailingCutoffKey:
    return (str(history_id), cutoff_month(month))


@dataclass(frozen=True, kw_only=True)
class TrailingCutoffArtifact:
    """The trailing top-quantile bar for one event month."""

    schema_version: str = TRAILING_CUTOFF_ARTIFACT_V1
    history_id: str
    month: str
    window_months: int
    quantile: float
    min_window: int
    cutoff: float | None
    lineage: Lineage
    content_hash: str

    def __str__(self) -> str:
        return f"{self.schema_version}:{self.content_hash}"

    @property
    def key(self) -> TrailingCutoffKey:
        return trailing_cutoff_key(self.history_id, self.month)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "history_id": self.history_id,
            "month": self.month,
            "window_months": self.window_months,
            "quantile": self.quantile,
            "min_window": self.min_window,
            "cutoff": self.cutoff,
            "lineage": self.lineage.document(),
        }


def _constants(window_months: Any, quantile: Any, min_window: Any) -> tuple[int, float, int]:
    constants = (int(window_months), float(quantile), int(min_window))
    if constants != (TRAILING_WINDOW_MONTHS, TRAILING_QUANTILE, TRAILING_MIN_WINDOW):
        raise TrailingCutoffError(
            "trailing cutoff constants differ from the legacy recipe "
            f"({TRAILING_WINDOW_MONTHS}, {TRAILING_QUANTILE}, {TRAILING_MIN_WINDOW})")
    return constants


def make_trailing_cutoff_artifact(
    *,
    month: Any,
    cutoff: Any,
    lineage: Lineage,
    history_id: str = PNL_HISTORY_ID,
    window_months: int = TRAILING_WINDOW_MONTHS,
    quantile: float = TRAILING_QUANTILE,
    min_window: int = TRAILING_MIN_WINDOW,
) -> TrailingCutoffArtifact:
    """Freeze one month's bar. A non-finite bar is refused: legacy's bar is a
    quantile of finite values or absent (``None``)."""
    window, share, minimum = _constants(window_months, quantile, min_window)
    value = None if cutoff is None else float(cutoff)
    if value is not None and not isfinite(value):
        raise TrailingCutoffError("a trailing cutoff must be finite or None")
    if not isinstance(lineage, Lineage) or not lineage.declared:
        raise TrailingCutoffError("a frozen trailing cutoff must declare its lineage")
    draft = TrailingCutoffArtifact(
        history_id=str(history_id), month=cutoff_month(month), window_months=window,
        quantile=share, min_window=minimum, cutoff=value, lineage=lineage.canonical(),
        content_hash="",
    )
    return replace(draft, content_hash=content_hash(draft.payload()))


def trailing_cutoff_from_document(document: Mapping[str, Any]) -> TrailingCutoffArtifact:
    """Rebuild from the JSON document (hash recomputed, never trusted)."""
    if document.get("schema_version") != TRAILING_CUTOFF_ARTIFACT_V1:
        raise TrailingCutoffError(
            "unsupported trailing cutoff schema_version: "
            f"{document.get('schema_version')!r}")
    month = document["month"]
    if cutoff_month(month) != month:
        raise TrailingCutoffError("trailing cutoff month must be a month start")
    return make_trailing_cutoff_artifact(
        month=month, cutoff=document.get("cutoff"),
        lineage=lineage_from_document(document.get("lineage")),
        history_id=document["history_id"], window_months=document["window_months"],
        quantile=document["quantile"], min_window=document["min_window"],
    )
