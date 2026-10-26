"""The forward forecast assembly's slice-A source (see ARCHITECTURE.md)."""
from __future__ import annotations

from dataclasses import dataclass

from engine.v2.models.contracts import ModelRelease
from engine.v2.models.loader import FrozenInference


@dataclass(frozen=True, kw_only=True)
class GateForecastSource:
    """One causally selected size fold as the gate's forecast source.

    Carries the fold's ``binding_id``, its own fold-view ``release``, the
    shared ``inference`` and the held-out ``pool_pred``/``pool_res``/
    ``interval_floor`` behind the gate's ``pred_abs_move`` family -- the two
    declarations ``gate_recipe.forecast`` and ``gate_forecast_pool`` (which
    it conflicts with) folded into one source. See ARCHITECTURE.md
    ("Forward forecast assembly").
    """

    binding_id: str
    artifact_hash: str
    release: ModelRelease
    inference: FrozenInference
    pool_pred: tuple[float, ...]
    pool_res: tuple[float, ...]
    interval_floor: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pool_pred", tuple(self.pool_pred))
        object.__setattr__(self, "pool_res", tuple(self.pool_res))
