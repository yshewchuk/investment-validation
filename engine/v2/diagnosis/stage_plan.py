"""Stage plans — what "the first differing stage" is measured against.

A finding that says "row 47 is red" is not a diagnosis. A finding that says
"`serialization`: `structure_params.width_moneyness` differs at the 7th
significant figure" is. The difference between the two is a stage plan: an
ordered list of the scorer's stages and, for each, the record fields that stage
produces.

The stage order is the §6.3 execution order, which is written as a sequence in
the design precisely so it can be used this way.

**The field set is never taken from here.** The comparator derives the compared
fields from the records themselves; this plan only says which stage each of
those fields belongs to. A field the plan does not know is still compared, in
the ``unassigned`` stage at the end — the 2026-09-11 explainer compared 41 of
the 70 fields the digest hashed, and a plan that could silently drop a field
would be the same defect in a new place.

Two assignments are deliberate and worth stating, because they are what make
the §8 negative controls land in the right stage:

* ``structure_params`` and ``structure_spec`` belong to **serialization**, not
  geometry. Geometry's outputs are the resolved legs, strike and expiry. These
  two are the *replay inputs* — values whose entire remaining job is to be
  written and read back — and both 2026-09-11 rounding defects corrupted them
  on exactly that path.
* ``model_versions`` belongs to **features**, the stage that infers the feature
  models, rather than to a stage of its own.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Stage", "StagePlan", "SCORER_V1", "UNASSIGNED", "load_stage_plan",
           "root_of"]

#: Where a field the plan does not name is compared. Last, so it never
#: shadows a real stage, and named so it is obvious in a receipt.
UNASSIGNED = "unassigned"


def root_of(field_path: str) -> str:
    """The unescaped root key of a typed field path.

    Paths come from :func:`engine.v2.diagnosis.record_comparator.flatten`:
    mapping segments are dot-joined with ``.`` and ``\\`` escaped inside each
    key, sequence segments are ``[i]``. The root of ``a\\.b[0].c`` is the key
    ``a.b``; splitting on the first ``.`` would cut an escaped key in half
    and assign it to a stage that does not exist.
    """
    out: list[str] = []
    i = 0
    while i < len(field_path):
        char = field_path[i]
        if char == "\\" and i + 1 < len(field_path):
            out.append(field_path[i + 1])
            i += 2
            continue
        if char in ".[":
            break
        out.append(char)
        i += 1
    return "".join(out)


@dataclass(frozen=True)
class Stage:
    stage_id: str
    #: Top-level record keys this stage produces. A nested path such as
    #: ``structure_params.width_moneyness`` is assigned by its root key.
    fields: tuple[str, ...]
    #: The stages whose outputs are this stage's INPUTS. Declared rather than
    #: assumed to be "every earlier stage", because the scorer is a graph and
    #: not a chain, and the difference decides whether findings can be proved
    #: independent. On 2026-09-11 a blanked forecast and a mis-seeded analog
    #: bootstrap were two unrelated bugs; a chain would have reported the
    #: second as a possible consequence of the first, which is the report that
    #: cost five nights. Analogs do not read the forecast, so they do not
    #: depend on it here.
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class StagePlan:
    plan_id: str
    stages: tuple[Stage, ...]

    def stage_ids(self) -> tuple[str, ...]:
        return tuple(s.stage_id for s in self.stages) + (UNASSIGNED,)

    def depends_on(self, stage_id: str) -> tuple[str, ...]:
        """The stages whose outputs are ``stage_id``'s inputs."""
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage.depends_on
        return ()

    def stage_of(self, field_path: str) -> str:
        """The stage owning a field path, by its root key."""
        root = root_of(field_path)
        for stage in self.stages:
            if root in stage.fields:
                return stage.stage_id
        return UNASSIGNED


#: The plan for the current scorer: `engine.score.ScoreResult.as_dict()` keys,
#: assigned to the §6.3 stages.
SCORER_V1 = StagePlan(
    plan_id="scorer.v1",
    stages=(
        Stage("resolve_context", (
            "ticker", "strategy", "as_of", "event_date", "session",
            "entry_date", "exit_date", "evidence_cutoff", "quote_date",
            "quote_age_sessions", "quote_max_age_sessions", "dte_entry",
            "snapshot_hash",
        )),
        Stage("features", (
            "model_inputs", "model_input_as_of", "implied_move",
            "implied_move_at_entry", "chain_last_obs", "chain_age_days",
            "model_versions",
        ), depends_on=("resolve_context",)),
        Stage("forecast", (
            "forecast_abs_move", "forecast_p10", "forecast_p90", "forecast_sd",
            "forecast_model", "forecast_fold", "driver_name",
            "driver_prediction", "driver_p10", "driver_p90",
            "runup_move_prediction", "runup_move_p10", "runup_move_p90",
            "runup_move_days", "runup_move_scale",
        ), depends_on=("features",)),
        Stage("geometry", (
            "legs", "strike", "requested_strike", "expiry", "variant",
            "structure_width", "structure_peak",
        ), depends_on=("resolve_context", "forecast")),
        Stage("pricing", ("entry_cost", "spot", "fill", "fill_alpha", "rel_spread"),
              depends_on=("geometry",)),
        Stage("analogs", (
            "exp_pnl_analog", "win_analog", "ci_low", "ci_high", "n_analogs",
            "analog_widened", "analog_buckets",
        ), depends_on=("resolve_context", "features")),
        Stage("simulation", (
            "exp_pnl_model", "win_model", "win_model_raw", "model_p10",
            "model_p90", "exp_pnl_sim", "win_sim", "payoff",
        ), depends_on=("forecast", "pricing", "analogs")),
        Stage("gate", (
            "gate_score", "gate_threshold", "gate_pass", "extrapolated", "flags",
        ), depends_on=("features", "pricing", "simulation", "analogs")),
        Stage("chooser", ("chooser_score",),
              depends_on=("features", "geometry", "simulation")),
        Stage("serialization", (
            "structure_params", "structure_spec", "detail", "schema_version",
            "score_id", "payload_hash", "request_hash", "canonical_request",
            "digest",
        ), depends_on=("geometry",)),
    ),
)

_PLANS = {SCORER_V1.plan_id: SCORER_V1}


def load_stage_plan(plan_id: str) -> StagePlan:
    """Resolve a ``stage_plan_ref``. Unknown ids raise rather than defaulting."""
    try:
        return _PLANS[plan_id]
    except KeyError:
        raise KeyError(
            f"unknown stage plan {plan_id!r}; known plans: {sorted(_PLANS)}"
        ) from None
