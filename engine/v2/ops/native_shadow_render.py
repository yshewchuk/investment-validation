"""The ops half of the native shadow-serving seam (spec_ns_b, G1/G5).

``native_shadow_serving_mode`` is the one explicit plan/config switch that
decides whether the shadow board is served from native scoring or from the
legacy rows the projection already carries.  The switch lives on the plan
document itself -- ``build_nightly_plan`` records ``shadow_serving_scorer`` on
the same dict every stage reads -- never on an environment variable or a CLI
flag with no plan record (G5).

The row-building half of the seam lives in
``engine.v2.serving.native_shadow_render`` and this module cannot import it:
``engine.v2.ops`` and ``engine.v2.serving`` are both layer 7 peers
(``system_rearchitecture.md`` §4.1, enforced by ``checks/import_layers.py``)
and a peer is not "down".  The serving side owns the native rows and its own
copy of the same two-string validation; the composing ``tools`` layer -- the
one place allowed to import both (guide §2, exactly as
``tools/v2_dashboard_project.py`` already imports ``engine.v2.ops.bootstrap``
and ``engine.v2.serving.projections`` together) -- is what binds the two
halves.  Both halves report the same ``INVALID_REQUEST`` refusal.

This module writes nothing: it only reads a plan dict and refuses a value it
cannot implement.  It names no ``engine.v2.ops.decision_commit``,
``...ledger_history_import``, ``...legacy_actions`` or
``engine.dashboard.nightly`` symbol (G1; the static test walks this module's
AST).
"""
from __future__ import annotations

from engine.v2.ops.errors import fail

__all__ = ["SHADOW_SERVING_SCORERS", "native_shadow_serving_mode"]

#: The only values ``shadow_serving_scorer`` may carry.
SHADOW_SERVING_SCORERS = ("native", "legacy")

_DEFAULT_SCORER = "native"


def native_shadow_serving_mode(plan: dict) -> str:
    """Return the plan's explicit shadow-serving scorer.

    ``"native"`` is the default for the shadow board: its rendered rows are
    built from real ``engine.v2.scoring.application.score_one`` records.
    ``"legacy"`` serves the legacy ``bundle_rows_by_ticker`` unchanged.  Any
    other value is a typed ``INVALID_REQUEST`` refusal -- never a silent
    default to either branch, because the switch must be explicit and
    recorded rather than inferred from an unrecognized value (G5).  The
    refusal reuses ``engine.v2.ops.errors.fail``, the same envelope every
    other plan validation uses, never a bare ``ValueError``.
    """
    mode = plan.get("shadow_serving_scorer", _DEFAULT_SCORER)
    if mode not in SHADOW_SERVING_SCORERS:
        raise fail("INVALID_REQUEST", "shadow_serving_scorer must be legacy or native")
    return mode
