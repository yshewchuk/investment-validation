"""Explicit pinned native D14 training; no mutable heads or legacy dataset reader."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from engine.v2.data.errors import DataError
from engine.v2.models.training.d14_inputs import build_d14_dataset as build_dataset


def main(argv=None):
    """Explicit native monthly training; existing supervised routes stay unchanged."""
    from engine import paths
    from engine.v2.models import no_fit
    from engine.v2.models.training import (
        RecipeKey,
        TrainingRefused,
        current_recipes,
        run_training_job,
    )
    from engine.v2.models.training.estimators import forbid_fitting

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("implied_t1", "runup_move"), required=True)
    for field in ("catalog", "artifact-root", "snapshot-id", "receipt-id", "out"):
        parser.add_argument(f"--{field}", required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    data_roots = (paths.DATA.resolve(), Path(args.artifact_root).resolve())
    if any(out == root or root in out.parents for root in data_roots):
        parser.error("--out must be outside data/ and the input artifact store")
    no_fit.forbid_fitting("tools.phase5_d14_inputs.main")
    forbid_fitting("tools.phase5_d14_inputs.main")
    try:
        frame = build_dataset(args.role, catalog=args.catalog, artifact_root=args.artifact_root,
                              snapshot_id=args.snapshot_id, receipt_id=args.receipt_id)
        recipe = current_recipes()[RecipeKey(args.role, "*", "tier4_monthly")]
        result = run_training_job(recipe, frame, out, plan_only=args.plan_only,
                                  input_identity=frame.attrs["training_inputs"])
    except (DataError, TrainingRefused) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"{recipe.recipe_id}: rows={len(frame)} folds={len(result.outcomes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
