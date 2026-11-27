"""EXP-190 Q2b folds (design sections 4, 7). Pure over the dict from `load_sealed_panel`; reads no
files and no labels. No production caller yet (Q3a fits are the first).

Labels are qualified later by `size_quantile_sources.training_labels(sealed, fold["training_event_ids"],
fold["training_labels_before"])`."""
import sys
from datetime import date

from engine.v2.data.errors import DataError
from engine.v2.data.errors import fail as data_fail
from engine.v2.ops.errors import fail
from engine.v2.research.magnitude_folds import annual_fold
from engine.v2.research.progress import stage
from experiments.label_availability import available_before, label_cutoff_reason
from experiments.size_quantile_sources import _MALFORMED, _refuse

SEALED_YEARS = ("2023", "2024", "2025")


def _check_years(years):
    if not (isinstance(years, (list, tuple)) and years
            and all(isinstance(y, str) and len(y) == 4 and y.isascii() and y.isdecimal() and y <= "2025"
                    for y in years)
            and all(a < b for a, b in zip(years, years[1:]))):
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "years must be a non-empty ascending list of unique 4-digit years <= 2025",
                   details={"reason": "years_invalid"})


def _embargoed(sealed):
    full = (date.min.isoformat(), date.max.isoformat())
    return {w["ticker"] for w in sealed["context"].get("eligibility_plan", {}).get("random_windows", [])
            if (w["start"], w["end"]) == full}


def _rule_fold(year, sealed, targets, embargoed):
    """The rule's annual fold, with training kept only if its label is available and its ticker is not embargoed."""
    fold = annual_fold(year, sealed["fold_events"])
    kept = [e for e in fold["training_candidate_event_ids"]
            if label_cutoff_reason(targets[e]["label_end"], fold["training_labels_before"]) is None
            and targets[e]["ticker"] not in embargoed]
    return fold, kept


def _evaluation(fold, sealed, targets, embargoed):
    return [e for e in fold["evaluation_event_ids"]
            if targets[e]["ticker"] not in embargoed
            and available_before(targets[e]["label_end"], sealed["evaluation_labels_before"])]


def _parity(sealed, targets, embargoed):
    # The sealed SIZE plan nests 2023 and 2024 as inner folds of the 2025 outer fold.
    folds = sorted([fold for g in sealed["plan"]["folds"] for fold in [*g["inner"], g["outer"]]],
                   key=lambda f: f["year"])
    if [fold["year"] for fold in folds] != list(SEALED_YEARS):
        raise _refuse("fold_parity_mismatch", "the sealed fold years are not 2023, 2024, 2025, each once")
    for fold in folds:
        year = fold["year"]
        try:
            _, kept = _rule_fold(year, sealed, targets, embargoed)
        except DataError:
            raise _refuse("fold_parity_mismatch", f"the {year} fold cannot be rebuilt from the sealed events") from None
        if kept != fold["training_candidate_event_ids"]:
            raise _refuse("fold_parity_mismatch", f"the {year} training list differs from the sealed plan")


def _folds(sealed, years):
    targets = {e["event_id"]: e for e in sealed["context"]["target_events"]}
    embargoed = _embargoed(sealed)
    _parity(sealed, targets, embargoed)
    print(f"[quantile_folds] parity=ok sealed_years={','.join(SEALED_YEARS)}", file=sys.stderr)
    admitted = {e["event_id"] for e in sealed["events"]}
    out = {}
    for year in years:
        fold, kept = _rule_fold(year, sealed, targets, embargoed)
        evaluation = _evaluation(fold, sealed, targets, embargoed)
        training_ids = [e for e in kept if e in admitted]
        evaluation_ids = [e for e in evaluation if e in admitted]
        if not training_ids:
            raise data_fail("POPULATION_COLLAPSED", f"the {year} fold has no admitted training events after the filters")
        if not evaluation_ids:
            raise data_fail("POPULATION_COLLAPSED", f"the {year} fold has no admitted evaluation events after the filters")
        print(f"[quantile_folds] year={year} cutoff={fold['training_labels_before']} training={len(training_ids)} "
              f"evaluation={len(evaluation_ids)} "
              f"dropped_training={len(fold['training_candidate_event_ids']) - len(training_ids)} "
              f"dropped_evaluation={len(fold['evaluation_event_ids']) - len(evaluation_ids)}", file=sys.stderr)
        out[year] = {"year": year, "training_labels_before": fold["training_labels_before"],
                     "training_event_ids": training_ids, "evaluation_event_ids": evaluation_ids}
    return out


def annual_folds(sealed, years):
    """Annual outer folds for the requested years, returned only after the rule reproduces the sealed 2023-2025
    fold (2025 outer, 2023 and 2024 inner) TRAINING id lists exactly (ids and order). Evaluation ids are not compared."""
    _check_years(years)
    with stage("quantile folds"):
        try:
            return _folds(sealed, years)
        except DataError:
            raise
        except _MALFORMED as exc:
            raise _refuse("sealed_malformed", f"a sealed document is malformed ({type(exc).__name__})") from None
