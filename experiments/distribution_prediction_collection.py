"""Reconstruct compact prediction summaries from freshly verified native evidence."""
from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from engine.v2.research.progress import Progress, stage
from experiments.distribution_prediction_diagnostic_report import prediction_diagnostic_report
from experiments.distribution_prediction_evaluation import read_evaluated_outer
from experiments.distribution_prediction_report import ARMS, HORIZONS
from experiments.distribution_prediction_uncertainty import prediction_uncertainty

__all__ = ["report_evidence"]


def _require(condition):
    if not condition:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution prediction collection evidence differs")


def _complete(rows, ids, year):
    expected = {(event, horizon, arm) for event in ids for horizon in HORIZONS for arm in ARMS}
    _require(type(rows) is list and len(rows) == len(expected))
    observations = {}
    for row in rows:
        key = row["event_id"], row["horizon"], row["arm"]
        _require(type(row["horizon"]) is int and key in expected and key not in observations and row["outer_year"] == year)
        observations[key] = row["observation"]
    _require(set(observations) == expected)
    return observations


def _compact(report, schema):
    report.pop("rows")
    report.pop("events")
    report["schema_version"] = schema
    for horizon in report["horizons"].values():
        for contrast in horizon.get("contrasts", {}).values():
            contrast["excluded_events"] = len(contrast.pop("excluded_event_ids"))
            contrast.pop("paired_deltas")
    return report


@stage("prediction report reconstruction")
def report_evidence(run, fits):
    """Return (evaluation-record bindings, compact score/diagnostic summaries).

    Fresh native admission and evaluated CAS replay precede reporting. Original
    sections retain row evidence; summaries never package the whole population
    in one row. No fitting, source outcome read, publication or final authority.
    """
    sealed = run.admit()
    plan, native = run.plans()
    try:
        _require(plan["plan_id"] == sealed["admission_plan_id"])
        folds = [group["outer"] for group in native["folds"]]
        years = [fold["year"] for fold in folds]
        ids = [event for fold in folds for event in fold["evaluation_event_ids"]]
        _require(bool(years) and years == sorted(set(years)) and all(type(year) is str for year in years))
        _require(bool(ids) and len(ids) == len(set(ids)) and ids == sealed["evaluation_event_ids"]
                 and len(ids) == sealed["event_count"])
        source_events = run.trials.source.document["size_panel"]["context"]["target_events"]
        _require(type(source_events) is list)
        metadata = {}
        for event in source_events:
            key = event["event_id"]
            _require(type(key) is str and bool(key) and key not in metadata)
            metadata[key] = event
        selected = set(ids)
        events = [{field: event[field] for field in ("event_id", "ticker", "event_date")}
                  for key, event in metadata.items() if key in selected]
        _require({event["event_id"] for event in events} == selected)
        groups, scores, diagnostics = [], [], []
        progress = Progress("prediction outer report evidence", len(folds))
        for fold in folds:
            year = fold["year"]
            record, scored, diagnosed = read_evaluated_outer(run, fits, year)
            _require(record["outer_year"] == year and record["run_plan_id"] == sealed["run_plan_id"])
            _require(_complete(scored, fold["evaluation_event_ids"], year)
                     == _complete(diagnosed, fold["evaluation_event_ids"], year))
            groups.append({"outer_year": year, "evaluation_id": content_hash(record)})
            scores.extend(scored)
            diagnostics.extend(diagnosed)
            progress.tick()
        forecast = prediction_uncertainty(scores, events, **sealed["request"]["uncertainty"])
        diagnostic = prediction_diagnostic_report(diagnostics, events)
        return groups, {
            "forecast": _compact(forecast, "distribution_prediction_score_summary.v1"),
            "diagnostics": _compact(diagnostic, "distribution_prediction_diagnostic_summary.v1"),
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution prediction collection evidence differs") from None
