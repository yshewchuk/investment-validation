"""Prediction-only request/seal and bounded evidence sections, not an execution loop."""
from collections.abc import Mapping
from copy import deepcopy
from datetime import date
from math import isfinite
from pathlib import Path

from engine.v2.foundation import canonical_json, content_hash
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import source_closure
from engine.v2.research.progress import Progress, stage
from experiments.distribution_plan import prepare_plan
from experiments.magnitude_search_execution import validate_execution_limits
from experiments.magnitude_search_records import append_record, read_record
from experiments.magnitude_search_shards import iter_section, write_section

__all__ = ["PredictionPlan", "validate_request"]


def require(condition):
    if not condition:
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution prediction contract")


def validate_request(request):
    """Detach explicit resource/report policy; never admit data or execution."""
    try:
        require(type(request) is dict and set(request) == {"schema_version", "limits", "max_events", "uncertainty"}
                and request["schema_version"] == "distribution_prediction_request.v1")
        require(type(request["max_events"]) is int and request["max_events"] > 0)
        limits = request["limits"]
        require(type(limits) is dict and set(limits) == {"parent", "job_wall_seconds"})
        validate_execution_limits(limits["parent"])
        require(type(limits["job_wall_seconds"]) is int and limits["job_wall_seconds"] > 0)
        policy = request["uncertainty"]
        require(type(policy) is dict and set(policy) == {"seed", "resamples", "block_days", "block_origin", "confidence"})
        require(type(policy["seed"]) is int and policy["seed"] >= 0
                and all(type(policy[key]) is int and policy[key] > 0 for key in ("resamples", "block_days")))
        confidence = policy["confidence"]
        require(type(confidence) in (int, float) and isfinite(confidence) and 0 < confidence < 1
                and type(policy["block_origin"]) is str
                and date.fromisoformat(policy["block_origin"]).isoformat() == policy["block_origin"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution prediction request") from None
    return deepcopy(request)


class PredictionPlan:
    """Native-source metadata, single-use evidence slots, and bounded CAS sections.

    A subclass owns execution, complete forecast/target ordering and verified
    collection. This class never opens outcomes, fits, prices or grants a run.
    """

    source_roots = ("experiments/distribution_prediction_plan.py", "experiments/distribution_worker.py",
        "experiments/distribution_prediction_evidence.py", "experiments/distribution_prediction_baseline.py",
        "experiments/distribution_prediction_evaluation.py")

    def __init__(self, trials, request):
        self.trials, self.request = trials, validate_request(request)

    def record(self, slot, body=None):
        """Create-only prediction namespace; no in-place update or retry reuse."""
        require(type(slot) is str and bool(slot) and (body is None or type(body) is dict))
        args = (self.trials.conn, self.trials.store, self.trials.own.run_id, "distribution:prediction-" + slot)
        return read_record(*args) if body is None else append_record(*args, body, replay=False)

    def plans(self):
        t = self.trials
        return prepare_plan(t.conn, t.store, t.own, t.source, t.request, t.source_plan_ref, t.recipe, t.caps)

    def preflight(self):
        request = validate_request(self.request)
        plan, native = self.plans()
        years = [group["outer"]["year"] for group in native["folds"]]
        events = [event for group in native["folds"] for event in group["outer"]["evaluation_event_ids"]]
        require(bool(years) and years == sorted(set(years)) and bool(events)
                and len(events) == len(set(events)) and len(events) <= request["max_events"]
                and set(plan["recipe"]["horizons"]) == {1, 3, 5})
        require(plan["budget"]["child_limits"]["memory_bytes"] <= request["limits"]["parent"]["memory_bytes"])
        body = {"schema_version": "distribution_prediction_plan.v1", "admission_plan_id": plan["plan_id"],
            "request": request, "event_count": len(events), "evaluation_event_ids": events,
            "fit_budget": plan["budget"], "source_closure": source_closure(Path(__file__).resolve().parents[1], list(self.source_roots)),
            "arms": ["histogram", "same_mean", "matched_point", "training_empirical"],
            "primary": "h1_same_mean_minus_histogram_crps", "calibration_identity": "uncalibrated",
            "final_read_authority": "none"}
        return {**body, "run_plan_id": content_hash(body)}

    def seal(self, *, clock):
        plan = self.preflight()
        self.trials.seal(clock=clock)
        return self.record("seal", plan)

    def admit(self):
        plan = self.preflight()
        if canonical_json(plan) != canonical_json(self.record("seal")):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution prediction seal differs")
        return plan

    @stage("prediction evidence publication")
    def publish(self, name, rows, *, count):
        """Bounded ordered section publication; caller owns admission/order claims."""
        require(type(name) is str and name.startswith("distribution_prediction_")
                and type(count) is int and count >= 0)
        try:
            rows = iter(rows)
        except TypeError:
            raise fail("INVALID_EXPERIMENT_SPEC", "prediction section rows must be iterable") from None
        def positioned():
            progress = Progress("prediction evidence publication", count)
            for index, row in enumerate(rows):
                require(isinstance(row, Mapping) and "position" not in row and index < count)
                yield {"position": index, **row}
                progress.tick()
            require(progress.done == count)
        return {"name": name, "descriptor": write_section(self.trials.store, name, positioned(), lambda row: (row["position"],))}

    def read_section(self, section):
        """Verify bounded stored evidence; return no new-source read authority."""
        require(type(section) is dict and set(section) == {"name", "descriptor"}
                and type(section["name"]) is str and section["name"].startswith("distribution_prediction_"))
        require(type(section["descriptor"]) is dict and type(section["descriptor"].get("row_count")) is int
                and section["descriptor"]["row_count"] >= 0)
        progress = Progress("prediction evidence replay", section["descriptor"]["row_count"])
        with stage("prediction evidence replay"):
            for index, row in enumerate(iter_section(self.trials.store, section["name"], section["descriptor"], lambda row: (row["position"],))):
                require(type(row["position"]) is int and row["position"] == index)
                yield {key: value for key, value in row.items() if key != "position"}
                progress.tick()
