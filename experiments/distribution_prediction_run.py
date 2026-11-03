"""Contained prediction-only development run and verified no-fit collection."""
import time
from datetime import timedelta
from math import isfinite

from engine.models.no_fit import forbid_fitting as forbid_legacy_fitting
from engine.v2.data.errors import DataError
from engine.v2.foundation import SystemClock, canonical_json, format_timestamp
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.ops.errors import OpsError, fail
from engine.v2.research.progress import Progress, decision, stage
from experiments.distribution_containment import contained
from experiments.distribution_fitted import DistributionFits
from experiments.distribution_prediction_collection import report_evidence
from experiments.distribution_prediction_evaluation import evaluate_outer
from experiments.distribution_prediction_evidence import freeze_outer
from experiments.distribution_prediction_plan import PredictionPlan, validate_request

__all__ = ["PredictionRun"]


def _require(condition):
    if not condition:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution prediction execution evidence differs")


class PredictionRun(PredictionPlan):
    """One consumed attempt; all-horizon forecast freeze precedes each outer read."""

    source_roots = ("experiments/distribution_prediction_run.py", "experiments/distribution_worker.py")

    def preflight(self):
        plan = super().preflight()
        _require(self.trials.own.document["execution_plan"]["runner"] == self.source_roots[0])
        return plan

    def run(self, *, clock=None):
        """Dedicated Linux main-thread process; direct callers must already seal."""
        began = time.monotonic()
        request, seal = validate_request(self.request), self.record("seal")
        _require(type(seal) is dict and canonical_json(request) == canonical_json(seal.get("request")))
        seconds = request["limits"]["job_wall_seconds"] - (time.monotonic() - began)
        with contained(request["limits"]["parent"], seconds), stage("distribution prediction experiment"):
            return self._run_contained(clock or SystemClock(), began)

    def _run_contained(self, clock, began):
        """CLI owns the existing containment and original whole-job start time."""
        if self.record("completed") is not None:
            return self.collect()
        forbid_legacy_fitting("PredictionRun.run")
        forbid_fitting("PredictionRun.run")
        plan = self.admit()
        seconds = plan["request"]["limits"]["job_wall_seconds"] - (time.monotonic() - began)
        if seconds <= 0:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "distribution prediction whole-job deadline exceeded")
        now = clock.now()
        self.record("started", {"run_plan_id": plan["run_plan_id"], "started_at": format_timestamp(now),
            "deadline": format_timestamp(now + timedelta(seconds=seconds))})
        try:
            groups, reports, summaries = self._execute()
            result = {"schema_version": "distribution_prediction_run.v1", "run_plan_id": plan["run_plan_id"],
                "groups": groups, "reports": reports, "elapsed_seconds": time.monotonic() - began,
                "final_read_authority": "none", "promotion_authority": "none"}
            self.record("completed", result)
            return {**result, "summary": summaries}
        except BaseException as error:
            try:
                code = error.code if isinstance(error, (OpsError, DataError)) else "INTERRUPTED"
                self.record("failed", {"run_plan_id": plan["run_plan_id"], "code": code})
            except BaseException:
                try:
                    decision("prediction", "failure journal unavailable; original failure retained; attempt remains consumed")
                except BaseException:
                    pass
            raise

    def _execute(self):
        _, native = self.plans()
        fits = DistributionFits(self.trials)
        progress = Progress("prediction outer groups", len(native["folds"]))
        for group in native["folds"]:
            year = group["outer"]["year"]
            for role, inner in [("inner", fold["year"]) for fold in group["inner"]] + [("outer", None)]:
                self.trials.start(role, year, inner)
                for horizon in self.trials.recipe["horizons"]:
                    fits.fit(role, year, horizon, inner, bounded=True)
            freeze_outer(self, fits, year)
            evaluate_outer(self, fits, year)
            progress.tick()
        groups, summaries = report_evidence(self, fits)
        reports = {name: self.publish("distribution_prediction_report_" + name, [value], count=1)
            for name, value in summaries.items()}
        return groups, reports, summaries

    @stage("prediction completed collection")
    def collect(self):
        """Rebuild native training/fit and numerical evidence, never read new outcomes."""
        plan, result, start = self.admit(), self.record("completed"), self.record("started")
        _require(type(result) is dict and set(result) == {"schema_version", "run_plan_id", "groups", "reports",
            "elapsed_seconds", "final_read_authority", "promotion_authority"}
            and result["schema_version"] == "distribution_prediction_run.v1"
            and result["run_plan_id"] == plan["run_plan_id"]
            and result["final_read_authority"] == result["promotion_authority"] == "none"
            and type(result["elapsed_seconds"]) in (int, float)
            and isfinite(result["elapsed_seconds"]) and result["elapsed_seconds"] >= 0
            and type(start) is dict and start.get("run_plan_id") == plan["run_plan_id"])
        groups, summaries = report_evidence(self, DistributionFits(self.trials))
        _require(canonical_json(groups) == canonical_json(result["groups"])
            and type(result["reports"]) is dict and set(result["reports"]) == set(summaries))
        for name, expected in summaries.items():
            _require(canonical_json(list(self.read_section(result["reports"][name]))) == canonical_json([expected]))
        return {**result, "summary": summaries}


if __name__ == "__main__":
    from experiments.distribution_prediction_cli import main
    raise SystemExit(main())
