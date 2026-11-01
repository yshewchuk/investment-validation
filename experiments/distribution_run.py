"""Explicit local development runner; no registration, provider or production path."""
import time
from datetime import timedelta

from engine.models.no_fit import forbid_fitting as forbid_legacy_fitting
from engine.v2.contracts import SnapshotRef
from engine.v2.foundation import SystemClock, format_timestamp, from_document
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.ops.errors import fail
from engine.v2.research._pricing import trading_calendar_from_snapshot
from engine.v2.research.progress import Progress, stage
from experiments.distribution_fitted import DistributionFits
from experiments.distribution_pipeline import evaluate_outer, finish_reports
from experiments.distribution_run_plan import DistributionRunPlan, contained
from experiments.distribution_run_request import PRIMARY_ARMS

__all__ = ["DistributionRun"]


class DistributionRun(DistributionRunPlan):
    """One fresh admitted run; completed collection never fits or prices."""

    source_roots = ("experiments/distribution_run.py", "experiments/distribution_worker.py")

    def preflight(self):
        """Bind this executable to its already-verified native registration."""
        plan = super().preflight()
        if self.trials.own.document["execution_plan"]["runner"] != "experiments/distribution_run.py":
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution run requires its registered runner")
        return plan

    def run(self, *, clock=None):
        """Explicitly contain this dedicated process and fresh per-fold bounded children."""
        clock = clock or SystemClock()
        seconds = self.request["limits"]["job_wall_seconds"]
        began = time.monotonic()
        with contained(self.request["limits"]["parent"], seconds), stage("distribution experiment"):
            return self._run_contained(clock, began)

    def _run_contained(self, clock, began):
        """Internal dispatch under the caller's whole-job containment; never reset it."""
        if self.record("completed") is not None:
            return self.collect()
        forbid_legacy_fitting("DistributionRun.run")
        forbid_fitting("DistributionRun.run")
        plan = self.admit()
        now = clock.now()
        seconds = self.request["limits"]["job_wall_seconds"] - (time.monotonic() - began)
        if seconds <= 0:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "distribution whole-job deadline exceeded")
        start = {"run_plan_id": plan["run_plan_id"], "started_at": format_timestamp(now),
                 "deadline": format_timestamp(now + timedelta(seconds=seconds))}
        self.record("started", start, replay=False)
        try:
            result = self._execute(plan)
            return self.record("completed", {**result, "run_plan_id": plan["run_plan_id"],
                "elapsed_seconds": time.monotonic() - began, "final_read_authority": "none"}, replay=False)
        except BaseException as error:
            self.record("failed", {"run_plan_id": plan["run_plan_id"], "code": getattr(error, "code", "INTERRUPTED"),
                "elapsed_seconds": time.monotonic() - began}, replay=False)
            raise

    def _execute(self, plan):
        _, native = self.plans()
        fits, sections, opportunities = DistributionFits(self.trials), [], []
        books = {alpha: {arm: [] for arm in (*PRIMARY_ARMS, "incumbent", "gate")} for alpha in plan["request"]["alphas"]}
        quotes = {alpha: {} for alpha in books}
        calendar = trading_calendar_from_snapshot(self.trials.repository, from_document(SnapshotRef, plan["quote_snapshot"]), extend_days=0)
        progress = Progress("distribution outer groups", len(native["folds"]))
        for group in native["folds"]:
            year, results = group["outer"]["year"], {}
            for role, inner in [("inner", fold["year"]) for fold in group["inner"]] + [("outer", None)]:
                self.trials.start(role, year, inner)
                for horizon in self.trials.recipe["horizons"]:
                    fitted = fits.fit(role, year, horizon, inner, bounded=True)
                    self.record("fit:" + fitted["receipt"]["claim"]["trial"]["trial_id"],
                        {"result_id": fitted["receipt"]["result_id"], "status": "bounded_completed"}, replay=False)
                    if role == "outer":
                        results[horizon] = fitted
            if set(results) != {1, 3, 5}:
                raise fail("INVALID_EXPERIMENT_SPEC", "distribution outer results require horizons 1,3,5")
            evaluate_outer(self, group, fits, results, plan, books, quotes, opportunities, sections, calendar)
            progress.tick()
        return finish_reports(self, books, quotes, opportunities, sections)


if __name__ == "__main__":
    from experiments.distribution_cli import main
    raise SystemExit(main())
