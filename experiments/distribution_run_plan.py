"""Metadata-only run policy/seal and dedicated-process containment primitives."""
import math
import resource
import signal
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

from engine.v2.foundation import canonical_json, content_hash, to_document
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import source_closure
from experiments.distribution_market import read_benchmark, read_bridge
from experiments.distribution_plan import prepare_plan
from experiments.distribution_run_request import require, validate_request
from experiments.magnitude_search_execution import validate_execution_limits
from experiments.magnitude_search_records import append_record, read_record
from experiments.magnitude_search_shards import build_manifest, iter_section, write_section
from experiments.memory_guard import rss_watchdog


def _completed(result):
    require(isinstance(result, dict) and {"run_plan_id", "sections"} <= result.keys()
        and type(result["run_plan_id"]) is str and bool(result["run_plan_id"])
        and type(result["sections"]) is list)
    for section in result["sections"]:
        require(isinstance(section, dict) and {"name", "descriptor"} <= section.keys()
            and type(section["name"]) is str and bool(section["name"])
            and isinstance(section["descriptor"], dict))
        build_manifest({}, {section["name"]: section["descriptor"]})


@contextmanager
def contained(limits, seconds):
    """Dedicated main-thread process only; restore soft limits, never raise hard limits."""
    validate_execution_limits(limits)
    require(type(seconds) in (int, float) and 0 < seconds < math.inf)
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))
    old = resource.getrlimit(resource.RLIMIT_CPU)
    handler = signal.getsignal(signal.SIGALRM)
    def expired(*args):
        raise fail("RESOURCE_LIMIT_EXCEEDED", "distribution whole-job deadline exceeded")
    try:
        used = resource.getrusage(resource.RUSAGE_SELF)
        requested = math.ceil(used.ru_utime + used.ru_stime) + limits["cpu_seconds"]
        soft, hard = old
        ceilings = [requested] + [value for value in old if value != resource.RLIM_INFINITY]
        resource.setrlimit(resource.RLIMIT_CPU, (min(ceilings), hard))
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, min(limits["wall_seconds"], seconds))
        with rss_watchdog(limits["memory_bytes"]):
            yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)
        resource.setrlimit(resource.RLIMIT_CPU, old)


class DistributionRunPlan:
    """Metadata/section journal only; no fitting, pricing or run authority."""

    source_roots = ("experiments/distribution_run_plan.py", "experiments/distribution_worker.py")

    def __init__(self, trials, request):
        self.trials, self.request = trials, validate_request(request)

    def record(self, slot, body=None, **options):
        require(type(slot) is str and bool(slot) and (body is None or isinstance(body, dict)))
        if slot == "completed" and body is not None:
            _completed(body)
            seal = self.record("seal")
            require(isinstance(seal, dict) and body["run_plan_id"] == seal.get("run_plan_id"))
        args = (self.trials.conn, self.trials.store, self.trials.own.run_id, "distribution:run-" + slot)
        result = read_record(*args) if body is None else append_record(*args, body, **options)
        if slot == "completed" and result is not None:
            _completed(result)
        return result

    def plans(self):
        t = self.trials
        return prepare_plan(t.conn, t.store, t.own, t.source, t.request, t.source_plan_ref, t.recipe, t.caps)

    def preflight(self):
        """Verify metadata, existing bridge/baseline inputs and declared bounds; no fit/quotes."""
        request = validate_request(self.request)
        plan, native = self.plans()
        years = [group["outer"]["year"] for group in native["folds"]]
        require(years == sorted(set(years)))
        count = sum(len(group["outer"]["evaluation_event_ids"]) for group in native["folds"])
        require(count <= request["max_opportunities"])
        require(plan["budget"]["child_limits"]["memory_bytes"] <= request["limits"]["parent"]["memory_bytes"])
        require(len(request["bridges"]) == len(years) * len(request["alphas"])
            and {(item["outer_year"], item["alpha"]) for item in request["bridges"]} ==
                {(year, alpha) for year in years for alpha in request["alphas"]})
        benchmarks = {entry["outer_year"]: entry for entry in request["benchmarks"]}
        available = {"incumbent": True, "gate": True}
        require(len(benchmarks) == len(request["benchmarks"]) and set(benchmarks) <= set(years))
        for group in native["folds"]:
            fold = group["outer"]
            for entry in request["bridges"]:
                if entry["outer_year"] == fold["year"]:
                    read_bridge(self.trials.store, entry, fold)
            rows = read_benchmark(self.trials.store, benchmarks.get(fold["year"]), fold)
            required = set(request["required_arms"]) | ({"incumbent", "gate"} if request["strict_benchmarks"] else set())
            for arm, field in (("incumbent", "incumbent_magnitudes"), ("gate", "gate_selected")):
                available[arm] &= bool(rows) and all(row[field] is not None for row in rows.values())
                require(arm not in required or available[arm])
        snapshot = self.trials.repository.resolve(request["quote_snapshot_id"])
        roots = list(self.source_roots)
        body = {"schema_version": "distribution_run_plan.v1", "admission_plan_id": plan["plan_id"],
            "request": request, "quote_snapshot": to_document(snapshot), "opportunities": count, "available_benchmarks": available,
            "fit_budget": plan["budget"], "source_closure": source_closure(Path(__file__).resolve().parents[1], roots),
            "final_read_authority": "none"}
        return {**body, "run_plan_id": content_hash(body)}

    def seal(self, *, clock):
        self.trials.seal(clock=clock)
        return self.record("seal", self.preflight())

    def admit(self):
        plan = self.preflight()
        require(canonical_json(plan) == canonical_json(self.record("seal")))
        return plan

    def collect(self):
        """Verify the sealed installation and all published report sections; no outcome reads."""
        plan, result = self.admit(), self.record("completed")
        require(result is not None and result["run_plan_id"] == plan["run_plan_id"])
        for section in result["sections"]:
            for _ in iter_section(self.trials.store, section["name"], section["descriptor"], lambda row: (row["position"],)):
                pass
        return result

    def publish(self, name, rows):
        require(type(name) is str and bool(name))
        try:
            rows = iter(rows)
        except TypeError:
            raise fail("INVALID_EXPERIMENT_SPEC", "distribution section rows must be iterable") from None
        def positioned():
            for index, row in enumerate(rows):
                require(isinstance(row, Mapping) and "position" not in row)
                yield {"position": index, **row}
        return {"name": name, "descriptor": write_section(self.trials.store, name,
            positioned(), lambda row: (row["position"],))}
