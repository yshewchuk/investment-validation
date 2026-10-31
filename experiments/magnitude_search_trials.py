"""Immutable search recording, bounded native inner fits and forecast-based selection."""
import copy
import json
import sqlite3
from pathlib import Path

from engine.models.no_fit import forbid_fitting as forbid_legacy_fitting
from engine.v2.contracts import DATA_FAILURE_CODES, FAILURE_CODES, Problem, SnapshotRef
from engine.v2.foundation import (
    ArtifactError,
    canonical_json,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.fingerprints import source_closure
from engine.v2.ops.migrations import migrate, require_current
from engine.v2.research.magnitude_inputs import load_magnitude_targets
from experiments.magnitude_report import magnitude_metrics
from experiments.magnitude_search_evidence import (
    admit_fold_inputs,
    fit_evidence,
    normalize_forecasts,
    validate_targets,
)
from experiments.magnitude_search_execution import (
    execution_limits,
    run_fitted_trial,
    validate_execution_limits,
)
from experiments.magnitude_search_inputs import load_inputs, publish_inputs
from experiments.magnitude_search_plan import (
    PLAN_V2,
    PLAN_V3,
    canonical_inner,
    cohort_key,
    read_search_plan,
    trial_key,
)
from experiments.magnitude_search_records import MIGRATIONS, OWNER, append_record, read_record
from experiments.size_fit_diagnostics import evidence as diagnostic_evidence
from experiments.size_fit_diagnostics import keys as diagnostic_keys
from experiments.size_job_budget import resolved_limits
from experiments.size_search_binding import execution_for_candidate


def _problem(problem):
    """Persist only registered policy and recorder-owned text, never caller diagnostics."""
    registry = {**DATA_FAILURE_CODES, **FAILURE_CODES}
    if not isinstance(problem, Problem) or not isinstance(problem.code, str) or problem.code not in registry:
        raise fail("INVALID_EXPERIMENT_SPEC", "trial failure requires a registered Problem")
    category, retryable = registry[problem.code]
    return to_document(Problem(code=problem.code, category=category, retryable=retryable,
                               message="search trial did not complete"))


def _candidate_failure(problem):
    return problem["code"] == "EXPERIMENT_VARIANT_FAILED"


_NOT_ADMITTED = object()


class SearchTrials:
    """Native inner fits or submitted forecasts; no outer evaluation or final authority."""

    def __init__(self, conn, store, repository, registration, request, plan_ref):
        self.conn, self.store, self.repository = conn, store, repository
        self.registration, self.plan_ref = registration, plan_ref
        self._memo = {}
        self._published = {}  # cohort -> (manifest, inputs_id) of this instance's successful publication.
        try:
            self.request = json.loads(json.dumps(request, allow_nan=False))
            if self.request != request:
                raise ValueError
        except (TypeError, ValueError, RecursionError):
            raise fail("INVALID_EXPERIMENT_SPEC", "search request must be canonical JSON") from None

    def _read(self, slot):
        return read_record(self.conn, self.store, self.registration.run_id, slot)

    def _append(self, slot, body, **kwargs):
        return append_record(self.conn, self.store, self.registration.run_id, slot, body, **kwargs)

    def _refused(self, key, error):
        problem = _problem(error.problem)
        try:
            self._append("terminal:" + key, {"status": "refused", "problem": problem})
        except (OpsError, OSError, sqlite3.Error) as publication_error:
            raise error from publication_error

    def _binding(self):
        plan = read_search_plan(self.conn, self.store, self.registration, self.request, self.plan_ref)
        try:
            sources = source_closure(Path(__file__).resolve().parents[1], ["experiments/magnitude_search_trials.py",
                "experiments/magnitude_search_worker.py"])
        except (OpsError, ArtifactError, OSError, SyntaxError, ValueError):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search recorder sources cannot be verified") from None
        return plan, {"plan_ref": to_document(self.plan_ref), "plan_id": plan["plan_id"],
                      "recorder_sources": sources, "estimator_execution": "unverified_submitted_forecasts"}

    def seal(self, *, clock):
        """Explicitly initialize journal ownership and reserve one plan before label reads."""
        _, binding = self._binding()
        migrate(self.conn, OWNER, MIGRATIONS, clock=clock)
        return self._append("seal", binding)

    def _admit(self):
        require_current(self.conn, OWNER, MIGRATIONS)
        plan, binding = self._binding()
        if self._read("seal") != binding:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search recording requires its exact sealed plan")
        return plan

    @staticmethod
    def _trial(plan, outer_year, inner_year, candidate_id):
        try:
            outer = next(fold for fold in plan["folds"] if fold["outer"]["year"] == outer_year)
            requested = next(fold for fold in outer["inner"] if fold["year"] == inner_year)
            candidate = next(item for item in plan["candidates"] if item["candidate_id"] == candidate_id)
        except StopIteration:
            raise fail("INVALID_EXPERIMENT_SPEC", "trial is outside the sealed search plan") from None
        if plan.get("schema_version") == PLAN_V3:
            owner_year, inner = canonical_inner(plan, outer_year, inner_year)
            return trial_key(plan, owner_year, inner_year, candidate_id), inner, candidate
        return trial_key(plan, outer_year, inner_year, candidate_id), requested, candidate

    def _inputs(self, inner):
        return admit_fold_inputs(self.repository, self.registration.document, inner,
            history_failure=fail("EXPERIMENT_IDENTITY_CONFLICT", "search native history binding changed"))

    def _cohort_inputs(self, cohort, inner):
        """Keep only the latest cohort resident; a revisit of a published cohort reloads its manifest; failures are never memoised."""
        if cohort not in self._memo:
            self._memo.clear()  # at most one cohort resident: release the previous one before loading
            if cohort in self._published:
                manifest, inputs_id = self._published[cohort]
                self._memo[cohort] = load_inputs(self.store, manifest, inputs_id)
            else:
                self._memo[cohort] = self._inputs(inner)
        return self._memo[cohort]

    def _cohort(self, admitted):
        """Load a stored cohort (v2 manifest or v1 whole record) bound to its admitted id."""
        record = self._read("cohort:" + admitted["cohort"])
        if record is None:
            raise ValueError
        try:
            return load_inputs(self.store, record, admitted["inputs_id"])
        except OpsError as error:
            if error.problem.retryable:
                raise
            # Identity conflicts take each caller's own evidence-chain message.
            raise ValueError from None

    def start(self, outer_year, inner_year, candidate_id):
        """Claim once, then bind native inputs; interrupted starts cannot be restarted."""
        return self._start(outer_year, inner_year, candidate_id)

    def _start(self, outer_year, inner_year, candidate_id, execution=None, admitted=_NOT_ADMITTED):
        plan = self._admit() if admitted is _NOT_ADMITTED else admitted
        key, inner, candidate = self._trial(plan, outer_year, inner_year, candidate_id)
        owner_year, _ = canonical_inner(plan, outer_year, inner_year)
        started = {"trial_id": key, "outer_year": owner_year, "inner_year": inner_year, "candidate": candidate}
        if execution is not None:
            started["execution"] = execution
        replay = plan.get("schema_version") == PLAN_V3
        if replay and self._read("start:" + key) is not None:
            if execution is not None:
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search trial is unfinished")
            if canonical_json(self._read("start:" + key)) != canonical_json(started):
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search trial is unfinished")
        self._append("start:" + key, started, replay=replay)
        if execution is not None:
            self._append("execution-start:" + key, execution, replay=replay)
        cohort = cohort_key(plan, owner_year, inner_year)
        inputs = self._cohort_inputs(cohort, inner)
        # Shards first (outside the catalog transaction); the manifest append is the commit point.
        if cohort not in self._published:
            self._published[cohort] = publish_inputs(self.store, inputs)
        manifest, inputs_id = self._published[cohort]
        self._append("cohort:" + cohort, manifest)
        self._append("inputs:" + key, {"cohort": cohort, "inputs_id": inputs_id})
        # Callers may mutate their copy; the memo stays private.
        return {**started, "inputs": copy.deepcopy(inputs)}

    def execute(self, outer_year, inner_year, candidate_id, *, limits):
        """Run one bounded native inner trial once, or verify its completed replay."""
        plan = self._admit()
        key, inner, candidate = self._trial(plan, outer_year, inner_year, candidate_id)
        size_plan = plan.get("schema_version") in (PLAN_V2, PLAN_V3)
        if size_plan and limits != plan["request"]["limits"]["child"]:
            raise fail("INVALID_EXPERIMENT_SPEC", "v2 trial limits differ from sealed request")
        validate_execution_limits(limits)
        attempt = {"limits": dict(limits), "candidate_id": candidate_id, "trial_id": key}
        completed = self._read("execution:" + key)
        if completed is not None:
            return self._executed(plan, outer_year, inner_year, candidate_id, attempt)
        forbid_fitting("SearchTrials.execute")
        forbid_legacy_fitting("SearchTrials.execute")
        resolved = resolved_limits(self.conn, self.store, self.registration, plan, limits) if size_plan else execution_limits(limits)
        started = self._start(outer_year, inner_year, candidate_id, attempt, plan)
        inputs = started["inputs"]
        execution = execution_for_candidate(self.registration.document, candidate)
        fits, pending = [], []
        expected = self._fit_evidence(inputs, inner, plan, execution)

        def progress(state, fit):
            index = len(fits)
            if (index >= len(expected) or canonical_json(fit) != canonical_json(expected[index])
                    or (state == "started") == bool(pending)):
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search fit progress differs from admitted inputs")
            self._append("fit-" + state + ":" + key + ":" + str(fit["horizon_sessions"]), fit, replay=False)
            if state == "started":
                pending.append(fit)
            else:
                fits.append(pending.pop())

        # The child streams the admitted manifest's shards itself; no inputs cross the pipe.
        admitted = self._read("inputs:" + key)
        source = {"manifest": self._read("cohort:" + admitted["cohort"]), "inputs_id": admitted["inputs_id"],
                  "store_root": str(self.store.root)}
        if size_plan:
            resolved = resolved_limits(self.conn, self.store, self.registration, plan, limits)
        result = run_fitted_trial(source, inner, plan, execution, resolved, progress)
        succeeded = isinstance(result, dict) and result.get("event") == "succeeded"
        if (not isinstance(result, dict) or result.get("event") not in ("succeeded", "failed")
                or (succeeded and (pending or len(fits) != len(plan["horizons"]) or set(result) != {"event", "forecasts"} | diagnostic_keys(self.registration.document)))
                or (not succeeded and (len(pending) != 1 or set(result) != {"event"}))):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search fit completion differs from admitted attempts")
        # run_fitted_trial has returned only after checking exit and reaping the child.
        # Bind the admitted id (a manifest hash for sharded cohorts), the id reconcile compares.
        self._append("execution-result:" + key, {"attempt": attempt,
                     "inputs_id": self._read("inputs:" + key)["inputs_id"], "result": result}, replay=False)
        evidence = {"attempt": attempt, "fit_attempt_count": len(fits) + len(pending),
                    "completed_fit_count": len(fits), "fits": fits, "status": result["event"],
                    "forecasts_id": content_hash(result["forecasts"]) if succeeded else None,
                    **diagnostic_evidence(result, execution, plan["horizons"])}
        self._append("execution:" + key, evidence, replay=False)
        self._record(outer_year, inner_year, candidate_id, native=True, **({"forecasts": result["forecasts"]} if succeeded
            else {"failure": fail("EXPERIMENT_VARIANT_FAILED", "fitted magnitude estimator failed").problem}))
        return self._executed(plan, outer_year, inner_year, candidate_id, attempt)

    def reconcile(self, outer_year, inner_year, candidate_id, *, limits):
        """Finish publication from a durable child result; never retry any fitting."""
        plan = self._admit()
        key, _, _ = self._trial(plan, outer_year, inner_year, candidate_id)
        validate_execution_limits(limits)
        attempt = {"limits": dict(limits), "candidate_id": candidate_id, "trial_id": key}
        result, evidence = self._result_evidence(plan, outer_year, inner_year, candidate_id, attempt)
        completed = self._read("execution:" + key)
        if completed is not None and canonical_json(completed) != canonical_json(evidence):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search execution accounting differs")
        if self._read("terminal:" + key) is not None:
            _, terminal = self._terminal_record(plan, outer_year, inner_year, candidate_id)
            return self._execution_receipt(plan, outer_year, inner_year, candidate_id, terminal, attempt,
                                           verified_evidence=evidence)
        self._append("execution:" + key, evidence)
        self._record(outer_year, inner_year, candidate_id, native=True, reconcile=True,
            **({"forecasts": result["forecasts"]} if result["event"] == "succeeded" else
               {"failure": fail("EXPERIMENT_VARIANT_FAILED", "fitted magnitude estimator failed").problem}))
        _, terminal = self._terminal_record(plan, outer_year, inner_year, candidate_id)
        return self._execution_receipt(plan, outer_year, inner_year, candidate_id, terminal, attempt,
                                       verified_evidence=evidence)

    def _result_evidence(self, plan, outer_year, inner_year, candidate_id, attempt):
        """Independently verify durable result, admission, and exact progress prefix."""
        key, inner, candidate = self._trial(plan, outer_year, inner_year, candidate_id)
        owner_year, _ = canonical_inner(plan, outer_year, inner_year)
        cohort = cohort_key(plan, owner_year, inner_year)
        try:
            started = {"trial_id": key, "outer_year": owner_year, "inner_year": inner_year,
                       "candidate": candidate, "execution": attempt}
            if (canonical_json(self._read("start:" + key)) != canonical_json(started)
                    or canonical_json(self._read("execution-start:" + key)) != canonical_json(attempt)):
                raise ValueError
            admitted = self._read("inputs:" + key)
            if not isinstance(admitted, dict) or set(admitted) != {"cohort", "inputs_id"} or admitted["cohort"] != cohort:
                raise ValueError
            inputs = self._cohort(admitted)
            durable = self._read("execution-result:" + key)
            if (not isinstance(durable, dict) or set(durable) != {"attempt", "inputs_id", "result"}
                    or canonical_json(durable["attempt"]) != canonical_json(attempt)
                    or durable["inputs_id"] != admitted["inputs_id"]):
                raise ValueError
            result = durable["result"]
            if not isinstance(result, dict) or result.get("event") not in ("succeeded", "failed"):
                raise ValueError
            succeeded = result["event"] == "succeeded"
            if set(result) != (({"event", "forecasts"} | diagnostic_keys(self.registration.document)) if succeeded else {"event"}):
                raise ValueError
            forecasts = None
            if succeeded:
                try:
                    forecasts = self._forecasts(result["forecasts"], inner, plan["horizons"])
                except OpsError:
                    raise ValueError from None
                if canonical_json(forecasts) != canonical_json(result["forecasts"]):
                    raise ValueError
            submitted = self._read("forecasts:" + key)
            if submitted is not None and canonical_json(submitted) != canonical_json(forecasts):
                raise ValueError
            if succeeded and submitted is not None:
                validation = self._read("validation:" + cohort)
                if validation is not None:
                    self._validation(inner, admitted, inputs, validation)
            execution = execution_for_candidate(self.registration.document, candidate)
            expected = self._fit_evidence(inputs, inner, plan, execution)
            started_fits, completed_fits = [], []
            for fit in expected:
                suffix = ":" + key + ":" + str(fit["horizon_sessions"])
                started_fits.append(self._read("fit-started" + suffix))
                completed_fits.append(self._read("fit-completed" + suffix))
            count = sum(item is not None for item in started_fits)
            completed = count if succeeded else count - 1
            if not 1 <= count <= len(expected) or (succeeded and count != len(expected)):
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search execution accounting differs")
            for index, fit in enumerate(expected):
                if (canonical_json(started_fits[index]) != canonical_json(fit if index < count else None)
                        or canonical_json(completed_fits[index]) != canonical_json(fit if index < completed else None)):
                    raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search execution accounting differs")
        except (ValueError, TypeError, KeyError, OverflowError):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search durable execution result differs") from None
        return result, {"attempt": attempt, "fit_attempt_count": count, "completed_fit_count": completed,
                        "fits": expected[:completed], "status": result["event"],
                        "forecasts_id": content_hash(forecasts) if succeeded else None,
                        **diagnostic_evidence(result, execution, plan["horizons"])}

    @staticmethod
    def _fit_evidence(inputs, inner, plan, execution):
        return fit_evidence(inputs, inner, plan, execution)

    def _executed(self, plan, outer_year, inner_year, candidate_id, attempt):
        _, terminal = self._terminal_record(plan, outer_year, inner_year, candidate_id)
        return self._execution_receipt(plan, outer_year, inner_year, candidate_id, terminal, attempt)

    def _execution_receipt(self, plan, outer_year, inner_year, candidate_id, terminal, attempt, *, verified_evidence=None):
        key, _, _ = self._trial(plan, outer_year, inner_year, candidate_id)
        record = self._read("execution:" + key)
        if (not isinstance(record, dict) or not isinstance(attempt, dict)
                or set(attempt) != {"limits", "candidate_id", "trial_id"}
                or attempt["candidate_id"] != candidate_id or attempt["trial_id"] != key
                or canonical_json(record.get("attempt")) != canonical_json(attempt)):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search execution accounting differs")
        try:
            validate_execution_limits(attempt["limits"])
        except OpsError:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search execution accounting differs") from None
        # Reconcile may supply evidence freshly checked earlier in this same call.
        receipt = verified_evidence
        if receipt is None:
            _, receipt = self._result_evidence(plan, outer_year, inner_year, candidate_id, attempt)
        if (canonical_json(record) != canonical_json(receipt)
                or (receipt["status"] == "succeeded") != (terminal["status"] in ("succeeded", "refused"))):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search execution accounting differs")
        return {**receipt, "terminal": terminal, "evaluation_authority": "none"}

    @staticmethod
    def _forecasts(forecasts, inner, horizons):
        try:
            return json.loads(canonical_json(normalize_forecasts(forecasts, inner, horizons)))
        except (ValueError, TypeError, OverflowError):
            raise fail("INVALID_EXPERIMENT_SPEC", "trial forecasts require complete finite event-horizon coverage") from None

    def record(self, outer_year, inner_year, candidate_id, *, forecasts=None, failure=None):
        """Submit forecasts/failure, or replay a completed trial; never recover native work."""
        return self._record(outer_year, inner_year, candidate_id, forecasts=forecasts, failure=failure)

    def _record(self, outer_year, inner_year, candidate_id, *, forecasts=None, failure=None, native=False, reconcile=False):
        """Commit forecasts before labels; native reconciliation can finish publication."""
        plan = self._admit()
        key, inner, _ = self._trial(plan, outer_year, inner_year, candidate_id)
        if self._read("start:" + key) is None:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "trial must be started before recording")
        if (forecasts is None) == (failure is None):
            raise fail("INVALID_EXPERIMENT_SPEC", "record exactly one forecast submission or failure")
        execution = self._read("execution-start:" + key)
        submitted = None
        if execution is not None:
            if not native and self._read("terminal:" + key) is None:
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native execution must finish before recording")
            completed = self._read("execution:" + key)
            if not isinstance(completed, dict) or completed.get("status") != ("failed" if failure is not None else "succeeded"):
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native completion differs from recording")
            if forecasts is not None:
                submitted = self._forecasts(forecasts, inner, plan["horizons"])
                if content_hash(submitted) != completed.get("forecasts_id"):
                    raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native completion differs from recording")
        if failure is not None:
            problem = _problem(failure)
            if not _candidate_failure(problem):
                raise OpsError(from_document(Problem, problem))
            if self._read("inputs:" + key) is None:
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "trial has no admitted native inputs")
            terminal = self._append("terminal:" + key, {"status": "failed", "problem": problem}, absent=("forecasts:" + key,))
            # Native execute/reconcile verifies the receipt once at its return boundary.
            return terminal if native else self._terminal(plan, outer_year, inner_year, candidate_id)[1]
        if submitted is None:
            submitted = self._forecasts(forecasts, inner, plan["horizons"])
        terminal = self._read("terminal:" + key)
        if terminal is not None:
            if self._read("forecasts:" + key) != submitted:
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "terminal trial cannot accept another submission")
            return self._terminal(plan, outer_year, inner_year, candidate_id)[1]
        admitted = self._read("inputs:" + key)
        if admitted is None:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "trial has no admitted native inputs")
        self._append("forecasts:" + key, submitted, replay=reconcile, absent=("terminal:" + key,))
        validation = self._read("validation:" + admitted["cohort"]) if reconcile else None
        stored = None
        if validation is None:
            try:
                stored = self._cohort(admitted)
            except ValueError:
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "trial native inputs changed after start") from None
            if content_hash(self._cohort_inputs(admitted["cohort"], inner)) != content_hash(stored):
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "trial native inputs changed after start")
            binding = self.registration.document
            labels = load_magnitude_targets(self.repository, from_document(SnapshotRef, binding["snapshot"]),
                as_of_month=binding["holdouts"]["holdout_as_of_month"], labels_before=inner["evaluation_labels_before"],
                event_ids=inner["evaluation_event_ids"], omit_not_computed=True)
            validation = {"inputs_id": admitted["inputs_id"], "evaluation_inputs": dict(labels.attrs),
                          "targets": labels[["event_id", "absolute_move_pct"]].to_dict("records")}
        try:
            self._validation(inner, admitted, stored if stored is not None else self._cohort(admitted), validation)
        except (ValueError, TypeError, KeyError, OverflowError):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search validation evidence differs") from None
        validation = self._append("validation:" + admitted["cohort"], validation)
        targets = {row["event_id"]: row["absolute_move_pct"] for row in validation["targets"]}
        scored = [{**row, "absolute_move_pct": targets[row["event_id"]]}
                  for row in submitted if row["event_id"] in targets]
        try:
            terminal = {"status": "succeeded", "inputs_id": admitted["inputs_id"],
                        "validation_id": content_hash(validation), "evaluation_inputs": validation["evaluation_inputs"],
                        "predictions": scored, "metrics": magnitude_metrics(scored)}
        except OpsError as error:
            if _candidate_failure(_problem(error.problem)):
                self._refused(key, error)
            raise
        return self._append("terminal:" + key, terminal)

    @staticmethod
    def _validation(inner, admitted, inputs, validation):
        """Check a native-derived shared target authority against the admitted preflight."""
        if (set(validation) != {"inputs_id", "evaluation_inputs", "targets"}
                or validation["inputs_id"] != admitted["inputs_id"]):
            raise ValueError
        validate_targets(inputs["preflight"], inner, validation["evaluation_inputs"], validation["targets"])

    def _terminal(self, plan, outer_year, inner_year, candidate_id):
        key, record = self._terminal_record(plan, outer_year, inner_year, candidate_id)
        attempt = self._read("execution-start:" + key)
        if attempt is not None:
            self._execution_receipt(plan, outer_year, inner_year, candidate_id, record, attempt)
        return key, record

    def _terminal_record(self, plan, outer_year, inner_year, candidate_id):
        """Verify terminal evidence; the caller also verifies any native receipt."""
        key, inner, candidate = self._trial(plan, outer_year, inner_year, candidate_id)
        owner_year, _ = canonical_inner(plan, outer_year, inner_year)
        cohort_id = cohort_key(plan, owner_year, inner_year)
        record = self._read("terminal:" + key)
        if record is None:
            raise fail("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt")
        try:
            expected = {"trial_id": key, "outer_year": owner_year, "inner_year": inner_year, "candidate": candidate}
            attempt = self._read("execution-start:" + key)
            if attempt is not None:
                expected["execution"] = attempt
            if canonical_json(self._read("start:" + key)) != canonical_json(expected):
                raise ValueError
            admitted = self._read("inputs:" + key)
            if admitted is None or set(admitted) != {"cohort", "inputs_id"}:
                raise ValueError
            if admitted["cohort"] != cohort_id:
                raise ValueError
            cohort = self._cohort(admitted)
            submitted = self._read("forecasts:" + key)
            if submitted is not None:
                try:
                    submitted = self._forecasts(submitted, inner, plan["horizons"])
                except OpsError:
                    raise ValueError from None
                validation = self._read("validation:" + admitted["cohort"])
                self._validation(inner, admitted, cohort, validation)
            if record["status"] == "succeeded":
                if set(record) != {"status", "inputs_id", "validation_id", "evaluation_inputs", "predictions", "metrics"}:
                    raise ValueError
                forecasts = [{name: row[name] for name in ("event_id", "horizon_sessions", "predicted_absolute_move_pct")}
                             for row in record["predictions"]]
                targets = {row["event_id"]: row["absolute_move_pct"] for row in validation["targets"]}
                scored_forecasts = ([row for row in submitted if row["event_id"] in targets]
                                    if submitted is not None else None)
                if (submitted is None or record["inputs_id"] != admitted["inputs_id"]
                        or forecasts != scored_forecasts
                        or record["validation_id"] != content_hash(validation)
                        or record["evaluation_inputs"] != validation["evaluation_inputs"]):
                    raise ValueError
                expected_rows = [{**row, "absolute_move_pct": targets[row["event_id"]]} for row in scored_forecasts]
                if canonical_json(record["predictions"]) != canonical_json(expected_rows):
                    raise ValueError
                try:
                    metrics = magnitude_metrics(record["predictions"])
                except (OpsError, OverflowError):
                    raise ValueError from None
                if record["metrics"] != metrics:
                    raise ValueError
            elif record["status"] not in ("failed", "refused"):
                raise ValueError
            else:
                try:
                    normalized = _problem(from_document(Problem, record["problem"]))
                except OpsError:
                    raise ValueError from None
                if (set(record) != {"status", "problem"} or normalized != record["problem"]
                        or not _candidate_failure(normalized)
                        or (record["status"] == "failed") != (submitted is None)):
                    raise ValueError
        except (ValueError, TypeError, KeyError, OverflowError):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search trial evidence chain differs") from None
        return key, record

    def freeze(self):
        """Freeze every outer choice from all terminal attempts; no evaluation authority."""
        plan = self._admit()
        selections, attempts = [], []
        for outer in plan["folds"]:
            scores = []
            for candidate in plan["candidates"]:
                rows, failed = [], False
                for inner in outer["inner"]:
                    key, record = self._terminal(plan, outer["outer"]["year"], inner["year"], candidate["candidate_id"])
                    attempts.append({"trial_id": key, "terminal_id": content_hash(record)})
                    failed |= record["status"] != "succeeded"
                    if record["status"] == "succeeded":
                        rows.extend(record["predictions"])
                scores.append({"candidate_id": candidate["candidate_id"], "metrics": None if failed else magnitude_metrics(rows)})
            eligible = [score for score in scores if score["metrics"] is not None]
            if not eligible:
                raise fail("EXPERIMENT_VARIANT_FAILED", "all candidates failed for an outer fold")
            selected = min(eligible, key=lambda score: (score["metrics"]["rmse"], score["candidate_id"]))
            selections.append({"outer_year": outer["outer"]["year"], "scores": scores,
                               "selected_candidate_id": selected["candidate_id"]})
        result = {"plan_id": plan["plan_id"], "attempts": attempts, "selections": selections,
                  "prospective_candidate_id": selections[-1]["selected_candidate_id"],
                  "estimator_execution": "unverified_submitted_forecasts", "evaluation_authority": "none"}
        return self._append("selection", result)
