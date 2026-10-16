"""All-attempt recording and selection from forecasts; estimator execution is unverified."""
import json
import math
from pathlib import Path

from engine.v2.contracts import DATA_FAILURE_CODES, FAILURE_CODES, Problem, SnapshotRef
from engine.v2.data.errors import DataError
from engine.v2.foundation import (
    ArtifactError,
    canonical_json,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.fingerprints import source_closure
from engine.v2.ops.migrations import migrate, require_current
from engine.v2.research.magnitude_history_store import read_stored
from engine.v2.research.magnitude_inputs import (
    load_magnitude_targets,
    preflight_magnitude_targets,
    qualify_magnitude_training_targets,
)
from experiments.magnitude_history_inputs import history_binding
from experiments.magnitude_report import _metrics
from experiments.magnitude_search_plan import read_search_plan
from experiments.magnitude_search_records import MIGRATIONS, OWNER, append_record, read_record


def _problem(problem):
    """Persist only registered policy and recorder-owned text, never caller diagnostics."""
    registry = {**DATA_FAILURE_CODES, **FAILURE_CODES}
    if not isinstance(problem, Problem) or not isinstance(problem.code, str) or problem.code not in registry:
        raise fail("INVALID_EXPERIMENT_SPEC", "trial failure requires a registered Problem")
    category, retryable = registry[problem.code]
    return to_document(Problem(code=problem.code, category=category, retryable=retryable,
                               message="search trial did not complete"))


class SearchTrials:
    """No fits, CLI, outer evaluation, deployment, ledger or final-read authority."""

    def __init__(self, conn, store, repository, registration, request, plan_ref):
        self.conn, self.store, self.repository = conn, store, repository
        self.registration, self.plan_ref = registration, plan_ref
        self.request = json.loads(canonical_json(request))

    def _read(self, slot):
        return read_record(self.conn, self.store, self.registration.run_id, slot)

    def _append(self, slot, body, **kwargs):
        return append_record(self.conn, self.store, self.registration.run_id, slot, body, **kwargs)

    def _binding(self):
        plan = read_search_plan(self.conn, self.store, self.registration, self.request, self.plan_ref)
        try:
            sources = source_closure(Path(__file__).resolve().parents[1], ["experiments/magnitude_search_trials.py"])
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
            inner = next(fold for fold in outer["inner"] if fold["year"] == inner_year)
            candidate = next(item for item in plan["candidates"] if item["candidate_id"] == candidate_id)
        except StopIteration:
            raise fail("INVALID_EXPERIMENT_SPEC", "trial is outside the sealed search plan") from None
        key = content_hash([plan["plan_id"], outer_year, inner_year, candidate_id])
        return key, inner, candidate

    def _inputs(self, inner):
        binding = self.registration.document
        snapshot = from_document(SnapshotRef, binding["snapshot"])
        month = binding["holdouts"]["holdout_as_of_month"]
        history = history_binding(self.repository, snapshot, binding["magnitude_plan"],
            binding["magnitude_history_request"], as_of_month=month, event_ids=binding["event_ids"])
        if history != binding["magnitude_history"]:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search native history binding changed")
        participants = sorted(set(inner["training_candidate_event_ids"] + inner["evaluation_event_ids"]))
        preflight = preflight_magnitude_targets(self.repository, snapshot, as_of_month=month, event_ids=participants)
        stored = read_stored(self.repository, from_document(SnapshotRef, history["feature_snapshot"]), history["context"])
        train = qualify_magnitude_training_targets(self.repository, snapshot, as_of_month=month,
            labels_before=inner["training_labels_before"], event_ids=inner["training_candidate_event_ids"])
        return {"preflight": preflight, "stored_inputs_id": stored["inputs_id"], "training_inputs": dict(train.attrs),
                "training": train[["event_id", "absolute_move_pct"]].to_dict("records"),
                "features": [row for row in stored["rows"] if row["event_id"] in participants]}

    def start(self, outer_year, inner_year, candidate_id):
        """Claim once, then bind native inputs; interrupted starts cannot be restarted."""
        plan = self._admit()
        key, inner, candidate = self._trial(plan, outer_year, inner_year, candidate_id)
        started = {"trial_id": key, "outer_year": outer_year, "inner_year": inner_year, "candidate": candidate}
        self._append("start:" + key, started, replay=False)
        try:
            inputs = self._inputs(inner)
            cohort = content_hash([plan["plan_id"], outer_year, inner_year])
            self._append("cohort:" + cohort, inputs)
            self._append("inputs:" + key, {"cohort": cohort, "inputs_id": content_hash(inputs)})
        except (OpsError, DataError) as error:
            self._append("terminal:" + key, {"status": "refused", "problem": _problem(error.problem)})
            raise
        return {**started, "inputs": inputs}

    @staticmethod
    def _forecasts(forecasts, inner, horizons):
        expected = {(event, horizon) for event in inner["evaluation_event_ids"] for horizon in horizons}
        try:
            if not isinstance(forecasts, list) or len(forecasts) != len(expected):
                raise ValueError
            found = {}
            for row in forecasts:
                if not isinstance(row, dict) or set(row) != {"event_id", "horizon_sessions", "predicted_absolute_move_pct"}:
                    raise ValueError
                pair = row["event_id"], row["horizon_sessions"]
                value = row["predicted_absolute_move_pct"]
                if (type(pair[1]) is not int or pair not in expected or pair in found
                        or type(value) not in (float, int) or not math.isfinite(value) or value < 0):
                    raise ValueError
                found[pair] = row
            return json.loads(canonical_json([found[pair] for pair in sorted(expected)]))
        except (ValueError, TypeError, OverflowError):
            raise fail("INVALID_EXPERIMENT_SPEC", "trial forecasts require complete finite event-horizon coverage") from None

    def record(self, outer_year, inner_year, candidate_id, *, forecasts=None, failure=None):
        """Commit submitted forecasts before any validation label read; never accept scores."""
        plan = self._admit()
        key, inner, _ = self._trial(plan, outer_year, inner_year, candidate_id)
        if self._read("start:" + key) is None:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "trial must be started before recording")
        if (forecasts is None) == (failure is None):
            raise fail("INVALID_EXPERIMENT_SPEC", "record exactly one forecast submission or failure")
        if failure is not None:
            return self._append("terminal:" + key, {"status": "failed", "problem": _problem(failure)},
                                absent=("forecasts:" + key,))
        submitted = self._forecasts(forecasts, inner, plan["horizons"])
        terminal = self._read("terminal:" + key)
        if terminal is not None:
            if self._read("forecasts:" + key) != submitted:
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "terminal trial cannot accept another submission")
            return self._terminal(plan, outer_year, inner_year, candidate_id)[1]
        admitted = self._read("inputs:" + key)
        if admitted is None:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "trial has no admitted native inputs")
        self._append("forecasts:" + key, submitted, replay=False, absent=("terminal:" + key,))
        try:
            if content_hash(self._inputs(inner)) != admitted["inputs_id"]:
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "trial native inputs changed after start")
            binding = self.registration.document
            labels = load_magnitude_targets(self.repository, from_document(SnapshotRef, binding["snapshot"]),
                as_of_month=binding["holdouts"]["holdout_as_of_month"], labels_before=inner["evaluation_labels_before"],
                event_ids=inner["evaluation_event_ids"])
            targets = dict(zip(labels.event_id, labels.absolute_move_pct))
            scored = [{**row, "absolute_move_pct": float(targets[row["event_id"]])} for row in submitted]
            terminal = {"status": "succeeded", "inputs_id": admitted["inputs_id"],
                        "evaluation_inputs": dict(labels.attrs), "predictions": scored, "metrics": _metrics(scored)}
        except (OpsError, DataError) as error:
            self._append("terminal:" + key, {"status": "refused", "problem": _problem(error.problem)})
            raise
        return self._append("terminal:" + key, terminal)

    def _terminal(self, plan, outer_year, inner_year, candidate_id):
        key, inner, candidate = self._trial(plan, outer_year, inner_year, candidate_id)
        record = self._read("terminal:" + key)
        if record is None:
            raise fail("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt")
        try:
            expected = {"trial_id": key, "outer_year": outer_year, "inner_year": inner_year, "candidate": candidate}
            if self._read("start:" + key) != expected:
                raise ValueError
            if record["status"] == "succeeded":
                admitted = self._read("inputs:" + key)
                cohort = self._read("cohort:" + admitted["cohort"])
                submitted = self._forecasts(self._read("forecasts:" + key), inner, plan["horizons"])
                forecasts = [{name: row[name] for name in ("event_id", "horizon_sessions", "predicted_absolute_move_pct")}
                             for row in record["predictions"]]
                if (cohort is None or content_hash(cohort) != admitted["inputs_id"]
                        or record["inputs_id"] != admitted["inputs_id"] or forecasts != submitted
                        or record["metrics"] != _metrics(record["predictions"])):
                    raise ValueError
            elif record["status"] not in ("failed", "refused"):
                raise ValueError
        except (ValueError, TypeError, KeyError):
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
                scores.append({"candidate_id": candidate["candidate_id"], "metrics": None if failed else _metrics(rows)})
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
