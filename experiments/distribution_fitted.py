"""Admitted, immutable one-horizon distribution/point fits; no job containment.

Only installed numeric fitting can publish completion. Replay verifies source
admission, ordered matrix identity and CAS results without refitting or outcomes.
"""
from math import isfinite, isnan

from engine.models.no_fit import forbid_fitting as forbid_legacy_fitting
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.ops.errors import fail
from engine.v2.research.magnitude_distribution import bin_index, histogram
from engine.v2.research.progress import Progress, stage
from experiments.distribution_candidate_fit import fit_candidate
from experiments.distribution_class_support import training_support
from experiments.magnitude_search_evidence import fit_evidence
from experiments.magnitude_search_records import append_record, read_record
from experiments.magnitude_search_shards import iter_section, write_section

__all__ = ["DistributionFits", "validate_fit_output", "publish_fit_output"]


def _require(condition):
    if not condition:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution fitted evidence differs")


def _event(row):
    return (row["event_id"],)


def _position(row):
    return (row["position"],)


def _tracked(name, rows, total):
    progress = Progress(name, total)
    for row in rows:
        yield row
        progress.tick()


def validate_fit_output(recipe, matrices, labels, events, result):
    """Validate installed numeric output before either caller publishes CAS rows."""
    _require(result["schema_version"] == "distribution_candidate_fit.v1" and result["recipe_id"] == recipe["recipe_id"])
    _require(len(result["probabilities"]) == len(result["point_predictions"]) == len(events))
    rows = [{"event_id": event, "probabilities": masses, "point_prediction": point}
            for event, masses, point in zip(events, result["probabilities"], result["point_predictions"])]
    DistributionFits._validate(recipe, matrices, labels, events, rows, result["tail"], result["diagnostics"])
    return rows


def publish_fit_output(store, rows, tail, diagnostics):
    """Publish bounded result sections only; no admission or journal authority."""
    with stage("distribution fitted publication"):
        forecasts = write_section(store, "distribution_fit_forecasts",
            _tracked("distribution forecast publication", sorted(rows, key=_event), len(rows)), _event)
        tail = {**tail, "excesses": write_section(store, "distribution_fit_tail",
            _tracked("distribution tail publication", ({"position": index, "excess": value}
                for index, value in enumerate(tail["excesses"])), tail["sample_count"]), _position)}
    return {"forecasts": forecasts, "tail": tail, "diagnostics": diagnostics}


class DistributionFits:
    """Wrap native-admitted trials; one attempt per horizon, receipt-only replay."""

    def __init__(self, trials):
        self.trials = trials

    def _record(self, slot, body=None):
        args = (self.trials.conn, self.trials.store, self.trials.own.run_id, "distribution:" + slot)
        return read_record(*args) if body is None else append_record(*args, body, replay=False)

    def _context(self, role, outer_year, horizon, inner_year):
        context = self.trials.read_fit_context(role, outer_year, inner_year)
        plan, fold, admission, inputs = (context[key] for key in ("plan", "fold", "admission", "inputs"))
        if type(horizon) is not int or horizon not in plan["recipe"]["horizons"]:
            raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution fit horizon")
        trial = next(item for item in admission["trials"] if item["horizon"] == horizon)
        columns = plan["recipe"]["point_recipe"]["features"]
        evidence = fit_evidence(inputs, fold, {"horizons": [horizon]}, {"features": columns})[0]
        claim = {"plan_id": plan["plan_id"], "admission_id": content_hash(admission), "trial": trial,
            "role": role, "outer_year": outer_year, "inner_year": inner_year,
            "evidence": {key: evidence[key] for key in
                ("horizon_sessions", "training_matrix_id", "prediction_matrix_id")},
            "ordered_events_id": content_hash([evidence["training_event_ids"], evidence["evaluation_event_ids"]])}
        indexed = {row["event_id"]: row for row in inputs["features"] if row["horizon_sessions"] == horizon}
        matrices = [[[indexed[event][name] for name in columns] for event in events]
                    for events in (evidence["training_event_ids"], evidence["evaluation_event_ids"])]
        labels = [row["absolute_move_pct"] for row in inputs["training"]]
        return (plan["recipe"], claim, matrices, labels, evidence["evaluation_event_ids"],
                {"admission": admission, "plan": plan})

    @staticmethod
    def _validate(recipe, matrices, labels, events, rows, tail, diagnostics):
        _require(len(rows) == len(events) and bool(rows)
                 and all(type(row) is dict and set(row) == {"event_id", "probabilities", "point_prediction"} for row in rows)
                 and [row["event_id"] for row in rows] == events)
        counts = [0] * 81
        for label in labels:
            counts[bin_index(float(label))] += 1
        support = training_support(counts)
        progress = Progress("distribution forecast verification", len(rows))
        for row in rows:
            histogram(row["probabilities"])
            _require(all(mass == 0 for mass, count in zip(row["probabilities"], counts) if not count))
            point = row["point_prediction"]
            _require(type(point) in (int, float) and 0 <= point <= 1.7976931348623157e308 and isfinite(point))
            progress.tick()
        expected_tail = histogram(rows[0]["probabilities"], tail_excesses=[float(y) - 20 for y in labels if y >= 20] or None)["tail"]
        _require(canonical_json(tail) == canonical_json(expected_tail))
        _require(type(diagnostics) is dict and type(diagnostics.get("classifier")) is dict)
        classes = diagnostics["classifier"].get("class_labels")
        _require(type(classes) is list and all(type(value) is int for value in classes)
                 and sorted(classes) == support["observed_class_labels"])
        iterations, trees = recipe["classifier"]["params"]["max_iter"], support["trees_per_iteration"]
        expected = {"training_support": support, "classifier": {"class_labels": classes,
            "iterations": iterations, "trees_per_iteration": trees, "total_trees": iterations * trees},
            "point": {"iterations_declared": recipe["point_recipe"]["estimator"]["params"]["max_iter"],
                      "trees_per_iteration_declared": 1}, "training_rows": len(labels), "prediction_rows": len(events),
            "all_null_training_columns": [name for index, name in enumerate(recipe["point_recipe"]["features"])
                if all(row[index] is None or isnan(row[index]) for row in matrices[0])]}
        _require(canonical_json(diagnostics) == canonical_json(expected))

    @staticmethod
    def _execution_binding(prepared):
        return {"plan_id": prepared[1]["plan_id"], "worker": "experiments/distribution_worker.py",
            "limits": prepared[5]["plan"]["budget"]["child_limits"]}

    def _read(self, prepared, record, *, bounded=False):
        recipe, claim, matrices, labels, events = prepared[:5]
        _require(canonical_json(self._record("fit-start:" + claim["trial"]["trial_id"])) == canonical_json(claim))
        _require(type(record) is dict)
        is_bounded = record.get("schema_version") == "distribution_fitted.v2"
        _require(set(record) == {"schema_version", "claim", "result_id",
            "forecasts", "tail", "diagnostics", "status", "final_read_authority"} | ({"execution"} if is_bounded else set())
            and record["schema_version"] in ("distribution_fitted.v1", "distribution_fitted.v2") and record["status"] == "completed"
            and record["final_read_authority"] == "none" and canonical_json(record["claim"]) == canonical_json(claim))
        _require(record["result_id"] == content_hash({key: value for key, value in record.items() if key != "result_id"}))
        _require(not bounded or is_bounded)
        binding = self._execution_binding(prepared) if is_bounded else None
        _require(canonical_json(self._record("fit-bound:" + claim["trial"]["trial_id"])) == canonical_json(binding))
        if is_bounded:
            _require(canonical_json(record["execution"]) == canonical_json(binding))
        descriptor = record["tail"]
        _require(type(descriptor) is dict and set(descriptor) == {"policy", "sample_count", "excesses"})
        _require(type(descriptor["sample_count"]) is int and 0 <= descriptor["sample_count"] <= len(labels))
        with stage("distribution fitted replay"):
            samples = list(_tracked("distribution tail replay", iter_section(self.trials.store,
                "distribution_fit_tail", descriptor["excesses"], _position), descriptor["sample_count"]))
            _require(all(set(row) == {"position", "excess"} and type(row["position"]) is int
                         and row["position"] == index for index, row in enumerate(samples)))
            tail = {**descriptor, "excesses": [row["excess"] for row in samples]}
            stored = list(_tracked("distribution forecast replay", iter_section(self.trials.store,
                "distribution_fit_forecasts", record["forecasts"], _event), len(events)))
            _require([row["event_id"] for row in stored] == sorted(events))
            indexed = {row["event_id"]: row for row in stored}
            rows = [indexed[event] for event in events]
            self._validate(recipe, matrices, labels, events, rows, tail, record["diagnostics"])
        return {"receipt": record, "rows": rows, "tail": tail}

    def fit(self, role, outer_year, horizon, inner_year=None, *, bounded=False):
        """Claim before either estimator; completed replay is read-only, interruption final."""
        forbid_legacy_fitting("DistributionFits.fit")
        forbid_fitting("DistributionFits.fit")
        if type(bounded) is not bool:
            raise fail("INVALID_EXPERIMENT_SPEC", "distribution bounded option must be Boolean")
        prepared = self._context(role, outer_year, horizon, inner_year)
        recipe, claim, matrices, labels, events = prepared[:5]
        key = claim["trial"]["trial_id"]
        previous = self._record("fit-result:" + key)
        if previous is not None:
            return self._read(prepared, previous, bounded=bounded)
        binding = self._execution_binding(prepared) if bounded else None
        self._record("fit-start:" + key, claim)
        if bounded:
            self._record("fit-bound:" + key, binding)
            from experiments.distribution_execution import run_distribution_fit
            admission = prepared[5]["admission"]
            output = run_distribution_fit({"manifest": admission["manifest"], "inputs_id": claim["trial"]["inputs_id"],
                "store_root": str(self.trials.store.root)}, recipe, events, horizon, prepared[5]["plan"]["budget"]["child_limits"])
            _require(type(output) is dict and set(output) == {"recipe_id", "evidence", "ordered_events_id",
                "forecasts", "tail", "diagnostics"} and output["recipe_id"] == recipe["recipe_id"]
                and output["evidence"] == claim["evidence"] and output["ordered_events_id"] == claim["ordered_events_id"])
            body = {"schema_version": "distribution_fitted.v2", "claim": claim, "execution": binding,
                **{key: output[key] for key in ("forecasts", "tail", "diagnostics")},
                "status": "completed", "final_read_authority": "none"}
            record = {**body, "result_id": content_hash(body)}
            verified = self._read(prepared, record)
            self._record("fit-result:" + key, record)
            return verified
        result = fit_candidate(recipe, matrices[0], labels, matrices[1], columns=recipe["point_recipe"]["features"])
        rows = validate_fit_output(recipe, matrices, labels, events, result)
        output = publish_fit_output(self.trials.store, rows, result["tail"], result["diagnostics"])
        body = {"schema_version": "distribution_fitted.v1", "claim": claim, **output, "status": "completed", "final_read_authority": "none"}
        record = self._record("fit-result:" + key, {**body, "result_id": content_hash(body)})
        return {"receipt": record, "rows": rows, "tail": result["tail"]}

    def read(self, role, outer_year, horizon, inner_year=None):
        """Verify immutable completion; never fall back to fitting or native outcomes."""
        prepared = self._context(role, outer_year, horizon, inner_year)
        return self._read(prepared, self._record("fit-result:" + prepared[1]["trial"]["trial_id"]))
