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

__all__ = ["DistributionFits"]


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
        return plan["recipe"], claim, matrices, labels, evidence["evaluation_event_ids"]

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

    def _read(self, prepared, record):
        recipe, claim, matrices, labels, events = prepared
        _require(canonical_json(self._record("fit-start:" + claim["trial"]["trial_id"])) == canonical_json(claim))
        _require(type(record) is dict and set(record) == {"schema_version", "claim", "result_id",
            "forecasts", "tail", "diagnostics", "status", "final_read_authority"}
            and record["schema_version"] == "distribution_fitted.v1" and record["status"] == "completed"
            and record["final_read_authority"] == "none" and canonical_json(record["claim"]) == canonical_json(claim))
        _require(record["result_id"] == content_hash({key: value for key, value in record.items() if key != "result_id"}))
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

    def fit(self, role, outer_year, horizon, inner_year=None):
        """Claim before either estimator; completed replay is read-only, interruption final."""
        forbid_legacy_fitting("DistributionFits.fit")
        forbid_fitting("DistributionFits.fit")
        prepared = self._context(role, outer_year, horizon, inner_year)
        recipe, claim, matrices, labels, events = prepared
        key = claim["trial"]["trial_id"]
        previous = self._record("fit-result:" + key)
        if previous is not None:
            return self._read(prepared, previous)
        self._record("fit-start:" + key, claim)
        result = fit_candidate(recipe, matrices[0], labels, matrices[1], columns=recipe["point_recipe"]["features"])
        _require(result["schema_version"] == "distribution_candidate_fit.v1" and result["recipe_id"] == recipe["recipe_id"])
        _require(len(result["probabilities"]) == len(result["point_predictions"]) == len(events))
        rows = [{"event_id": event, "probabilities": masses, "point_prediction": point}
                for event, masses, point in zip(events, result["probabilities"], result["point_predictions"])]
        self._validate(recipe, matrices, labels, events, rows, result["tail"], result["diagnostics"])
        with stage("distribution fitted publication"):
            forecasts = write_section(self.trials.store, "distribution_fit_forecasts",
                _tracked("distribution forecast publication", sorted(rows, key=_event), len(rows)), _event)
            tail = {**result["tail"], "excesses": write_section(self.trials.store, "distribution_fit_tail",
                _tracked("distribution tail publication", ({"position": index, "excess": value}
                    for index, value in enumerate(result["tail"]["excesses"])), result["tail"]["sample_count"]), _position)}
        body = {"schema_version": "distribution_fitted.v1", "claim": claim, "forecasts": forecasts, "tail": tail,
                "diagnostics": result["diagnostics"], "status": "completed", "final_read_authority": "none"}
        record = self._record("fit-result:" + key, {**body, "result_id": content_hash(body)})
        return {"receipt": record, "rows": rows, "tail": result["tail"]}

    def read(self, role, outer_year, horizon, inner_year=None):
        """Verify immutable completion; never fall back to fitting or native outcomes."""
        prepared = self._context(role, outer_year, horizon, inner_year)
        return self._read(prepared, self._record("fit-result:" + prepared[1]["trial"]["trial_id"]))
