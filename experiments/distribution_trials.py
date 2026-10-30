"""Immutable distribution-only admission journal; no fitting or completed trials.

Reuses native registration, fold admission, CAS shards and journal migrations.
This library is not an executable runner or a new holdout/registration framework.
"""
from copy import deepcopy

from engine.v2.data.errors import DataError
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.migrations import migrate, require_current
from engine.v2.research.progress import Progress, stage
from experiments.distribution_candidate_recipe import trial_identity
from experiments.distribution_plan import prepare_plan
from experiments.magnitude_search_inputs import load_fit_inputs, publish_inputs
from experiments.magnitude_search_records import MIGRATIONS, OWNER, append_record, read_record
from experiments.size_search_binding import fold_inputs

__all__ = ["DistributionTrials"]


def _require(condition):
    if not condition:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution admission evidence differs")


class DistributionTrials:
    """Own one seal and single-use fold claims; replay verifies stored admission only."""

    def __init__(self, conn, store, repository, own_registration, source_registration,
                 source_request, source_plan_ref, recipe, budget_caps):
        self.conn, self.store, self.repository = conn, store, repository
        self.own, self.source = own_registration, source_registration
        self.request, self.recipe, self.caps = deepcopy((source_request, recipe, budget_caps))
        self.source_plan_ref = source_plan_ref

    def _plans(self):
        with stage("distribution plan verification"):
            return prepare_plan(self.conn, self.store, self.own, self.source,
                                self.request, self.source_plan_ref, self.recipe, self.caps)

    def _read(self, slot):
        return read_record(self.conn, self.store, self.own.run_id, "distribution:" + slot)

    def _append(self, slot, body, **kwargs):
        return append_record(self.conn, self.store, self.own.run_id, "distribution:" + slot, body, **kwargs)

    def seal(self, *, clock):
        """Verify both registrations and explicitly reserve the exact installed plan."""
        plan, _ = self._plans()
        migrate(self.conn, OWNER, MIGRATIONS, clock=clock)
        return self._append("seal", plan)

    def _admit(self):
        require_current(self.conn, OWNER, MIGRATIONS)
        plan, native = self._plans()
        _require(canonical_json(self._read("seal")) == canonical_json(plan))
        return plan, native

    def _select(self, native, role, outer_year, inner_year):
        if (type(role) is not str or role not in ("inner", "outer")
                or type(outer_year) is not str
                or (role == "inner" and type(inner_year) is not str)
                or (role == "outer" and inner_year is not None)):
            raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution fold selection")
        try:
            group = next(group for group in native["folds"] if group["outer"]["year"] == outer_year)
            fold = (next(fold for fold in group["inner"] if fold["year"] == inner_year)
                    if role == "inner" else {**group["outer"], "evaluation_labels_before":
                        self.source.document["magnitude_execution"]["evaluation_labels_before"]})
        except StopIteration:
            raise fail("INVALID_EXPERIMENT_SPEC", "distribution fold is outside the sealed plan") from None
        key = content_hash({"fold": fold, "role": role, "outer_year": outer_year})
        return fold, {"fold_id": key, "role": role, "outer_year": outer_year, "inner_year": inner_year}

    @staticmethod
    def _receipt(plan, claim, manifest, inputs_id):
        trials = [trial_identity(plan["recipe"], context_id=plan["context_id"], fold_id=claim["fold_id"],
            inputs_id=inputs_id, budget_id=plan["budget"]["budget_id"], horizon=horizon, role=claim["role"])
            for horizon in plan["recipe"]["horizons"]]
        return {"schema_version": "distribution_admission.v1", **claim, "inputs_id": inputs_id,
                "manifest": manifest, "trials": trials, "status": "admitted", "final_read_authority": "none"}

    def start(self, role, outer_year, inner_year=None):
        """Claim before native training reads; interrupted claims can never restart."""
        plan, native = self._admit()
        fold, selected = self._select(native, role, outer_year, inner_year)
        if not callable(getattr(self.repository, "resolve", None)):
            raise fail("INVALID_EXPERIMENT_SPEC", "distribution admission requires a repository snapshot resolver")
        claim = {"plan_id": plan["plan_id"], **selected}
        key = claim["fold_id"]
        self._append("start:" + key, claim, replay=False)
        try:
            with stage("distribution native fold admission"):
                inputs = fold_inputs(self.repository, self.source.document, fold)
                columns = ("event_id", "context_id", "horizon_sessions", *plan["recipe"]["point_recipe"]["features"])
                projected = []
                progress = Progress("distribution feature projection", len(inputs["features"]))
                for row in inputs["features"]:
                    projected.append({name: row[name] for name in columns})
                    progress.tick()
                inputs = {**inputs, "features": projected}
                manifest, inputs_id = publish_inputs(self.store, inputs)
            receipt = self._receipt(plan, claim, manifest, inputs_id)
            self._append("admission:" + key, receipt)
        except (OpsError, DataError) as error:
            if error.problem.retryable:
                raise fail("EXPERIMENT_IDENTITY_CONFLICT",
                           "distribution admission interrupted after its single-use claim") from None
            raise
        return deepcopy(receipt)

    def read_admission(self, role, outer_year, inner_year=None):
        """Verify seal, unique claim and CAS inputs; return no labels or fit completion."""
        plan, native = self._admit()
        fold, selected = self._select(native, role, outer_year, inner_year)
        claim = {"plan_id": plan["plan_id"], **selected}
        _require(canonical_json(self._read("start:" + claim["fold_id"])) == canonical_json(claim))
        record = self._read("admission:" + claim["fold_id"])
        _require(isinstance(record, dict) and {"manifest", "inputs_id"} <= set(record))
        inputs_id = record["inputs_id"]
        with stage("distribution stored admission verification"):
            load_fit_inputs(self.store, record["manifest"], inputs_id, fold["evaluation_event_ids"],
                            plan["recipe"]["horizons"], plan["recipe"]["point_recipe"]["features"])
        expected = self._receipt(plan, claim, record["manifest"], inputs_id)
        _require(canonical_json(record) == canonical_json(expected))
        return expected
