"""R4-20 gap 1/2: frozen gate champions must see source_features base
columns (im_d1/im_d5/im_d10, ...) and entry_cost_pct before they execute.

Gap 1 -- ``_facts`` used to merge ``inputs.context``/``inputs.features``/the
flattened driver ``model_inputs``, but never ``inputs.features.get(
"source_features")`` (the captured base feature columns
``tools/capture_tier0_corpus.py`` writes). A frozen gate naming e.g.
``im_d5`` in its ``feature_order`` could never see it and always refused
MISSING_FEATURES.

Gap 2 -- ``entry_cost_pct`` is a required feature of the STR-THRU/STR-RUNUP
gate champions, but native only ever computed it post-hoc in
``financial.financial_diagnostics`` (``application.py``), long after the
gate stage in ``_append_late_stages`` had already run.
"""
import hashlib
import json

from engine.v2.models import FrozenInference, ModelBinding, ModelRelease
from engine.v2.models.contracts import ArtifactMember
from engine.v2.scoring.frozen_executor import FrozenStageExecutor
from engine.v2.scoring.native_analog import source_population_hash
from engine.v2.scoring.stages import (
    NativeScoreInputs,
    STAGE_NAMES,
    StageReceipt,
    assemble_native_values,
)

_RECEIPTS = tuple(
    StageReceipt(stage, "declared-input", "declared-output")
    for stage in STAGE_NAMES if stage != "diagnostics"
)
_QUOTES = {
    ("C", 100.0, "2026-09-18"): {"bid": 1.9, "ask": 2.1},
    ("P", 100.0, "2026-09-18"): {"bid": 1.9, "ask": 2.1},
}


def _inputs(*, source_features=None, model_inputs=None, gate=None, chooser=None,
           analogs=None, role_model_inputs=None, context_overrides=None) -> NativeScoreInputs:
    context = {
        "ticker": "AAA",
        "strategy": "STR-THRU",
        "event_date": "2026-09-16",
        "entry_date": "2026-09-16",
        "exit_date": "2026-09-18",
        "expiry": "2026-09-18",
        "strike": 100.0,
        "spot": 100.0,
        "quotes": dict(_QUOTES),
    }
    context.update(context_overrides or {})
    forecast = {
        "driver_name": "abs_move",
        "models": {"driver_prediction": {"intercept": 0.0, "coefficients": {}}},
    }
    features = {"model_inputs": dict(model_inputs or {})}
    if source_features is not None:
        features["source_features"] = dict(source_features)
    if role_model_inputs is not None:
        features["role_model_inputs"] = dict(role_model_inputs)
    return NativeScoreInputs(
        context=context,
        features=features,
        forecast=forecast,
        geometry=None,
        pricing=None,
        analogs=analogs if analogs is not None else {"recipe": None},
        simulation={"mode": "not_applicable"},
        gate=gate if gate is not None else {"mode": "not_applicable"},
        chooser=chooser or {},
        diagnostics={},
        source_ref="gate-input-gaps-fixture",
        stage_receipts=_RECEIPTS,
    )


class _FrozenGate:
    """Adapts a bare ``FrozenStageExecutor`` to the ``executors["gate_score"]``
    contract ``_execute_gate_executor`` expects: ``.feature_order`` and
    ``.predict(facts) -> Mapping``. Production code gets this from
    ``FrozenRecipeExecutor``; this is the same interface, built directly on
    the binding for a self-contained test.
    """

    def __init__(self, executor: FrozenStageExecutor, binding: ModelBinding) -> None:
        self._executor = executor
        self.feature_order = binding.feature_order

    def predict(self, features):
        return self._executor.predict(features)


def _frozen_gate(tmp_path, feature_order: tuple[str, ...], *, role="gate",
                 output_name="gate_score", coefficients=None) -> dict:
    """One linear gate artifact naming exactly ``feature_order``."""
    payload = {
        "schema_version": "linear_estimator.v1.0",
        "feature_order": list(feature_order),
        "outputs": [
            {
                "name": output_name,
                "intercept": 0.1,
                "coefficients": [float((coefficients or {}).get(name, 0.0))
                                 for name in feature_order],
            },
        ],
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    (tmp_path / "estimator.json").write_bytes(raw)
    artifact_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
    binding = ModelBinding(
        binding_id="gate-binding",
        model_id="gate-v1",
        role=role,
        strategy_id="STR-THRU",
        decision_clock_id="entry-close",
        adapter="json-linear.v1",
        feature_order=feature_order,
        output_names=(output_name,),
        members=(
            ArtifactMember(
                name="estimator", path="estimator.json",
                content_hash=artifact_hash,
            ),
        ),
    )
    release = ModelRelease(
        release_id="release-gate-v1",
        deployment_id="deployment-gate-v1",
        bindings=(binding,),
    )
    inference = FrozenInference(tmp_path)
    executor = FrozenStageExecutor(
        inference=inference, release=release, binding_id=binding.binding_id,
    )
    gate_executor = _FrozenGate(executor, binding)
    return {"executors": {output_name: gate_executor}, "threshold": 0.0}


# --- Gap 1: source_features must reach the gate ---------------------------

def test_nonfinite_source_feature_gives_missing_features(tmp_path):
    gate = _frozen_gate(tmp_path, ("im_d5",))
    values = assemble_native_values(_inputs(
        source_features={"im_d5": float("nan")}, gate=gate,
    ))
    assert "MISSING_FEATURES" in values["flags"]
    assert values.get("gate_score") is None


def test_derived_gate_analog_overrides_conflicting_captured_role_value(tmp_path):
    rows = [{"row_id": "analog-1", "features": {"move": 1.0},
             "realized_pnl": 0.3}]
    recipe = {"feature_names": ("move",), "neighbors": 1,
              "population_hash": source_population_hash(rows)}
    gate = _frozen_gate(tmp_path, ("analog_n",),
                        coefficients={"analog_n": 1.0})
    values = assemble_native_values(_inputs(
        gate=gate,
        analogs={"source_rows": rows, "query_features": {"move": 1.0},
                 "recipe": recipe},
        role_model_inputs={"gate": {"analog_n": 999.0}},
    ))
    assert values["n_analogs"] == 1
    assert values["gate_score"] == 1.1


def test_derived_chooser_value_overrides_conflicting_captured_role_value(
    tmp_path, monkeypatch,
):
    import engine.v2.scoring.native_chooser as native_chooser

    monkeypatch.setattr(
        native_chooser, "derive_chooser_columns",
        lambda *args, **kwargs: {"analog_n": 5.0},
    )
    chooser = _frozen_gate(
        tmp_path, ("analog_n",), role="chooser", output_name="chooser_score",
        coefficients={"analog_n": 1.0},
    )
    values = assemble_native_values(_inputs(
        chooser=chooser,
        role_model_inputs={"chooser": {"analog_n": 999.0}},
    ))
    assert values["chooser_score"] == 5.1


def test_finite_source_feature_lets_the_gate_execute(tmp_path):
    gate = _frozen_gate(tmp_path, ("im_d5",))
    values = assemble_native_values(_inputs(
        source_features={"im_d5": 3.5}, gate=gate,
    ))
    assert "MISSING_FEATURES" not in values["flags"]
    assert values.get("gate_score") is not None


def test_source_feature_absent_entirely_gives_missing_features(tmp_path):
    """Before the fix, ``source_features`` was never merged at all -- the
    key was ABSENT, not merely non-finite. Same outcome, different cause."""
    gate = _frozen_gate(tmp_path, ("im_d5",))
    values = assemble_native_values(_inputs(gate=gate))
    assert "MISSING_FEATURES" in values["flags"]


# --- Gap 2: entry_cost_pct must be computed before the gate runs ----------

def test_entry_cost_pct_available_to_the_gate(tmp_path):
    gate = _frozen_gate(tmp_path, ("entry_cost_pct",))
    values = assemble_native_values(_inputs(gate=gate))
    assert "MISSING_FEATURES" not in values["flags"]
    assert values.get("gate_score") is not None
    assert values.get("entry_cost_pct") is not None
