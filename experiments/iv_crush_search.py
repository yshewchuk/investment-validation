"""Thin IV registration and sealed-plan policy; no search/CLI execution."""
from pathlib import Path

from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT, IV_CRUSH_TABLE_NAME
from engine.v2.ops.errors import fail
from engine.v2.ops.experiments import ExperimentSpec, resolve_experiment_plan
from engine.v2.research.magnitude_panel_qualified_table import CONTRACT as PANEL_CONTRACT
from engine.v2.research.snapshot_inputs import snapshot_reference
from experiments.iv_crush_plan import target
from experiments.iv_crush_registration import validate_registration as validate_registration
from experiments.native_registration import prepare_regression_registration


def prepare(repository, spec, *, source, feature_snapshot, target_snapshot, scope, as_of_month,
            training_start, labels_before, event_ids=None):
    """Pin committed feature/target tables; later fold admission owns payload qualification."""
    if not isinstance(spec, ExperimentSpec):
        raise fail("INVALID_EXPERIMENT_SPEC", "regression registration requires an experiment specification")
    resolve_experiment_plan(spec)
    if (spec.runner != "experiments/iv_crush_search.py" or spec.price_source != IV_CRUSH_TABLE_NAME
            or type(spec.seed) is not int or spec.seed != 0 or spec.economic_params):
        raise fail("INVALID_EXPERIMENT_SPEC", "IV research requires its explicit runner, target, seed and no economics")
    snapshot_reference(feature_snapshot)
    snapshot_reference(target_snapshot)
    if (repository.table_contract(feature_snapshot, PANEL_CONTRACT.table_name) != PANEL_CONTRACT
            or repository.table_contract(target_snapshot, IV_CRUSH_TABLE_NAME) != IV_CRUSH_CONTRACT):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "IV research requires native feature and target tables")
    return prepare_regression_registration(repository, spec, code_root=Path(__file__).resolve().parents[1],
        source=source, feature_snapshot=feature_snapshot, target_snapshot=target_snapshot, target=target(),
        scope=scope, as_of_month=as_of_month, training_start=training_start, labels_before=labels_before,
        event_ids=event_ids)


def prepare_plan(conn, store, registration, request):
    """Supply IV target policy to the shared metadata-only search planner."""
    validate_registration(registration.document)
    from experiments.iv_crush_comparator import recipe
    from experiments.regression_search_binding import prepare_plan as prepare_regression_plan
    return prepare_regression_plan(conn, store, registration, request, target=target(), comparison=recipe())
