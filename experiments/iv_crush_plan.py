"""Metadata-only IV-crush temporal planning; no registration, label reads or fits."""
from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT
from engine.v2.foundation import content_hash
from engine.v2.models.regression_target import target_descriptor
from engine.v2.research.magnitude_folds import plan_regression_folds


def target():
    """Declare signed relative IV30 change, not fixed-contract or fixed-expiry IV."""
    definition = {"schema_version": "iv_crush_relative_iv30.v1",
        "formula": "100*(post_iv30/pre_iv30-1)", "units": "percent_relative_change",
        "contract_hash": IV_CRUSH_CONTRACT.definition_hash,
        "availability_basis": "modeled_post_close_plus_one_calendar_day"}
    return target_descriptor(column="crush_pct_iv30", prediction_column="predicted_crush_pct_iv30",
                             domain="signed", definition_id=content_hash(definition))


def plan(repository, snapshot, *, as_of_month, evaluation_years, training_start, event_ids=None):
    """Use admitted metadata/calendar only; the snapshot is not an IV-target pin."""
    return plan_regression_folds(repository, snapshot, target=target(), as_of_month=as_of_month,
        evaluation_years=evaluation_years, training_start=training_start, event_ids=event_ids)
