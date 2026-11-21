"""Closed, policy-owned classification of a failed search fit; never free-form caller text."""
from engine.v2.data.errors import DataError
from engine.v2.ops.errors import OpsError

FAILURE_REASONS = ("nonconvergence", "invalid_prediction", "value_error", "arithmetic_error", "data_error", "other")


def classify(error):
    """Map a fit exception to one registered reason using type and policy-owned messages only."""
    if isinstance(error, OpsError):
        text = error.problem.message
    else:
        text = str(error)
    if "convergence failed" in text:
        return "nonconvergence"
    if text == "forecast validation failed":
        return "invalid_prediction"
    if isinstance(error, ArithmeticError):
        return "arithmetic_error"
    if isinstance(error, ValueError):
        return "value_error"
    if isinstance(error, (DataError, OpsError)):
        return "data_error"
    return "other"


def valid_reason(value):
    return isinstance(value, str) and value in FAILURE_REASONS
