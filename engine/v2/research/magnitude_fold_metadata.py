"""Public metadata-only access to the unchanged native annual fold rule."""
from . import magnitude_folds


def annual_fold(year, events):
    """Group admitted event metadata; preserve native population-collapse refusals."""
    return magnitude_folds._fold(year, events)
