"""Source-free initial raw 81-bin class policy; no fitting or support provenance.

Caller-supplied counts/class labels prove neither eligibility nor causal source
admission. Absent classes retain zero mass: no smoothing, clipping or repair.
"""
from math import ceil

from engine.v2.ops.errors import fail
from engine.v2.research.magnitude_distribution import histogram

__all__ = ["training_support", "early_stopping_decision", "class_histogram"]


def training_support(bin_counts) -> dict:
    """Describe supplied class counts and binary/multiclass tree multiplicity.

    Malformed 81-bin counts refuse INVALID_EXPERIMENT_SPEC; fewer than two
    observed bins refuse EXPERIMENT_VARIANT_FAILED. No counts are inferred.
    """
    if (type(bin_counts) is not list or len(bin_counts) != 81
            or any(type(count) is not int or count < 0 for count in bin_counts)):
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution class support")
    observed = [index for index, count in enumerate(bin_counts) if count]
    if len(observed) < 2:
        raise fail("EXPERIMENT_VARIANT_FAILED", "distribution candidate requires two observed classes")
    return {"bin_counts": bin_counts.copy(), "observed_class_labels": observed,
            "sample_count": sum(bin_counts),
            "trees_per_iteration": 1 if len(observed) == 2 else len(observed)}


def early_stopping_decision(recipe, support):
    """Return the shared requested/used/fallback decision for fit and replay.

    A recipe that does not request early stopping is never used and has no
    fallback. A class below two rows cannot be stratified; otherwise the
    rounded-up validation count must be at least the observed class count.
    Absent zero-count bins are not observed classes and never trigger a
    singleton fallback.
    """
    if not recipe["classifier"]["params"]["early_stopping"]:
        return {"requested": False, "used": False, "fallback_reason": None}
    if any(support["bin_counts"][label] < 2 for label in support["observed_class_labels"]):
        return {"requested": True, "used": False, "fallback_reason": "class_count_below_2"}
    observed = len(support["observed_class_labels"])
    fraction = recipe["classifier"]["params"]["validation_fraction"]
    validation = ceil(support["sample_count"] * fraction)
    if validation < observed:
        return {"requested": True, "used": False, "fallback_reason": "validation_split_too_small"}
    return {"requested": True, "used": True, "fallback_reason": None}


def class_histogram(class_labels, probabilities, *, tail_excesses=None) -> dict:
    """Map aligned raw class probabilities to all 81 bins without sorting labels.

    Structural class/mapping errors refuse INVALID_EXPERIMENT_SPEC. The native
    histogram owns numerical probability/tail refusals and detached output;
    its DataError envelope propagates unchanged. An unresolved tail is valid.
    """
    if (type(class_labels) is not list or not 2 <= len(class_labels) <= 81
            or any(type(label) is not int or not 0 <= label <= 80 for label in class_labels)
            or len(set(class_labels)) != len(class_labels)
            or type(probabilities) is not list or len(probabilities) != len(class_labels)):
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution class mapping")
    masses = [0.0] * 81
    for label, probability in zip(class_labels, probabilities):
        masses[label] = probability
    return histogram(masses, tail_excesses=tail_excesses)
