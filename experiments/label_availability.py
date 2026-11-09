"""Shared label-end availability checks for experiment fold admission."""
from datetime import date, timedelta


def available_before(label_end, cutoff):
    return (date.fromisoformat(label_end) + timedelta(days=1)).isoformat() < cutoff


def label_cutoff_reason(label_end, cutoff):
    if label_end >= cutoff:
        return "label_end_at_or_after_cutoff"
    if not available_before(label_end, cutoff):
        return "label_available_at_or_after_cutoff"
    return None
