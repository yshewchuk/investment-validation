"""Shared score-plan identities and chooser-aware population comparison."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["population_key", "population_difference"]


def population_key(row: Mapping[str, Any]) -> str:
    """Key a planned row before its strike and expiry are known."""
    return "|".join(str(row.get(key, "")) for key in ("ticker", "strategy", "event_date"))


def population_difference(planned: Iterable[str], observed: Iterable[str]
                          ) -> tuple[list[str], list[str]]:
    """Return missing and unplanned keys, allowing derived chooser rows.

    ``score_calendar`` appends DYN-SV only when it ranks a planned event's
    menu. An explicitly planned chooser key still has to be observed.
    """
    expected, actual = set(planned), set(observed)
    events = {(parts[0], parts[2]) for parts in (key.split("|") for key in expected
                                              if isinstance(key, str))
              if len(parts) == 3 and parts[1] != "DYN-SV"}

    def derived(key: str) -> bool:
        parts = key.split("|")
        return len(parts) == 3 and parts[1] == "DYN-SV" and (parts[0], parts[2]) in events

    return sorted(expected - actual), sorted(key for key in actual - expected if not derived(key))
