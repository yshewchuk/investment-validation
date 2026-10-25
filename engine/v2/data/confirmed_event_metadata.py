"""Pure per-row confirmation reasons; not population or source admission."""
import pandas as pd

from engine.v2.data import computed_moves, errors

__all__ = ["classify_metadata"]


def classify_metadata(events: pd.DataFrame) -> tuple[str | None, ...]:
    """Validate metadata and preserve every row's native confirmation reason.

    No sorting, deduplication, price access or ticker-level eligibility applies.
    The caller owns input-size bounds and source/population qualification.
    """
    if not isinstance(events, pd.DataFrame):
        raise errors.fail("CONTRACT_MISMATCH", "event metadata requires a DataFrame")
    try:
        if (not events.columns.is_unique
                or not set(computed_moves._EVENT_COLUMNS) <= set(events.columns)):
            raise errors.fail("CONTRACT_MISMATCH",
                              "event metadata columns must be unique and complete")
        computed_moves._validate_inputs(events, (), (), min_scoreable=0)
        return tuple(computed_moves._event_reasons(events))
    except (TypeError, ValueError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH",
                          "event metadata has unsupported shape or values") from None
