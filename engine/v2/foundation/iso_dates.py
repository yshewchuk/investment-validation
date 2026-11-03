"""Pure-stdlib ISO date validation shared by fit transport and research inputs."""
from __future__ import annotations

from datetime import date


def validate_train_event_dates(dates, train_features) -> None:
    """Validate optional ordered training event dates against sized training rows."""
    if dates is None:
        return
    if not isinstance(dates, (list, tuple)):
        raise ValueError("training event dates must be a list or tuple")
    try:
        if len(dates) != len(train_features):
            raise ValueError("training event dates length does not match training rows")
    except TypeError:
        raise ValueError("training event dates require sized training rows") from None
    for value in dates:
        if not isinstance(value, str):
            raise ValueError("training event dates must be ISO date strings")
        try:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError
        except ValueError:
            raise ValueError("training event dates must be canonical ISO dates") from None
