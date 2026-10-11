"""One plain ledger row when a holdout-excluding read refuses (issue #489)."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from engine.v2.data.errors import DataError

_LOG = logging.getLogger(__name__)
_CODE = "HOLDOUT_ACCESS_DENIED"


def note_holdout_refusal(experiment_id: str, ledger_path: Path | str, error: BaseException) -> None:
    """Append one refusal row when error is HOLDOUT_ACCESS_DENIED; never raises.

    The row holds the experiment id, the UTC date and the refusal code only: no
    metrics, partial results or holdout data. A failed append is logged and never
    masks the refusal.
    """
    if not isinstance(error, DataError) or error.code != _CODE:
        return
    try:
        from experiments.lib import LEDGER_COLUMNS, ledger_append

        row = {"id": experiment_id, "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
               "stage": f"refused:{_CODE}"}
        ledger_append([{name: row.get(name, "") for name in LEDGER_COLUMNS}], path=Path(ledger_path))
    except Exception:
        _LOG.exception("holdout refusal ledger row for %s was not written", experiment_id)


@contextmanager
def record_holdout_refusal(experiment_id: str, ledger_path: Path | str):
    """Wrap one loader call; on HOLDOUT_ACCESS_DENIED record one row, then re-raise."""
    try:
        yield
    except DataError as exc:
        note_holdout_refusal(experiment_id, ledger_path, exc)
        raise
