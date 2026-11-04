"""S3a candidate-id and cutoff helpers."""
from __future__ import annotations

import pytest

from engine.v2.ops import retrain_candidate as rc
from engine.v2.ops.errors import OpsError
from engine.v2.ops.state_stage import _safe_id as stage_safe_id

MONTH = "2026-10"
SESSION = "2026-11-02"


def test_candidate_id_and_cutoff_are_canonical():
    candidate = rc.candidate_id(MONTH, SESSION)
    assert candidate == "pools-retrain-2026-10-2026-11-02"
    assert rc.retrain_cutoff(MONTH) == "2026-11-01"
    assert stage_safe_id(candidate)
    assert candidate.startswith("pools-")


@pytest.mark.parametrize("call", [
    pytest.param(lambda: rc.candidate_id("2026-13", SESSION), id="bad-month"),
    pytest.param(lambda: rc.candidate_id("2026-1", SESSION), id="noncanonical-month"),
    pytest.param(lambda: rc.candidate_id(MONTH, "2026-11-31"), id="bad-session"),
    pytest.param(lambda: rc.retrain_cutoff("2026-13"), id="bad-cutoff-month"),
    pytest.param(lambda: rc.retrain_cutoff("2026-1"), id="noncanonical-cutoff-month"),
])
def test_invalid_month_and_session_refuse(call):
    with pytest.raises(OpsError) as err:
        call()
    assert err.value.code == "INVALID_REQUEST"
