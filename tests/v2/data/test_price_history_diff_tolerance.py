"""#1285: a ``price_history`` refresh must not rewrite history for ``close_adj``
float jitter. Pure-pandas tests of ``engine.v2.data.price_history.diff_retrieval``
value comparison: ``close_raw``/``high_raw`` exact (NaN equals NaN), ``close_adj``
within ``CLOSE_ADJ_RTOL`` relative, always against the LATEST stored version so
accepted drift cannot accumulate. No ops store, no catalog, no filesystem."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from engine.v2.data import price_history
from engine.v2.data.errors import DataError
from engine.v2.data.price_history import CLOSE_ADJ_RTOL

_STORED_COLUMNS = ["date", "close_adj", "close_raw", "high_raw", "retrieved_at", "deleted",
                   "source_kind", "source_hash", "capture_id"]

_T0 = "2024-09-01T00:00:00+00:00"
_T1 = "2024-09-02T00:00:00+00:00"
_T2 = "2024-09-03T00:00:00+00:00"


def _stored(*rows):
    """rows: (date, close_adj, close_raw, high_raw, retrieved_at, deleted)."""
    return pd.DataFrame([{"date": d, "close_adj": adj, "close_raw": raw, "high_raw": high,
                          "retrieved_at": at, "deleted": deleted,
                          "source_kind": "legacy_px_csv", "source_hash": "h",
                          "capture_id": "cap0"}
                         for d, adj, raw, high, at, deleted in rows],
                        columns=_STORED_COLUMNS)


def _retrieval(*rows):
    """rows: (date, close_adj, close_raw, high_raw)."""
    return pd.DataFrame([{"date": d, "close_adj": adj, "close_raw": raw, "high_raw": high}
                         for d, adj, raw, high in rows])


def _diff(stored, retrieval, *, retrieved_at=_T1):
    return price_history.diff_retrieval(stored, retrieval, retrieved_at=retrieved_at,
                                        source_kind="tier1", source_hash="h1",
                                        capture_id="cap1")


def test_close_adj_jitter_within_tolerance_is_no_change():
    stored = _stored(("2024-01-01", 100.0, 100.0, 105.0, _T0, False),
                     ("2024-01-02", 99.0, 99.0, 103.0, _T0, False))
    jittered = _retrieval(("2024-01-01", 100.0 * (1 + 2e-6), 100.0, 105.0),
                          ("2024-01-02", 99.0 * (1 + 2e-6), 99.0, 103.0))
    assert _diff(stored, jittered).empty


def test_dividend_like_close_adj_shift_versions_every_date():
    stored = _stored(("2024-01-01", 100.0, 100.0, 105.0, _T0, False),
                     ("2024-01-02", 99.0, 99.0, 103.0, _T0, False))
    shifted = _retrieval(("2024-01-01", 100.0 * (1 + 1e-3), 100.0, 105.0),
                         ("2024-01-02", 99.0 * (1 + 1e-3), 99.0, 103.0))
    out = _diff(stored, shifted)
    assert list(out["date"]) == ["2024-01-01", "2024-01-02"]
    assert list(out["retrieved_at"]) == [_T1, _T1]
    assert not out["deleted"].any()
    assert out["close_adj"].tolist() == pytest.approx([100.0 * (1 + 1e-3), 99.0 * (1 + 1e-3)])
    assert list(out["close_raw"]) == [100.0, 99.0]
    assert list(out["high_raw"]) == [105.0, 103.0]


def test_one_ulp_close_raw_difference_versions_the_date():
    ulp = math.nextafter(100.0, math.inf)
    stored = _stored(("2024-01-01", 100.0, 100.0, 105.0, _T0, False),
                     ("2024-01-02", 99.0, 99.0, 103.0, _T0, False))
    retrieval = _retrieval(("2024-01-01", 100.0, ulp, 105.0),
                           ("2024-01-02", 99.0, 99.0, 103.0))
    out = _diff(stored, retrieval)
    assert list(out["date"]) == ["2024-01-01"]
    assert list(out["close_adj"]) == [100.0]
    assert list(out["close_raw"]) == [ulp]
    assert list(out["high_raw"]) == [105.0]
    assert list(out["retrieved_at"]) == [_T1]


def test_one_ulp_high_raw_difference_with_identical_raw_adj_versions_the_date():
    ulp = math.nextafter(105.0, math.inf)
    stored = _stored(("2024-01-01", 100.0, 100.0, 105.0, _T0, False),
                     ("2024-01-02", 99.0, 99.0, 103.0, _T0, False))
    retrieval = _retrieval(("2024-01-01", 100.0, 100.0, ulp),
                           ("2024-01-02", 99.0, 99.0, 103.0))
    out = _diff(stored, retrieval)
    assert list(out["date"]) == ["2024-01-01"]
    assert list(out["close_adj"]) == [100.0]
    assert list(out["close_raw"]) == [100.0]
    assert list(out["high_raw"]) == [ulp]
    assert list(out["retrieved_at"]) == [_T1]


def test_nan_close_adj_matches_only_nan():
    nan = float("nan")
    stored = _stored(("2024-01-01", nan, 100.0, 105.0, _T0, False),
                     ("2024-01-02", nan, 99.0, 103.0, _T0, False),
                     ("2024-01-03", 98.0, 98.0, 102.0, _T0, False))
    retrieval = _retrieval(("2024-01-01", nan, 100.0, 105.0),
                           ("2024-01-02", 99.0, 99.0, 103.0),
                           ("2024-01-03", nan, 98.0, 102.0))
    out = _diff(stored, retrieval)
    assert list(out["date"]) == ["2024-01-02", "2024-01-03"]
    assert out["close_adj"].iloc[0] == 99.0
    assert math.isnan(out["close_adj"].iloc[1])
    assert list(out["close_raw"]) == [99.0, 98.0]
    assert list(out["high_raw"]) == [103.0, 102.0]
    assert list(out["retrieved_at"]) == [_T1, _T1]


def test_accepted_drift_does_not_accumulate():
    v0 = 100.0
    stored = _stored(("2024-01-01", v0, v0, 105.0, _T0, False))
    drift_a = v0 * (1 + 8e-6)
    assert _diff(stored, _retrieval(("2024-01-01", drift_a, v0, 105.0))).empty
    # diff_retrieval is pure: the empty result means the caller appends
    # nothing, so the stored latest version stays v0 and B is compared to v0,
    # never to the refused-to-be-stored A.
    drift_b = v0 * (1 + 1.6e-5)
    out = _diff(stored, _retrieval(("2024-01-01", drift_b, v0, 105.0)))
    assert len(out) == 1
    assert out["close_adj"].tolist() == pytest.approx([drift_b])
    assert list(out["close_raw"]) == [v0]
    assert list(out["high_raw"]) == [105.0]
    assert list(out["retrieved_at"]) == [_T1]


@pytest.mark.parametrize("adj", [0.06, 10000.0])
def test_close_adj_tolerance_scales_relatively_not_absolutely(adj):
    # An absolute tolerance would pass 0.06*2e-6 = 1.2e-7 but reject
    # 10000.0*2e-5 = 0.2; the relative rule gives the same verdict at both
    # magnitudes: 2e-6 relative is jitter, 2e-5 relative versions.
    high = adj * 1.05
    stored = _stored(("2024-01-01", adj, adj, high, _T0, False))
    under = adj * (1 + 2e-6)
    assert _diff(stored, _retrieval(("2024-01-01", under, adj, high))).empty
    over = adj * (1 + 2e-5)
    out = _diff(stored, _retrieval(("2024-01-01", over, adj, high)))
    assert list(out["date"]) == ["2024-01-01"]
    assert out["close_adj"].tolist() == pytest.approx([over])
    assert list(out["close_raw"]) == [adj]
    assert list(out["high_raw"]) == [high]
    assert list(out["retrieved_at"]) == [_T1]


def test_close_adj_boundary_relative_diff_just_under_and_over_rtol():
    stored = _stored(("2024-01-01", 100.0, 100.0, 105.0, _T0, False))
    under = 100.0 * (1 + 0.9 * CLOSE_ADJ_RTOL)
    assert _diff(stored, _retrieval(("2024-01-01", under, 100.0, 105.0))).empty
    over = 100.0 * (1 + 1.1 * CLOSE_ADJ_RTOL)
    out = _diff(stored, _retrieval(("2024-01-01", over, 100.0, 105.0)))
    assert list(out["date"]) == ["2024-01-01"]
    assert out["close_adj"].tolist() == pytest.approx([over])
    assert list(out["close_raw"]) == [100.0]
    assert list(out["high_raw"]) == [105.0]
    assert list(out["retrieved_at"]) == [_T1]


def test_tombstoned_date_revived_with_identical_values_ignores_unchanged_live_date():
    stored = _stored(("2024-01-01", 50.0, 50.0, 52.0, _T0, False),
                     ("2024-01-01", 50.0, 50.0, 52.0, _T1, True),
                     ("2024-01-02", 100.0, 100.0, 105.0, _T1, False))
    retrieval = _retrieval(("2024-01-01", 50.0, 50.0, 52.0),
                           ("2024-01-02", 100.0 * (1 + 2e-6), 100.0, 105.0))
    out = _diff(stored, retrieval, retrieved_at=_T2)
    assert list(out["date"]) == ["2024-01-01"]
    assert not bool(out["deleted"].iloc[0])
    assert out["retrieved_at"].iloc[0] == _T2
    assert list(out["close_adj"]) == [50.0]
    assert list(out["close_raw"]) == [50.0]
    assert list(out["high_raw"]) == [52.0]


def test_omitted_live_date_still_refuses_whole_response():
    stored = _stored(("2024-01-01", 100.0, 100.0, 105.0, _T0, False),
                     ("2024-01-02", 99.0, 99.0, 103.0, _T0, False))
    with pytest.raises(DataError) as exc:
        _diff(stored, _retrieval(("2024-01-01", 100.0, 100.0, 105.0)))
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"missing_count": 1, "missing_dates": ["2024-01-02"]}
