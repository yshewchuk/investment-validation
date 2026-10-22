"""#1103 — the regime block must say so when the ^GSPC series trails the frame.

The legacy ``gspc_daily.csv`` was fetched once and nothing refreshed it, so a
frame whose decision dates run past the series end was silently served stale
market state. The fix is a warning only: one ``UserWarning`` and one stderr
line, once per ``(path, mtime)``, with every computed value unchanged. These
tests pin the warning's shape, the once-per-file-version suppression, the
strictly-seven-day threshold, and value identity across the call that warned
and the one that stayed quiet.

Synthetic data only: no network, nothing under the real data root.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

from engine.data.features import panel
from engine.data.features.panel import add_regime_features

SERIES_END = pd.Timestamp("2026-08-27")
STALE_DECISION = pd.Timestamp("2026-10-29")  # 63 days past the series end
WITHIN_WINDOW = pd.Timestamp("2026-09-03")  # exactly 7 days past: must not warn

STALE_MATCH = "regime features are stale"

#: The market-state block the staleness diagnostic must never alter.
SPY_COLS = ("spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol5",
            "spy_vol20", "spy_vol60", "spy_vol252", "spy_vol20_rel252")

#: Hand-computed market-state values at the anchor row — the series end, row
#: 299 of the 300-row synthetic file below, whose closes are c_i = 100 + 0.5*i,
#: so the spot is c_299 = 249.5. Derived once from the definitions in
#: ``panel.add_regime_features`` (a scratch run of exactly these formulas) and
#: pasted as literals, so the tests pin the computed block against arithmetic
#: rather than re-deriving the expectation from the implementation:
#:   spy_retW          = (c299 / c(299-W) - 1) * 100
#:                       ret21: 10.5/239.0; ret63: 31.5/218.0; ret252: 126/123.5
#:   spy_dd252         = (c299 / max(c48..c299) - 1) * 100 = 0.0 (rising series)
#:   spy_volW          = sample std (ddof=1) of the simple returns
#:                       c(k+1)/c(k) - 1 over k = 299-W .. 298,
#:                       times sqrt(252) * 100
#:   spy_vol20_rel252  = spy_vol20 / spy_vol252 - 1
SPY_AT_ANCHOR = {
    "spy_ret21": 4.3933054393305415,
    "spy_ret63": 14.449541284403677,
    "spy_ret252": 102.0242914979757,
    "spy_dd252": 0.0,
    "spy_vol5": 0.010202677511272283,
    "spy_vol20": 0.03936749340495202,
    "spy_vol60": 0.12668884515903708,
    "spy_vol252": 0.9111766241585297,
    "spy_vol20_rel252": -0.9567948821763202,
}


@pytest.fixture(autouse=True)
def _fresh_dedup_state(monkeypatch):
    """Once-per-(path, mtime) is module state; give each test an empty set."""
    monkeypatch.setattr(panel, "_STALE_REGIME_WARNED", set())


def _write_gspc(path, periods: int = 300) -> None:
    """The legacy multi-header ^GSPC layout, closes = 100 + 0.5*i rising."""
    dates = pd.date_range(end=SERIES_END, periods=periods, freq="D")
    closes = 100.0 + 0.5 * np.arange(len(dates), dtype=float)
    lines = [
        "Price,Adj Close,Close,High,Low,Open,Volume",
        "Ticker,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC",
        "Date,,,,,,",
    ]
    for day, close in zip(dates, closes):
        lines.append(f"{day.date()},{close},{close},{close},{close},{close},1000")
    path.write_text("\n".join(lines) + "\n")


def _frame(decision: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["AAA"], "date": pd.to_datetime([decision])})


def test_stale_frame_warns_once_and_a_second_call_stays_quiet(tmp_path):
    path = tmp_path / "gspc_daily.csv"
    _write_gspc(path)
    frame = _frame(STALE_DECISION)

    with pytest.warns(UserWarning, match=STALE_MATCH):
        first = add_regime_features(frame, gspc_path=path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        second = add_regime_features(frame, gspc_path=path)
    assert [w for w in caught if "stale" in str(w.message)] == []

    # Values are identical with and without the warning, and they are the real
    # anchor: the last close strictly before the decision date is the series
    # end (row 299). The staleness path is diagnostic only.
    pd.testing.assert_frame_equal(first, second)
    # Every market-state value is pinned to its hand-computed literal, so the
    # test checks arithmetic, not the implementation against itself.
    for col, pinned in SPY_AT_ANCHOR.items():
        assert first[col].iloc[0] == pytest.approx(pinned), col
    assert first["regime_asof"].iloc[0] == np.datetime64(str(SERIES_END.date()))

    # Every market-state value equals what the SAME file yields on a decision
    # date inside the 7-day window (which does not warn): the stale anchor is
    # the series end in both cases, so the warning is diagnostic only and never
    # moves a number.
    fresh = add_regime_features(_frame(WITHIN_WINDOW), gspc_path=path)
    for col in SPY_COLS:
        assert first[col].iloc[0] == pytest.approx(fresh[col].iloc[0])


def test_stale_frame_emits_exactly_one_stderr_line(tmp_path, capsys):
    path = tmp_path / "gspc_daily.csv"
    _write_gspc(path)
    expected_line = (
        "[regime] WARNING gspc series ends 2026-08-27, decision date "
        "2026-10-29: regime features are stale"
    )

    with pytest.warns(UserWarning, match=STALE_MATCH):
        add_regime_features(_frame(STALE_DECISION), gspc_path=path)
    err = capsys.readouterr().err
    assert expected_line in err
    assert err.count("[regime] WARNING") == 1

    # The same file version never repeats it, on stderr either.
    add_regime_features(_frame(STALE_DECISION), gspc_path=path)
    assert capsys.readouterr().err == ""


def test_decision_within_seven_days_does_not_warn(tmp_path, capsys):
    path = tmp_path / "gspc_daily.csv"
    _write_gspc(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = add_regime_features(_frame(WITHIN_WINDOW), gspc_path=path)

    assert [w for w in caught if "stale" in str(w.message)] == []
    assert "[regime] WARNING" not in capsys.readouterr().err
    # It still computed the block normally: the series end is the anchor, so
    # every value matches the same hand-computed literals as the stale call.
    assert out["regime_asof"].iloc[0] == np.datetime64(str(SERIES_END.date()))
    for col, pinned in SPY_AT_ANCHOR.items():
        assert out[col].iloc[0] == pytest.approx(pinned), col


def test_boundary_exactly_seven_days_quiet_eight_days_warns(tmp_path, capsys):
    """The threshold is strictly more than seven days: +7 stays quiet, +8 warns."""
    path = tmp_path / "gspc_daily.csv"
    _write_gspc(path)
    seven = SERIES_END + pd.Timedelta(days=7)
    eight = SERIES_END + pd.Timedelta(days=8)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        add_regime_features(_frame(seven), gspc_path=path)
    assert [w for w in caught if "stale" in str(w.message)] == []
    assert "[regime] WARNING" not in capsys.readouterr().err

    with pytest.warns(UserWarning, match=STALE_MATCH):
        add_regime_features(_frame(eight), gspc_path=path)


def test_refreshed_file_can_warn_again(tmp_path):
    """The suppression key is (path, mtime), not path: a refreshed file that is
    still stale behind a new decision date must be able to warn again."""
    path = tmp_path / "gspc_daily.csv"
    _write_gspc(path)
    frame = _frame(STALE_DECISION)

    with pytest.warns(UserWarning, match=STALE_MATCH):
        add_regime_features(frame, gspc_path=path)

    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    with pytest.warns(UserWarning, match=STALE_MATCH):
        add_regime_features(frame, gspc_path=path)
