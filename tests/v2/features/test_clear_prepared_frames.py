"""``tier4.clear_prepared_frames`` empties the process-global ``_PREPARED`` dict.

The nightly's fetch-only path calls this before ``warm_serving`` so the parent
process does not pin the Tier-4 prepared panel while the warm subprocess runs
(#1195). Populating the real dict with a dummy entry and clearing it in
``finally`` keeps other tests' cache state intact whatever the assertion does.
"""
from __future__ import annotations

import pandas as pd

from engine.data.features import tier4


def test_clear_prepared_frames_empties_prepared_dict():
    df = pd.DataFrame({"ticker": ["AAA"], "date": ["2026-10-25"]})
    tier4._PREPARED["m"] = (df, df, df)
    try:
        assert "m" in tier4._PREPARED
        tier4.clear_prepared_frames()
        assert tier4._PREPARED == {}
    finally:
        tier4._PREPARED.clear()