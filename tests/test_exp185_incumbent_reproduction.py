"""Tier-0 tests for EXP-185's population-restricted incumbent reproduction.

EXP-185 imports EXP-144's ``run.py`` as a module and its spec pins an
``incumbent.population_event_id_sha256``, which makes the shared
``incumbent_reproduction()`` restrict its exact-match checks to the frozen
registered OOS population instead of today's larger one. These tests load
that runner the same way ``experiments/EXP-185_str_runup_t14_corrected_
calendar_gate_rebaseline_registered/run.py`` does and call
``incumbent_reproduction`` directly; the pinned-population tests point
``REGISTERED_POPULATION_PATH`` at a ``tmp_path`` parquet of their own, so
they never read the committed EXP-144 results artifact. ``main()`` is
never run.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNNER_DIR = "EXP-144_str_runup_t14_corrected_calendar_gate_rebaseline"

#: A threshold whose 4-value quantile below is exact (the two top scores tie
#: at it), so the learned-vs-stored check can be asserted to the bit.
STORED_THRESHOLD = 0.08
REGISTERED_IDS = ["R-1", "R-2", "R-3", "R-4"]
REGISTERED_SCORES = [0.01, 0.02, STORED_THRESHOLD, STORED_THRESHOLD]
EXPANSION_IDS = ["X-1", "X-2", "X-3"]
EXPANSION_SCORES = [0.31, 0.42, 0.53]


def _exp144():
    """A fresh module object per call, so a test can point ``V2_CATALOG`` /
    ``V2_STORE_ROOT`` at its own ``tmp_path`` without touching another test's
    (or the real ``private/ops`` root's) view of them."""
    source = ROOT / "experiments" / RUNNER_DIR / "run.py"
    module_spec = importlib.util.spec_from_file_location(
        "exp144_helpers_under_test", source
    )
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module


def _registered_scores():
    """OOS events split into the 4 pinned-registered ids and 3 later
    expansions, all with finite incumbent scores."""
    return pd.DataFrame({
        "event_id": REGISTERED_IDS + EXPANSION_IDS,
        "year": [2021, 2021, 2022, 2022, 2023, 2023, 2024],
        "incumbent_complete_case": REGISTERED_SCORES + EXPANSION_SCORES,
    })


def _registered_spec(module, **overrides):
    incumbent = {
        "stored_threshold": STORED_THRESHOLD,
        "expected_oos_rows": len(REGISTERED_IDS),
        "expected_selected_at_stored_threshold": sum(
            score >= STORED_THRESHOLD for score in REGISTERED_SCORES
        ),
        "population_event_id_count": len(REGISTERED_IDS),
        "population_event_id_sha256": module._population_digest(REGISTERED_IDS),
    }
    incumbent.update(overrides)
    return {"incumbent": incumbent}


def _pin_population(module, tmp_path, event_ids):
    path = tmp_path / "registered_population.parquet"
    pd.DataFrame({"event_id": list(event_ids)}).to_parquet(path, index=False)
    module.REGISTERED_POPULATION_PATH = path
    return path


def test_incumbent_reproduction_without_population_key_is_unchanged():
    module = _exp144()
    scores = pd.DataFrame({
        "event_id": ["E-2019", "E-LOW", "E-MID", "E-AT", "E-ALSO-AT", "E-NAN"],
        "year": [2019, 2021, 2022, 2023, 2024, 2024],
        "incumbent_complete_case": [
            0.99, 0.01, 0.02, STORED_THRESHOLD, STORED_THRESHOLD, float("nan"),
        ],
    })
    spec = {"incumbent": {
        "stored_threshold": STORED_THRESHOLD,
        "expected_oos_rows": 4,
        "expected_selected_at_stored_threshold": 2,
    }}

    result = module.incumbent_reproduction(scores, spec)

    assert "population_restricted" not in result
    assert result["oos_rows"] == 4
    assert result["learned_threshold"] == pytest.approx(STORED_THRESHOLD)
    assert result["stored_threshold"] == STORED_THRESHOLD
    assert result["selected_at_stored_threshold"] == 2
    assert result["expected_oos_rows"] == 4
    assert result["expected_selected"] == 2


def test_incumbent_reproduction_raises_on_row_mismatch_without_population_key():
    module = _exp144()
    scores = pd.DataFrame({
        "event_id": ["E-LOW", "E-MID", "E-AT", "E-ALSO-AT"],
        "year": [2021, 2022, 2023, 2024],
        "incumbent_complete_case": [
            0.01, 0.02, STORED_THRESHOLD, STORED_THRESHOLD,
        ],
    })
    spec = {"incumbent": {
        "stored_threshold": STORED_THRESHOLD,
        "expected_oos_rows": 999,
        "expected_selected_at_stored_threshold": 2,
    }}

    with pytest.raises(RuntimeError):
        module.incumbent_reproduction(scores, spec)


def test_incumbent_reproduction_restricts_to_the_pinned_population(tmp_path):
    module = _exp144()
    _pin_population(module, tmp_path, REGISTERED_IDS)
    scores = _registered_scores()
    spec = _registered_spec(module)

    result = module.incumbent_reproduction(scores, spec)

    assert result["population_restricted"] is True
    assert result["oos_rows"] == 4
    assert result["threshold_matches_stored"] is True
    assert result["selected_matches_expected"] is True
    assert result["full_population"]["event_count"] == 7


def test_incumbent_reproduction_reports_not_raises_on_score_drift_in_the_overlap(
    tmp_path,
):
    module = _exp144()
    _pin_population(module, tmp_path, REGISTERED_IDS)
    scores = _registered_scores()
    # The registered rows actually select 2; the spec's pinned expectation
    # says 3, as if per-event scores had moved since registration. That is
    # reported, not enforced.
    spec = _registered_spec(module, expected_selected_at_stored_threshold=3)

    result = module.incumbent_reproduction(scores, spec)

    assert result["population_restricted"] is True
    assert result["oos_rows"] == 4
    assert result["selected_matches_expected"] is False
    assert result["selected_at_stored_threshold"] == 2
    assert result["expected_selected"] == 3


def test_incumbent_reproduction_raises_on_row_mismatch_within_the_population(
    tmp_path,
):
    module = _exp144()
    _pin_population(module, tmp_path, REGISTERED_IDS)
    scores = _registered_scores()
    spec = _registered_spec(module, expected_oos_rows=len(REGISTERED_IDS) + 1)

    with pytest.raises(RuntimeError):
        module.incumbent_reproduction(scores, spec)


def test_incumbent_reproduction_raises_on_population_artifact_drift(tmp_path):
    module = _exp144()
    drifted_ids = ["R-1", "R-2", "R-3", "R-9"]
    _pin_population(module, tmp_path, drifted_ids)
    scores = _registered_scores()
    spec = _registered_spec(module)

    with pytest.raises(RuntimeError, match="drifted"):
        module.incumbent_reproduction(scores, spec)
