# packages: engine.v2.models, engine.v2.research
"""SPLINE_COLUMNS is the ordered L-list subset; removals are named explicitly."""
from __future__ import annotations

from engine.v2.models.magnitude_elasticnet_spline_recipe import SPLINE_COLUMNS
from engine.v2.research.derived_features import LOG_COLUMNS

# These tie at k=6 on the v11 panel (design 6(c)).
REMOVED_TIED = ("implied_move", "tail_share_gt10", "imp_chg5", "imp_chg10", "iv30_chg20")


def test_spline_columns_are_the_named_log_column_subset():
    assert len(SPLINE_COLUMNS) == len(set(SPLINE_COLUMNS)) == 34
    assert set(SPLINE_COLUMNS) <= set(LOG_COLUMNS)
    assert set(LOG_COLUMNS) - set(SPLINE_COLUMNS) == set(REMOVED_TIED), (
        "SPLINE_COLUMNS must equal LOG_COLUMNS except for columns removed for tied "
        "quantile knots; name every removed column in REMOVED_TIED with its design "
        "6(c) reason")
    assert SPLINE_COLUMNS == tuple(
        name for name in LOG_COLUMNS if name not in set(REMOVED_TIED))
