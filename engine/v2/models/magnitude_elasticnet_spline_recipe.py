"""Closed spline-arm constants; no sklearn, no data reads."""

# Tied columns would be removed here per design 6(c), never per fold.
SPLINE_COLUMNS = (
    "mean_prior_abs_move", "ema2_prior_abs_move", "ema4_prior_abs_move",
    "ema8_prior_abs_move", "ema12_prior_abs_move", "implied_move",
    "iv10", "iv30", "exern_iv10", "exern_iv30", "fwd90_30",
    "fexern90_30", "rvol30", "spy_vol5", "spy_vol20", "spy_vol60",
    "season_heat_mean_abs_5", "season_heat_mean_abs_20",
    "season_heat_ratio_20_252", "tail_max_abs", "tail_std_abs",
    "tail_p75_abs", "tail_p90_abs", "tail_share_gt10", "tail_max4_abs",
    "tail_std_over_mean", "tail_max_over_mean", "iv10_30", "iv_rv",
    "ex_ratio", "imp_chg5", "imp_chg10", "iv30_chg5", "iv30_chg20",
    "iv10_chg5", "own_rvol5", "own_rvol10", "own_rvol60", "own_absret1",
)

__all__ = ["SPLINE_COLUMNS"]
