"""Header captured from a real serving_model cache; no real fitted state or values."""

HEADER = {
    "capture": {
        "source": "data/models/tier4/size_v1_4_202610_76d3b1b795a4.joblib",
        "captured_on": "2026-09-30",
        "producer": "engine.data.features.tier4.serving_model",
        "sha256": "3ee4008983557fcc817bba8d2e7f8179566475660820cb698ba160cc78475255",
        "estimator_type": "BlendModel",
        "keys": ["estimator", "features", "fold_start", "model_id", "pool_pred", "pool_res", "tier3_snapshot"],
        "pool_shapes": {"pool_pred": [91650], "pool_res": [91650]},
        "trimming": "Header and shape only; tests replace estimator and pools with synthetic inputs.",
    },
    "features": ["has_implied_quote", "mean_prior_abs_move", "abs_dist_ema", "ema12r_abs",
                 "mean_prior_move", "signed_streak", "dist_high", "dist_ema", "spy_vol20",
                 "spy_dd252", "mean_prior_or_implied", "or_implied", "or_rvol30", "mcap_log"],
    "fold_start": "2026-10-01",
    "model_id": "size_v1_4",
    "tier3_snapshot": "76d3b1b795a48b65ecbe9c01a5cec9632ec32779f0464683366517162abc0d2f",
}
