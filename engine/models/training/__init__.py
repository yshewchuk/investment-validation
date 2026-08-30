"""Training scripts for the registry's champions.

    common.py       the walk-forward harness every model is evaluated by
    size_model.py   v1.3 OLS+NN(64,32) blend → predicted |earnings move|
    implied_t1.py   GBM → quoted implied move at the last pre-print close
    gate.py         GBM → per-trade mid-fill return (the selection signal)
    train_all.py    train, evaluate, and register all three

Artifacts are written to ``data/models/`` and registered in
``engine/models/registry.json``. Every script is seeded and re-runnable: the
same data snapshot and seed reproduce the same artifact hash.
"""
