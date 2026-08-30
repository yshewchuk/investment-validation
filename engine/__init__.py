"""Engine — the shared core every phase of the earnings-vol program builds on.

Layout::

    engine/paths.py       logical name → physical path registry
    engine/fills.py       FillModel: the worst→best execution interpolation
    engine/calendar.py    canonical earnings calendar + session-aware day math
    engine/structures.py  trade structures → leg lists → the one pricing path
    engine/data/          the three-tier data architecture (raw/curated/features)

    engine/audit.py       leak discipline, asserted on every scoring path
    engine/features.py    as-of feature vectors (panel path and live path)
    engine/replay.py      structures × real chains → priced trades
    engine/payoff.py      predicted quantity → exit value, empirically calibrated
    engine/analogs.py     matched historical trades + bootstrap intervals
    engine/score.py       the Phase 1 public API
    engine/calibrate.py   reliability curves, Brier scores, decile tables
    engine/models/        the registry and the champions it describes

``engine/SCORING.md`` is the Phase 1 map.

Import discipline: everything downstream reads Tier 2/3 through
``engine.paths``; direct reads of a raw source format outside
``engine/data/normalize/`` are a code smell. Nothing outside
``engine/models/`` opens a model artifact.
"""

__version__ = "0.1.0"
