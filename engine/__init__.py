"""Engine — the shared core every phase of the earnings-vol program builds on.

Layout::

    engine/paths.py       logical name → physical path registry
    engine/fills.py       FillModel: the worst→best execution interpolation
    engine/calendar.py    canonical earnings calendar + session-aware day math
    engine/structures.py  trade structures → leg lists → the one pricing path
    engine/data/          the three-tier data architecture (raw/curated/features)

Import discipline: everything downstream reads Tier 2/3 through
``engine.paths``; direct reads of a raw source format outside
``engine/data/normalize/`` are a code smell.
"""

__version__ = "0.1.0"
