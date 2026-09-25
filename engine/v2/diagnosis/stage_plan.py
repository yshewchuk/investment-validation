"""Re-export of ``engine.v2.parity.stage_plan`` -- the same objects, not a copy.

The comparator core moved to ``engine/v2/parity`` (layer 6.5) so the nightly
parity report in ``engine/v2/ops`` can run it: no package may import this
sink. This module keeps the ``engine.v2.diagnosis.stage_plan`` path working for
checks, tools and tests, the way ``canonical.py`` re-exports foundation.
"""
from __future__ import annotations

from engine.v2.parity.stage_plan import *  # noqa: F401,F403
from engine.v2.parity.stage_plan import __all__  # noqa: F401
