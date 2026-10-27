"""Leaf guard: nothing executes from inside an experiment research workspace.

``workspace.overlay`` writes ``MARKER`` at a workspace root. The supervisor service and the
job launcher call ``refuse_workspace_execution`` with the roots they are about to use; any path
that is, or lies under, a marked directory (by its literal or its resolved spelling) is refused
``WORKSPACE_REFUSED`` before a process starts. Stdlib and ``ops.errors`` only, so the supervisor
and executor can import it without pulling in the data layer.
"""
from __future__ import annotations

import os
from pathlib import Path

from engine.v2.ops.errors import fail

__all__ = ["MARKER", "refuse_workspace_execution"]

MARKER = ".research_workspace"


def refuse_workspace_execution(*paths) -> None:
    for raw in paths:
        if raw is None:
            continue
        for spelling in {os.path.abspath(raw), os.path.realpath(raw)}:
            for directory in (Path(spelling), *Path(spelling).parents):
                if os.path.lexists(directory / MARKER):
                    raise fail("WORKSPACE_REFUSED",
                               "no supervisor or job may run from a research workspace",
                               details={"reason": "execution_in_workspace"})
