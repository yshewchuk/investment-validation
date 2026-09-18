"""The one declared v2 -> legacy edge of ``engine.v2.models.training``.

``engine.models.no_fit`` holds the process-wide no-fit switch every legacy
scoring path is rigged with (P5-2). Legacy may never import v2
(``checks/import_layers.py`` rule 3), so a v2-local guard would never be set
by the legacy scorer's ``no_fit_guard()``. Sharing the one switch is what
makes a v2 fit reached from any guarded scoring path trip the same way a
legacy fit does. Declared in ``checks/legacy_adapters.json``.
"""
from __future__ import annotations

from engine.models.no_fit import forbid_fitting

__all__ = ["forbid_fitting"]
