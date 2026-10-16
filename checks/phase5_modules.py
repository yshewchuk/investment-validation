"""Optional-module probe shared by tools/phase5_prepare_release.py and checks/phase5_acceptance.py.

Lives in checks/ (not engine.v2.models) because the layer rules forbid dynamic imports in production code.
"""
from __future__ import annotations

import importlib


def modules_available(modules: tuple[str, ...]) -> tuple[bool, str]:
    """``(True, "")`` if every module imports, else ``(False, first missing)``."""
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError:
            return False, name
    return True, ""
