"""Server composition for the preview launcher (package-internal).

Holds the only ``engine.v2.serving.operations.create_server`` call in this
package and, when an explicit ops root is configured, wires the authenticated
``POST /actions/refresh`` route's ``submit_refresh`` callback to
``engine.v2.ops.cli.refresh_action`` on exactly that root. ``engine.v2.ops`` is
imported lazily, only when a refresh root is given, so a plain read-only
preview never loads the supervisor. Kept separate from ``preview`` so the
launcher's own dependency footprint stays at the ``checks/code_budgets.py`` fan
out budget; this is not part of the package's public interface.
"""
from __future__ import annotations

from engine.v2.serving.operations import create_server

__all__ = ["build_server"]


def _refresh_callback(ops_root):
    """A ``submit_refresh`` callback bound to ``ops_root``, or ``None``.

    Returns ``None`` for a missing or empty ops root so ``create_server`` keeps
    the read-only 503 ``refresh not configured`` behaviour; it never guesses a
    root from the release root. When set, it calls ``refresh_action`` on
    exactly ``ops_root`` — a shadow nightly plan submission only, never an
    inline refresh or a production authority switch.
    """
    if not ops_root:
        return None
    from pathlib import Path

    from engine.v2.ops.cli import refresh_action

    root = Path(ops_root)

    def submit_refresh(payload):
        return refresh_action(root, payload)

    return submit_refresh


def build_server(*, host, port, token, health_path, release_root, frozen_at,
                 model_release_root, ops_root=None, calibration_health_path=None,
                 serving_index_path=None):
    """Compose the operations server exactly as the launcher needs it."""
    return create_server((host, port), token=token, health_path=health_path,
                         release_root=release_root, frozen_at=frozen_at,
                         model_release_root=model_release_root,
                         calibration_health_path=calibration_health_path,
                         serving_index_path=serving_index_path,
                         submit_refresh=_refresh_callback(ops_root))
