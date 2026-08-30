"""Model artifacts and the registry that is the only way to reach them.

    engine/models/registry.py     schema, loader, and the integrity checks
    engine/models/registry.json   the manifest: one entry per trained model
    engine/models/training/       the scripts that produce artifacts

Nothing outside this package opens a model file. A champion is looked up by
``(strategy, role)``, the loader verifies the artifact still matches what the
manifest claims about it, and the caller gets an object that knows its own
feature list. The alternative — a path in a config somewhere — is how a model
gets retrained without its recorded metrics moving, which makes every number
downstream a claim about a file nobody can identify any more.

Trained artifacts live under ``data/models/`` rather than here: they are
regenerable from ``training/`` at a fixed seed, they are large, and the public
repo takes code, not binaries.
"""

from engine.models.registry import (  # noqa: F401
    ModelArtifact,
    Registry,
    RegistryEntry,
    RegistryError,
    champion,
    load_registry,
)

__all__ = [
    "ModelArtifact",
    "Registry",
    "RegistryEntry",
    "RegistryError",
    "champion",
    "load_registry",
]
