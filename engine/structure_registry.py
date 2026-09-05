"""Which SHAPE a strategy trades — champion and challenger, for structures.

The model registry answers "which gate, which size model" and enforces exactly
one champion per ``(strategy, role)``. It cannot answer "which shape", because
a structure is not a model: it has no artifact to hash, no features, no
training window. Forcing one into :class:`~engine.models.registry.RegistryEntry`
would mean writing falsehoods into five required fields.

So structures get their own champion, with the same discipline and none of the
model machinery. The unit is a **family** — a set of structures that are
alternatives for the same idea, competing for the same events with the same
forecast. Exactly one member of a family is live; the others are *superseded*.

**Superseded is not disabled, and the difference matters.**
:data:`engine.score.DISABLED_STRATEGIES` means "the scorer refuses to put a
number on this" and stamps ``UNVALIDATED_STRUCTURE`` — the right answer for
CAL-P, whose exact spec has never been backtested. It is the wrong answer for a
structure with three completed experiments behind it that simply lost to a
better one. A superseded structure is fully validated; it is just not the one
being traded, and the board should say so with the successor's name attached.

Nothing is deleted on promotion. The loser stays in ``STRUCTURES`` because
old experiments replay against it and a structure that vanishes cannot be
re-measured — and because a champion that turns out to be wrong needs somewhere
to roll back to.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from engine import paths

__all__ = [
    "FAMILIES",
    "CHAMPIONS_PATH",
    "StructureChampionError",
    "StructureChampion",
    "load_champions",
    "family_of",
    "champion_for",
    "superseded_by",
    "live_strategies",
    "promote_structure",
]

#: Structures that are alternatives for the same idea. Membership is a claim
#: that two shapes compete for the SAME events with the SAME forecast, so that
#: running both would double the position without doubling the idea. A
#: structure named in no family is unaffected by any of this and is live on its
#: own terms.
#:
#: ``twin-peak`` — a doubled ATM long, shorts either side, wings that cancel the
#: tails to exactly zero. TWIN-P spends seven listed strikes on it, TWIN-P5
#: five. Same thesis (earnings moves are usually small but rarely zero), same
#: forecast sizing the peak, same three-term arithmetic entry rule.
FAMILIES: dict[str, tuple[str, ...]] = {
    "twin-peak": ("TWIN-P", "TWIN-P5"),
}

CHAMPIONS_PATH = paths.ENGINE / "models" / "structures.json"


class StructureChampionError(RuntimeError):
    """The structure champion manifest is missing, ambiguous, or inconsistent."""


@dataclass(frozen=True)
class StructureChampion:
    """One family's live shape, and the receipt for why it is live."""

    family: str
    strategy: str
    promoted: str = ""
    evidence: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        members = FAMILIES.get(self.family)
        if members is None:
            raise StructureChampionError(
                f"unknown structure family {self.family!r}; known: {sorted(FAMILIES)}"
            )
        if self.strategy not in members:
            raise StructureChampionError(
                f"{self.strategy!r} is not a member of family {self.family!r} "
                f"({', '.join(members)}) — a champion must be one of the "
                "alternatives it was compared against"
            )

    def as_dict(self) -> dict:
        return asdict(self)


def _default_champions() -> dict[str, StructureChampion]:
    """The first-declared member of each family, when no manifest exists.

    A family with no manifest entry is not an error — it means nobody has run
    the comparison yet, and the incumbent keeps trading. Falling back to the
    first member rather than to nothing is what stops a missing file from
    silently emptying the board.
    """
    return {
        name: StructureChampion(
            family=name, strategy=members[0],
            evidence="no promotion recorded; incumbent by declaration order",
        )
        for name, members in FAMILIES.items()
    }


def load_champions(path: Path | None = None) -> dict[str, StructureChampion]:
    """Every family's champion, keyed by family."""
    target = Path(path or CHAMPIONS_PATH)
    champions = _default_champions()
    if not target.exists():
        return champions
    try:
        doc = json.loads(target.read_text())
    except (OSError, ValueError) as exc:
        raise StructureChampionError(f"{target}: unreadable manifest — {exc}") from exc
    seen: set[str] = set()
    for row in doc.get("champions", []):
        entry = StructureChampion(**row)
        if entry.family in seen:
            raise StructureChampionError(
                f"family {entry.family!r} names more than one champion — two live "
                "shapes for one idea is the ambiguity this file exists to prevent"
            )
        seen.add(entry.family)
        champions[entry.family] = entry
    return champions


def family_of(strategy: str) -> str | None:
    """The family ``strategy`` competes in, or ``None`` if it stands alone."""
    for name, members in FAMILIES.items():
        if strategy in members:
            return name
    return None


def champion_for(family: str, path: Path | None = None) -> StructureChampion:
    champions = load_champions(path)
    if family not in champions:
        raise StructureChampionError(f"unknown structure family {family!r}")
    return champions[family]


def superseded_by(strategy: str, path: Path | None = None) -> StructureChampion | None:
    """The champion that displaced ``strategy``, or ``None`` if it is live.

    ``None`` for a structure in no family, and ``None`` for the champion
    itself — only a beaten member of a family gets an answer here.
    """
    family = family_of(strategy)
    if family is None:
        return None
    reigning = champion_for(family, path)
    return None if reigning.strategy == strategy else reigning


def live_strategies(strategies, path: Path | None = None) -> list[str]:
    """``strategies`` with every superseded family member removed, order kept."""
    return [s for s in strategies if superseded_by(s, path) is None]


def promote_structure(
    family: str,
    strategy: str,
    *,
    evidence: str,
    notes: str = "",
    promoted: str | None = None,
    path: Path | None = None,
) -> StructureChampion:
    """Make ``strategy`` the live shape for ``family`` and write the manifest.

    The decision itself belongs to ``experiments/promote.py`` — this only
    records it, and refuses a strategy that is not one of the family's declared
    alternatives. ``evidence`` is required: a champion with no receipt is a
    preference, and six months from now nobody will remember which it was.
    """
    if not evidence.strip():
        raise StructureChampionError(
            f"promoting {strategy!r} needs evidence — name the experiment that "
            "decided it"
        )
    entry = StructureChampion(
        family=family, strategy=strategy,
        promoted=promoted or date.today().isoformat(),
        evidence=evidence, notes=notes,
    )
    champions = load_champions(path)
    champions[family] = entry
    target = Path(path or CHAMPIONS_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(
        {
            "_comment": (
                "Which SHAPE each structure family trades. One champion per "
                "family; the others are superseded, not disabled — they stay "
                "in engine.structures.STRUCTURES so old experiments replay and "
                "so a promotion can be rolled back. See "
                "engine/structure_registry.py."
            ),
            "champions": [champions[f].as_dict() for f in sorted(champions)],
        },
        indent=2,
    ) + "\n")
    return entry
