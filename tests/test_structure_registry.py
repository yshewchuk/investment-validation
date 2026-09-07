"""The structure champion registry — one live shape per family, with a receipt."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.structure_registry import (  # noqa: E402
    FAMILIES,
    StructureChampion,
    StructureChampionError,
    champion_for,
    family_of,
    live_strategies,
    load_champions,
    promote_structure,
    superseded_by,
)
from engine.structures import STRUCTURES  # noqa: E402


class TestFamilies:
    def test_every_declared_member_is_a_real_structure(self):
        """A family naming a structure that does not exist would silently
        supersede nothing, or hide a live shape behind a typo."""
        unknown = {m for members in FAMILIES.values() for m in members
                   if m not in STRUCTURES}
        assert not unknown, f"families name structures absent from STRUCTURES: {unknown}"

    def test_a_structure_belongs_to_at_most_one_family(self):
        seen: dict[str, str] = {}
        for name, members in FAMILIES.items():
            for m in members:
                assert m not in seen, f"{m} is in both {seen[m]} and {name}"
                seen[m] = name

    def test_a_family_needs_at_least_two_alternatives(self):
        """A family of one is not a competition; it is a structure, and the
        supersede machinery has nothing to say about it."""
        for name, members in FAMILIES.items():
            assert len(members) >= 2, f"{name} declares only {members}"


#: The real ``twin-peak`` family (TWIN-P vs TWIN-P5) was retired 2026-09-06:
#: a further comparison found the two shapes are not mutually exclusive — each
#: wins on different events — so both are live instead of one superseding the
#: other (see ``engine/structure_registry.py`` and
#: ``engine.entry_rules.TWIN_P_LEGACY_RULE``). The promotion/champion
#: MECHANISM this file exercises still needs an example family to operate on,
#: so the classes below inject a synthetic one rather than reuse production
#: state that no longer exists.
def _inject_synthetic_family(monkeypatch):
    monkeypatch.setattr(
        "engine.structure_registry.FAMILIES",
        {"twin-peak": ("TWIN-P", "TWIN-P5")},
    )


class TestTheManifest:
    @pytest.fixture(autouse=True)
    def _synthetic_family(self, monkeypatch):
        _inject_synthetic_family(monkeypatch)

    def test_a_missing_manifest_keeps_the_incumbent_rather_than_emptying_the_board(
        self, tmp_path
    ):
        champions = load_champions(tmp_path / "absent.json")
        assert champions["twin-peak"].strategy == "TWIN-P"

    def test_promotion_writes_the_receipt(self, tmp_path):
        path = tmp_path / "structures.json"
        promote_structure("twin-peak", "TWIN-P5", evidence="EXP-126",
                          notes="n", promoted="2026-09-04", path=path)
        doc = json.loads(path.read_text())
        row = doc["champions"][0]
        assert row["strategy"] == "TWIN-P5"
        assert row["evidence"] == "EXP-126"
        assert row["promoted"] == "2026-09-04"

    def test_promotion_without_evidence_is_refused(self, tmp_path):
        """A champion with no receipt is a preference. Six months from now
        nobody will remember which it was."""
        with pytest.raises(StructureChampionError, match="needs evidence"):
            promote_structure("twin-peak", "TWIN-P5", evidence="  ",
                              path=tmp_path / "s.json")

    def test_a_champion_must_be_one_of_the_alternatives_it_beat(self):
        with pytest.raises(StructureChampionError, match="not a member"):
            StructureChampion(family="twin-peak", strategy="STR-THRU")

    def test_an_unknown_family_is_refused(self):
        with pytest.raises(StructureChampionError, match="unknown structure family"):
            StructureChampion(family="nope", strategy="TWIN-P")

    def test_two_champions_for_one_family_is_an_error(self, tmp_path):
        """The ambiguity the file exists to prevent: two live shapes for one
        idea means the board doubles the position on a single thesis."""
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"champions": [
            {"family": "twin-peak", "strategy": "TWIN-P"},
            {"family": "twin-peak", "strategy": "TWIN-P5"},
        ]}))
        with pytest.raises(StructureChampionError, match="more than one champion"):
            load_champions(path)

    def test_an_unreadable_manifest_raises_rather_than_defaulting(self, tmp_path):
        """Falling back to the incumbent on a CORRUPT file would trade a shape
        nobody chose while reporting success."""
        path = tmp_path / "s.json"
        path.write_text("{not json")
        with pytest.raises(StructureChampionError, match="unreadable"):
            load_champions(path)


class TestSupersede:
    @pytest.fixture(autouse=True)
    def _synthetic_family(self, monkeypatch):
        _inject_synthetic_family(monkeypatch)

    def test_the_champion_is_not_superseded(self, tmp_path):
        path = tmp_path / "s.json"
        promote_structure("twin-peak", "TWIN-P5", evidence="EXP-126", path=path)
        assert superseded_by("TWIN-P5", path) is None
        assert superseded_by("TWIN-P", path).strategy == "TWIN-P5"

    def test_promotion_is_reversible(self, tmp_path):
        """A one-way door is not a decision. The beaten shape stays in
        STRUCTURES precisely so a rollback is a manifest edit."""
        path = tmp_path / "s.json"
        promote_structure("twin-peak", "TWIN-P5", evidence="EXP-126", path=path)
        assert champion_for("twin-peak", path).strategy == "TWIN-P5"
        promote_structure("twin-peak", "TWIN-P", evidence="rolled back", path=path)
        assert champion_for("twin-peak", path).strategy == "TWIN-P"
        assert superseded_by("TWIN-P5", path).strategy == "TWIN-P"

    def test_family_membership_is_looked_up_both_ways(self):
        assert family_of("TWIN-P") == "twin-peak"
        assert family_of("TWIN-P5") == "twin-peak"
        assert family_of("CND-P") is None

    def test_live_strategies_keeps_order_and_non_family_members(self, tmp_path):
        path = tmp_path / "s.json"
        promote_structure("twin-peak", "TWIN-P5", evidence="EXP-126", path=path)
        got = live_strategies(["CAL-P", "TWIN-P", "STR-THRU", "TWIN-P5"], path)
        assert got == ["CAL-P", "STR-THRU", "TWIN-P5"]


class TestTheShippedManifest:
    """The real repo state: no family left, both twin-peak shapes live.

    Distinct from ``TestTheManifest``/``TestSupersede`` above, which exercise
    the generic mechanism against an injected family — these read the actual
    ``FAMILIES`` dict and ``engine/models/structures.json`` unpatched, to pin
    the 2026-09-06 retirement itself.
    """

    def test_no_family_is_declared(self):
        assert FAMILIES == {}

    def test_the_shipped_manifest_has_no_champions(self):
        assert load_champions() == {}

    def test_neither_twin_peak_shape_is_superseded(self):
        assert family_of("TWIN-P") is None
        assert family_of("TWIN-P5") is None
        assert superseded_by("TWIN-P") is None
        assert superseded_by("TWIN-P5") is None

    def test_both_twin_peak_shapes_are_live(self):
        assert live_strategies(["TWIN-P", "TWIN-P5", "CAL-P"]) == [
            "TWIN-P", "TWIN-P5", "CAL-P"]
