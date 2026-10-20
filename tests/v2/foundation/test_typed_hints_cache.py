"""from_document resolves a class's annotations once, with identical decode results."""
from __future__ import annotations

import dataclasses

import pytest

from engine.v2.foundation import typed


@dataclasses.dataclass(frozen=True)
class _Leaf:
    name: str
    weight: float = 1.0


@dataclasses.dataclass(frozen=True)
class _Doc:
    leaves: tuple[_Leaf, ...]
    label: str | None = None


def test_repeated_decodes_resolve_hints_once_and_agree(monkeypatch):
    typed._class_hints.cache_clear()
    calls = []
    original = typed.get_type_hints
    monkeypatch.setattr(typed, "get_type_hints", lambda cls: calls.append(cls) or original(cls))
    doc = {"leaves": [{"name": "a"}, {"name": "b", "weight": 2}], "label": "x"}
    first = typed.from_document(_Doc, doc)
    second = typed.from_document(_Doc, doc)
    assert first == second == _Doc(leaves=(_Leaf("a"), _Leaf("b", 2)), label="x")
    assert calls.count(_Doc) == 1 and calls.count(_Leaf) == 1


def test_refusals_unchanged_after_cache_is_warm():
    typed._class_hints.cache_clear()
    good = {"leaves": [{"name": "a"}]}
    typed.from_document(_Doc, good)
    with pytest.raises(typed.DocumentError) as bad_type:
        typed.from_document(_Doc, {"leaves": [{"name": 3}]})
    assert bad_type.value.code == "BAD_TYPE" and bad_type.value.path == "$.leaves[0].name"
    with pytest.raises(typed.DocumentError) as unknown:
        typed.from_document(_Doc, {"leaves": [], "extra": 1})
    assert unknown.value.code == "UNKNOWN_FIELD"
    with pytest.raises(typed.DocumentError) as missing:
        typed.from_document(_Doc, {})
    assert missing.value.code == "MISSING_FIELD"
