from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.v2.scoring.frozen_record import (
    FROZEN_RECORD_FIELDS,
    deep_freeze,
    freeze_record_fields,
)


def test_deep_freeze_recursively_freezes_mappings_sequences_and_sets():
    source = {"items": [{"value": 1}], "tags": {"a", "b"}}

    frozen = deep_freeze(source)

    assert frozen["items"] == ({"value": 1},)
    assert isinstance(frozen["items"][0], dict)
    assert isinstance(frozen["tags"], frozenset)
    with pytest.raises(TypeError):
        frozen["items"][0]["value"] = 2
    with pytest.raises(AttributeError):
        frozen["items"].append({"value": 3})
    with pytest.raises(AttributeError):
        frozen["tags"].add("c")


def test_freeze_record_fields_covers_declared_fields_and_returns_same_record():
    source = {name: {"nested": [{"value": 1}]} for name in FROZEN_RECORD_FIELDS}
    record = SimpleNamespace(**source, untouched={"mutable": True})

    result = freeze_record_fields(record)

    assert result is record
    for name in FROZEN_RECORD_FIELDS:
        value = getattr(record, name)
        assert isinstance(value, dict)
        assert isinstance(value["nested"], tuple)
        with pytest.raises(TypeError):
            value["nested"][0]["value"] = 2
    assert record.untouched == {"mutable": True}
    record.untouched["mutable"] = False
