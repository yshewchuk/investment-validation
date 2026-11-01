"""Cold/warm projections must use the same authoritative native column admission."""
from copy import deepcopy

import numpy as np
import pytest

from engine.v2.data.errors import DataError
from engine.v2.research.magnitude_panel_qualified_table import admit_columns
from experiments import size_search_binding as binding
from tests.v2.research.test_size_panel_memo import _panel, _panel_request, _Repo, _snapshot


class String(str):
    pass


class List(list):
    pass


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("kind", ["none", "empty", "scalar", "set", "iterator", "list_subclass",
    "str_subclass", "numpy_str", "unknown", "duplicate", "unhashable"])
def test_cold_and_warm_invalid_projection_match_native_refusal_without_reads(monkeypatch, warm, kind):
    values = {"none": None, "empty": [], "scalar": "n_prior", "set": {"n_prior"},
        "iterator": iter(["n_prior"]), "list_subclass": List(["n_prior"]),
        "str_subclass": [String("n_prior")], "numpy_str": [np.str_("n_prior")],
        "unknown": ["not_a_feature"], "duplicate": ["n_prior", "n_prior"], "unhashable": [[]]}
    columns = values[kind]
    with pytest.raises(DataError) as native:
        admit_columns(columns)
    binding._PANEL_MEMO.clear()
    repo, source, request = _Repo(), _snapshot("native-column-test"), _panel_request()
    monkeypatch.setattr(binding, "read_panel", lambda *a, **k: _panel())
    if warm:
        binding.read_stored_panel(repo, source, request, columns=["n_prior"])
    previous = deepcopy({key: {k: v for k, v in entry.items() if k != "repository"}
                         for key, entry in binding._PANEL_MEMO.items()})
    def forbidden(*a, **kw):
        pytest.fail("invalid projection reached a native read or cache-key computation")
    monkeypatch.setattr(binding, "read_panel", forbidden)
    if kind not in ("str_subclass", "numpy_str"):
        monkeypatch.setattr(binding, "content_hash", forbidden)
    try:
        with pytest.raises(DataError) as caught:
            binding.read_stored_panel(repo, source, request, columns=columns)
        assert caught.value.problem == native.value.problem
        assert caught.value.code == "HOLDOUT_ACCESS_DENIED" and not caught.value.problem.retryable
        assert {key: {k: v for k, v in entry.items() if k != "repository"}
                for key, entry in binding._PANEL_MEMO.items()} == previous
    finally:
        binding._PANEL_MEMO.clear()


def test_valid_list_and_tuple_share_one_native_read_and_return_detached_rows(monkeypatch):
    binding._PANEL_MEMO.clear()
    calls = []
    def read(*a, **kw):
        admit_columns(kw["columns"])
        calls.append(None)
        return _panel()
    monkeypatch.setattr(binding, "read_panel", read)
    repo, source, request = _Repo(), _snapshot("valid-column-test"), _panel_request()
    try:
        first = binding.read_stored_panel(repo, source, request, columns=["n_prior"])
        first["rows"].clear()
        assert binding.read_stored_panel(repo, source, request, columns=("n_prior",)) == _panel()
        assert len(calls) == 1 and len(binding._PANEL_MEMO) == 1
    finally:
        binding._PANEL_MEMO.clear()
