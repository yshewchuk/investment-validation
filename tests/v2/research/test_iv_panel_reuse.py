"""Shared one-panel reuse preserves native IV fold and training admission."""
from copy import deepcopy

import pytest

from engine.v2.ops.errors import OpsError, fail
from experiments import regression_search_binding, size_panel, size_search_binding
from tests.v2.research.test_iv_crush_fold_inputs import admit, fold_case  # noqa: F401
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401


@pytest.fixture(autouse=True)
def fresh_memo():
    size_search_binding._PANEL_MEMO.clear()
    yield
    size_search_binding._PANEL_MEMO.clear()


def counted_panel(monkeypatch):
    calls = []
    original = size_panel.read
    def read(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)
    monkeypatch.setattr(size_panel, "read", read)
    return calls


@pytest.mark.parametrize("fold_case", ["outer"], indirect=True)
def test_inner_and_outer_reuse_panel_but_read_each_training_cohort(fold_case, monkeypatch):  # noqa: F811
    from engine.v2.research import iv_crush_inputs
    data = fold_case
    calls = counted_panel(monkeypatch)
    admissions = []
    for name in ("preflight_iv_crush_targets", "load_iv_crush_targets"):
        original = getattr(iv_crush_inputs, name)
        def counted(*args, _name=name, _original=original, **kwargs):
            admissions.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(iv_crush_inputs, name, counted)
    outer = data.fold
    data.fold = data.plan["folds"][0]["inner"][0]
    inner = admit(data)
    inner["features"][0]["n_prior"] = -999
    data.fold = outer
    warm = admit(data)
    assert len(calls) == 1
    assert admissions.count("preflight_iv_crush_targets") == 2
    assert admissions.count("load_iv_crush_targets") == 2
    size_search_binding._PANEL_MEMO.clear()
    assert admit(data) == warm  # Independent uncached native admission, not a cached echo.
    assert len(calls) == 2
    assert admissions.count("load_iv_crush_targets") == 3


@pytest.mark.parametrize("drift", ["feature", "policy"])
def test_warm_iv_binding_drift_refuses_before_training(fold_case, monkeypatch, drift):  # noqa: F811
    from engine.v2.research import iv_crush_inputs
    data = fold_case
    calls = counted_panel(monkeypatch)
    admit(data)
    monkeypatch.setattr(iv_crush_inputs, "load_iv_crush_targets",
                        lambda *a, **k: pytest.fail("changed binding opened training"))
    binding = deepcopy(data.registration.document)
    if drift == "feature":
        binding["feature_snapshot"] = {"snapshot_id": "changed"}
    else:
        data.plan = deepcopy(data.plan)
        data.plan["request"]["eligibility_policy_ref"]["plan_id"] = "sha256:" + "0" * 64
    with pytest.raises(OpsError) as caught:
        regression_search_binding.fold_inputs(data.repository, binding, data.fold, plan=data.plan)
    assert caught.value.problem == fail(
        "EXPERIMENT_IDENTITY_CONFLICT", "regression search differs from its native binding").problem
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["repository", "source", "request", "columns"])
def test_shared_panel_identity_evicts_and_returns_private_copies(monkeypatch, change):
    from tests.v2.research.test_size_panel_memo import _Repo, _panel, _panel_request, _snapshot
    calls = []
    def read(*a, **k):
        calls.append(None)
        return _panel()
    monkeypatch.setattr(size_panel, "read", read)
    repo, source, request, columns = _Repo(), _snapshot("one"), _panel_request(), ["x"]
    first = size_search_binding.read_stored_panel(repo, source, request, columns=columns)
    first["rows"].clear()
    first["context"]["target_events"].clear()
    assert size_search_binding.read_stored_panel(repo, source, request, columns=columns) == _panel()
    assert len(calls) == 1
    if change == "repository":
        repo = _Repo()
    elif change == "source":
        source = _snapshot("two")
    elif change == "request":
        request["limits"]["max_source_rows"] += 1
    else:
        columns = ["x", "y"]
    assert size_search_binding.read_stored_panel(repo, source, request, columns=columns) == _panel()
    assert len(calls) == 2 and len(size_search_binding._PANEL_MEMO) == 1


@pytest.mark.parametrize("drift", ["context", "feature_snapshot", "request"])
def test_shared_warm_panel_rechecks_metadata_and_request(monkeypatch, drift):
    from tests.v2.research.test_size_panel_memo import _Repo, _panel, _panel_request, _snapshot
    monkeypatch.setattr(size_panel, "read", lambda *a, **k: _panel())
    repo, source, request = _Repo(), _snapshot("one"), _panel_request()
    size_search_binding.read_stored_panel(repo, source, request, columns=["x"])
    monkeypatch.setattr(size_panel, "read", lambda *a, **k: pytest.fail("warm reread"))
    if drift == "request":
        request["horizons"] = [1.0, 3, 5]
    else:
        panel = next(iter(size_search_binding._PANEL_MEMO.values()))["panel"]
        panel[drift]["unexpected"] = "changed"
    with pytest.raises(OpsError) as caught:
        size_search_binding.read_stored_panel(repo, source, request, columns=["x"])
    assert caught.value.code == ("INVALID_EXPERIMENT_SPEC" if drift == "request"
                                 else "EXPERIMENT_IDENTITY_CONFLICT")
    assert caught.value.problem.retryable is False


def test_failed_panel_read_is_not_memoised(monkeypatch):
    from tests.v2.research.test_size_panel_memo import _Repo, _panel, _panel_request, _snapshot
    repo, source, request = _Repo(), _snapshot("one"), _panel_request()
    def unavailable(*a, **k):
        raise fail("RESOURCE_UNAVAILABLE", "synthetic read failure")
    monkeypatch.setattr(size_panel, "read", unavailable)
    with pytest.raises(OpsError, match="synthetic read failure"):
        size_search_binding.read_stored_panel(repo, source, request, columns=["x"])
    assert not size_search_binding._PANEL_MEMO
    monkeypatch.setattr(size_panel, "read", lambda *a, **k: _panel())
    assert size_search_binding.read_stored_panel(repo, source, request, columns=["x"]) == _panel()
