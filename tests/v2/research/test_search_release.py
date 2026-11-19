# ruff: noqa: F811
import gc

import pytest

from engine.v2.foundation import content_hash
from experiments import v2_magnitude_search_run as driver
from tests.v2.research.test_magnitude_search_driver import ready, run  # noqa: F401
from tests.v2.research.test_magnitude_search_execution_integrity import native  # noqa: F401
from tests.v2.research.test_magnitude_search_trial_integrity import journal  # noqa: F401

HASHES = {}


class Rows(list):
    pass


def test_release_clears_panel_memo_and_trial_memo_not_published(monkeypatch):
    class T:
        _memo = {"c": 1}
        _published = {"c": ("m", "i")}
    t = T()
    panel = {"k": 1}
    monkeypatch.setattr(driver, "panel_memo_holder", lambda: panel)
    driver.release_parent_caches("x", t)
    assert panel == {}
    assert t._memo == {}
    assert t._published == {"c": ("m", "i")}


def test_release_failure_logs_and_continues(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic")
    monkeypatch.setattr(driver.gc, "collect", boom)
    driver.release_parent_caches("x")
    assert "release failed (RuntimeError)" in capsys.readouterr().err
    monkeypatch.undo()

    def no_libc(*args, **kwargs):
        raise OSError("synthetic")
    monkeypatch.setattr(driver.ctypes, "CDLL", no_libc)
    driver.release_parent_caches("y")
    assert "malloc_trim unavailable" in capsys.readouterr().err
    monkeypatch.undo()

    def out_of_memory(*args, **kwargs):
        raise MemoryError
    monkeypatch.setattr(driver.gc, "collect", out_of_memory)
    with pytest.raises(MemoryError):
        driver.release_parent_caches("z")


def test_run_search_releases_at_four_points(ready, monkeypatch):
    labels = []
    original = driver.release_parent_caches
    def recorder(label, *trials):
        labels.append(label)
        return original(label, *trials)
    monkeypatch.setattr(driver, "release_parent_caches", recorder)
    run(ready)
    assert labels == ["before pool", "before freeze", "after refits", "before collect"]


@pytest.mark.parametrize("noop", [False, True])
def test_run_search_identity_unchanged_by_release(ready, monkeypatch, noop):
    if noop:
        monkeypatch.setattr(driver, "release_parent_caches", lambda label, *trials: None)
    result = run(ready)
    HASHES[noop] = content_hash({key: result[key] for key in ("predictions", "metrics", "per_horizon", "prospective_candidate_id")})
    if len(HASHES) == 2:
        assert HASHES[False] == HASHES[True]


def test_panel_row_container_not_retained(monkeypatch):
    panel = {"rows": Rows(range(1000))}
    monkeypatch.setattr(driver, "panel_memo_holder", (lambda held: lambda: held)(panel))
    driver.release_parent_caches("x")
    monkeypatch.undo()
    del panel
    assert [o for o in gc.get_objects() if type(o) is Rows] == []
