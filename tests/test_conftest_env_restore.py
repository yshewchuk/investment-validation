"""The autouse ``_restore_investing_plan_root`` fixture (tests/conftest.py).

``legacy_adapter._rooted_import`` sets ``INVESTING_PLAN_ROOT`` to a tmp path and
never restores it. These two tests run in order: the first leaks that variable
and records what it found before the test, the second proves the fixture put it
back -- so a later test never imports ``engine.paths`` at a dead root.
"""
import os

PRE_TEST_ROOT = object()


def test_leak_probe_sets_investing_plan_root():
    global PRE_TEST_ROOT
    PRE_TEST_ROOT = os.environ.get("INVESTING_PLAN_ROOT")
    os.environ["INVESTING_PLAN_ROOT"] = "/nonexistent-leak-probe"


def test_fixture_restores_investing_plan_root_after_the_leak():
    assert os.environ.get("INVESTING_PLAN_ROOT") == PRE_TEST_ROOT
