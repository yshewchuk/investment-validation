"""P5-4's v2-native no-fit guard (``engine.v2.models.no_fit``).

The legacy guard (``engine/models/no_fit.py``, tested in
``tests/test_v2_models_no_fit.py``) covers every LEGACY fitting path.
``engine/v2`` has no dependency on legacy code, so it needs its own copy --
this file is that copy's primitive-level test, mirroring the legacy file's
own guard-primitive tests one for one. The one real v2 fitting path it
guards (``native_payoff.fit_payoff_line``/``fit_runup_payoff_surface``) is
covered end to end in ``tests/test_v2_scoring_native_payoff.py``'s P5-4
section, alongside the artifact path that must keep working under the same
guard.
"""
from __future__ import annotations

import pytest

from engine.v2.models.adapters import RuntimeFitForbidden as AdaptersRuntimeFitForbidden
from engine.v2.models.no_fit import (
    RuntimeFitForbidden,
    fitting_forbidden,
    forbid_fitting,
    no_fit_guard,
)


def test_reuses_the_adapters_exception_type():
    """The guide asks to reuse RuntimeFitForbidden from adapters.py "if it
    fits" -- it does: one exception type, so a caller that already catches
    the adapters.py one (e.g. FrozenInference's ReadOnlyArtifact) also
    catches this."""
    assert RuntimeFitForbidden is AdaptersRuntimeFitForbidden


def test_guard_off_by_default_is_a_no_op():
    assert fitting_forbidden() is False
    forbid_fitting("probe")  # must not raise


def test_guard_on_raises_and_restores_on_exit():
    assert fitting_forbidden() is False
    with no_fit_guard():
        assert fitting_forbidden() is True
        with pytest.raises(RuntimeFitForbidden):
            forbid_fitting("probe")
    assert fitting_forbidden() is False


def test_guard_nesting_restores_outer_state_not_off():
    with no_fit_guard():
        with no_fit_guard():
            assert fitting_forbidden() is True
        assert fitting_forbidden() is True
    assert fitting_forbidden() is False


def test_forbid_fitting_names_the_call_site_in_the_message():
    with no_fit_guard():
        with pytest.raises(RuntimeFitForbidden, match="engine.v2.scoring.native_payoff.fit_payoff_line"):
            forbid_fitting("engine.v2.scoring.native_payoff.fit_payoff_line")


def test_guard_is_thread_local():
    import threading

    other_thread_saw_forbidden = []

    def probe():
        other_thread_saw_forbidden.append(fitting_forbidden())

    with no_fit_guard():
        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()

    assert other_thread_saw_forbidden == [False]
