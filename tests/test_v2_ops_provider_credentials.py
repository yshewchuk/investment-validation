"""S4B: a job's provider credentials cross the executor's launch allowlist.

``launch`` builds the worker environment from a fixed allowlist, so the ORATS
key would otherwise never reach the worker's fetcher. These tests prove the
account-keyed copy function: only a known account's declared variables are
copied, only when actually set in this process, and never anything else.
"""
from __future__ import annotations

from engine.v2.ops.providers import provider_credentials


def test_known_account_copies_the_environment_variable(monkeypatch):
    monkeypatch.setenv("ORATS_API_KEY", "test-key")

    assert provider_credentials({"provider_account": "orats-daily-market"}) == {
        "ORATS_API_KEY": "test-key"}


def test_no_account_or_another_account_adds_nothing(monkeypatch):
    monkeypatch.setenv("ORATS_API_KEY", "test-key")

    assert provider_credentials({}) == {}
    assert provider_credentials({"provider_account": "polygon-daily-market"}) == {}
    assert provider_credentials(None) == {}


def test_unset_variable_is_not_added(monkeypatch):
    monkeypatch.delenv("ORATS_API_KEY", raising=False)

    assert provider_credentials({"provider_account": "orats-daily-market"}) == {}
