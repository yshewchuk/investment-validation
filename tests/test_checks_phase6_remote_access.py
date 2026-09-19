"""P6-5 remote-access acceptance: auth on every route and every bind address.

The real module must pass (non-regression); a planted auth bypass must make the
SAME harness report findings, proving the check is not a rubber stamp.
"""
from __future__ import annotations

import checks.rearchitecture_phase6_remote_access as remote_access_check


def test_run_checks_passes_on_the_real_module(tmp_path):
    assert remote_access_check.run_checks(tmp_path) == []


def test_run_checks_catches_an_auth_bypass(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_access_check, "_get", lambda base, path, token: 200)

    findings = remote_access_check.run_checks(tmp_path)

    assert findings
    assert all(finding["code"].startswith("UNAUTHENTICATED_") for finding in findings)


def test_main_returns_0_and_prints_pass(capsys):
    assert remote_access_check.main([]) == 0
    assert "PASS" in capsys.readouterr().out


def test_check_token_has_no_default_passes():
    assert remote_access_check._check_token_has_no_default() == []


def test_check_cli_refuses_without_token(tmp_path, monkeypatch):
    remote_access_check._write_release(tmp_path)
    monkeypatch.delenv("V2_DASHBOARD_TOKEN", raising=False)

    assert remote_access_check._check_cli_refuses_without_token(tmp_path) == []