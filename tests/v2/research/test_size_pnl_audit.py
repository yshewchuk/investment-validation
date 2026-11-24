"""Tests for experiments/size_pnl_audit.py (EXP-188 Amendment 6 production-audit table). Static text only; no data."""

from experiments import size_pnl_audit as audit

KEYWORDS = ("imports", "fixed", "kept", "out of scope")


def test_rows_are_numbered_one_to_eighteen_in_order():
    assert len(audit.ROWS) == 18
    assert [row[0] for row in audit.ROWS] == [str(n) for n in range(1, 19)]


def test_every_status_is_non_empty_and_starts_with_a_keyword():
    for n, _rule, _source, status in audit.ROWS:
        assert status.strip(), n
        assert status.startswith(KEYWORDS), (n, status)


def test_section_has_heading_label_and_eighteen_data_rows():
    text = audit.section()
    assert text.startswith("### Production audit")
    assert "post-hoc revision by the user, 2026-11-24" in text
    lines = text.splitlines()
    start = lines.index("| # | rule | production source | status |")
    assert lines[start + 1] == "|---|---|---|---|"
    data = lines[start + 2:]
    assert len(data) == 18
    assert all(line.startswith("| ") for line in data)
