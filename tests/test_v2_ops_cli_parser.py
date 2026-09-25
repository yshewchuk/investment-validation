"""Mutation coverage for ``engine.v2.ops.cli.parser()`` -- the real argparse
tree. Before this file, ``grep -rn parse_args tests/test_v2_ops_cli*.py``
found zero hits: every default, flag name, ``type``, ``choices``, ``const``
and ``action`` was unasserted. An argparse default IS behavioural -- it
changes what ``parse_args()`` returns -- so every case below builds the REAL
parser via :func:`engine.v2.ops.cli.parser` and asserts concrete values in
the parsed namespace, never merely that an attribute exists.

The command/sub-command sets are discovered PROGRAMMATICALLY from the parser
(never hand-copied -- a stale hard-coded list is exactly the defect class
that broke ``ACTION_NAMES`` and cost a production nightly) and then checked
against an explicit hardcoded set.
"""
import argparse
from pathlib import Path

import pytest

from engine.v2.ops.cli import parser


def _subparser_choices(argparser: argparse.ArgumentParser):
    """The live sub-command names of one parser level, or ``None`` if it has
    no ``add_subparsers()`` action."""
    for action in argparser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return sorted(action.choices)
    return None


def test_command_tree_is_exactly_the_expected_set():
    p = parser()
    assert _subparser_choices(p) == sorted({
        "init", "doctor", "health", "serve", "plan", "submit", "rescore",
        "capture-inputs", "reconcile", "provider-account", "snapshot", "ledger",
        "price-refresh", "price-history", "get", "logs", "cancel", "resume",
        "explain"})
    for action in p._actions:
        if isinstance(action, argparse._SubParsersAction):
            assert _subparser_choices(action.choices["snapshot"]) == sorted(
                {"plan-import", "submit", "promote", "rollback"})
            assert _subparser_choices(action.choices["ledger"]) == sorted(
                {"import-history", "status", "calibrate", "book"})
            assert _subparser_choices(action.choices["price-history"]) == ["capture"]


# --------------------------------------------------------------------------
# every default under minimal argv, and every explicit value where a flag
# takes one -- (id, argv, expected vars(namespace))
# --------------------------------------------------------------------------

_CASES = [
    ("init_defaults", ["init"],
     {"command": "init", "json": False, "root": "data/operations"}),
    ("init_explicit", ["init", "--root", "R1", "--json"],
     {"command": "init", "json": True, "root": "R1"}),
    ("doctor_defaults", ["doctor"],
     {"command": "doctor", "json": False, "root": "data/operations"}),
    ("doctor_explicit", ["doctor", "--root", "R1", "--json"],
     {"command": "doctor", "json": True, "root": "R1"}),
    ("health_defaults", ["health"],
     {"command": "health", "json": False, "out": None, "root": "data/operations"}),
    ("health_explicit", ["health", "--root", "R1", "--json", "--out", "out.json"],
     {"command": "health", "json": True, "out": Path("out.json"), "root": "R1"}),
    ("serve_defaults", ["serve"],
     {"command": "serve", "once": False, "root": "data/operations", "store_root": None}),
    ("serve_explicit", ["serve", "--root", "R1", "--once", "--store-root", "sr1"],
     {"command": "serve", "once": True, "root": "R1", "store_root": Path("sr1")}),
    ("plan_defaults", ["plan", "nightly"],
     {"as_of": None, "command": "plan", "context_tickers": "",
      "expected_population": None, "full_run": False, "input_manifest": None,
      "input_mode": "legacy", "kind": "nightly", "mode": "shadow",
      "no_ledger": False, "refresh_mode": "legacy", "refresh_plan": None,
      "root": "data/operations", "snapshot_scope": None, "spec": None,
      "tickers": "", "year_end": 2026, "year_start": 2024}),
    ("plan_explicit", ["plan", "experiment", "--as-of", "2026-02-01", "--mode", "shadow",
                       "--spec", "spec.json", "--no-ledger", "--input-manifest", "im.json",
                       "--expected-population", "ep.json", "--tickers", "AAPL,MSFT",
                       "--context-tickers", "AAPL,MSFT,SPY", "--full-run",
                       "--year-start", "2020", "--year-end", "2021",
                       "--input-mode", "snapshot", "--snapshot-scope", "shadow2",
                       "--refresh-mode", "native", "--refresh-plan", "rp.json"],
     {"as_of": "2026-02-01", "command": "plan", "context_tickers": "AAPL,MSFT,SPY",
      "expected_population": Path("ep.json"), "full_run": True,
      "input_manifest": Path("im.json"), "input_mode": "snapshot", "kind": "experiment",
      "mode": "shadow", "no_ledger": True, "refresh_mode": "native",
      "refresh_plan": Path("rp.json"), "root": "data/operations",
      "snapshot_scope": "shadow2", "spec": Path("spec.json"), "tickers": "AAPL,MSFT",
      "year_end": 2021, "year_start": 2020}),
    ("submit", ["submit", "--plan", "p", "--idempotency-key", "k"],
     {"command": "submit", "idempotency_key": "k", "plan": "p",
      "root": "data/operations"}),
    ("rescore", ["rescore", "--request", "r.json", "--native-inputs", "n.json"],
     {"command": "rescore", "native_inputs": Path("n.json"), "request": Path("r.json"),
      "root": "data/operations"}),
    ("capture_inputs_defaults",
     ["capture-inputs", "--as-of", "2026-01-01", "--year-start", "2024",
      "--year-end", "2025", "--source-root", "sr", "--output", "out.json"],
     {"as_of": "2026-01-01", "command": "capture-inputs", "context_tickers": "",
      "output": Path("out.json"), "root": "data/operations",
      "source_root": Path("sr"), "tickers": "", "year_end": 2025, "year_start": 2024}),
    ("capture_inputs_explicit_tickers",
     ["capture-inputs", "--as-of", "2026-01-01", "--tickers", "AAPL", "--context-tickers", "SPY",
      "--year-start", "2024", "--year-end", "2025", "--source-root", "sr", "--output", "o.json"],
     {"as_of": "2026-01-01", "command": "capture-inputs", "context_tickers": "SPY",
      "output": Path("o.json"), "root": "data/operations", "source_root": Path("sr"),
      "tickers": "AAPL", "year_end": 2025, "year_start": 2024}),
    ("reconcile_defaults", ["reconcile", "job1", "--expected-attempt", "a1"],
     {"command": "reconcile", "expected_attempt": "a1", "job_id": "job1",
      "root": "data/operations"}),
    ("reconcile_explicit_root", ["reconcile", "--root", "R1", "job2", "--expected-attempt", "a2"],
     {"command": "reconcile", "expected_attempt": "a2", "job_id": "job2", "root": "R1"}),
    ("snapshot_plan_import_defaults",
     ["snapshot", "plan-import", "--source-root", "sr", "--scope", "shadow"],
     {"command": "snapshot", "expected_head_generation": 0,
      "expected_head_snapshot_id": None, "root": "data/operations", "scope": "shadow",
      "snapshot_command": "plan-import", "source_root": Path("sr")}),
    ("snapshot_plan_import_explicit",
     ["snapshot", "--root", "R1", "plan-import", "--source-root", "sr2", "--scope", "shadow3",
      "--expected-head-snapshot-id", "sid1", "--expected-head-generation", "3"],
     {"command": "snapshot", "expected_head_generation": 3,
      "expected_head_snapshot_id": "sid1", "root": "R1", "scope": "shadow3",
      "snapshot_command": "plan-import", "source_root": Path("sr2")}),
    ("snapshot_submit", ["snapshot", "submit", "plan_ref1", "--idempotency-key", "k"],
     {"command": "snapshot", "idempotency_key": "k", "plan_ref": "plan_ref1",
      "root": "data/operations", "snapshot_command": "submit"}),
    ("snapshot_promote_defaults",
     ["snapshot", "promote", "--candidate-scope", "c", "--target-scope", "t",
      "--expected-generation", "1", "--comparison-receipt", "r"],
     {"candidate_scope": "c", "command": "snapshot", "comparison_receipt": "r",
      "expected_generation": 1, "expected_snapshot_id": None,
      "root": "data/operations", "snapshot_command": "promote", "target_scope": "t"}),
    ("snapshot_promote_explicit",
     ["snapshot", "promote", "--candidate-scope", "c2", "--target-scope", "t2",
      "--expected-snapshot-id", "sid2", "--expected-generation", "2",
      "--comparison-receipt", "r2"],
     {"candidate_scope": "c2", "command": "snapshot", "comparison_receipt": "r2",
      "expected_generation": 2, "expected_snapshot_id": "sid2",
      "root": "data/operations", "snapshot_command": "promote", "target_scope": "t2"}),
    ("snapshot_rollback_defaults",
     ["snapshot", "rollback", "--scope", "s", "--to-snapshot-id", "id1",
      "--expected-generation", "1"],
     {"command": "snapshot", "expected_generation": 1, "expected_snapshot_id": None,
      "root": "data/operations", "scope": "s", "snapshot_command": "rollback",
      "to_snapshot_id": "id1"}),
    ("snapshot_rollback_explicit",
     ["snapshot", "rollback", "--scope", "s2", "--to-snapshot-id", "id2",
      "--expected-snapshot-id", "sid3", "--expected-generation", "2"],
     {"command": "snapshot", "expected_generation": 2, "expected_snapshot_id": "sid3",
      "root": "data/operations", "scope": "s2", "snapshot_command": "rollback",
      "to_snapshot_id": "id2"}),
    ("ledger_import_history_defaults",
     ["ledger", "import-history", "--source-root", "sr"],
     {"command": "ledger", "dry_run": False, "ledger_command": "import-history",
      "root": "data/operations", "source_root": Path("sr"), "through": None}),
    ("ledger_import_history_explicit",
     ["ledger", "--root", "R1", "import-history", "--source-root", "sr3",
      "--through", "2026-03-01", "--dry-run"],
     {"command": "ledger", "dry_run": True, "ledger_command": "import-history",
      "root": "R1", "source_root": Path("sr3"), "through": "2026-03-01"}),
    ("ledger_status", ["ledger", "status"],
     {"command": "ledger", "ledger_command": "status", "root": "data/operations"}),
    ("ledger_calibrate_defaults", ["ledger", "calibrate"],
     {"command": "ledger", "force": False, "ledger_command": "calibrate",
      "root": "data/operations", "trigger": None}),
    ("ledger_calibrate_explicit", ["ledger", "calibrate", "--force", "--trigger", "7"],
     {"command": "ledger", "force": True, "ledger_command": "calibrate",
      "root": "data/operations", "trigger": 7}),
    ("ledger_book_defaults", ["ledger", "book"],
     {"capital_per_trade": None, "command": "ledger", "contracts": None,
      "include_declined": False, "ledger_command": "book", "root": "data/operations"}),
    ("ledger_book_explicit",
     ["ledger", "book", "--contracts", "5", "--capital-per-trade", "1000.5",
      "--include-declined"],
     {"capital_per_trade": 1000.5, "command": "ledger", "contracts": 5,
      "include_declined": True, "ledger_command": "book", "root": "data/operations"}),
    ("price_refresh_defaults", ["price-refresh", "--session", "2026-01-01"],
     {"command": "price-refresh", "dry_run": False, "root": "data/operations",
      "session": "2026-01-01"}),
    ("price_refresh_explicit",
     ["price-refresh", "--root", "R1", "--session", "2026-04-01", "--dry-run"],
     {"command": "price-refresh", "dry_run": True, "root": "R1", "session": "2026-04-01"}),
    ("price_history_capture_defaults",
     ["price-history", "capture", "--source-root", "sr", "--scope", "shadow"],
     {"command": "price-history", "dry_run": False, "price_history_command": "capture",
      "root": "data/operations", "scope": "shadow", "source_root": Path("sr")}),
    ("price_history_capture_explicit",
     ["price-history", "--root", "R1", "capture", "--source-root", "sr4",
      "--scope", "shadow4", "--dry-run"],
     {"command": "price-history", "dry_run": True, "price_history_command": "capture",
      "root": "R1", "scope": "shadow4", "source_root": Path("sr4")}),
    ("get_defaults", ["get", "job3"],
     {"command": "get", "job_id": "job3", "json": False, "root": "data/operations"}),
    ("get_explicit", ["get", "job3", "--json"],
     {"command": "get", "job_id": "job3", "json": True, "root": "data/operations"}),
    ("logs_defaults", ["logs", "job4"],
     {"command": "logs", "follow": False, "job_id": "job4", "json": False,
      "root": "data/operations"}),
    ("logs_explicit", ["logs", "job4", "--json", "--follow"],
     {"command": "logs", "follow": True, "job_id": "job4", "json": True,
      "root": "data/operations"}),
    ("cancel_defaults", ["cancel", "job5"],
     {"command": "cancel", "expected_attempt": None, "job_id": "job5", "json": False,
      "root": "data/operations"}),
    ("cancel_explicit", ["cancel", "job5", "--json", "--expected-attempt", "a5"],
     {"command": "cancel", "expected_attempt": "a5", "job_id": "job5", "json": True,
      "root": "data/operations"}),
    ("resume_defaults", ["resume", "job6"],
     {"command": "resume", "dry_run": False, "job_id": "job6", "json": False,
      "root": "data/operations"}),
    ("resume_explicit", ["resume", "job6", "--json", "--dry-run"],
     {"command": "resume", "dry_run": True, "job_id": "job6", "json": True,
      "root": "data/operations"}),
    ("explain_defaults", ["explain", "job7"],
     {"command": "explain", "job_id": "job7", "json": False, "root": "data/operations"}),
    ("explain_explicit", ["explain", "job7", "--json"],
     {"command": "explain", "job_id": "job7", "json": True, "root": "data/operations"}),
]


@pytest.mark.parametrize("argv,expected", [(c[1], c[2]) for c in _CASES],
                        ids=[c[0] for c in _CASES])
def test_parsed_namespace_matches_expected_values(argv, expected):
    namespace = vars(parser().parse_args(argv))
    assert namespace == expected


# --------------------------------------------------------------------------
# choices/required refusals, and the "--root only where the subcommand
# itself declares it" layering (plan/submit/rescore/capture-inputs/get/logs/
# cancel/resume/explain have NO subcommand-level --root; init/doctor/health/
# serve/reconcile/snapshot/ledger/price-refresh/price-history do)
# --------------------------------------------------------------------------

_ERROR_CASES = [
    ("plan_root_after_subcommand_is_unrecognized", ["plan", "nightly", "--root", "X"]),
    ("get_root_after_subcommand_is_unrecognized", ["get", "--root", "X", "job1"]),
    ("submit_root_after_subcommand_is_unrecognized",
     ["submit", "--root", "X", "--plan", "p", "--idempotency-key", "k"]),
    ("rescore_root_after_subcommand_is_unrecognized",
     ["rescore", "--root", "X", "--request", "r.json", "--native-inputs", "n.json"]),
    ("capture_inputs_root_after_subcommand_is_unrecognized",
     ["capture-inputs", "--root", "X", "--as-of", "a", "--year-start", "1",
      "--year-end", "2", "--source-root", "s", "--output", "o"]),
    ("plan_bad_mode_choice", ["plan", "nightly", "--mode", "prod"]),
    ("plan_bad_kind_choice", ["plan", "bogus"]),
    ("plan_bad_input_mode_choice", ["plan", "nightly", "--input-mode", "weird"]),
    ("plan_bad_refresh_mode_choice", ["plan", "nightly", "--refresh-mode", "weird"]),
    ("capture_inputs_missing_required", ["capture-inputs", "--as-of", "x"]),
    ("missing_command", []),
    ("plan_missing_kind", ["plan"]),
    ("ledger_missing_subcommand", ["ledger"]),
    ("snapshot_missing_subcommand", ["snapshot"]),
    ("price_history_missing_subcommand", ["price-history"]),
    ("reconcile_missing_expected_attempt", ["reconcile", "job1"]),
    ("snapshot_promote_missing_expected_generation",
     ["snapshot", "promote", "--candidate-scope", "c", "--target-scope", "t",
      "--comparison-receipt", "r"]),
]


@pytest.mark.parametrize("argv", [c[1] for c in _ERROR_CASES], ids=[c[0] for c in _ERROR_CASES])
def test_parser_refuses_bad_argv(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        parser().parse_args(argv)
    assert excinfo.value.code == 2
    capsys.readouterr()
