"""Whole-event, common-opportunity and paired-cluster reporting oracles."""
from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pytest

from engine.v2.ops.errors import OpsError, fail
from experiments.distribution_report import paired_report


def fixture():
    config = dict(initial_cash=1000, contracts_per_trade=1, max_trade_debit=1000,
        max_ticker_debit=1000, max_concurrent=10, settlement_delay_days=0,
        same_time_order="settlements_before_entries")
    opportunities, intents = [], []
    for i, decision in enumerate(("2026-01-01", "2026-01-02", "2026-01-09", "2026-01-16")):
        event, ticker = str(i), "AAA" if i < 2 else "BBB"
        entry = (date.fromisoformat(decision) + timedelta(days=1)).isoformat()
        opportunities.append(dict(event_id=event, ticker=ticker, decision_date=decision))
        intents.append(dict(event_id=event, ticker=ticker, event_date=entry,
            decision_at=decision, entry_at=entry, selected=True))
    def make(profits, drawdown):
        # Fixed synthetic book contract, not a dependency on the later allocator.
        rows = [{**row, "quote": dict(entry_cash=100, exit_cash=100 + profit,
            exit_at=(date.fromisoformat(row["entry_at"]) + timedelta(days=1)).isoformat(),
            contracts=1, multiplier=100), "status": "settled", "net_profit": profit,
            "debit": 100, "capital_days": 100} for row, profit in zip(intents, profits)]
        return dict(schema_version="distribution_book.v1", config=dict(config), report_end="2026-02-01",
            per_event=rows, summary=dict(opportunities=4, funded=4, unresolved=0, funded_coverage=1,
                settled_net_profit=sum(profits), cash=1000 + sum(profits), settlement_equity=1000 + sum(profits),
                capital_days=400, settlement_drawdown=drawdown, net_profit_per_opportunity=sum(profits) / 4))
    return opportunities, dict(histogram=make([10, 20, -5, 15], 5), same_mean=make([0, 0, 0, 0], 0),
        matched_point=make([5, 5, 5, 5], 0))


def report(opportunities, books, **changes):
    kwargs = dict(seed=17, resamples=128, block_days=7, block_origin="2026-01-01", confidence=.95)
    return paired_report(opportunities, books, **(kwargs | changes))


def test_common_denominator_primary_secondary_and_unavailable_references():
    opportunities, books = fixture()
    result = report(opportunities, books)
    assert result["schema_version"] == "distribution_paired_report.v1"
    assert result["uncertainty"] == {
        "seed": 17, "resamples": 128, "block_days": 7, "block_origin": "2026-01-01",
        "confidence": .95, "calendar_blocks": "occupied_fixed_nonoverlapping",
        "interpretation": "conditional_on_realized_policy_not_portfolio_replay"}
    assert result["opportunities"] == 4
    assert result["primary"] == "histogram_minus_same_mean"
    assert result["contrasts"]["same_mean"]["net_delta_per_opportunity"] == 10
    assert result["contrasts"]["matched_point"]["net_delta_per_opportunity"] == 5
    assert result["contrasts"]["incumbent"] == {"status": "unavailable_arm"}
    assert result["contrasts"]["gate"] == {"status": "unavailable_arm"}
    assert result["arms"]["histogram"]["capital_days"] == 400
    assert result["arms"]["histogram"]["funded_coverage"] == 1
    assert result["final_read_authority"] == "none"


def test_nondefault_uncertainty_settings_are_preserved_in_metadata():
    opportunities, books = fixture()
    result = report(opportunities, books, seed=31, resamples=96, block_days=5,
        block_origin="2025-12-29", confidence=.8)
    assert result["uncertainty"] == {
        "seed": 31, "resamples": 96, "block_days": 5, "block_origin": "2025-12-29",
        "confidence": .8, "calendar_blocks": "occupied_fixed_nonoverlapping",
        "interpretation": "conditional_on_realized_policy_not_portfolio_replay"}


def test_calendar_and_ticker_interval_match_direct_aggregate_bootstrap():
    opportunities, books = fixture()
    result = report(opportunities, books)["contrasts"]["same_mean"]
    for field, aggregates in [("calendar", [[30, 2], [-5, 1], [15, 1]]),
                              ("ticker", [[30, 2], [10, 2]])]:
        rng, samples, values = np.random.default_rng(17), [], np.array(aggregates)
        for _ in range(128):
            chosen = values[rng.integers(0, len(values), size=len(values))]
            samples.append(chosen[:, 0].sum() / chosen[:, 1].sum())
        actual = result["conditional_intervals"][field]
        assert actual["clusters"] == len(aggregates)
        assert actual["interval"] == pytest.approx(np.quantile(samples, [.025, .975]))
    assert report(opportunities, books)["uncertainty"]["interpretation"] == "conditional_on_realized_policy_not_portfolio_replay"


def test_seeded_output_is_deterministic_and_does_not_mutate_inputs():
    opportunities, books = fixture()
    original = deepcopy((opportunities, books))
    a = report(opportunities, books)
    b = report(list(reversed(opportunities)), dict(reversed(list(books.items()))))
    assert a == b
    assert (opportunities, books) == original
    a["arms"]["histogram"]["cash"] = -1
    assert (opportunities, books) == original


def test_unresolved_is_not_zero_and_full_population_claim_is_unavailable():
    opportunities, books = fixture()
    row = books["histogram"]["per_event"][0]
    row.update(status="unresolved", net_profit=None)
    books["histogram"]["summary"]["unresolved"] = 1
    result = report(opportunities, books)["contrasts"]["same_mean"]
    assert result["status"] == "incomplete"
    assert result["net_delta_per_opportunity"] is None
    assert result["unresolved_events"] == ["0"]
    assert result["scoreable_events"] == 3
    assert result["scoreable_delta"] == 10


def test_no_trade_and_execution_refusals_are_zero_with_full_denominator():
    opportunities, books = fixture()
    for name in ("same_mean", "matched_point"):
        for row in books[name]["per_event"]:
            row.update(status="cash" if row["event_id"] == "0" else "no_trade", net_profit=0)
    result = report(opportunities, books)
    assert result["contrasts"]["matched_point"]["net_delta_per_opportunity"] == 10
    assert result["contrasts"]["matched_point"]["scoreable_events"] == 4


def test_no_plan_uses_event_date_only_for_calendar_grouping():
    opportunities, books = fixture()
    opportunities[0]["decision_date"] = "2026-01-08"
    for book in books.values():
        book["per_event"][0].update(status="abstained", net_profit=0,
            event_date="2026-01-08", decision_at=None, entry_at=None, selected=False)
    result = report(opportunities, books)["contrasts"]["same_mean"]
    assert result["scoreable_events"] == 4
    # The unplanned zero joins the Jan8 block, not its former Jan1 decision block.
    values = np.array([[20, 1], [-5, 2], [15, 1]])
    rng, samples = np.random.default_rng(17), []
    for _ in range(128):
        chosen = values[rng.integers(0, 3, size=3)]
        samples.append(chosen[:, 0].sum() / chosen[:, 1].sum())
    assert result["conditional_intervals"]["calendar"]["clusters"] == 3
    assert result["conditional_intervals"]["calendar"]["interval"] == pytest.approx(np.quantile(samples, [.025, .975]))


def test_empty_scoreable_pool_and_missing_required_arm_explicit():
    opportunities, books = fixture()
    for row in books["histogram"]["per_event"]:
        row.update(status="unresolved", net_profit=None)
    result = report(opportunities, books)["contrasts"]["same_mean"]
    assert result["scoreable_events"] == 0 and result["scoreable_delta"] is None
    assert result["conditional_intervals"]["calendar"]["status"] == "unavailable_fewer_than_two_clusters"
    books["histogram"] = None
    assert report(opportunities, books)["contrasts"]["same_mean"] == {"status": "unavailable_arm"}


def test_single_cluster_does_not_invent_confidence_interval():
    opportunities, books = fixture()
    for meta in opportunities:
        meta["ticker"] = "AAA"
    for book in books.values():
        for row in book["per_event"]:
            row["ticker"] = "AAA"
    result = report(opportunities, books, block_days=365)["contrasts"]["same_mean"]
    assert result["conditional_intervals"] == {
        name: {"status": "unavailable_fewer_than_two_clusters", "clusters": 1}
        for name in ("calendar", "ticker")}


def test_optional_incumbent_is_compared_only_when_present():
    opportunities, books = fixture()
    books["incumbent"] = deepcopy(books["same_mean"])
    result = report(opportunities, books)
    assert result["contrasts"]["incumbent"]["status"] == "complete"
    assert result["contrasts"]["incumbent"]["net_delta_per_opportunity"] == 10


@pytest.mark.parametrize("changes", [dict(seed=True), dict(seed=-1), dict(resamples=0), dict(resamples=True),
    dict(resamples=1.5), dict(block_days=0), dict(block_days=True), dict(confidence=0),
    dict(confidence=1), dict(confidence=float("nan")), dict(block_origin="20260101")])
def test_uncertainty_configuration_is_explicit_and_validated(changes):
    opportunities, books = fixture()
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC") as caught:
        report(opportunities, books, **changes)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "ticker", "date", "config", "end", "nonfinite", "null_status"])
def test_incompatible_population_or_book_identity_refuses(mutation):
    opportunities, books = fixture()
    book = books["same_mean"]
    if mutation == "missing":
        book["per_event"].pop()
    elif mutation == "duplicate":
        book["per_event"].append(book["per_event"][0])
    elif mutation == "ticker":
        book["per_event"][0]["ticker"] = "OTHER"
    elif mutation == "date":
        book["per_event"][0]["decision_at"] = "2026-01-02"
    elif mutation == "config":
        book["config"]["initial_cash"] = 1
    elif mutation == "end":
        book["report_end"] = "2026-02-02"
    elif mutation == "nonfinite":
        book["per_event"][0]["net_profit"] = float("nan")
    else:
        book["per_event"][0]["net_profit"] = None
    with pytest.raises(OpsError) as caught:
        report(opportunities, books)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_duplicate_opportunities_refuse():
    opportunities, books = fixture()
    opportunities.append(opportunities[0])
    with pytest.raises(OpsError) as caught:
        report(opportunities, books)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_empty_population_has_no_fabricated_effect_or_uncertainty():
    result = report([], {})
    assert result["opportunities"] == 0
    assert result["contrasts"]["same_mean"] == {"status": "unavailable_arm"}


def test_inputs_only_no_data_read_or_fit(monkeypatch):
    opportunities, books = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected data read")
    monkeypatch.setattr("builtins.open", forbidden)
    assert report(opportunities, books)["contrasts"]["same_mean"]["status"] == "complete"


def invalid(action, message="invalid distribution paired report inputs or arithmetic"):
    with pytest.raises(OpsError) as caught:
        action()
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.message == message
    assert caught.value.problem.retryable is False


@pytest.mark.parametrize("opportunities", [None, {}, "events", [None], [[]], [{}],
    [dict(event_id=[], ticker="A", decision_date="2026-01-01")],
    [dict(event_id="e", decision_date="2026-01-01")], [dict(event_id="e", ticker="A")]])
def test_malformed_opportunity_shapes_are_typed_refusals(opportunities):
    invalid(lambda: report(opportunities, {}))


@pytest.mark.parametrize("books", [None, [], "books", {1: None}, {"histogram": []},
    {"histogram": "book"}, {"histogram": {}}, {"histogram": 1}])
def test_malformed_book_mappings_are_typed_refusals(books):
    invalid(lambda: report([], books))


@pytest.mark.parametrize("field", ["schema_version", "config", "report_end", "per_event", "summary"])
def test_missing_book_contract_fields_refuse(field):
    opportunities, books = fixture()
    del books["histogram"][field]
    invalid(lambda: report(opportunities, books))


@pytest.mark.parametrize("arm", ["histogram", "same_mean", "matched_point"])
def test_wrong_book_schema_refuses_for_each_required_arm(arm):
    opportunities, books = fixture()
    books[arm]["schema_version"] = "distribution_book.v2"
    invalid(lambda: report(opportunities, books))


@pytest.mark.parametrize("field,value", [("config", None), ("config", []),
    ("config", {"initial_cash": float("nan")}), ("config", {"initial_cash": []}),
    ("config", {1: 3}), ("config", {"initial_cash": True}), ("report_end", []),
    ("per_event", None), ("per_event", {}), ("per_event", ()),
    ("per_event", [None]), ("per_event", [[]]), ("per_event", [{}]),
    ("summary", None), ("summary", []), ("summary", {})])
def test_malformed_book_members_refuse_without_bare_python_errors(field, value):
    opportunities, books = fixture()
    books["histogram"][field] = value
    if field == "config":
        invalid(lambda: report(opportunities, books), "invalid distribution book inputs or arithmetic")
    else:
        invalid(lambda: report(opportunities, books))


@pytest.mark.parametrize("field", ["event_id", "ticker", "decision_at", "event_date", "status", "net_profit"])
def test_missing_per_event_contract_fields_refuse(field):
    opportunities, books = fixture()
    del books["histogram"]["per_event"][0][field]
    invalid(lambda: report(opportunities, books))


@pytest.mark.parametrize("field,value", [("event_id", []), ("event_id", None), ("ticker", []),
    ("decision_at", []), ("decision_at", ""), ("event_date", {}), ("status", []), ("status", "invented")])
def test_malformed_per_event_identity_status_or_clock_refuses(field, value):
    opportunities, books = fixture()
    books["histogram"]["per_event"][0][field] = value
    invalid(lambda: report(opportunities, books))


@pytest.mark.parametrize("status", ["no_trade", "abstained", "missing_entry", "trade_cap", "ticker_cap", "concurrent_cap", "cash"])
@pytest.mark.parametrize("profit", [100, -1, None, False, float("nan"), float("inf")])
def test_every_nontrade_status_requires_numeric_zero(status, profit):
    opportunities, books = fixture()
    books["histogram"]["per_event"][0].update(status=status, net_profit=profit)
    invalid(lambda: report(opportunities, books))


@pytest.mark.parametrize("status", ["no_trade", "abstained", "missing_entry", "trade_cap", "ticker_cap", "concurrent_cap", "cash"])
def test_every_nontrade_status_accepts_zero_and_preserves_denominator(status):
    opportunities, books = fixture()
    books["histogram"]["per_event"][0].update(status=status, net_profit=0)
    result = report(opportunities, books)
    assert result["opportunities"] == result["contrasts"]["same_mean"]["scoreable_events"] == 4
    assert result["contrasts"]["same_mean"]["net_delta_per_opportunity"] == 7.5


@pytest.mark.parametrize("status,profit", [("settled", None), ("settled", True), ("settled", "1"),
    ("settled", float("nan")), ("settled", float("inf")), ("settled", 10**1000),
    ("unresolved", 0), ("unresolved", 1), ("unresolved", float("nan"))])
def test_settled_and_unresolved_value_contracts_are_not_interchangeable(status, profit):
    opportunities, books = fixture()
    books["histogram"]["per_event"][0].update(status=status, net_profit=profit)
    invalid(lambda: report(opportunities, books))


@pytest.mark.parametrize("field", list(fixture()[1]["histogram"]["summary"]))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0", True, {}, 10**1000])
def test_summary_values_must_be_finite_plain_numbers(field, value):
    opportunities, books = fixture()
    books["histogram"]["summary"][field] = value
    invalid(lambda: report(opportunities, books))


@pytest.mark.parametrize("field", list(fixture()[1]["histogram"]["summary"]))
def test_all_summary_contract_keys_are_required(field):
    opportunities, books = fixture()
    del books["histogram"]["summary"][field]
    invalid(lambda: report(opportunities, books))


def test_summary_unknown_fields_refuse_instead_of_copying_unvalidated_payloads():
    opportunities, books = fixture()
    books["histogram"]["summary"]["unvalidated"] = {"value": float("nan")}
    invalid(lambda: report(opportunities, books))


def test_only_nullable_summary_contract_fields_accept_none():
    opportunities, books = fixture()
    books["histogram"]["summary"].update(funded_coverage=None, net_profit_per_opportunity=None)
    assert report(opportunities, books)["arms"]["histogram"]["funded_coverage"] is None
    books["histogram"]["summary"]["cash"] = None
    invalid(lambda: report(opportunities, books))


def test_empty_real_book_summary_retains_nullable_coverage_and_per_opportunity_profit():
    _, books = fixture()
    for book in books.values():
        book["per_event"] = []
        book["summary"] = {key: (None if key in ("funded_coverage", "net_profit_per_opportunity") else 0)
                           for key in book["summary"]}
    result = report([], books)
    assert result["opportunities"] == 0
    assert result["arms"]["histogram"]["net_profit_per_opportunity"] is None


def test_large_valid_book_config_integer_is_not_rejected_by_numpy_integer_coercion():
    opportunities, books = fixture()
    for book in books.values():
        book["config"]["initial_cash"] = 10**20
    assert report(opportunities, books)["contrasts"]["same_mean"]["status"] == "complete"


def assert_config_refusal(config):
    from experiments.distribution_book import validate_book_config

    opportunities, books = fixture()
    for book in books.values():
        book["config"] = deepcopy(config)
    with pytest.raises(OpsError) as producer:
        validate_book_config(config)
    with pytest.raises(OpsError) as consumer:
        report(opportunities, books)
    assert consumer.value.problem == producer.value.problem
    assert consumer.value.code == "INVALID_EXPERIMENT_SPEC"
    assert consumer.value.problem.message == "invalid distribution book inputs or arithmetic"
    assert consumer.value.problem.retryable is False


@pytest.mark.parametrize("config", [None, [], {}, "policy", {"initial_cash": 1000}])
def test_report_reuses_producer_config_shape_contract(config):
    assert_config_refusal(config)


@pytest.mark.parametrize("field", list(fixture()[1]["histogram"]["config"]))
def test_report_requires_every_producer_policy_field(field):
    _, books = fixture()
    config = books["histogram"]["config"]
    del config[field]
    assert_config_refusal(config)


def test_report_refuses_unknown_policy_fields():
    _, books = fixture()
    assert_config_refusal(books["histogram"]["config"] | {"unknown": 1})


@pytest.mark.parametrize("field", ["initial_cash", "max_trade_debit", "max_ticker_debit"])
@pytest.mark.parametrize("value", [0, -1, True, "100", None, float("nan"), float("inf"), 10**1000])
def test_report_reuses_positive_finite_monetary_limits(field, value):
    _, books = fixture()
    assert_config_refusal(books["histogram"]["config"] | {field: value})


@pytest.mark.parametrize("field", ["contracts_per_trade", "max_concurrent", "settlement_delay_days"])
@pytest.mark.parametrize("value", [-1, .5, True, "1", None, float("nan"), float("inf")])
def test_report_reuses_nonfractional_count_and_delay_policy(field, value):
    _, books = fixture()
    assert_config_refusal(books["histogram"]["config"] | {field: value})


@pytest.mark.parametrize("field", ["contracts_per_trade", "max_concurrent"])
def test_report_refuses_zero_contracts_and_capacity(field):
    _, books = fixture()
    assert_config_refusal(books["histogram"]["config"] | {field: 0})


@pytest.mark.parametrize("order", ["arbitrary", "", None, True, 1, [], {}])
def test_report_reuses_closed_settlement_order_policy(order):
    _, books = fixture()
    assert_config_refusal(books["histogram"]["config"] | {"same_time_order": order})


@pytest.mark.parametrize("order", ["settlements_before_entries", "entries_before_settlements"])
def test_report_accepts_both_actual_producer_settlement_policies(order):
    opportunities, books = fixture()
    for book in books.values():
        book["config"].update(same_time_order=order, settlement_delay_days=0)
    assert report(opportunities, books)["contrasts"]["same_mean"]["net_delta_per_opportunity"] == 10


@pytest.mark.parametrize("arm", ["histogram", "same_mean", "matched_point", "incumbent", "gate"])
def test_cross_arm_event_date_alignment_is_required_even_with_matching_decision_clock(arm):
    opportunities, books = fixture()
    if arm not in books:
        books[arm] = deepcopy(books["same_mean"])
    books[arm]["per_event"][0]["event_date"] = "2026-01-03"
    invalid(lambda: report(opportunities, books))


def test_event_dates_may_differ_from_decision_dates_but_not_between_arms():
    opportunities, books = fixture()
    for book in books.values():
        book["per_event"][0]["event_date"] = "2026-01-03"
    assert report(opportunities, books)["contrasts"]["same_mean"]["net_delta_per_opportunity"] == 10


@pytest.mark.parametrize("scenario", ["settled", "no_trade", "no_plan", "unresolved"])
def test_actual_funded_book_producer_preserves_clock_status_and_economic_contract(scenario):
    from experiments.distribution_book import funded_book

    opportunities, expected = fixture()
    actual = {}
    fields = ("event_id", "ticker", "event_date", "decision_at", "entry_at", "selected")
    for arm, reference in expected.items():
        intents = [{field: row[field] for field in fields} for row in reference["per_event"]]
        quotes = {row["event_id"]: deepcopy(row["quote"]) for row in reference["per_event"]}
        if scenario in ("no_trade", "no_plan"):
            intents[0]["selected"] = False
        if scenario == "no_plan":
            intents[0].update(decision_at=None, entry_at=None)
            del quotes["0"]
        if scenario == "unresolved":
            quotes["0"]["exit_cash"] = None
        actual[arm] = funded_book(intents, quotes, config=reference["config"], report_end=reference["report_end"])
        if scenario == "settled":
            assert actual[arm]["per_event"] == reference["per_event"]
            assert actual[arm]["summary"] == reference["summary"]
        row, summary = actual[arm]["per_event"][0], actual[arm]["summary"]
        assert row["status"] == ("abstained" if scenario == "no_plan" else scenario)
        assert row["event_date"] == "2026-01-02"
        first_profit = {"histogram": 10, "same_mean": 0, "matched_point": 5}[arm]
        expected_profit = first_profit if scenario == "settled" else None if scenario == "unresolved" else 0
        assert row["net_profit"] == expected_profit
        remaining_profit = {"histogram": 30, "same_mean": 0, "matched_point": 15}[arm]
        settled_profit = remaining_profit + (first_profit if scenario == "settled" else 0)
        assert summary["settled_net_profit"] == settled_profit
        assert summary["settlement_equity"] == 1000 + settled_profit
        assert summary["cash"] == 1000 + settled_profit - (100 if scenario == "unresolved" else 0)
        assert summary["funded"] == (3 if scenario in ("no_trade", "no_plan") else 4)
        assert summary["unresolved"] == (1 if scenario == "unresolved" else 0)
        assert summary["capital_days"] == {"settled": 400, "no_trade": 300, "no_plan": 300, "unresolved": 3300}[scenario]
    if scenario == "no_plan":
        # The pipeline uses this exact grouping fallback; it is not an availability clock.
        for meta, row in zip(opportunities, actual["histogram"]["per_event"]):
            meta["decision_date"] = row["decision_at"] or row["event_date"]
        assert opportunities[0]["decision_date"] == "2026-01-02"
    result = report(opportunities, actual)
    primary = result["contrasts"]["same_mean"]
    assert result["opportunities"] == 4
    assert result["arms"] == {arm: book["summary"] for arm, book in actual.items()}
    assert primary["scoreable_events"] == (3 if scenario == "unresolved" else 4)
    assert primary["unresolved_events"] == (["0"] if scenario == "unresolved" else [])
    assert primary["status"] == ("incomplete" if scenario == "unresolved" else "complete")
    expected_delta = {"settled": 10, "no_trade": 7.5, "no_plan": 7.5, "unresolved": None}[scenario]
    assert primary["net_delta_per_opportunity"] == expected_delta
    assert primary["scoreable_delta"] == (10 if scenario == "unresolved" else expected_delta)
    if scenario == "settled":
        assert result == report(opportunities, expected)


def test_actual_producer_event_date_mismatch_refuses_without_inventing_decision_dates():
    from experiments.distribution_book import funded_book

    opportunities, expected = fixture()
    actual = {}
    fields = ("event_id", "ticker", "event_date", "decision_at", "entry_at", "selected")
    for arm, reference in expected.items():
        intents = [{field: row[field] for field in fields} for row in reference["per_event"]]
        if arm == "same_mean":
            intents[0]["event_date"] = "2026-01-03"
        quotes = {row["event_id"]: row["quote"] for row in reference["per_event"]}
        actual[arm] = funded_book(intents, quotes, config=reference["config"], report_end=reference["report_end"])
    assert actual["histogram"]["per_event"][0]["decision_at"] == actual["same_mean"]["per_event"][0]["decision_at"]
    invalid(lambda: report(opportunities, actual))


def assert_invalid_clock_report(opportunities, books):
    with pytest.raises(OpsError) as caught:
        report(opportunities, books)
    assert caught.value.problem == fail(
        "INVALID_EXPERIMENT_SPEC", "invalid distribution paired report inputs or arithmetic").problem


@pytest.mark.parametrize("field", ["report_end", "event_date"])
@pytest.mark.parametrize("value", ["", "not-a-date", "20260102", "2026-W01-5", "2026-1-02",
    "2026-02-29", "2024-02-30", "0000-01-01", "10000-01-01", "2026-01-02T00:00:00", " 2026-01-02"])
def test_matching_malformed_book_clocks_refuse_without_cross_arm_mismatch(field, value):
    opportunities, books = fixture()
    for book in books.values():
        if field == "report_end":
            book[field] = value
        else:
            # Keep valid planned decision_at so the grouping clock cannot mask this guard.
            book["per_event"][0][field] = value
    assert_invalid_clock_report(opportunities, books)


@pytest.mark.parametrize("field", ["report_end", "event_date"])
@pytest.mark.parametrize("value", [None, True, 20260102, [], {}, date(2026, 1, 2)])
def test_matching_nonstring_book_clocks_have_exact_typed_refusal(field, value):
    opportunities, books = fixture()
    for book in books.values():
        if field == "report_end":
            book[field] = value
        else:
            book["per_event"][0][field] = value
    assert_invalid_clock_report(opportunities, books)


@pytest.mark.parametrize("value", ["", "20260102", "2026-02-29"])
def test_empty_producer_book_still_requires_canonical_report_end(value):
    from experiments.distribution_book import funded_book

    _, fixtures = fixture()
    book = funded_book([], {}, config=fixtures["histogram"]["config"], report_end="2026-02-01")
    book["report_end"] = value
    assert_invalid_clock_report([], {"histogram": book, "same_mean": deepcopy(book)})


@pytest.mark.parametrize("day", ["0001-01-01", "2024-02-29", "9999-12-31"])
def test_empty_producer_report_end_accepts_canonical_calendar_boundaries(day):
    from experiments.distribution_book import funded_book

    _, fixtures = fixture()
    book = funded_book([], {}, config=fixtures["histogram"]["config"], report_end=day)
    assert report([], {"histogram": book})["opportunities"] == 0


@pytest.mark.parametrize("day", ["0001-01-01", "2024-02-29", "9999-12-31"])
def test_matching_canonical_event_dates_accept_calendar_boundaries(day):
    opportunities, books = fixture()
    for book in books.values():
        book["report_end"] = "9999-12-31"
        book["per_event"][0]["event_date"] = day
    assert report(opportunities, books)["contrasts"]["same_mean"]["net_delta_per_opportunity"] == 10
