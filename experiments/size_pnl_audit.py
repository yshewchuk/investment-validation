"""Amendment 6 production-audit table for the EXP-188 report. Research-only: reached from size_pnl_assemble. Static text; no I/O."""

from experiments.size_pnl_report import md_table

# (n, rule, production_source, status). Status starts with imports, fixed, kept or out of scope.
ROWS = (
    ("1", "Width rule (divisors)", "engine/forecast_sizing.py forecast_params and constants",
     "imports; drop divisor c, call forecast_params(name, pred)"),
    ("2", "WIDTH_MIN/MAX, decline", "forecast_params rules; Scorer._size_from_forecast (NO_FORECAST)",
     "kept; width_declined scored as P&L 0 with the event kept in E, counted by reason"),
    ("3", "Strike selection and snapping", "engine/structures.py StrikeSelector.select via price_structure",
     "fixed; resolve through price_structure on a ChainSnapshot, delete the local loop"),
    ("4", "ladder_too_coarse", "price_structure raises LadderTooCoarse on any leg collision",
     "fixed; falls out of row 3, local check deleted, decline is a non-trade at P&L 0"),
    ("5", "Zero-floor re-check", "none: production trades what price_structure returns",
     "fixed; delete the negative-floor drop, count negative_floor_taken as a diagnostic"),
    ("6", "Entry cost not a debit", "engine/pnl_sim.py expected_pnl returns None for entry_cost <= 0",
     "fixed; cost <= 0 is undetermined, MIN_DEBIT deleted"),
    ("7", "exp_pnl_sim", "engine/pnl_sim.py expected_pnl, ResidualPool, DRAWS",
     "kept; imports the constants, held to expected_pnl within 1e-9; seed key=event_id"),
    ("8", "Residual pool", "Scorer._residual_pool: all Tier-4 history from 2018",
     "kept; per-model pool from 2023, production-length pool shown as T9 sensitivity"),
    ("9", "Trailing cutoff", "pnl_sim.trailing_cutoff (WINDOW_MONTHS, QUANTILE, min_window=100)",
     "imports; MIN_WINDOW copy deleted, per-model history kept"),
    ("10", "MCAP_FLOOR, MAX_REL_SPREAD, mean relative spread", "engine/entry_rules.py; engine.score._mean_relative_spread",
     "fixed; spread computed by _mean_relative_spread on the row-3 StructurePrice"),
    ("11", "CAPITAL_PER_TRADE, lots", "engine/portfolio.py build_book lot floor",
     "kept; build_book test on synthetic ledger rows requires equal contracts"),
    ("12", "Entry/exit timing", "factory entry_offset=0, exit_offset=1; TradingCalendar.last_pre_print",
     "imports; test that AMC and BMO sessions match last_pre_print"),
    ("13", "Expiry selection", "each factory leg's ExpirySelector (first_post_event)",
     "fixed; pinned() requires leg selectors equal the ladder's, else factory_default_mismatch"),
    ("14", "Straddle universe", "engine/score.py GATE_MCAP_FLOOR, BAD_QUOTE_COST_PCT",
     "imports (Amendment 5)"),
    ("15", "Straddle gate", "engine/models/registry.json gate.fit, refit through 2024",
     "imports; user-approved refit (Amendment 3)"),
    ("16", "Straddle legs and entry quotes", "STR-THRU ATM via price_structure",
     "fixed; resolve through price_structure; zero-bid and locked quotes priced, crossed or non-finite undetermined"),
    ("17", "Marks and exits", "engine/replay.py price_structure(closing=True), structure_return",
     "kept by user decision; marks_mid is the production rule, primary is the pre-registered conservative one"),
    ("18", "DYN-SV chooser", "engine/score.py chooser",
     "out of scope; seven independent books, stated in the report"),
)

NOTE = ("Every divergence from production is fixed in the code or kept by the user's decision; kept rows are open questions "
        "for the user (design section 18, Open questions).")


def section():
    table = md_table(["#", "rule", "production source", "status"], [list(row) for row in ROWS])
    return "\n\n".join(["### Production audit (Amendment 6, post-hoc revision by the user, 2026-11-24)", NOTE, table])
