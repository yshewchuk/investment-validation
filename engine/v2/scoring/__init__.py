"""Scoring application forecasts, shape, pricing, gate/chooser decisions, diagnostics

Layer 5 of `system_rearchitecture.md` §4.1. Replaces `score.py split by the stages in §6.3`, `entry_rules.py`, `replay.py`, `trailing_cutoff from pnl_sim.py`.

Empty by construction: phase 0 writes no production logic
(`guides/rearchitecture_phase0_baseline.md` §10). See ``README.md`` for what
this package will own, what it deliberately will not, and which packages may
import it.
"""
