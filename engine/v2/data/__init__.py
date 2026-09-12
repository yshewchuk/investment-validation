"""Ingestion fetch receipts, normalization, coverage, finality, source revisions

Layer 1 of `system_rearchitecture.md` §4.1. Replaces `data/sources/`, `data/normalize/`, `data/pulls/`, `store.py`, `fetch.py`, `throttle.py`, `finality.py`, `rebuild.py`, `calendar sourcing from calendar.py`.

Empty by construction: phase 0 writes no production logic
(`guides/rearchitecture_phase0_baseline.md` §10). See ``README.md`` for what
this package will own, what it deliberately will not, and which packages may
import it.
"""
