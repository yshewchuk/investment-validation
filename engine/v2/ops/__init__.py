"""Supervisor/catalog transactions, leases, dependencies, capacity, retry history

Layer 7 of `system_rearchitecture.md` §4.1. Replaces `new supervisor and catalog`, `dashboard/nightly.py becomes a job graph`, `tools/bounded_run.py becomes an executor adapter`.

Empty by construction: phase 0 writes no production logic
(`guides/rearchitecture_phase0_baseline.md` §10). See ``README.md`` for what
this package will own, what it deliberately will not, and which packages may
import it.
"""
