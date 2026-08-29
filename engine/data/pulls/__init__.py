"""Quota-spending pull plans.

Every module here follows the same discipline: `--dry-run` prints the cost and
the coverage it would buy, `--confirm` is required to spend, and all traffic
goes through the Tier-1 fetch wrapper so a repeated request is free.
"""
