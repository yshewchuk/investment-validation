"""The three-tier data architecture.

Tier 1 (``data/raw/``)      immutable cache — every byte ever fetched, verbatim
Tier 2 (``data/curated/``)  one normalized cross-source schema, Parquet by year
Tier 3 (``data/features/``) derived panels and matrices, snapshot-hashed

Each tier is rebuildable from the one below it. Tier 2 rebuilds from Tier 1 with
no network access at all, which is what makes ``rebuild.py`` a regression test
rather than another data pull.
"""
