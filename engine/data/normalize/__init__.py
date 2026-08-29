"""Tier-1 → Tier-2 normalizers.

One module per table. Every normalizer is:

* **pure and offline** — it reads the raw cache and writes frames, and makes no
  network call, so a Tier-2 rebuild works with the network unplugged;
* **idempotent** — running it twice produces the same frames;
* **provenance-carrying** — every row records the raw file and source it came
  from.

Direct reads of a raw source format anywhere outside this package are a code
smell: that is what Tier 2 exists to stop.
"""
