"""Research tooling — offline analysis CLIs and snapshot-pinned replay.

Layer 7.0 of `system_rearchitecture.md` §4.1: a peer of `engine.v2.serving` and
`engine.v2.ops`, added by Phase 6 slice 6 (decision UD-4) rather than a
documented §4 owner-table row. The pure analysis cores of
`tools/signal_screen.py` and `tools/fill_quality.py`, and the read path of
`engine/data/pulls/polygon_fills.py`, move here; every read goes through
`engine.v2.data.Repository` on one explicitly resolved snapshot.

This package also carries the snapshot-pinned native replay and the trade
table it publishes, reading committed snapshots through
``engine.v2.data.Repository``; it never touches the legacy mutable store. See
``README.md`` for the public interface, the consumers, and what this
package deliberately does not do.
"""
from __future__ import annotations

__all__: list[str] = []
