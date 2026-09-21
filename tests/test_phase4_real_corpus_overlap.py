"""Regression for the phase4_real.py memory-ceiling OOM (2026-09-21).

``build_evidence`` used to bind its own persistent ``corpus = load(resolved)``
to a named local BEFORE calling ``run_corpus(resolved)`` --
``checks.tier0_corpus.run`` -> ``_run`` -> ``_run_loaded``, which loads an
INDEPENDENT second full ``Corpus`` internally just to verify the tier-0 round
trip (958388f already made that internal copy release itself before it
returns). With the old ordering, both copies were resident at once for the
whole span of the ``run_corpus`` call: build_evidence's own ``corpus`` local
outlives that call, and the internal one is fully built before it starts
releasing anything. Measured on the real 3.2 GB / 20-pair corpus: a single
process climbs past an 8.5 GB cap, and past a 9.5 GB cap with active
swapping, still rising when killed.

Fix: swap the two lines so ``run_corpus`` runs to completion (and drops its
own internal corpus) before ``load`` is called for the copy the rest of the
function holds -- the two calls are independent (neither's result feeds the
other), so this changes nothing about what either verifies.

This test proves the ORDERING/lifetime property directly against the real
(unmodified) ``build_evidence``, using a corpus with zero pairs (just a
valid ``INDEX.json``) so it stays cheap: ``_run_loaded``'s ``if not
corpus.pairs`` branch (checks/tier0_corpus.py) returns before building any
case, so no subprocess ever forks and the loader itself does no per-pair
work -- the load/lifetime ordering this test checks is independent of
corpus size. ``_champion_artifacts_verified`` is stubbed out since it reads
the real model registry, which is unrelated to corpus loading.
"""
from __future__ import annotations

import gc
import weakref
from pathlib import Path

from checks import phase4_real
from checks import tier0_corpus as t0


def _build_empty_corpus(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "INDEX.json").write_text('{"pairs": {}}')
    (root / "pairs").mkdir(exist_ok=True)
    return root


def test_build_evidence_never_holds_two_full_corpora_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(phase4_real, "_champion_artifacts_verified", lambda: True)

    root = _build_empty_corpus(tmp_path / "tier0")
    real_load = t0.load
    refs: list[weakref.ReferenceType] = []
    overlap_detected = {"flag": False}

    def tracking_load(path):
        gc.collect()
        if any(ref() is not None for ref in refs):
            overlap_detected["flag"] = True
        loaded = real_load(path)
        refs.append(weakref.ref(loaded))
        return loaded

    # Patch BOTH names: `checks.tier0_corpus.run` (aliased into phase4_real
    # as `run_corpus`) resolves `load` through tier0_corpus's OWN module
    # globals internally, while phase4_real's direct `corpus = load(...)`
    # call resolves through its own separately-bound `load` name -- both
    # must point at the same tracking wrapper to see one unified call order.
    monkeypatch.setattr(t0, "load", tracking_load)
    monkeypatch.setattr(phase4_real, "load", tracking_load)

    evidence = phase4_real.build_evidence(root, tmp_path / "artifacts")

    assert len(refs) == 2, (
        "expected exactly one load() inside run_corpus and one persistent "
        f"load() in build_evidence, saw {len(refs)}"
    )
    assert overlap_detected["flag"] is False, (
        "a second load() started while an earlier loaded Corpus was still "
        "referenced -- the two full corpora overlapped in memory again"
    )
    gc.collect()
    assert refs[0]() is None, (
        "run_corpus's internal corpus was not released before build_evidence "
        "returned"
    )
    # Sanity: the real function ran end to end on the empty corpus, not a
    # stub standing in for it.
    assert evidence["corpus_hash"] is None
    assert evidence["population"]["expected"] == 0
