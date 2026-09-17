# Phase 3A status: provenance-correct launch evidence

Updated 2026-09-16. Phase 3A is in final evidence assembly. It is not marked
complete until the current Phase 3 gate validates one package. Phase 3B
incremental data, native scoring, and cutover remain out of scope.

## Accepted source boundary

The candidate is built only from Phase 2 release
`reld9a81e6ded7f45c13b5afe21`, published at
`2026-09-16T22:16:53.600037Z`. Its release-manifest hash is
`sha256:24121235ccd4a0e8b096ddfb946c465b1316e3dacdfc274e1903026bbbabf27d`.
The source score document defines 111 saved-score population keys and the
bridge, API, and browser checks use that same 111-key population. The two
ladder rows are validated separately and are not substituted for source-score
population evidence.

Fresh D19 receipts are green. Earlier notes that replay remained red are
historical and must not be used to block or justify this candidate.

## Launch evidence being assembled

The final package must contain all L01-L14 receipts at the final source hash,
not copied receipts from an earlier candidate. In particular it contains:

- A verified `PreviewInput` whose source release, snapshot, score, bundle,
  finality, model evidence, and comparison references match the accepted
  Phase 2 handoff.
- Two distinct projection generations, A and B, from that input boundary.
  A real fenced operations sequence published A, then B, then a fresh
  rollback operation release bound to A. The retained operations releases are
  `relacf1e9f45f4d9a96edb3081a`, `rel2a1ae238cdb90e3c80779db6`, and
  `rel7ed9da6817b9bb580b95afce`; the last is the current pointer.
- Browser evidence through the real API chain: operations `CURRENT`, that
  release's `projection_binding.json`, then the serving index. It independently
  compares the complete source population rather than treating an API result
  as its own expected set.

The private rebuild workspace is `/tmp/phase3a-rebuild/`. It is an evidence
work area, not a production data root. The final immutable package is written
under `/root/phase3-evidence/` after its final implementation commit is known.

## Operator handoff

The serving API must be launched with the fenced scope root, otherwise it has
no current release by design:

```bash
V2_DASHBOARD_TOKEN=<private-token> python3 -m engine.v2.serving.api \
  --host 127.0.0.1 --port 8765 \
  --serving-db <serving-root>/serving.sqlite \
  --store-root <serving-root>/objects \
  --serving-root <serving-root> \
  --publication-root <ops-root>/releases/<scope>
```

To update, project a verified new Phase 2 input and publish it through
`engine.v2.ops.effects_graph.publication_effect` under a valid fenced claim.
To roll back, bind the previously retained projection as a new input generation
and publish through that same effect. Do not copy release directories or write
`CURRENT` directly; both bypass gate and fence evidence.

## Deferred work

Phase 3B owns incremental ingestion and correction handling. Phases 4-6 own
native scoring, model extraction, and consumer authority. This read-only
shadow preview does not change legacy prediction, settlement, or publication
authority.
