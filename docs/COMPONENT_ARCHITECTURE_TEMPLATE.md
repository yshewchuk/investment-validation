# Component ARCHITECTURE.md template

Every production component (a package under `engine/v2/**`, or a legacy
package under `engine/**`) that gets a detailed architecture write-up keeps
its own `ARCHITECTURE.md` next to its code — e.g. `engine/v2/ops/ARCHITECTURE.md`.
The root `/ARCHITECTURE.md` stays high-level (components, the layer map,
production entrypoints, cross-cutting invariants and anti-patterns); a
component doc is where the per-package detail lives, so the root doc does not
grow without bound as components are documented one at a time.

There is no per-task design doc anymore: a task that changes a component's
interfaces, dependencies, inputs/outputs or failure semantics updates that
component's `ARCHITECTURE.md` (and the root doc, if the change is
cross-cutting) in the same PR, and states its options and rationale in the
PR body instead of a separate file. A doc that lives beside code that keeps
changing stays current; a doc written once per task and never touched again
drifts.

## Sections, in order

1. **Purpose.** What this component owns, in the terms of the root doc's
   layer table (its layer number, what it replaces in legacy, one sentence
   of scope). Cite the root doc, don't repeat its whole table.
2. **Primary contracts and public interfaces.** The names another package,
   CLI, or route may actually call — the same list a package README's
   `Public interface` section enforces, where one exists. For a CLI
   component, this is its subcommand tree, derived from the real argparse
   definitions, not a remembered list.
3. **Inputs.** What this component reads, and from where (an artifact
   store, a catalog table, a request document, a legacy file).
4. **Outputs.** What it writes or returns, and to where.
5. **Dependencies.** Which lower layers/components it imports (checked
   against the root doc's layer table and `checks/import_layers.py`'s real
   rules, not assumed), and which legacy adapter (if any) it reaches
   through. Who calls this component — real call sites, found by grep, not
   just the ones a design doc happened to touch.
6. **External systems and libraries.** Databases, filesystems, third-party
   APIs, subprocesses — anything outside this repo's own code that this
   component talks to.
7. **Failure semantics.** The 4c R1–R6 template: missing input, cache,
   retry, transaction, partial write, idempotency. "It raises" is not a
   failure semantic.
8. **Invariants.** The invariants from the root doc's §5 that this
   component is responsible for enforcing, plus any invariant specific to
   this component alone.
9. **Diagrams.** A mermaid diagram for anything non-trivial this component
   owns: a stage/job graph, a data flow, a state machine. Every node and
   edge must be checked against the real code (the graph literal, the
   state transitions) before publishing — a diagram that has drifted from
   the code it claims to describe is worse than no diagram.

## Public-safe

Every component doc, like the root doc, is public: no strategy thresholds,
gate-logic numbers, edge figures, or local filesystem paths.
