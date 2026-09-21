"""P5-3 mutation-survivor cluster, Part 2: the golden recipe fingerprint.

``engine/v2/models/training/recipes.py``'s five ``_*_recipes`` builders
(``_size_recipes``, ``_decision_recipes``, ``_crush_recipes``,
``_gate_recipes``, ``_chooser_recipes``) carry fields that a prior mutation
analysis called unkillable equivalent mutants: ``dataset_source``, ``notes``,
``legacy_refs``, ``residuals.*``, ``target.signed``/``target.definition``,
``label.definition``, ``produces``, ``output_id``, ``clock_id``,
``missing_mask``, ``upstream[i].source``, ``recipe_id``. That classification
is wrong: ``recipe_fingerprint`` (``recipes.py`` ``content_hash(recipe.as_dict())``)
hashes the WHOLE dataclass, and ``job.py`` ``_summary`` writes that hash into
every training job's ``summary.json`` as ``recipe_fingerprint``, which is the
identity ``RECEIPT_MISMATCH`` compares against on resume
(``receipts.receipt_issues``). So every field in ``TrainingRecipe`` --
including the ones above -- is observed in production through this one value.

THIS TEST IS A CHANGE-DETECTOR BY DESIGN, and that is correct here: for a
field whose whole job is provenance and identity, "did this change" IS the
semantics of the field. There is nothing else to assert about
``dataset_source`` or ``notes`` -- they carry no other consumer. Pinning the
fingerprint is the same discipline ``RECEIPT_MISMATCH`` itself applies to a
resumed training job: an unexpected change to identity is refused, not
silently absorbed.

If this test fails after a deliberate recipe edit (a new feature, a changed
filter, a rewritten note), that is not a bug in the test: recompute the
expected hash with

    python3 -c "from engine.v2.models.training.recipes import current_recipes, \\
        recipe_fingerprint; [print(k.label(), recipe_fingerprint(r)) \\
        for k, r in sorted(current_recipes().items(), key=lambda kv: kv[0].label()) \\
        if k.output != 'calibration']"

and update ``EXPECTED_FINGERPRINTS`` below deliberately, in the same commit as
the recipe change, so the diff shows a human decided the identity should
move. A silent drift here is exactly what ``RECEIPT_MISMATCH`` exists to
catch downstream; this test catches it one step earlier, at the source.

Covers the 11 recipes built by the five mutation-survivor functions above
(``current_recipes()`` filtered to ``output != "calibration"``; the
calibration surfaces come from ``.calibration.calibration_recipes`` and
carry no survivors in this cluster). Discovered programmatically -- never a
hand-copied key list.
"""
from __future__ import annotations

import pytest

from engine.v2.models.training.recipes import current_recipes, recipe_fingerprint

# Pinned 2026-09-21 against engine/models/registry.json at commit 21b9c16.
# Recompute with the snippet in the module docstring after a deliberate
# recipe change; do not "fix" a failure by copying whatever the code now
# produces without checking the change was intended.
EXPECTED_FINGERPRINTS = {
    "size:*:champion": "sha256:414c7b5330545515fd836e15d481321af8f296330d64088c32c7689785463f90",
    "size:*:tier4_monthly": "sha256:92a16364cc1d8cce65f3374ca96287f05f0e67cb32b7b157af086ae0b7bf68b4",
    "implied_t1:*:champion": "sha256:d022b893a6f1c25923abd329bcbe85a3cde5a899e23272490e49ba2953230cb6",
    "implied_t1:*:tier4_monthly": "sha256:e53989398934c5c109ccd9a7b58a21f130420b7040dcbe91deed6df3aba72059",
    "runup_move:*:champion": "sha256:d3e16b3c31d5b19bafd0eeb509a14109aaeb283963e2b2f64d7a191474b1091b",
    "runup_move:*:tier4_monthly": "sha256:76bddde3b15dcffed64bd4ec2ede0df96ebb44eab735acf59ee98c01e21f9c8d",
    "iv_crush:*:champion": "sha256:59c73e3ac0488252e502a40447e19938fac0d926eec2aac59a6db2eec2147d70",
    "iv_crush:*:tier4_monthly": "sha256:1406ab6e9e681bbbc6357c0c9dffb17999249a241c8c7c76689dcf80854ef1ff",
    "gate:STR-THRU:champion": "sha256:a786e7f76dc444bd3606334aeb7d8444778233e814174ef69fa84dc0bc053dc1",
    "gate:STR-RUNUP:champion": "sha256:f5f99f7bd2bc6c6dfc15d7affca21b72b7e84c3b0918b5125c01419111e65a1e",
    "chooser:DYN-SV:champion": "sha256:64eeb167e8b715c38ee7c7683f1585e5b631d40c2200335268845433fef1b779",
}


def _non_calibration_recipes():
    """The 11 recipes the five mutation-survivor functions build, found
    programmatically (never a hand-copied list)."""
    return {k: r for k, r in current_recipes().items() if k.output != "calibration"}


def test_the_eleven_recipes_are_discovered_programmatically():
    # A change here (a role added/removed, an output added/removed) means
    # EXPECTED_FINGERPRINTS needs the same update -- fail loud, not quiet.
    recipes = _non_calibration_recipes()
    assert len(recipes) == 11
    assert {k.label() for k in recipes} == set(EXPECTED_FINGERPRINTS)


@pytest.mark.parametrize("label", sorted(EXPECTED_FINGERPRINTS))
def test_recipe_fingerprint_is_pinned(label):
    """The identity ``job.py`` ``_summary`` writes into every training job's
    ``summary.json``, and that ``receipts.receipt_issues`` compares a resumed
    fold's stored receipt against (``RECEIPT_MISMATCH``). This is the ONLY
    place ``dataset_source``, ``notes``, ``legacy_refs``, ``residuals.*``,
    ``target.signed``/``target.definition``, ``label.definition``,
    ``produces``, ``output_id``, ``clock_id``, ``missing_mask``,
    ``upstream[i].source`` and ``recipe_id`` are ever observed outside the
    source file, so this is the only place they can be tested: through the
    identity they all feed.
    """
    recipes = _non_calibration_recipes()
    recipe = next(r for k, r in recipes.items() if k.label() == label)
    actual = recipe_fingerprint(recipe)
    assert actual == EXPECTED_FINGERPRINTS[label], (
        f"{label}: recipe_fingerprint changed from the pinned value.\n"
        f"  expected {EXPECTED_FINGERPRINTS[label]}\n"
        f"  actual   {actual}\n"
        "If this recipe changed on purpose, this is exactly the identity "
        "move RECEIPT_MISMATCH exists to catch downstream -- update "
        "EXPECTED_FINGERPRINTS in this file deliberately, in the same "
        "commit, using the recompute snippet in the module docstring. Do "
        "not update it without checking the change was intended: a silent "
        "drift here is a silent drift into every resumed training job's "
        "identity check."
    )
