"""S9H hash preservation: analog row ids ride a non-hashed display channel.

``checks/phase4_real.py`` compares every runtime stage receipt against the
captured corpus receipts. The analog stage's per-event row ids are display
data a serving projection needs, so they travel on
``StageObservation.display_document`` -- which ``_emit_stage`` deliberately
never hashes -- instead of the stage output. This is the golden control: the
same fixed input scored with the ids present and with them absent produces
byte-identical ``analogs`` and ``simulation`` receipts, while the ids are
still observable on the display channel.
"""
from __future__ import annotations

from dataclasses import replace

from engine.v2.foundation import content_hash
from engine.v2.scoring.source_inputs import build_native_score_inputs
from engine.v2.scoring.stages import assemble_native_values
from tests.test_v2_scoring_source_inputs import _bundle


def _observations(monkeypatch, *, strip_ids: bool):
    """Score the fixed bundle, optionally with the matcher's ids stripped."""
    if strip_ids:
        import engine.v2.scoring.native_analog as native_analog

        real = native_analog.evaluate_analogs

        def without_ids(**kwargs):
            return replace(real(**kwargs), selected_row_ids=(),
                           contributing_row_ids=())

        monkeypatch.setattr(native_analog, "evaluate_analogs", without_ids)
    inputs = build_native_score_inputs(_bundle())
    seen: dict = {}
    values = assemble_native_values(
        inputs, strategy="STR-THRU",
        observer=lambda item: seen.setdefault(item.receipt.stage, item))
    return values, seen


def test_analog_ids_ride_the_display_channel_without_moving_stage_hashes(monkeypatch):
    values, with_ids = _observations(monkeypatch, strip_ids=False)
    _, without_ids = _observations(monkeypatch, strip_ids=True)

    assert with_ids["analogs"].display_document == {
        "selected_row_ids": ("a1", "a2"),
        "contributing_row_ids": ("a1", "a2"),
    }
    assert without_ids["analogs"].display_document == {
        "selected_row_ids": (), "contributing_row_ids": (),
    }
    # The expected hashes ARE the ids-absent run: removing the side-channel
    # field changes nothing in the hashed payload, for the analog stage and
    # for the next stage chained to it via ``prior``.
    for stage in ("analogs", "simulation"):
        assert (with_ids[stage].receipt.output_hash
                == without_ids[stage].receipt.output_hash)
        assert (with_ids[stage].receipt.input_hash
                == without_ids[stage].receipt.input_hash)
        assert with_ids[stage].output_document == without_ids[stage].output_document
    # The hashed analog document and the record's values carry no row ids.
    assert set(with_ids["analogs"].output_document) == {
        "exp_pnl_analog", "win_analog", "ci_low", "ci_high", "n_analogs"}
    assert "selected_row_ids" not in values
    assert "contributing_row_ids" not in values
    assert content_hash(with_ids["analogs"].output_document) == \
        with_ids["analogs"].receipt.output_hash
