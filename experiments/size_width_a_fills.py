"""EXP-189 Amendment 2: fill model A (mid-anchored, fill design section 9) behind the F seam. Consumed by size_width_f_select.

``load(constants_id, variant)`` reads CONSTANTS_PATH once, before any quote is read, and refuses in order:
``constants_unreadable`` (not a readable regular file), ``input_too_large`` (over MAX_CONSTANTS_BYTES, RESOURCE_LIMIT_EXCEEDED),
``constants_unreadable`` (not JSON), then fair_value_apply.load_constants' ``constants_invalid``, ``constants_id_mismatch``,
``variant_unknown``. It returns the ModelA and the extra receipt fields: constants_id, variant, constants_file_sha256
(the bytes parsed) and model_a_source_sha256 (one hash per SOURCE_FILES file, each read capped at MAX_SOURCE_BYTES: input_too_large). constants_file_sha256 is recorded in each stage's inputs receipt (top level) and compared with nothing across stages (the pinned constants_id already refuses any changed constant).
``AFills`` is FFills with model A entry and exit; the columns debit_F and proceeds_F hold model A values. The leg table
carries constants_id and variant (#3104 choice B). Entry and exit each refuse ``legs_unbound`` or
``legs_constants_mismatch`` unless every row carries the pinned pair, so a trade's two tables cannot differ.
"""

import hashlib
import json
import typing
from pathlib import Path

from experiments import fair_value_apply as fva
from experiments import fair_value_inputs as fvi
from experiments.fair_value_legs import FairValueRefusal
from experiments.size_width_f_fills import FFills

HERE = Path(__file__).resolve().parent
CONSTANTS_PATH = HERE / "EXP-189_short_vol_width_tuning" / "model_a_constants.json"  # read at call time (tests monkeypatch it)
MAX_CONSTANTS_BYTES = 1 << 16
MAX_SOURCE_BYTES = 1 << 20
# every file of model A's pricing path outside fvf_id: the apply code, this module, the F seam (quote tables, structure dispatch), and the shared entry, Skip/Exit and pricing-seam modules
SOURCE_FILES = ("fair_value_apply.py", "fair_value_constants.py", "size_width_a_fills.py", "size_width_f_fills.py",
                "size_width_f_quotes.py", "size_pnl_shortvol.py", "size_width_fills.py", "size_width_pricing.py")


class ModelA(typing.NamedTuple):
    document: dict
    constants_id: str
    variant: str

    def leg_fills(self, quotes, vol_table):
        return fva.leg_fills(quotes, vol_table, *self)

    def columns(self):
        return dict(zip(fva.BINDING_COLUMNS, (self.constants_id, self.variant)))


def load(constants_id, variant):
    path = Path(CONSTANTS_PATH)
    if not path.is_file():
        raise FairValueRefusal("constants_unreadable")
    try:
        blob = fvi._read_capped(path, MAX_CONSTANTS_BYTES, path.name)
    except OSError:
        raise FairValueRefusal("constants_unreadable") from None
    try:
        document = json.loads(blob)
    except ValueError:
        raise FairValueRefusal("constants_unreadable") from None
    fva.load_constants(document, constants_id, variant)
    return ModelA(document, constants_id, variant), {
        "constants_id": constants_id, "variant": variant, "constants_file_sha256": hashlib.sha256(blob).hexdigest(),
        "model_a_source_sha256": {name: hashlib.sha256(fvi._read_capped(HERE / name, MAX_SOURCE_BYTES, name)).hexdigest() for name in SOURCE_FILES}}


class AFills(FFills):
    def __init__(self, quotes, model):
        super().__init__(quotes)
        self.model = model

    def _structure(self, table, exit):
        if exit:
            return fva.structure_exit(table, *self.model)
        bound = self.model.columns()
        if any(c not in table.columns or table[c].isna().any() for c in bound):
            raise FairValueRefusal("legs_unbound")
        if any((table[c] != v).any() for c, v in bound.items()):
            raise FairValueRefusal("legs_constants_mismatch")
        return fva.structure_entry(table)

    def exit_value_at_eff_mid(self, trade):
        if self.exit_result(trade).outcome == "unpriceable": return float("nan")
        legs = [leg for leg in trade.legs if leg["qty"] > 0]
        table = self.quotes.legs(trade.chunk, trade.i, legs, exit=True)
        mid = ((table["eff_bid"] + table["eff_ask"]) / 2).to_numpy(dtype=float)
        return float(sum(leg["qty"] * (1.0 if leg["side"] == "buy" else -1.0) * x for leg, x in zip(legs, mid)))
