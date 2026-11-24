"""EXP-188 receipts: content-hashed, write-once receipt files. Synthetic bodies only; no repository.

The divisor re-tune (choose_c, tune, C_GRID) was removed by Amendment 6, so only the receipt library is tested here.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from experiments import size_pnl_receipts as receipts
from experiments import size_pnl_shortvol as sv

BODY = {"stage": "tuning", "model_id": "M", "c": 1.1}
CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
UNREADABLE = "tuning_receipt_unreadable"


def test_receipt_hash_refuses_a_body_carrying_content_hash():
    with pytest.raises(ValueError):
        receipts.receipt_hash({"content_hash": "x"})


def test_receipt_hash_is_sha256_of_the_canonical_json():
    # The canonical text of {"a": 1} is the bytes '{"a":1}'; the digest is taken over those bytes directly.
    assert receipts.receipt_hash({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()


def test_write_refuses_a_non_dict_body_and_creates_no_file(tmp_path):
    path = tmp_path / "r.json"
    with pytest.raises(ValueError):
        receipts.write_receipt(path, [])
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_nested_int_keys_are_admitted_and_round_trip(tmp_path):
    # JSON stringifies the keys, so 2 and 10 sort as "10" < "2" unless the body is admitted first.
    path = tmp_path / "r.json"
    body = {"t": {2: 1, 10: 2}}
    written = receipts.write_receipt(path, body)
    assert written["t"] == {"2": 1, "10": 2}
    assert {k: v for k, v in written.items() if k != "content_hash"} == {"t": {"2": 1, "10": 2}}
    assert receipts.load_receipt(path) == written


def test_nan_body_is_refused_as_value_error(tmp_path):
    path = tmp_path / "r.json"
    with pytest.raises(ValueError):
        receipts.write_receipt(path, {"c": math.nan})
    assert not path.exists()


def test_write_then_load_round_trip(tmp_path):
    path = tmp_path / "sub" / "receipt.json"
    written = receipts.write_receipt(path, dict(BODY))
    assert written["content_hash"] == receipts.receipt_hash(BODY)
    assert {k: v for k, v in written.items() if k != "content_hash"} == BODY
    assert receipts.load_receipt(path) == written


def test_synthetic_model_id_is_refused_before_any_file_is_written(tmp_path):
    path = tmp_path / "r.json"
    with pytest.raises(sv.Refusal) as caught:
        receipts.write_receipt(path, {**BODY, "model_id": receipts.SYNTHETIC_PREFIX + "abc"})
    assert caught.value.code == CONFLICT
    assert caught.value.details == {"reason": "SYNTHETIC_IN_TUNING"}
    assert list(tmp_path.iterdir()) == []


def test_replay_with_the_same_body_leaves_the_file_untouched(tmp_path):
    path = tmp_path / "r.json"
    first = receipts.write_receipt(path, dict(BODY))
    before = path.stat()
    again = receipts.write_receipt(path, dict(BODY))
    after = path.stat()
    assert again == first
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_changed_body_refuses_and_leaves_the_file_alone(tmp_path):
    path = tmp_path / "r.json"
    receipts.write_receipt(path, dict(BODY))
    original = path.read_bytes()
    with pytest.raises(sv.Refusal) as caught:
        receipts.write_receipt(path, {**BODY, "c": 0.9})
    assert caught.value.code == CONFLICT
    assert caught.value.details == {"reason": "tuning_receipt_changed", "path": str(path)}
    assert path.read_bytes() == original


def test_corrupt_file_refuses_and_is_never_repaired(tmp_path):
    path = tmp_path / "r.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(sv.Refusal) as caught:
        receipts.write_receipt(path, dict(BODY))
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == UNREADABLE
    assert path.read_text(encoding="utf-8") == "{not json"


def test_no_temp_files_are_left_behind(tmp_path):
    path = tmp_path / "r.json"
    receipts.write_receipt(path, dict(BODY))
    receipts.write_receipt(path, dict(BODY))
    with pytest.raises(sv.Refusal):
        receipts.write_receipt(path, {**BODY, "c": 0.9})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.json"]


def test_lost_link_race_rereads_the_winner(tmp_path, monkeypatch):
    seed = tmp_path / "seed.json"
    winner = receipts.write_receipt(seed, dict(BODY))
    path = tmp_path / "r.json"

    def race(src, dst):  # a concurrent writer lands the same receipt just before our link
        Path(dst).write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
        raise FileExistsError(dst)

    monkeypatch.setattr(receipts.os, "link", race)
    assert receipts.write_receipt(path, dict(BODY)) == winner


def test_load_missing_receipt_refuses(tmp_path):
    with pytest.raises(sv.Refusal) as caught:
        receipts.load_receipt(tmp_path / "nope.json")
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "tuning_receipt_missing"


def test_load_tampered_receipt_refuses(tmp_path):
    path = tmp_path / "r.json"
    receipts.write_receipt(path, dict(BODY))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["c"] = 0.9  # body changed, content_hash left as written
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(sv.Refusal) as caught:
        receipts.load_receipt(path)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == UNREADABLE


def test_oversize_receipt_is_refused_at_write_and_unreadable_at_load(tmp_path):
    big = {**BODY, "pad": "x" * receipts.MAX_RECEIPT_BYTES}
    path = tmp_path / "big.json"
    # The writer refuses an oversize body before touching the filesystem: no target, no temp file.
    with pytest.raises(sv.Refusal) as caught:
        receipts.write_receipt(path, big)
    assert caught.value.code == CONFLICT
    assert caught.value.details == {"reason": "tuning_receipt_too_large", "path": str(path)}
    assert list(tmp_path.iterdir()) == []
    # A well-formed, correctly hashed oversize receipt written by hand: the size cap alone must refuse it on load.
    full = dict(big, content_hash=receipts.receipt_hash(big))
    path.write_bytes((json.dumps(full, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    assert path.stat().st_size > receipts.MAX_RECEIPT_BYTES
    with pytest.raises(sv.Refusal) as caught:
        receipts.load_receipt(path)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == UNREADABLE


def test_symlinked_receipt_refuses_as_unreadable(tmp_path):
    seed = tmp_path / "seed.json"
    receipts.write_receipt(seed, dict(BODY))
    link = tmp_path / "link.json"
    os.symlink(seed, link)
    with pytest.raises(sv.Refusal) as caught:
        receipts.load_receipt(link)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == UNREADABLE


def test_missing_hash_refuses_as_unreadable(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(dict(BODY)), encoding="utf-8")
    with pytest.raises(sv.Refusal) as caught:
        receipts.load_receipt(path)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == UNREADABLE


def test_nan_receipt_text_refuses_as_unreadable(tmp_path):
    path = tmp_path / "r.json"
    path.write_text('{"a": NaN, "content_hash": "x"}', encoding="utf-8")
    with pytest.raises(sv.Refusal) as caught:
        receipts.load_receipt(path)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == UNREADABLE


@pytest.mark.parametrize("text", ["[1, 2]", '"text"', "7"])
def test_non_dict_json_refuses_as_unreadable(tmp_path, text):
    path = tmp_path / "r.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(sv.Refusal) as caught:
        receipts.load_receipt(path)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == UNREADABLE
