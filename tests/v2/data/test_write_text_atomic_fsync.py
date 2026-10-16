import os

from engine.data import manifest


def test_write_text_atomic_fsyncs_full_content_before_replace(tmp_path, monkeypatch):
    target = tmp_path / "out.json"
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append(("fsync", os.fstat(fd).st_size))
        return real_fsync(fd)

    def replace(src, dst):
        events.append(("replace", None))
        return real_replace(src, dst)

    monkeypatch.setattr(manifest.os, "fsync", fsync)
    monkeypatch.setattr(manifest.os, "replace", replace)
    manifest.write_text_atomic(target, "héllo")
    assert events == [("fsync", len("héllo".encode())), ("replace", None)]
    assert target.read_text(encoding="utf-8") == "héllo"
