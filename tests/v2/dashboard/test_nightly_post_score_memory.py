"""_log_rss stderr line and Scorer.release_score_caches cache clearing."""

from engine.dashboard import nightly
from engine.score import Scorer


def test_log_rss_writes_stderr_line(capsys, monkeypatch):
    monkeypatch.setattr(nightly.memprobe, "rss_bytes", lambda: 2_500_000_000)
    monkeypatch.setattr(nightly.memprobe, "peak_rss_bytes", lambda: 3_100_000_000)
    nightly._log_rss("x")
    captured = capsys.readouterr()
    assert captured.err == "[nightly-mem] x rss=2.50G peak=3.10G\n"
    assert captured.out == ""


def test_release_score_caches_empties_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "engine.score._release_free_pages", lambda: calls.append(1)
    )
    scorer = Scorer.__new__(Scorer)
    scorer._live_features_cache = {("a",): {"x": 1.0}}
    scorer.release_score_caches()
    assert scorer._live_features_cache == {}
    assert len(calls) == 1
