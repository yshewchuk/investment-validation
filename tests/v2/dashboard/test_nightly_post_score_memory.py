"""_log_rss stderr line and Scorer.release_score_caches cache clearing."""

import re

from engine.dashboard.nightly import _log_rss
from engine.score import Scorer


def test_log_rss_writes_stderr_line(capsys):
    _log_rss("x")
    captured = capsys.readouterr()
    assert re.search(r"\[nightly-mem\] x rss=\d+\.\d\dG peak=", captured.err)
    assert captured.out == ""


def test_release_score_caches_empties_cache():
    scorer = Scorer.__new__(Scorer)
    scorer._live_features_cache = {("a",): {"x": 1.0}}
    scorer.release_score_caches()
    assert scorer._live_features_cache == {}
