"""``ReadCounters`` snapshots are isolated copies, scoped to one instance."""
from engine.v2.data.read_counters import ReadCounters

_FIELDS = ("scans", "fragments_opened", "rows_decoded", "rows_returned",
           "bytes_read", "seconds")


def test_snapshot_is_isolated_copy():
    c = ReadCounters()
    c.add("t", scans=1)
    s = c.snapshot()
    c.add("t", scans=2)
    assert s == {"t": {"scans": 1, "fragments_opened": 0, "rows_decoded": 0,
                       "rows_returned": 0, "bytes_read": 0, "seconds": 0}}
    assert c.snapshot()["t"]["scans"] == 3


def test_mutating_a_snapshot_leaves_stored_counters_unchanged():
    c = ReadCounters()
    c.add("t", scans=1)
    s = c.snapshot()
    s["t"]["scans"] = 99
    s["other"] = {}
    expected = dict.fromkeys(_FIELDS, 0)
    expected["scans"] = 1
    assert c.snapshot() == {"t": expected}
    s1 = c.snapshot()
    s2 = c.snapshot()
    assert s1 is not s2
    assert s1["t"] is not s2["t"]


def test_table_totals_are_independent():
    c = ReadCounters()
    c.add("a", scans=1, rows_decoded=5)
    c.add("b", scans=2)
    snap = c.snapshot()
    assert snap["a"]["scans"] == 1
    assert snap["a"]["rows_decoded"] == 5
    assert snap["b"]["scans"] == 2
    assert snap["b"]["rows_decoded"] == 0


def test_repository_instances_do_not_share_counters():
    one = ReadCounters()
    other = ReadCounters()
    one.add("t", scans=1)
    assert other.snapshot() == {}


def test_unknown_field_is_ignored():
    c = ReadCounters()
    c.add("t", bogus=3)
    assert "bogus" not in c.snapshot()["t"]
