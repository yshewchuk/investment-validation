"""D3: the withheld-release banner must clear once delivery catches up (§5)."""
from engine.v2.ops.catalog import transaction
from engine.v2.ops.health import health
from tests.ops_support import catalog


def _release(conn, release_id, occurrence, *, eligible, delivered):
    with transaction(conn):
        conn.execute(
            "INSERT INTO releases(release_id,occurrence,manifest_json,manifest_hash,"
            "expected_current,eligible,published_at,delivered_at) VALUES (?,?,?,?,?,?,?,?)",
            (release_id, occurrence, "{}", "hash-" + release_id, None, int(eligible),
             occurrence if delivered else None, occurrence if delivered else None))


def test_withheld_with_no_delivery_is_shown(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    _release(conn, "r5", "2026-09-05", eligible=False, delivered=False)
    document = health(conn, clock=clock)
    assert document["withheld_release"]["release_id"] == "r5"


def test_withheld_on_night_5_delivered_on_night_6_is_cleared(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    _release(conn, "r5", "2026-09-05", eligible=False, delivered=False)
    _release(conn, "r6", "2026-09-06", eligible=True, delivered=True)
    document = health(conn, clock=clock)
    assert document["withheld_release"] is None
    assert document["current_release"]["release_id"] == "r6"


def test_delivered_on_night_5_withheld_on_night_6_is_shown(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    _release(conn, "r5", "2026-09-05", eligible=True, delivered=True)
    _release(conn, "r6", "2026-09-06", eligible=False, delivered=False)
    document = health(conn, clock=clock)
    assert document["withheld_release"]["release_id"] == "r6"


def test_withheld_and_delivered_on_the_same_occurrence_is_cleared(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    _release(conn, "r5-withheld", "2026-09-05", eligible=False, delivered=False)
    _release(conn, "r5-delivered", "2026-09-05", eligible=True, delivered=True)
    document = health(conn, clock=clock)
    assert document["withheld_release"] is None
