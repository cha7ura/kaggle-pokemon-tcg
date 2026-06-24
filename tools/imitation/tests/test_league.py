import json, glob
from tools.imitation.league import opponent_pilot, build_roster, _learned_slugs
from tools import replays_db


def test_opponent_pilot_routing():
    assert glob.glob("tools/imitation/policies/*.json"), "need a trained policy (Task 4)"
    manifest = json.load(open("tools/imitation/data/manifest.json"))
    sig = next(m["deck_sig"] for m in manifest if m["games"] >= 30)
    assert opponent_pilot(sig)[0] == "imitation"
    assert opponent_pilot("999999_does_not_exist")[0] == "typh"


def test_build_roster_loads_policy():
    slugs = _learned_slugs()
    assert slugs, "expected at least one field deck with a learned policy"
    roster = build_roster(slugs[:2])
    r = roster[0]
    assert len(r["deck"]) == 60
    assert r["pilot"] in ("imitation", "typh")
    if r["pilot"] == "imitation":
        assert isinstance(r["policy"], dict) and "feature" in r["policy"]


def test_league_storage_roundtrip():
    batch = "__test_batch__"
    db = replays_db._connect()
    db.execute("DELETE FROM league_games WHERE batch=?", (batch,)); db.commit(); db.close()
    rows = [
        {"deck_a": "f00", "deck_b": "f01", "pilot_a": "imitation", "pilot_b": "typh", "winner": 0},
        {"deck_a": "f00", "deck_b": "f01", "pilot_a": "imitation", "pilot_b": "typh", "winner": 1},
        {"deck_a": "f00", "deck_b": "f01", "pilot_a": "imitation", "pilot_b": "typh", "winner": 0},
    ]
    assert replays_db.store_league_games(rows, batch) == 3
    m = replays_db.league_matrix(batch)
    cell = m[("f00", "f01")]
    assert cell["n"] == 3 and cell["a"] == 2 and cell["b"] == 1
    db = replays_db._connect()
    db.execute("DELETE FROM league_games WHERE batch=?", (batch,)); db.commit(); db.close()
