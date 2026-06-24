# tools/imitation/tests/test_features.py
import math, itertools
from tools.imitation.features import state_features, option_features, STATE_DIM, OPTION_DIM
from tools.replays_db import iter_replays

def _a_decision():
    for _ep, d in itertools.islice(iter_replays(), 50):
        for s in d["steps"]:
            for seat in (0, 1):
                obs = s[seat].get("observation", {})
                sel = obs.get("select")
                cur = obs.get("current")
                act = s[seat].get("action")
                if (sel and cur and isinstance(act, list) and sel.get("option")
                        and len(sel["option"]) >= 2
                        and all(isinstance(x, int) and x < len(sel["option"]) for x in act)):
                    return cur, sel, seat
    raise RuntimeError("no decision found")

def test_state_features_shape_and_finite():
    cur, sel, seat = _a_decision()
    v = state_features(cur, seat)
    assert len(v) == STATE_DIM
    assert all(isinstance(x, (int, float)) for x in v)
    assert all(math.isfinite(x) for x in v)

def test_option_features_shape():
    cur, sel, seat = _a_decision()
    v = option_features(sel["option"][0], sel.get("context"), sel)
    assert len(v) == OPTION_DIM

def test_state_features_handles_empty_active():
    # a player with empty active list must not crash, returns zeros for active fields
    cur = {"turn": 1, "yourIndex": 0, "supporterPlayed": False, "stadiumPlayed": False,
           "turnActionCount": 0,
           "players": [{"active": [], "bench": [], "hand": [], "prize": [None]*6,
                        "handCount": 0, "asleep": False, "confused": False,
                        "paralyzed": False, "poisoned": False, "burned": False},
                       {"active": [], "bench": [], "hand": [], "prize": [None]*6,
                        "handCount": 0, "asleep": False, "confused": False,
                        "paralyzed": False, "poisoned": False, "burned": False}]}
    v = state_features(cur, 0)
    assert len(v) == STATE_DIM
