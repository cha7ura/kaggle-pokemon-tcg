# tools/imitation/tests/test_extract.py
import json, glob
from tools.imitation.extract_decisions import iter_decisions, deck_sig_of

def test_iter_decisions_yields_valid_records():
    f = sorted(glob.glob("json/*.json"))[0]
    g = json.load(open(f))
    recs = list(iter_decisions(g))
    assert recs, "expected at least one decision"
    r = recs[0]
    assert set(r) >= {"deck_sig", "player", "reward", "context", "state", "options", "chosen"}
    assert 0 <= r["chosen"] < len(r["options"])
    assert len(r["options"]) >= 2
    # every option vector same width
    assert len({len(o) for o in r["options"]}) == 1

def test_deck_sig_stable():
    a = deck_sig_of([3, 1, 2, 1])
    b = deck_sig_of([1, 1, 2, 3])
    assert a == b
