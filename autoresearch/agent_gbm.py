"""GBM-imitator Alakazam pilot: scores each legal option with a gradient-boosted tree trained on
TOP-agent replay decisions (tools/build_gbm.py). Pure-python tree traversal — no xgboost/numpy
needed at inference. Crash-safe. Feature extraction MUST match build_gbm.py exactly.
"""
import os, json
from cg.api import to_observation_class, OptionType, all_card_data, all_attack

NE, NOPT = 12, 17
PLAY, ATTACH, EVOLVE, ATTACK = 7, 8, 9, 13
SF = 6 + NE + 2 + 4
CD = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: a for a in all_attack()}


def _oh(i, n):
    v = [0.0] * n
    if i is not None and 0 <= int(i) < n: v[int(i)] = 1.0
    return v


def _emh(lst):
    v = [0.0] * NE
    for e in (lst or []):
        i = int(e)
        if 0 <= i < NE: v[i] = 1.0
    return v


def _cardfeat(cid):
    c = CD.get(cid)
    if not c: return [0.0] * SF
    bd = max([ATK[a].damage for a in c.attacks if a in ATK], default=0)
    return (_oh(int(c.cardType), 6) + _oh(int(c.energyType), NE)
            + [1.0 if c.ex else 0.0, 1.0 if c.megaEx else 0.0]
            + [c.hp / 300, c.retreatCost / 4, len(c.attacks) / 4, bd / 300])
CF0 = [0.0] * SF


def _mons(p):
    return (([p.active[0]] if p.active and p.active[0] else []) + [b for b in p.bench if b])[:6]


def _mean(vs):
    n = len(vs)
    return [sum(col) / n for col in zip(*vs)] if n else None


def _context(o):
    st = o.current; me = st.yourIndex; my = st.players[me]; op = st.players[1 - me]

    def pool(p):
        ms = _mons(p)
        if not ms: return CF0 + [0.0, 0.0]
        cf = _mean([_cardfeat(m.id) for m in ms])
        return cf + [sum(m.hp / 300 for m in ms) / len(ms), sum(len(m.energies) / 4 for m in ms) / len(ms)]
    state = [len(my.prize) / 6, len(op.prize) / 6, len(my.bench) / 5, len(op.bench) / 5,
             len(my.hand or []) / 10, st.turn / 40,
             float(my.poisoned), float(my.burned), float(my.asleep), float(my.paralyzed), float(my.confused)]
    return pool(my) + pool(op) + state


def _optfeat(o, opt):
    st = o.current; my = st.players[st.yourIndex]
    t = int(opt.type); ot = _oh(t, NOPT); ocf = CF0; oaf = [0.0] * (1 + NE)
    idx = getattr(opt, "index", None)
    if t in (PLAY, ATTACH, EVOLVE) and idx is not None and my.hand and idx < len(my.hand) and my.hand[idx]:
        ocf = _cardfeat(my.hand[idx].id)
    if t == ATTACK:
        a = ATK.get(getattr(opt, "attackId", -1))
        if a: oaf = [a.damage / 300] + _emh(a.energies)
    return ot + ocf + oaf


def _load_trees():
    for p in ("weights/gbm_trees.json", "gbm_trees.json", "/kaggle_simulations/agent/gbm_trees.json",
              os.path.join(os.path.dirname(__file__), "gbm_trees.json")):
        if os.path.exists(p):
            d = json.load(open(p)); return d["trees"], d["base"]
    return None, 0.0
_TREES, _BASE = _load_trees()


def _tree_score(node, feat):
    while "leaf" not in node:
        idx = int(node["split"][1:])               # "f37" -> 37
        nxt = node["yes"] if feat[idx] < node["split_condition"] else node["no"]
        node = next(c for c in node["children"] if c["nodeid"] == nxt)
    return node["leaf"]


def _score(feat):
    return _BASE + sum(_tree_score(t, feat) for t in _TREES)


def _legal_fallback(sel):
    n = len(sel.option); k = min(max(1, sel.minCount), n) if n else 0
    return list(range(k))


def agent(obs_dict):
    try:
        o = to_observation_class(obs_dict)
        if o.select is None:
            return _DECK
        sel = o.select
        if not _TREES:
            return _legal_fallback(sel)
        ctx = _context(o)
        scores = [_score(ctx + _optfeat(o, opt)) for opt in sel.option]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        k = min(sel.maxCount, len(sel.option)); k = max(k, min(max(1, sel.minCount), len(sel.option)))
        return order[:k] if order else _legal_fallback(sel)
    except Exception:
        try:
            return _legal_fallback(to_observation_class(obs_dict).select)
        except Exception:
            return [0]


def _read_deck():
    for p in ("deck.csv", "/kaggle_simulations/agent/deck.csv", "decks/alakazam.csv"):
        if os.path.exists(p):
            return [int(x) for x in open(p) if x.strip()][:60]
    return []
_DECK = _read_deck()
