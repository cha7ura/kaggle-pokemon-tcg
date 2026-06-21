"""Value-guided Alakazam pilot (exceed-path #2): at clean MAIN single-choice decisions, forward-sim
each candidate one step via the engine model and pick the resulting state with the highest learned
win-probability (value net trained on top replays, AUC 0.976). Other decisions defer to the strong
public Alakazam rule agent. Crash-safe; value net is pure-python (no deps at inference).
"""
import os, json
from cg.api import (to_observation_class, OptionType, SelectType,
                    search_begin, search_step, search_end)
import agent_alakazam as H   # strong rule agent: default policy + deck

NE, SF = 12, 6 + 12 + 2 + 4
ENERGY_ID, BASIC = 3, 721
from cg.api import all_card_data, all_attack
CD = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: a for a in all_attack()}


def _oh(i, n):
    v = [0.0] * n
    if i is not None and 0 <= int(i) < n: v[int(i)] = 1.0
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


def _context(st):
    me = st.yourIndex; my = st.players[me]; op = st.players[1 - me]

    def pool(p):
        ms = _mons(p)
        if not ms: return CF0 + [0.0, 0.0]
        cols = list(zip(*[_cardfeat(m.id) for m in ms]))
        cf = [sum(c) / len(ms) for c in cols]
        return cf + [sum(m.hp / 300 for m in ms) / len(ms), sum(len(m.energies) / 4 for m in ms) / len(ms)]
    state = [len(my.prize) / 6, len(op.prize) / 6, len(my.bench) / 5, len(op.bench) / 5,
             len(my.hand or []) / 10, st.turn / 40,
             float(my.poisoned), float(my.burned), float(my.asleep), float(my.paralyzed), float(my.confused)]
    return pool(my) + pool(op) + state


def _load(p_):
    for p in (p_, os.path.join(os.path.dirname(__file__), p_), "/kaggle_simulations/agent/" + p_):
        if os.path.exists(p):
            d = json.load(open(p)); return d["trees"], d["base"]
    return None, 0.0
_VT, _VB = _load("value_trees.json")


def _tscore(node, f):
    while "leaf" not in node:
        nxt = node["yes"] if f[int(node["split"][1:])] < node["split_condition"] else node["no"]
        node = next(c for c in node["children"] if c["nodeid"] == nxt)
    return node["leaf"]


def _value(st, me):
    """win-prob for player `me` from state st."""
    import math
    raw = _VB + sum(_tscore(t, _context(st)) for t in _VT)
    v = 1 / (1 + math.exp(-raw))
    return v if st.yourIndex == me else 1 - v


def _determinize(state):
    me = state.yourIndex; my, opp = state.players[me], state.players[1 - me]
    oa = [BASIC] if (opp.active and opp.active[0] is None) else []
    return (H._DECK[:], [ENERGY_ID] * len(my.prize), H._DECK[:],
            [ENERGY_ID] * len(opp.prize), [ENERGY_ID] * opp.handCount, oa)


def _eval_action(obs, action, me):
    ss = search_begin(obs, *_determinize(obs.current)); sid = ss.searchId
    try:
        ss = search_step(sid, action)
        cur = ss.observation.current
        if cur is None: return 0.0
        if cur.result == me: return 1.0
        if cur.result == (1 - me): return 0.0
        return _value(cur, me)
    finally:
        search_end()


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return H._DECK
        sel = obs.select
        cands = [i for i in range(len(sel.option)) if int(sel.option[i].type) != int(OptionType.END)]
        if (_VT and int(sel.type) == int(SelectType.MAIN) and sel.maxCount == 1 and sel.minCount == 1
                and len(cands) >= 2):
            me = obs.current.yourIndex
            best_i, best_v = None, -1.0
            for i in cands:
                try:
                    v = _eval_action(obs, [i], me)
                except Exception:
                    v = -1.0
                if v > best_v:
                    best_v, best_i = v, i
            if best_i is not None:
                return [best_i]
        return H.select_indices(obs) if hasattr(H, "select_indices") else H.agent(obs_dict)
    except Exception:
        try:
            return H.agent(obs_dict)
        except Exception:
            return [0]
