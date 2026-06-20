"""Train a gradient-boosted-tree imitator of TOP Alakazam agents from replay decisions.

Pointwise action-scoring: one training row per (decision, option); label=1 if the top agent chose
that option. XGBoost scores each option; the agent picks argmax legal. Trees -> no encoder wall,
ships as plain JSON + a tiny pure-python traverser (no xgboost at inference). Run in Docker:
  pip install -q xgboost numpy; PYTHONPATH=/app/sdk python tools/build_gbm.py
"""
import json, glob, os, csv, numpy as np
from cg.api import to_observation_class, OptionType, all_card_data, all_attack

ROOT = "/app"
CD = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: a for a in all_attack()}
NE, NOPT = 12, 17
PLAY, ATTACH, EVOLVE, ATTACK = 7, 8, 9, 13


def emh(lst):
    v = np.zeros(NE, np.float32)
    for e in (lst or []):
        i = int(e)
        if 0 <= i < NE: v[i] = 1
    return v


def oh(i, n):
    v = np.zeros(n, np.float32)
    if i is not None and 0 <= int(i) < n: v[int(i)] = 1
    return v


SF = 6 + NE + 2 + 4  # cardType, energyType, ex/megaEx, hp/retreat/natk/dmgPerE
def cardfeat(cid):
    c = CD.get(cid)
    if not c: return np.zeros(SF, np.float32)
    bd = max([ATK[a].damage for a in c.attacks if a in ATK], default=0)
    return np.concatenate([oh(int(c.cardType), 6), oh(int(c.energyType), NE),
        [1.0 if c.ex else 0, 1.0 if c.megaEx else 0],
        [c.hp/300, c.retreatCost/4, len(c.attacks)/4, bd/300]]).astype(np.float32)
CF0 = np.zeros(SF, np.float32)


def mons(p):
    return (([p.active[0]] if p.active and p.active[0] else []) + [b for b in p.bench if b])[:6]


def context_feats(o):
    st = o.current; me = st.yourIndex; my = st.players[me]; op = st.players[1-me]
    def pool(p):
        ms = mons(p)
        if not ms: return np.concatenate([CF0, [0, 0]])
        cf = np.mean([cardfeat(m.id) for m in ms], 0)
        return np.concatenate([cf, [np.mean([m.hp/300 for m in ms]), np.mean([len(m.energies)/4 for m in ms])]])
    state = np.array([len(my.prize)/6, len(op.prize)/6, len(my.bench)/5, len(op.bench)/5,
                      len(my.hand or [])/10, st.turn/40,
                      my.poisoned, my.burned, my.asleep, my.paralyzed, my.confused], np.float32)
    return np.concatenate([pool(my), pool(op), state])  # fixed-size per decision


def option_feats(o, opt):
    st = o.current; my = st.players[st.yourIndex]
    t = int(opt.type); ot = oh(t, NOPT)
    ocf = CF0.copy(); oaf = np.zeros(1 + NE, np.float32)
    idx = getattr(opt, "index", None)
    if t in (PLAY, ATTACH, EVOLVE) and idx is not None and my.hand and idx < len(my.hand) and my.hand[idx]:
        ocf = cardfeat(my.hand[idx].id)
    if t == ATTACK:
        a = ATK.get(getattr(opt, "attackId", -1))
        if a: oaf = np.concatenate([[a.damage/300], emh(a.energies)])
    return np.concatenate([ot, ocf, oaf])


def deck_of(steps, p):
    for s in steps:
        a = s[p].get("action")
        if isinstance(a, list) and len(a) == 60 and all(isinstance(x, int) for x in a): return a


def main():
    X, y = [], []
    files = glob.glob(f"{ROOT}/json/*.json")
    n_dec = 0
    for f in files:
        try: d = json.load(open(f))
        except Exception: continue
        for p in (0, 1):
            dk = deck_of(d["steps"], p)
            if not dk or 743 not in set(dk): continue  # Alakazam agents only
            for s in d["steps"]:
                o = s[p].get("observation"); a = s[p].get("action")
                if not isinstance(o, dict) or not isinstance(a, list): continue
                sel = o.get("select")
                if not isinstance(sel, dict) or sel.get("type") != 0: continue
                opts = sel.get("option") or []
                if len(opts) < 2: continue
                try: ob = to_observation_class(o)
                except Exception: continue
                ctx = context_feats(ob)
                chosen = set(a)
                for i, opt in enumerate(ob.select.option):
                    X.append(np.concatenate([ctx, option_feats(ob, opt)]))
                    y.append(1 if i in chosen else 0)
                n_dec += 1
    X = np.array(X, np.float32); y = np.array(y, np.float32)
    print(f"decisions={n_dec} rows={len(y)} pos={int(y.sum())} feat_dim={X.shape[1]}", flush=True)
    import xgboost as xgb
    dtr = xgb.DMatrix(X, label=y)
    params = {"objective": "binary:logistic", "max_depth": 6, "eta": 0.2,
              "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss"}
    bst = xgb.train(params, dtr, num_boost_round=120)
    bst.save_model(f"{ROOT}/weights/gbm_alakazam.json")
    # quick train accuracy (argmax-per-decision would need grouping; report AUC-ish via threshold)
    pred = bst.predict(dtr)
    acc = ((pred > 0.5) == (y > 0.5)).mean()
    print(f"saved weights/gbm_alakazam.json | row acc={acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
