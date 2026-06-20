"""Self-contained cg-net trainer: BC warm-start -> mixed-opponent self-play A2C.

Runs in the linux/amd64 engine container OR on Colab. Engine-bound (CPU), so local Docker is a
fine substrate. Checkpoints to /app/weights. wandb optional (--wandb + WANDB_API_KEY env).

  PYTHONPATH=/app/sdk python colab/cg_a2c_train.py --iters 1000 --bc-games 150 --batch 8 --wandb
"""
import argparse, os, sys, time, importlib.util, random
import numpy as np, torch, torch.nn as nn

REPO = "/app"
sys.path.insert(0, REPO + "/sdk")
os.environ.setdefault("AGENT_DECK", REPO + "/autoresearch/decks/lucario_meta.csv")
dev = "cuda" if torch.cuda.is_available() else "cpu"
import cg.game as G
from cg.api import to_observation_class, OptionType, all_card_data, all_attack

CARD2IX = {c.cardId: i + 1 for i, c in enumerate(all_card_data())}; NCARD = len(CARD2IX) + 1
ATK = {a.attackId: a.damage for a in all_attack()}
PLAY, ATTACH, EVOLVE, ATTACK = (int(OptionType.PLAY), int(OptionType.ATTACH), int(OptionType.EVOLVE), int(OptionType.ATTACK))
NOPT, K, EMB, NE = 16, 12, 24, 12
DECK = [int(l) for l in open(os.environ["AGENT_DECK"]) if l.strip()][:60]
_spec = importlib.util.spec_from_file_location("agl", REPO + "/autoresearch/agent_lucario.py")
hl = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(hl); heuristic = hl.agent
_CD = {c.cardId: c for c in all_card_data()}
SF = 6 + NE + 13 + 13 + 6 + 3; AF = 1 + NE + 1
SLOT = EMB + SF + NE + 3; OPTF = NOPT + EMB + SF + AF; STATE = 2 + 2 + 1 + 1 + 5 + 5


def _emh(lst):
    v = np.zeros(NE, np.float32)
    for e in (lst or []):
        i = int(e)
        if 0 <= i < NE: v[i] = 1
    return v


def _oh(i, n):
    v = np.zeros(n, np.float32)
    if i is not None and 0 <= int(i) < n: v[int(i)] = 1
    return v


def _cardfeat(cid):
    c = _CD.get(cid)
    if c is None: return np.zeros(SF, np.float32)
    wk = int(c.weakness) if c.weakness is not None else 12
    rs = int(c.resistance) if c.resistance is not None else 12
    flags = np.array([c.basic, c.stage1, c.stage2, c.ex, c.megaEx, c.aceSpec], np.float32)
    sc = np.array([c.hp / 300, c.retreatCost / 4, len(c.attacks) / 4], np.float32)
    return np.concatenate([_oh(int(c.cardType), 6), _oh(int(c.energyType), NE), _oh(wk, 13), _oh(rs, 13), flags, sc])


CARDFEAT = {cid: _cardfeat(cid) for cid in _CD}; CF0 = np.zeros(SF, np.float32)
ATKFEAT = {a.attackId: np.concatenate([[a.damage / 100], _emh(a.energies), [len(a.energies) / 4]]).astype(np.float32) for a in all_attack()}
AF0 = np.zeros(AF, np.float32)


def _mons(p): return (([p.active[0]] if p.active and p.active[0] else []) + [b for b in p.bench if b])[:6]


def encode3(obs_dict):
    o = to_observation_class(obs_dict); sel = o.select; st = o.current
    me = st.yourIndex; my = st.players[me]; op = st.players[1 - me]
    ids = np.zeros(K, np.int64); feats = np.zeros((K, SF + NE + 3), np.float32)
    for i, m in enumerate(_mons(my)): ids[i] = CARD2IX.get(m.id, 0); feats[i] = np.concatenate([CARDFEAT.get(m.id, CF0), _emh(m.energies), [m.hp / max(1, m.maxHp), len(m.energies) / 4, 1.0]])
    for j, m in enumerate(_mons(op)): ids[6 + j] = CARD2IX.get(m.id, 0); feats[6 + j] = np.concatenate([CARDFEAT.get(m.id, CF0), _emh(m.energies), [m.hp / max(1, m.maxHp), len(m.energies) / 4, 0.0]])
    state = np.concatenate([[len(my.prize) / 6, len(op.prize) / 6], [len(my.bench) / 5, len(op.bench) / 5], [len(my.hand or []) / 10], [st.turn / 40], [my.poisoned, my.burned, my.asleep, my.paralyzed, my.confused], [op.poisoned, op.burned, op.asleep, op.paralyzed, op.confused]]).astype(np.float32)
    n = len(sel.option); otype = np.zeros((n, NOPT), np.float32); ocard = np.zeros(n, np.int64); ocf = np.zeros((n, SF), np.float32); oaf = np.zeros((n, AF), np.float32)
    for i, opt in enumerate(sel.option):
        t = int(opt.type); otype[i] = _oh(t, NOPT); idx = getattr(opt, "index", None)
        if t in (PLAY, ATTACH, EVOLVE) and idx is not None and my.hand and idx < len(my.hand) and my.hand[idx]:
            cid = my.hand[idx].id; ocard[i] = CARD2IX.get(cid, 0); ocf[i] = CARDFEAT.get(cid, CF0)
        if t == ATTACK: oaf[i] = ATKFEAT.get(getattr(opt, "attackId", -1), AF0)
    return dict(ids=ids, feats=feats, state=state, otype=otype, ocard=ocard, ocf=ocf, oaf=oaf, sel=sel)


class Net3(nn.Module):
    def __init__(s, ncard, h=256):
        super().__init__(); s.emb = nn.Embedding(ncard, EMB)
        s.board = nn.Sequential(nn.Linear(SLOT, h), nn.ReLU())
        s.score = nn.Sequential(nn.Linear(h + STATE + OPTF, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        s.value = nn.Sequential(nn.Linear(h + STATE, h), nn.ReLU(), nn.Linear(h, 1))

    def _ctx(s, e, T):
        bv = s.board(torch.cat([s.emb(T(e["ids"])), T(e["feats"])], -1)).mean(-2)
        return torch.cat([bv, T(e["state"])], -1)

    def logits(s, e):
        T = lambda a: torch.as_tensor(a, device=dev)
        ctx = s._ctx(e, T); n = e["otype"].shape[0]
        x = torch.cat([ctx[None, :].expand(n, -1), T(e["otype"]), s.emb(T(e["ocard"])), T(e["ocf"]), T(e["oaf"])], -1)
        return s.score(x).squeeze(-1)

    def value_of(s, e):
        T = lambda a: torch.as_tensor(a, device=dev)
        return s.value(s._ctx(e, T)).squeeze(-1)


def _kpick(net, e, sample):
    sel = e["sel"]; lg = net.logits(e)
    k = min(sel.maxCount, len(sel.option)); k = max(k, min(max(1, sel.minCount), len(sel.option)))
    if sample:
        return torch.multinomial(torch.softmax(lg.detach(), 0), k).tolist()
    return torch.topk(lg, k).indices.tolist()


def net_agent(net, sample=False):
    def act(o):
        ob = to_observation_class(o)
        if ob.select is None: return DECK
        return _kpick(net, encode3(o), sample)
    return act


def random_agent(o):
    ob = to_observation_class(o)
    if ob.select is None: return DECK
    n = len(ob.select.option); return random.sample(range(n), min(max(1, ob.select.minCount), n))


def play(a0, a1):
    obs, _ = G.battle_start(DECK, DECK)
    if obs is None: return None
    s = 0
    try:
        while True:
            cur = obs["current"]
            if cur is not None and cur.get("result", -1) != -1: return cur["result"]
            if obs["select"] is None: return None
            pl = cur["yourIndex"]
            try: action = (a0 if pl == 0 else a1)(obs)
            except Exception: return 1 - pl
            obs = G.battle_select(action); s += 1
            if s > 10000: return 2
    finally: G.battle_finish()


def winrate(net, opp, n=40):
    w = d = l = 0
    for i in range(n):
        seat = i % 2; r = play(net_agent(net), opp) if seat == 0 else play(opp, net_agent(net))
        if r is None: continue
        if r == 2: d += 1
        elif r == seat: w += 1
        else: l += 1
    return (w + 0.5 * d) / max(1, w + d + l)


def play_sp(net):
    obs, _ = G.battle_start(DECK, DECK)
    if obs is None: return [], {0: 0.0, 1: 0.0}
    recs = []; leads = {0: 0.0, 1: 0.0}; steps = 0
    try:
        while True:
            cur = obs["current"]
            if cur is not None and cur.get("result", -1) != -1:
                r = cur["result"]; return recs, {s: ((0.0 if r == 2 else (1.0 if r == s else -1.0)) + 0.3 * leads[s]) for s in (0, 1)}
            if obs["select"] is None: return recs, {0: 0.0, 1: 0.0}
            pl = cur["yourIndex"]; e = encode3(obs); sel = e["sel"]
            lg = net.logits(e); logp = torch.log_softmax(lg, 0); p = torch.softmax(lg, 0); v = net.value_of(e)
            leads[pl] = float(e["state"][1] - e["state"][0])
            k = min(sel.maxCount, len(sel.option)); k = max(k, min(max(1, sel.minCount), len(sel.option)))
            idx = torch.multinomial(p.detach(), k).tolist()
            recs.append((logp[idx].sum(), -(p * logp).sum(), v, pl))
            obs = G.battle_select(idx); steps += 1
            if steps > 10000: return recs, {0: 0.3 * leads[0], 1: 0.3 * leads[1]}
    finally: G.battle_finish()


def play_vsheur(net):
    seat = random.randint(0, 1); obs, _ = G.battle_start(DECK, DECK)
    if obs is None: return [], 0.0
    recs = []; lead = 0.0; steps = 0
    try:
        while True:
            cur = obs["current"]
            if cur is not None and cur.get("result", -1) != -1:
                r = cur["result"]; return recs, (0.0 if r == 2 else (1.0 if r == seat else -1.0)) + 0.3 * lead
            if obs["select"] is None: return recs, 0.0
            pl = cur["yourIndex"]
            if pl == seat:
                e = encode3(obs); sel = e["sel"]; lg = net.logits(e); logp = torch.log_softmax(lg, 0); p = torch.softmax(lg, 0); v = net.value_of(e)
                lead = float(e["state"][1] - e["state"][0])
                k = min(sel.maxCount, len(sel.option)); k = max(k, min(max(1, sel.minCount), len(sel.option)))
                idx = torch.multinomial(p.detach(), k).tolist()
                recs.append((logp[idx].sum(), -(p * logp).sum(), v, seat)); action = idx
            else:
                try: action = heuristic(obs)
                except Exception: return recs, (1.0 if pl != seat else -1.0) + 0.3 * lead
            obs = G.battle_select(action); steps += 1
            if steps > 10000: return recs, 0.3 * lead
    finally: G.battle_finish()


def a2c_mixed(net, opt, games, p_heur=0.5, vc=0.5, bc=0.01):
    loss = torch.zeros((), device=dev); hr = []
    for _ in range(games):
        if random.random() < p_heur:
            recs, rew = play_vsheur(net); pairs = [(lp, ent, v, rew) for (lp, ent, v, _) in recs]
            if recs: hr.append(1.0 if rew > 0 else (0.5 if abs(rew) < 1e-6 else 0.0))
        else:
            recs, rd = play_sp(net); pairs = [(lp, ent, v, rd[s]) for (lp, ent, v, s) in recs]
        for lp, ent, v, R in pairs:
            adv = R - v.detach(); loss = loss - adv * lp + vc * (v - R) ** 2 - bc * ent
    if loss.requires_grad:
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    return float(np.mean(hr)) if hr else 0.0


def bc_warmstart(net, opt, games, epochs):
    data = []
    for g in range(games):
        obs, _ = G.battle_start(DECK, DECK)
        if obs is None: continue
        s = 0
        try:
            while True:
                cur = obs["current"]
                if cur is not None and cur.get("result", -1) != -1: break
                if obs["select"] is None: break
                e = encode3(obs)
                try: a = heuristic(obs)
                except Exception: break
                data.append((e, a[0])); obs = G.battle_select(a); s += 1
                if s > 10000: break
        finally: G.battle_finish()
    print(f"  BC data: {len(data)} samples", flush=True)
    for ep in range(epochs):
        random.shuffle(data); tot = 0.0
        for e, y in data:
            lg = net.logits(e); loss = nn.functional.cross_entropy(lg[None], torch.tensor([y], device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"  BC ep{ep} loss={tot/len(data):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--bc-games", type=int, default=150)
    ap.add_argument("--bc-epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    os.makedirs(REPO + "/weights", exist_ok=True)
    run = None
    if args.wandb and os.environ.get("WANDB_API_KEY"):
        import wandb; run = wandb.init(project="ptcg-rnad", name="local-mixed-a2c", config=vars(args))
    net = Net3(NCARD).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    print(f"device={dev} params={sum(p.numel() for p in net.parameters())}", flush=True)
    print("== BC warm-start ==", flush=True); bc_warmstart(net, opt, args.bc_games, args.bc_epochs)
    print("warm wr_vs_heur", round(winrate(net, heuristic, 40), 3), flush=True)
    for g in opt.param_groups: g["lr"] = 3e-4
    best = 0.0
    for it in range(args.iters):
        t0 = time.time(); thw = a2c_mixed(net, opt, args.batch)
        dt = time.time() - t0
        if it % 25 == 0 or it == args.iters - 1:
            wr = winrate(net, heuristic, 40)
            print(f"it{it:4d} train_hwr={thw:.3f} wr_vs_heur={wr:.3f} {dt:.1f}s/it", flush=True)
            if run: run.log({"iter": it, "train_hwr": thw, "wr_vs_heuristic": wr, "s_per_it": dt})
            if wr > best: best = wr; torch.save(net.state_dict(), REPO + "/weights/local_best.pt")
        elif run:
            run.log({"iter": it, "train_hwr": thw, "s_per_it": dt})
        if (it + 1) % 100 == 0:
            torch.save(net.state_dict(), REPO + "/weights/local_ck.pt"); print(f"  [ckpt it{it+1} best={best:.3f}]", flush=True)
    if run: run.finish()
    print(f"DONE best wr_vs_heur={best:.3f}", flush=True)


if __name__ == "__main__":
    main()
