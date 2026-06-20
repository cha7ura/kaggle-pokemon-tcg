"""cg-scale net training: behavior-clone the heuristic, then (TODO) R-NaD self-play fine-tune.

Runs on Colab (engine pulled as a wandb artifact `ptcg-engine`; sdk on sys.path). The shipped
agent stays NumPy/stdlib — this trains the net; agent_net.py would run the exported weights.

Status (run on T4, wandb project inc-llm/ptcg-rnad):
  behavior cloning of agent_lucario -> imitation_acc 0.78, wr_vs_random 0.75, wr_vs_heuristic 0.10.
  BC plays real TCG but loses to the teacher (imitation error compounds over a game) -> needs
  R-NaD self-play fine-tune (the win objective, not per-move imitation). See wiki/12, wiki/13.

Net2 = action-scoring policy: card-id embeddings over the board + per-option features
(type one-hot, target-card embedding, attack damage) -> one logit per legal option -> softmax.
Handles the variable TCG action space (AlphaStar-style pointer head, wiki/13).
"""
import os, sys, importlib.util
import numpy as np, torch, torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NOPT, K, EMB = 16, 12, 16


def setup(ptcg_dir):
    """Put the cg engine + heuristic on the path. ptcg_dir = downloaded ptcg-engine artifact."""
    sys.path.insert(0, os.path.join(ptcg_dir, "sdk"))
    os.environ["AGENT_DECK"] = os.path.join(ptcg_dir, "decks/lucario_meta.csv")
    spec = importlib.util.spec_from_file_location("agl", os.path.join(ptcg_dir, "autoresearch/agent_lucario.py"))
    hl = importlib.util.module_from_spec(spec); spec.loader.exec_module(hl)
    return hl                                   # hl.agent = heuristic, hl._DECK = deck


def _maps():
    from cg.api import all_card_data, all_attack
    c2i = {c.cardId: i + 1 for i, c in enumerate(all_card_data())}
    atk = {a.attackId: a.damage for a in all_attack()}
    return c2i, atk


def make_encode(c2i, atk):
    from cg.api import to_observation_class, OptionType
    PLAY, ATTACH, EVOLVE, ATTACK = (int(OptionType.PLAY), int(OptionType.ATTACH),
                                    int(OptionType.EVOLVE), int(OptionType.ATTACK))

    def mons(p):
        out = ([p.active[0]] if p.active and p.active[0] else []) + [b for b in p.bench if b]
        return out[:6]

    def encode(obs_dict):
        o = to_observation_class(obs_dict); sel = o.select; st = o.current
        me = st.yourIndex; my = st.players[me]; op = st.players[1 - me]
        bids = np.zeros(K, np.int64); bfeat = np.zeros((K, 2), np.float32)
        for i, m in enumerate(mons(my)):
            bids[i] = c2i.get(m.id, 0); bfeat[i] = [m.hp / 300, len(m.energies) / 4]
        for j, m in enumerate(mons(op)):
            bids[6 + j] = c2i.get(m.id, 0); bfeat[6 + j] = [m.hp / 300, len(m.energies) / 4]
        state = np.array([len(my.prize) / 6, len(op.prize) / 6, len(my.bench) / 5,
                          len(op.bench) / 5, len(my.hand or []) / 10], np.float32)
        n = len(sel.option)
        ot = np.zeros((n, NOPT), np.float32); oc = np.zeros(n, np.int64); osc = np.zeros((n, 1), np.float32)
        for i, opt in enumerate(sel.option):
            t = int(opt.type)
            if t < NOPT: ot[i, t] = 1
            idx = getattr(opt, "index", None)
            if t in (PLAY, ATTACH, EVOLVE) and idx is not None and my.hand and idx < len(my.hand) and my.hand[idx]:
                oc[i] = c2i.get(my.hand[idx].id, 0)
            if t == ATTACK: osc[i, 0] = atk.get(getattr(opt, "attackId", -1), 0) / 100
        return dict(bids=bids, bfeat=bfeat, state=state, otype=ot, ocard=oc, oscal=osc, sel=sel)
    return encode


class Net2(nn.Module):
    def __init__(s, ncard, h=128):
        super().__init__()
        s.emb = nn.Embedding(ncard, EMB)
        s.board = nn.Sequential(nn.Linear(EMB + 2, h), nn.ReLU())
        s.score = nn.Sequential(nn.Linear(h + 5 + NOPT + EMB + 1, h), nn.ReLU(), nn.Linear(h, 1))

    def logits(s, e):
        T = lambda a: torch.as_tensor(a, device=DEV)
        bvec = s.board(torch.cat([s.emb(T(e["bids"])), T(e["bfeat"])], 1)).mean(0)
        ctx = torch.cat([bvec, T(e["state"])])
        n = e["otype"].shape[0]
        x = torch.cat([ctx.expand(n, -1), T(e["otype"]), s.emb(T(e["ocard"])), T(e["oscal"])], 1)
        return s.score(x).squeeze(-1)


# game loop + agents + behavior cloning: see the Colab notebook cells (gen_bc_data / BC train loop).
# This module holds the reusable encoder + net; the driver lives in the notebook for live iteration.
# Next: R-NaD self-play fine-tune from the bc-net warm-start (wandb artifact bc-net).
