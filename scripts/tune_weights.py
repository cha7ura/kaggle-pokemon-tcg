"""Black-box weight tuning for the Grimmsnarl pilot (stdlib-only (mu+lambda) evolution strategy).

Optimizes agent_grimmsnarl's W weight table by win-rate vs the generic pilot on the SAME deck,
using the pilot_gate engine loop. This is policy search ON TOP of the proven heuristic
representation -- NOT learning from raw features (which hit a wall: clone 45-47% < 50.8% baseline).

Why this and not deep RL: the bottleneck is the feature representation, not the algorithm. RL over
raw features inherits that wall; tuning ~10 weights on a working heuristic sidesteps it, is tractable
on CPU (no GPU), and directly optimizes engine win-rate.

    PYTHONPATH=sdk python tune_weights.py --deck autoresearch/decks/grimmsnarl.csv \
        --games 60 --gens 12 --pop 8 --out tuned_weights.json

Each candidate is evaluated by playing --games games (seat-alternated) of tuned-pilot vs
generic agent_lucario. Fitness = win-rate. Prints progress each generation; writes best weights.
"""
import os, sys, json, random, argparse, importlib, math
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "sdk"))
sys.path.insert(0, os.path.join(ROOT, "autoresearch"))
from cg.game import battle_start, battle_select, battle_finish

def read_deck(path):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    return [int(l) for l in open(p) if l.strip()][:60]

# tunable subset of W (the knobs that shape develop-vs-attack ordering). ko_win/end left fixed.
TUNE_KEYS = ["ability", "evolve", "play_pokemon", "attach", "draw", "boss", "stadium", "attack", "retreat"]

def play_one(a_seat, deck, agentA, agentB, max_steps=20000):
    obs, _ = battle_start(deck, deck)
    if obs is None: return None
    steps = 0
    try:
        while True:
            cur = obs["current"]
            if cur is not None and cur.get("result", -1) != -1:
                w = cur["result"]
                if w == 2: return "draw"
                return "A" if (w == a_seat) else "B"
            if obs["select"] is None: return None
            seat = cur.get("yourIndex") if cur else None
            pilot = agentA if seat == a_seat else agentB
            obs = battle_select(pilot(obs))
            steps += 1
            if steps > max_steps: return None
    finally:
        battle_finish()

def fitness(weights, deck, A, B, games, seed):
    A.set_weights(dict(zip(TUNE_KEYS, weights)))
    random.seed(seed)
    aw = dec = err = 0
    for g in range(games):
        r = play_one(g % 2, deck, A.agent, B.agent)
        if r == "A": aw += 1; dec += 1
        elif r == "B": dec += 1
        elif r is None: err += 1
    return (100.0 * aw / dec if dec else 0.0), err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.35, help="log-space mutation scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "tuned_weights.json"))
    a = ap.parse_args()
    random.seed(a.seed)

    os.environ["AGENT_DECK"] = a.deck if os.path.isabs(a.deck) else os.path.join(ROOT, a.deck)
    A = importlib.import_module("agent_grimmsnarl")
    B = importlib.import_module("agent_lucario")
    deck = read_deck(a.deck)

    base = [A.W[k] for k in TUNE_KEYS]
    print(f"[tune] deck={os.path.basename(a.deck)} | keys={TUNE_KEYS}")
    print(f"[tune] baseline weights: {dict(zip(TUNE_KEYS, base))}")
    base_wr, base_err = fitness(base, deck, A, B, a.games, a.seed)
    print(f"[tune] baseline (as-shipped) win-rate vs generic: {base_wr:.1f}%  (err {base_err})\n")

    # (mu+lambda) ES in log-space (weights are positive magnitudes; retreat can be small)
    def mutate(vec):
        return [max(1.0, v * math.exp(random.gauss(0, a.sigma))) for v in vec]

    best = (base_wr, base[:])
    parent = base[:]
    for gen in range(a.gens):
        cands = [mutate(parent) for _ in range(a.pop)]
        scored = []
        for i, c in enumerate(cands):
            wr, err = fitness(c, deck, A, B, a.games, a.seed + gen*100 + i)
            scored.append((wr, c, err))
        scored.sort(key=lambda x: x[0], reverse=True)
        gbest = scored[0]
        if gbest[0] > best[0]:
            best = (gbest[0], gbest[1][:])
            parent = gbest[1][:]
        print(f"[tune] gen {gen+1:2}/{a.gens}: best-this-gen {gbest[0]:.1f}% (err {gbest[2]}) | overall best {best[0]:.1f}%")

    out = {"tune_keys": TUNE_KEYS, "baseline_winrate": round(base_wr,1),
           "best_winrate": round(best[0],1), "best_weights": dict(zip(TUNE_KEYS, best[1])),
           "improvement_pts": round(best[0]-base_wr,1), "games_per_eval": a.games}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n[tune] baseline {base_wr:.1f}% -> tuned {best[0]:.1f}%  ({best[0]-base_wr:+.1f} pts)")
    print(f"[tune] best weights: {out['best_weights']}")
    print(f"[tune] wrote {a.out}")
    if best[0] - base_wr < 2:
        print("[tune] VERDICT: tuning did not meaningfully help -> ship as-is")
    else:
        print("[tune] VERDICT: tuned weights improve win-rate -> update W in agent_grimmsnarl.py")

if __name__ == "__main__":
    main()
