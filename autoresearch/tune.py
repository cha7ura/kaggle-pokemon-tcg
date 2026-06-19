"""Autoresearch weight-tuner for the Lucario heuristic.

Hill-climbs agent_lucario.W via mirror gauntlets. Both seats run the SAME
agent function; we just swap the global W per move, so a candidate W plays the
champion W on the identical meta deck. A candidate is promoted only if it beats
the champion with a Wilson lower bound > 0.5 over GAMES seat-swapped games.

Deterministic-ish: we vary perturbations by iteration index (no RNG seeding API).
Run:  PYTHONPATH=/app/sdk python tune.py 16 300
"""
import json
import math
import sys
import random

import agent_lucario as base
from cg.game import battle_start, battle_select, battle_finish

DECK = base.read_deck("decks/lucario_meta.csv")
TUNE_KEYS = ["attack", "draw", "boss", "switch", "energy_need", "attach",
             "play_pokemon", "evolve", "ability", "retreat"]
MULTS = [0.3, 0.5, 0.7, 1.5, 2.0, 3.0]


def wilson_lb(wins, n, z=1.96):
    if n == 0:
        return 0.0
    p = wins / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    marg = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - marg)


def play(Wa, Wb, a_seat):
    obs, sd = battle_start(DECK, DECK)
    if obs is None:
        return None
    try:
        for _ in range(10000):
            cur = obs["current"]
            if cur is not None and cur.get("result", -1) != -1:
                return cur["result"]
            sel = obs["select"]
            if sel is None:
                obs = battle_select(base._DECK); continue
            base.W = Wa if cur["yourIndex"] == a_seat else Wb
            try:
                act = base.agent(obs)
            except Exception:
                act = [0]
            obs = battle_select(act)
        return 2
    finally:
        battle_finish()


def gauntlet(cand, champ, games):
    w = d = l = 0
    for i in range(games):
        a_seat = i % 2
        r = play(cand, champ, a_seat)
        if r is None:
            continue
        if r == 2:
            d += 1
        elif r == a_seat:
            w += 1
        else:
            l += 1
    return w, d, l


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    champ = dict(base.W)
    accepted = 0
    for it in range(iters):
        rng = random.Random(it * 7919 + 17)        # reproducible per-iteration
        cand = dict(champ)
        for k in rng.sample(TUNE_KEYS, 2):
            cand[k] = cand[k] * rng.choice(MULTS)
        w, d, l = gauntlet(cand, champ, games)
        n = w + d + l
        lb = wilson_lb(w + 0.5 * d, n)
        tag = "ACCEPT" if lb > 0.5 else "reject"
        changed = {k: round(cand[k], 1) for k in TUNE_KEYS if cand[k] != champ[k]}
        print(f"[{it:02d}] {tag} score={(w+0.5*d)/n:.3f} lb={lb:.3f} "
              f"({w}-{d}-{l})  delta={changed}", flush=True)
        if lb > 0.5:
            champ = cand
            accepted += 1
    print("\nACCEPTED", accepted, "improvements")
    print("BEST_W=" + json.dumps(champ))
    with open("tuned_W.json", "w") as f:
        json.dump(champ, f)


if __name__ == "__main__":
    main()
