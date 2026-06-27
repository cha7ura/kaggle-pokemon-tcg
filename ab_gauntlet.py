"""A/B policy gauntlet: SAME deck on both sides, two policies, seat-swapped — isolates policy
skill (not deck luck). Reports A's win-rate with a Wilson lower bound. Run in linux/amd64:
    GAMES=120 python ab_gauntlet.py
A = deck-matched policy (submission_trev_meta), B = proven policy (submission_cardpolicy_trev).
"""
import os, sys, math

ROOT = "/app"
sys.path.insert(0, os.path.join(ROOT, "submission_trev_meta"))   # cardpol + cg
os.chdir(os.path.join(ROOT, "submission_trev_meta"))

from cardpol.gameplay_policy import CardPolicy
from cg.game import battle_start, battle_select, battle_finish

DECK = [int(x) for x in open("deck.csv") if x.strip().lstrip("-").isdigit()][:60]
assert len(DECK) == 60

MODEL_A = os.path.join(ROOT, "submission_trev_meta/cardpol/data/card_policy.npz")        # deck-matched 0.490
MODEL_B = os.path.join(ROOT, "submission_cardpolicy_trev/cardpol/data/card_policy.npz")   # scaffold proven
polA = CardPolicy(DECK, MODEL_A)
polB = CardPolicy(DECK, MODEL_B)
print(f"A=deck-matched  B=proven  | deck={len(DECK)} cards", flush=True)


def play_one(a0, a1, max_steps=10000):
    obs, _ = battle_start(DECK, DECK)
    if obs is None:
        return None
    steps = 0
    try:
        while True:
            cur = obs["current"]
            if cur is not None and cur.get("result", -1) != -1:
                return cur["result"]
            sel = obs["select"]
            if sel is None:
                return None
            player = cur["yourIndex"]
            try:
                action = (a0 if player == 0 else a1)(obs)
            except Exception:
                return 1 - player          # crash = forfeit
            obs = battle_select(action)
            steps += 1
            if steps > max_steps:
                return 2
    finally:
        battle_finish()


def wilson_lb(w, n, z=1.96):
    if n == 0:
        return 0.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - m)


GAMES = int(os.environ.get("GAMES", "120"))
wA = d = wB = 0
for i in range(GAMES):
    if i % 2 == 0:                          # A is player 0
        r = play_one(polA.act, polB.act); a_seat = 0
    else:                                   # A is player 1
        r = play_one(polB.act, polA.act); a_seat = 1
    if r is None:
        continue
    if r == 2:
        d += 1
    elif r == a_seat:
        wA += 1
    else:
        wB += 1
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{GAMES}: A {wA} - {wB} B  ({d} draws)", flush=True)

n = wA + wB + d
score = wA + 0.5 * d
print(f"\nRESULT over {n} games: A(deck-matched) {wA} wins, B(proven) {wB} wins, {d} draws")
print(f"A win-rate = {100*score/n:.1f}%  (Wilson LB {100*wilson_lb(score, n):.1f}%)")
print("SHIP:", "deck-matched (A)" if score / n >= 0.5 else "proven (B)")
