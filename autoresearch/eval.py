"""FIXED harness — the autoresearch loop must NOT edit this file.

Mirrors Karpathy autoresearch's `prepare.py`: it defines the single comparable
metric and never changes, so every experiment is measured the same way.

Metric: challenger win-rate over a SEAT-SWAPPED gauntlet vs a champion agent,
reported with a Wilson lower confidence bound. The engine can't be seeded, so
soundness comes from (a) large N, (b) playing every matchup from both seats to
cancel the first-player advantage, (c) the same deck on both sides so we measure
*policy* skill, not deck luck.

Run inside the linux/amd64 container (see run.sh):
    PYTHONPATH=/app/sdk python eval.py --games 400 --champion champion_agent.py
"""
import argparse
import importlib.util
import json
import math
import os
import time
from collections import Counter

from cg.game import battle_start, battle_select, battle_finish

DEFAULT_DECK = "decks/champion.csv"


def load_callable(path: str, name: str = "agent"):
    spec = importlib.util.spec_from_file_location(f"mod_{os.path.basename(path)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def read_deck(path: str) -> list[int]:
    with open(path) as f:
        deck = [int(line) for line in f if line.strip()][:60]
    if len(deck) != 60:
        raise ValueError(f"{path}: deck has {len(deck)} cards, need 60")
    return deck


def play_one(deck0, deck1, agent0, agent1, max_steps=10000):
    """Returns winner index (0/1), 2 for draw, or None if the start/round errored."""
    obs, sd = battle_start(deck0, deck1)
    if obs is None:
        return None
    steps = 0
    try:
        while True:
            cur = obs["current"]
            if cur is not None and cur.get("result", -1) != -1:
                return cur["result"]
            sel = obs["select"]
            if sel is None:                      # deck phase — not reached (decks passed up front)
                return None
            player = cur["yourIndex"]
            try:
                action = (agent0 if player == 0 else agent1)(obs)
            except Exception:
                # a crashing agent forfeits this game (mirrors Kaggle's error handling)
                return 1 - player
            obs = battle_select(action)
            steps += 1
            if steps > max_steps:
                return 2                          # treat a stuck game as a draw
    finally:
        battle_finish()


def gauntlet(challenger, champion, deck_c, deck_o, games):
    """Seat-swapped match. Returns (wins, draws, losses) from challenger's view."""
    w = d = l = 0
    for i in range(games):
        if i % 2 == 0:                            # challenger is player 0
            r = play_one(deck_c, deck_o, challenger, champion)
            ch_seat = 0
        else:                                     # challenger is player 1
            r = play_one(deck_o, deck_c, champion, challenger)
            ch_seat = 1
        if r is None:
            continue
        if r == 2:
            d += 1
        elif r == ch_seat:
            w += 1
        else:
            l += 1
    return w, d, l


def wilson_lb(wins: float, n: int, z: float = 1.96) -> float:
    """Lower bound of a Wilson score interval. Draws should be counted as 0.5 wins."""
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenger", default="agent.py", help="agent file the loop edits")
    ap.add_argument("--champion", default=None,
                    help="frozen agent file to beat; omit to face the random baseline")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--deck", default=DEFAULT_DECK, help="challenger deck")
    ap.add_argument("--deck-champion", default=None,
                    help="champion deck (defaults to --deck). Differ to A/B test DECKS.")
    ap.add_argument("--promote-lb", type=float, default=0.5,
                    help="Wilson LB above this => PROMOTE. This is the keep/discard rule.")
    args = ap.parse_args()

    challenger = load_callable(args.challenger)
    if args.champion:
        champion = load_callable(args.champion)
        champ_name = args.champion
    else:
        from baselines import random_agent
        champion = random_agent
        champ_name = "random_baseline"

    deck_c = read_deck(args.deck)
    deck_o = read_deck(args.deck_champion) if args.deck_champion else deck_c

    t0 = time.time()
    w, d, l = gauntlet(challenger, champion, deck_c, deck_o, args.games)
    dt = time.time() - t0

    n = w + d + l
    score = (w + 0.5 * d) / n if n else 0.0
    lb = wilson_lb(w + 0.5 * d, n)
    verdict = "PROMOTE" if lb > args.promote_lb else "REJECT"

    out = {
        "challenger": args.challenger, "champion": champ_name,
        "deck": args.deck, "deck_champion": args.deck_champion or args.deck,
        "games": n, "wins": w, "draws": d, "losses": l,
        "score": round(score, 4), "wilson_lb": round(lb, 4),
        "promote_lb": args.promote_lb, "verdict": verdict,
        "secs": round(dt, 1), "games_per_s": round(n / dt, 1) if dt else None,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
