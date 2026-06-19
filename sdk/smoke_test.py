"""End-to-end smoke test: run random self-play games via the cg engine directly.

Proves the native libcg.so loads and a full episode completes inside the container.
Also times episodes so we know the autoresearch experiment budget (games/sec).
"""
import random
import time
import sys

from cg.game import battle_start, battle_select, battle_finish


def read_deck(path="deck.csv"):
    with open(path) as f:
        return [int(line) for line in f if line.strip()][:60]


def random_agent(obs: dict) -> list[int]:
    sel = obs["select"]
    n = len(sel["option"])
    k = sel["maxCount"]
    return random.sample(range(n), k)


def play_one(deck0, deck1, max_steps=10000):
    obs, sd = battle_start(deck0, deck1)
    if obs is None:
        return {"error": (sd.errorPlayer, sd.errorType)}
    steps = 0
    try:
        while True:
            cur = obs["current"]
            if cur is not None and cur.get("result", -1) != -1:
                return {"winner": cur["result"], "steps": steps, "turns": cur.get("turn")}
            sel = obs["select"]
            if sel is None:                       # deck-selection phase (shouldn't hit: decks passed)
                obs = battle_select(read_deck())
                continue
            player = cur["yourIndex"]
            action = random_agent(obs)            # both seats random for the smoke test
            obs = battle_select(action)
            steps += 1
            if steps > max_steps:
                return {"error": "max_steps", "steps": steps}
    finally:
        battle_finish()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    deck = read_deck()
    print(f"deck size: {len(deck)}  | running {n} random self-play games\n")
    results = []
    t0 = time.time()
    for i in range(n):
        r = play_one(deck, deck)
        results.append(r)
        print(f"  game {i+1:>3}: {r}")
    dt = time.time() - t0
    wins = [r.get("winner") for r in results if "winner" in r]
    print(f"\n{len(wins)}/{n} completed | {dt:.2f}s total | {n/dt:.1f} games/s | "
          f"avg {sum(r.get('steps',0) for r in results)/max(1,n):.0f} steps/game")
    from collections import Counter
    print("winner dist:", Counter(wins))


if __name__ == "__main__":
    main()
