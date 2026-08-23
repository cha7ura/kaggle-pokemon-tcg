"""Engine verify for ANY submission bundle (rule-based or cardpol): run the agent through real
games via the bundled libcg.so. Confirms deck=60, deck-phase return, and never crashes/forfeits.
    SUB=submission_megastarmie python verify_submission.py
"""
import os, sys

SUB = os.environ.get("SUB", "submission_megastarmie")
SUBABS = os.path.abspath(SUB)
sys.path.insert(0, SUBABS)                 # cg (+ cardpol if present)
os.chdir(SUBABS)                           # agents read deck.csv via cwd

deck = [int(x) for x in open("deck.csv") if x.strip().lstrip("-").isdigit()][:60]
print(f"deck cards: {len(deck)}")
assert len(deck) == 60, "deck must be 60"

import main                                # loads the agent (rule-based or cardpol)
d = main.agent({})
print(f"deck-phase agent(deck) -> {len(d)} ints")
assert len(d) == 60, "deck phase must return 60"

from cg.game import battle_start, battle_select, battle_finish


def play_one(a0, a1, max_steps=10000):
    obs, _ = battle_start(deck, deck)
    if obs is None:
        return ("START_ERR", 0)
    steps = 0
    try:
        while True:
            cur = obs["current"]
            if cur is not None and cur.get("result", -1) != -1:
                return (cur["result"], steps)
            sel = obs["select"]
            if sel is None:
                return ("NO_SELECT", steps)
            player = cur["yourIndex"]
            try:
                action = (a0 if player == 0 else a1)(obs)
            except Exception as e:
                return (("AGENT_CRASH", player, repr(e)[:100]), steps)
            obs = battle_select(action)
            steps += 1
            if steps > max_steps:
                return (2, steps)
    finally:
        battle_finish()


crashes = 0
for g in range(4):
    r, steps = play_one(main.agent, main.agent)
    crashed = isinstance(r, tuple) and r and r[0] == "AGENT_CRASH"
    crashes += 1 if crashed else 0
    print(f"game {g}: result={r} steps={steps}")

print("VERIFY:", "PASS — no agent crash, games completed" if crashes == 0
      else f"FAIL — {crashes} agent crashes")
