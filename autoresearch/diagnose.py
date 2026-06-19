"""Diagnostic self-play: how do games end, and where does the agent spend choices?

Run in the container:  PYTHONPATH=/app/sdk python diagnose.py 300
"""
import sys
from collections import Counter

from cg.api import to_observation_class, OptionType
from cg.game import battle_start, battle_select, battle_finish
from agent import agent, read_deck

REASON = {1: "all-prizes-taken", 2: "deck-out", 3: "no-active-pokemon", 4: "card-effect"}
OPT_NAME = {int(v): v.name for v in OptionType}


def play(deck):
    obs, sd = battle_start(deck, deck)
    if obs is None:
        return None
    reason = None
    turns = 0
    multi_attack_decisions = 0
    chosen_types = Counter()
    try:
        while True:
            for lg in obs.get("logs", []):
                if lg.get("type") == 23:                  # RESULT log
                    reason = lg.get("reason")
            cur = obs["current"]
            if cur is not None and cur.get("result", -1) != -1:
                turns = cur.get("turn", turns)
                return {"winner": cur["result"], "reason": reason, "turns": turns,
                        "multi_attack": multi_attack_decisions, "types": chosen_types}
            sel = obs["select"]
            if sel is None:
                obs = battle_select(read_deck()); continue
            n_attacks = sum(1 for o in sel["option"] if o.get("type") == int(OptionType.ATTACK))
            if n_attacks > 1:
                multi_attack_decisions += 1
            action = agent(obs)
            for i in action:
                chosen_types[sel["option"][i]["type"]] += 1
            obs = battle_select(action)
    finally:
        battle_finish()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    deck = read_deck()
    reasons = Counter()
    turn_hist = []
    multi = 0
    types = Counter()
    done = 0
    for _ in range(n):
        r = play(deck)
        if not r:
            continue
        done += 1
        reasons[REASON.get(r["reason"], f"?{r['reason']}")] += 1
        turn_hist.append(r["turns"])
        multi += r["multi_attack"]
        types += r["types"]
    print(f"games: {done}")
    print(f"avg turns: {sum(turn_hist)/len(turn_hist):.1f}  (min {min(turn_hist)} max {max(turn_hist)})")
    print(f"\nWIN/LOSS REASON distribution:")
    for k, v in reasons.most_common():
        print(f"  {v/done*100:5.1f}%  {k}")
    print(f"\nmulti-attack decision points (>1 attack offered): {multi} total, "
          f"{multi/done:.2f}/game  <- exp002 relevance")
    print(f"\nchosen option-type mix (per decision):")
    tot = sum(types.values())
    for t, c in types.most_common():
        print(f"  {c/tot*100:5.1f}%  {OPT_NAME.get(t, t)}")


if __name__ == "__main__":
    main()
