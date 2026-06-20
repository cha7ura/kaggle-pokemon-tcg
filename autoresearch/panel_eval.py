"""Diverse-opponent panel gauntlet — a ladder proxy that the mirror metric isn't.

The mirror gate (eval.py) is misleading: it ranked v4>v2, but the real ladder ranked v2>v4.
The ladder is a DIVERSE field, so we score a candidate against a panel of varied opponents
(each on its own deck) and aggregate. Validate against known ladder scores before trusting it
(see validate_panel.py / spec). Reuses eval.py; never edits it.

  python panel_eval.py --agent agent_lucario.py --deck decks/lucario_meta.csv --games 300
"""
import argparse
import json

from eval import load_callable, read_deck, gauntlet, wilson_lb
from baselines import random_agent, greedy_agent

# name, make-agent, opponent-deck, include-in-aggregate
PANEL = [
    ("random",    lambda: random_agent,                       "decks/lucario_meta.csv", False),
    ("greedy",    lambda: greedy_agent,                        "decks/lucario_meta.csv", True),
    ("dragapult", lambda: load_callable("dragapult_agent.py"), "decks/dragapult.csv",    True),
    ("v4_mirror", lambda: load_callable("champion_lucario.py"),"decks/lucario_meta.csv", True),
    ("crustle",   lambda: load_callable("crustle_agent.py"),   "decks/crustle.csv",      False),  # rare stall outlier
]


def panel_eval(agent_path, deck_path, games):
    cand = load_callable(agent_path)
    deck_c = read_deck(deck_path)
    rows = {}
    agg = []
    for name, mk, odeck, meaningful in PANEL:
        opp = mk()
        deck_o = read_deck(odeck)
        w, d, l = gauntlet(cand, opp, deck_c, deck_o, games)
        n = w + d + l
        wr = (w + 0.5 * d) / n if n else 0.0
        rows[name] = {"wins": w, "draws": d, "losses": l,
                      "wr": round(wr, 4), "lb": round(wilson_lb(w + 0.5 * d, n), 4),
                      "in_aggregate": meaningful}
        if meaningful:
            agg.append(wr)
    panel_score = sum(agg) / len(agg) if agg else 0.0
    return {"agent": agent_path, "deck": deck_path, "games_each": games,
            "panel_score": round(panel_score, 4),
            "random_floor": rows["random"]["wr"], "crustle": rows["crustle"]["wr"],
            "opponents": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="agent_lucario.py")
    ap.add_argument("--deck", default="decks/lucario_meta.csv")
    ap.add_argument("--games", type=int, default=300, help="games per opponent (seat-swapped)")
    a = ap.parse_args()
    print(json.dumps(panel_eval(a.agent, a.deck, a.games), indent=2))


if __name__ == "__main__":
    main()
