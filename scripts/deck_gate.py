"""Deck-vs-deck evaluation on the real cg engine.

Same heuristic pilot (agent_lucario) on BOTH seats — the only variable is the DECK. Measures
whether deck A beats deck B head-to-head, controlling for pilot skill. Seat alternates each game
to cancel first-player advantage.

    PYTHONPATH=sdk python deck_gate.py --deckA decks/cinderace.csv --deckB submission_megastarmie/deck.csv --games 100

Writes deck_gate_result.json.
"""
import os, sys, json, time, random, argparse
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "sdk"))
sys.path.insert(0, os.path.join(ROOT, "autoresearch"))

from cg.game import battle_start, battle_select, battle_finish
import agent_lucario as H


def read_deck(path):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    return [int(l) for l in open(p) if l.strip()][:60]


def play_one(a_seat, deckA, deckB, max_steps=20000):
    """deckA plays seat a_seat, deckB the other. Return winning DECK label 'A'/'B', 'draw', or None."""
    d0, d1 = (deckA, deckB) if a_seat == 0 else (deckB, deckA)
    obs, sd = battle_start(d0, d1)
    if obs is None:
        return None
    steps = 0
    try:
        while True:
            cur = obs["current"]
            if cur is not None and cur.get("result", -1) != -1:
                w = cur["result"]  # winning seat 0/1, or 2 draw
                if w == 2:
                    return "draw"
                winning_deck_is_A = (w == a_seat)
                return "A" if winning_deck_is_A else "B"
            sel = obs["select"]
            if sel is None:
                return None
            action = H.agent(obs)
            obs = battle_select(action)
            steps += 1
            if steps > max_steps:
                return None
    finally:
        battle_finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deckA", required=True)
    ap.add_argument("--deckB", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labelA", default="A")
    ap.add_argument("--labelB", default="B")
    ap.add_argument("--out", default=os.path.join(ROOT, "deck_gate_result.json"))
    a = ap.parse_args()
    random.seed(a.seed)

    deckA, deckB = read_deck(a.deckA), read_deck(a.deckB)
    assert len(deckA) == 60 and len(deckB) == 60, f"decks must be 60: A={len(deckA)} B={len(deckB)}"
    print(f"[deck-gate] {a.labelA} ({os.path.basename(a.deckA)}) vs {a.labelB} ({os.path.basename(a.deckB)})")
    print(f"[deck-gate] {a.games} games | same heuristic pilot both seats | seat alternates\n")

    aw = bw = draw = err = 0
    per = []
    seatwins = {0: [0, 0], 1: [0, 0]}  # a_seat -> [A_wins, games]
    for g in range(a.games):
        a_seat = g % 2
        r = play_one(a_seat, deckA, deckB)
        seatwins[a_seat][1] += 1
        if r == "A":
            aw += 1; seatwins[a_seat][0] += 1
        elif r == "B":
            bw += 1
        elif r == "draw":
            draw += 1
        else:
            err += 1
        per.append({"game": g, "a_seat": a_seat, "winner": r})
        if (g + 1) % 10 == 0:
            dec = aw + bw
            wr = 100 * aw / dec if dec else 0
            print(f"   {g+1:4}/{a.games}  {a.labelA} {aw}  {a.labelB} {bw}  draw {draw}  err {err}  | {a.labelA} WR {wr:.1f}%")

    dec = aw + bw
    wrA = 100 * aw / dec if dec else 0
    print(f"\n[deck-gate] {a.labelA} win-rate vs {a.labelB}: {wrA:.1f}%  ({aw}/{dec} decisive)")
    print(f"[deck-gate] {a.labelA} as seat0: {seatwins[0]}  as seat1: {seatwins[1]}")
    out = {"labelA": a.labelA, "labelB": a.labelB, "deckA": a.deckA, "deckB": a.deckB,
           "games": a.games, "A_wins": aw, "B_wins": bw, "draws": draw, "errors": err,
           "A_winrate_pct": round(wrA, 1), "seatwins": seatwins, "per_game": per}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[deck-gate] wrote {a.out}")


if __name__ == "__main__":
    main()
