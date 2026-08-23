"""Pilot-vs-pilot on the SAME deck — measures whether a deck-aware pilot beats the generic one.

Both seats play the SAME deck; the only variable is the PILOT (agentA vs agentB). Seat of the
test pilot alternates each game to cancel first-player advantage.

    PYTHONPATH=sdk python pilot_gate.py --deck autoresearch/decks/grimmsnarl.csv \
        --agentA agent_grimmsnarl --agentB agent_lucario --games 100

agentA is the "test" pilot (win-rate reported is A's). Writes pilot_gate_result.json.
"""
import os, sys, json, time, random, argparse, importlib
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "sdk"))
sys.path.insert(0, os.path.join(ROOT, "autoresearch"))

from cg.game import battle_start, battle_select, battle_finish

def read_deck(path):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    return [int(l) for l in open(p) if l.strip()][:60]

def play_one(a_seat, deck, agentA, agentB, max_steps=20000):
    """agentA pilots seat a_seat, agentB the other seat. Both play `deck`.
    Returns 'A'/'B'/'draw'/None."""
    obs, sd = battle_start(deck, deck)
    if obs is None:
        return None
    steps = 0
    try:
        while True:
            cur = obs["current"]
            if cur is not None and cur.get("result", -1) != -1:
                w = cur["result"]
                if w == 2:
                    return "draw"
                return "A" if (w == a_seat) else "B"
            sel = obs["select"]
            if sel is None:
                return None
            # whose turn? the seat being asked = obs current yourIndex
            seat = obs["current"].get("yourIndex") if obs.get("current") else None
            pilot = agentA if seat == a_seat else agentB
            action = pilot(obs)
            obs = battle_select(action)
            steps += 1
            if steps > max_steps:
                return None
    finally:
        battle_finish()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True)
    ap.add_argument("--agentA", default="agent_grimmsnarl")
    ap.add_argument("--agentB", default="agent_lucario")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "pilot_gate_result.json"))
    a = ap.parse_args()
    random.seed(a.seed)

    # force both agent modules to load the SAME deck (override AGENT_DECK)
    os.environ["AGENT_DECK"] = a.deck if os.path.isabs(a.deck) else os.path.join(ROOT, a.deck)
    # compat: pilots that hardcode "deck.csv" (e.g. agent_lb950) read cwd/deck.csv — write it too
    import shutil as _sh
    _sh.copy(os.environ["AGENT_DECK"], os.path.join(ROOT, "deck.csv"))
    A = importlib.import_module(a.agentA)
    B = importlib.import_module(a.agentB)
    deck = read_deck(a.deck)
    assert len(deck) == 60, f"deck must be 60: {len(deck)}"
    print(f"[pilot-gate] {a.agentA} (A) vs {a.agentB} (B) on {os.path.basename(a.deck)}")
    print(f"[pilot-gate] {a.games} games | same deck both seats | only PILOT differs | seat alternates\n")

    aw = bw = draw = err = 0
    seatwins = {0: [0, 0], 1: [0, 0]}
    per = []
    for g in range(a.games):
        a_seat = g % 2
        r = play_one(a_seat, deck, A.agent, B.agent)
        seatwins[a_seat][1] += 1
        if r == "A": aw += 1; seatwins[a_seat][0] += 1
        elif r == "B": bw += 1
        elif r == "draw": draw += 1
        else: err += 1
        per.append({"game": g, "a_seat": a_seat, "winner": r})
        if (g + 1) % 10 == 0:
            dec = aw + bw
            wr = 100 * aw / dec if dec else 0
            print(f"   {g+1:4}/{a.games}  {a.agentA} {aw}  {a.agentB} {bw}  draw {draw}  err {err}  | A WR {wr:.1f}%")

    dec = aw + bw
    wrA = 100 * aw / dec if dec else 0
    print(f"\n[pilot-gate] {a.agentA} win-rate vs {a.agentB}: {wrA:.1f}%  ({aw}/{dec} decisive)")
    print(f"[pilot-gate] {a.agentA} as seat0: {seatwins[0]}  as seat1: {seatwins[1]}")
    if wrA > 55: print("[pilot-gate] CLEAR WIN: deck-aware pilot beats generic -> ship it")
    elif wrA >= 50: print("[pilot-gate] MARGINAL: roughly even")
    else: print("[pilot-gate] LOSS: deck-aware pilot does not beat generic -> do not ship")
    out = {"agentA": a.agentA, "agentB": a.agentB, "deck": a.deck, "games": a.games,
           "A_wins": aw, "B_wins": bw, "draws": draw, "errors": err,
           "A_winrate_pct": round(wrA, 1), "seatwins": seatwins, "per_game": per}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[pilot-gate] wrote {a.out}")

if __name__ == "__main__":
    main()
