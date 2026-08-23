"""Step 3 validation gate: prior-ISMCTS vs heuristic on the real cg engine.

Controlled experiment — BOTH seats play the SAME deck, so the only variable is the pilot
(search vs heuristic). This isolates "does better play matter", the question step 3 answers.
Seat assignment alternates each game to cancel first-player advantage.

Writes step3_result.json (which the assistant reads to pick up after the run).

Requires linux/amd64 + libcg.so (Docker locally, or Colab). Run:
    PYTHONPATH=sdk python run_gate.py --games 100 --sims 48 --depth 12
"""
import os, sys, json, time, random, argparse
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "sdk"))
sys.path.insert(0, os.path.join(ROOT, "autoresearch"))

from cg.game import battle_start, battle_select, battle_finish
import agent_lucario as H
import agent_ismcts_prior as S


def read_deck(path):
    return [int(l) for l in open(path) if l.strip()][:60]


def play_one(search_seat, deck0, deck1, sims, depth, max_steps=20000):
    """Return winner seat (0/1), 2 for draw, or None on engine error / step blowout."""
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
            if sel is None:
                return None
            seat = cur["yourIndex"]
            if seat == search_seat:
                action = S._ismcts(S.to_observation_class(obs), sims=sims, depth=depth) \
                    if _is_main(sel) else H.select_indices(S.to_observation_class(obs))
                action = S.agent(obs) if action is None else action
            else:
                action = H.agent(obs)
            obs = battle_select(action)
            steps += 1
            if steps > max_steps:
                return None
    finally:
        battle_finish()


def _is_main(sel):
    try:
        from cg.api import SelectType
        return int(sel.get("type")) == int(SelectType.MAIN) and sel.get("maxCount") == 1 and sel.get("minCount") == 1
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--deck", default=os.path.join(ROOT, "autoresearch", "decks", "lucario_meta.csv"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "step3_result.json"))
    a = ap.parse_args()
    random.seed(a.seed)

    deck = read_deck(a.deck)
    assert len(deck) == 60, f"deck must be 60 cards, got {len(deck)}"
    os.environ.setdefault("AGENT_DECK", a.deck)

    print(f"[gate] {a.games} games | sims={a.sims} depth={a.depth} | deck={os.path.basename(a.deck)}")
    print(f"[gate] both seats play the same deck; only the pilot differs (ISMCTS vs heuristic)\n")

    t0 = time.time()
    search_wins = heur_wins = draws = errors = 0
    per_game = []
    for g in range(a.games):
        search_seat = g % 2                     # alternate to cancel first-move edge
        r = play_one(search_seat, deck, deck, a.sims, a.depth)
        if r is None:
            errors += 1; outcome = "error"
        elif r == 2:
            draws += 1; outcome = "draw"
        elif r == search_seat:
            search_wins += 1; outcome = "search"
        else:
            heur_wins += 1; outcome = "heuristic"
        per_game.append({"game": g, "search_seat": search_seat, "winner": outcome})
        if (g + 1) % 10 == 0:
            dec = search_wins + heur_wins
            wr = 100 * search_wins / dec if dec else 0
            print(f"  {g+1:>4}/{a.games}  search {search_wins}  heur {heur_wins}  draw {draws}  err {errors}  | search WR {wr:.1f}%")

    dt = time.time() - t0
    decisive = search_wins + heur_wins
    wr = 100 * search_wins / decisive if decisive else None
    result = {
        "games": a.games, "sims": a.sims, "depth": a.depth,
        "deck": os.path.basename(a.deck), "seconds": round(dt, 1),
        "search_wins": search_wins, "heuristic_wins": heur_wins,
        "draws": draws, "errors": errors,
        "search_winrate_pct": round(wr, 1) if wr is not None else None,
        "gate_passed": (wr is not None and wr > 50.0),
        "verdict": ("PASS: determinization fix makes search beat heuristic -> R-NaD self-play worth starting"
                    if (wr is not None and wr > 50.0) else
                    "FAIL: search still not > heuristic -> deck selection remains the lever; skip self-play"),
        "per_game": per_game,
        "diag": dict(S.DIAG),
    }
    json.dump(result, open(a.out, "w"), indent=1)
    print(f"\n[gate] search win-rate vs heuristic: {wr:.1f}%  ({search_wins}/{decisive} decisive)")
    d = S.DIAG
    tot_roll = d["rollout_ok"] + d["rollout_err"]
    err_pct = 100 * d["rollout_err"] / tot_roll if tot_roll else 0
    print(f"[gate] DIAG: ismcts engaged {d['engage']}x, deferred {d['defer']}x | "
          f"rollouts {d['rollout_ok']} ok / {d['rollout_err']} err ({err_pct:.0f}% err)")
    print(f"[gate] DIAG2: truncated {d.get('rollout_trunc',0)}x, first-step-fail {d.get('first_step_err',0)}x, "
          f"empty-select {d.get('empty_select',0)}x")
    if d.get("fail_detail"):
        print(f"[gate] FAIL DETAIL (first rejected step): {d['fail_detail']}")
    if d["rollout_err"] and d["last_err"]:
        print(f"[gate] FIRST ROLLOUT ERROR:\n{d['last_err']}")
    print(f"[gate] {result['verdict']}")
    print(f"[gate] wrote {a.out}")


if __name__ == "__main__":
    main()
