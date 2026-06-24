"""In-docker round-robin: every deck in the roster plays every other N games (seat-swapped),
emitting one JSONL row per game to stdout. The engine is unseeded, so the N games of a pairing
vary by shuffle + coin flip on their own. Runs inside linux/amd64 (cwd autoresearch, PYTHONPATH=sdk).
STDLIB + cg only. Reuses eval.play_one (no edit to the fixed harness).

  python league_play.py roster.json --games 30      # roster = [{slug,deck,policy,pilot}]
"""
import json, sys, argparse
from eval import play_one  # the frozen harness; we only reuse its single-game runner

sys.path.insert(0, "/app")  # tools.imitation.* for the imitation pilot path
from tools.imitation.features import state_features, option_features
from tools.imitation.policy import pick


def make_pilot(deck, tree):
    """Return an agent(obs)->list[int]. tree=None -> generic typh fallback pilot."""
    if tree is None:
        from agent_typh import agent as typh
        return typh

    def agent(obs):
        try:
            sel = obs.get("select")
            if sel is None:
                return deck
            opts = sel.get("option") or []
            if not opts:
                return []
            cur = obs.get("current") or {}
            seat = cur.get("yourIndex", 0)
            state = state_features(cur, seat)
            ovecs = [option_features(o, sel.get("context"), sel) for o in opts]
            return pick(tree, ovecs, state, sel.get("minCount", 1), sel.get("maxCount", 1))
        except Exception:
            sel = (obs or {}).get("select") or {}
            n = len(sel.get("option") or [])
            return list(range(min(max(0, sel.get("minCount", 0)), n)))
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roster")
    ap.add_argument("--games", type=int, default=30)
    args = ap.parse_args()
    roster = json.load(open(args.roster))
    for r in roster:
        r["_agent"] = make_pilot(r["deck"], r.get("policy"))

    for i in range(len(roster)):
        for k in range(i + 1, len(roster)):
            A, B = roster[i], roster[k]
            for g in range(args.games):
                if g % 2 == 0:  # A is seat 0
                    res = play_one(A["deck"], B["deck"], A["_agent"], B["_agent"])
                    winner = 2 if res == 2 else (0 if res == 0 else 1) if res in (0, 1) else None
                else:           # B is seat 0 -> remap seat winner to deck index
                    res = play_one(B["deck"], A["deck"], B["_agent"], A["_agent"])
                    winner = 2 if res == 2 else (1 if res == 0 else 0) if res in (0, 1) else None
                if winner is None:
                    continue
                print(json.dumps({"deck_a": A["slug"], "deck_b": B["slug"],
                                  "pilot_a": A.get("pilot"), "pilot_b": B.get("pilot"),
                                  "winner": winner}), flush=True)


if __name__ == "__main__":
    main()
