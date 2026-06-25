"""In-docker A/B: CardPolicy vs the typh baseline pilot on the SAME deck. If CardPolicy wins >0.5,
the learned policy out-pilots the baseline we currently ship. Run inside linux/amd64.

  python /app/tools/imitation/validate_policy.py f01 f02 f00 --games 20
"""
import sys, argparse
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/autoresearch")
from eval import play_one
from agent_typh import agent as typh
from tools.imitation.gameplay_policy import CardPolicy


def deck_of(slug):
    return [int(x) for x in open(f"/app/autoresearch/decks/field/{slug}.csv") if x.strip().isdigit()][:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="+")
    ap.add_argument("--games", type=int, default=20)
    a = ap.parse_args()
    print(f"CardPolicy vs typh, {a.games} games/deck (seat-swapped)", flush=True)
    for slug in a.slugs:
        deck = deck_of(slug)
        pol = CardPolicy(deck)
        agent = lambda obs: pol.act(obs)
        w = n = 0
        for g in range(a.games):
            if g % 2 == 0:
                r = play_one(deck, deck, agent, typh); win = 1.0 if r == 0 else 0.0
            else:
                r = play_one(deck, deck, typh, agent); win = 1.0 if r == 1 else 0.0
            if r == 2:
                w += 0.5; n += 1
            elif r in (0, 1):
                w += win; n += 1
        wr = w / n if n else 0.0
        print(f"  {slug}: CardPolicy {wr:.3f} vs typh  ({w:.1f}/{n})", flush=True)


if __name__ == "__main__":
    main()
