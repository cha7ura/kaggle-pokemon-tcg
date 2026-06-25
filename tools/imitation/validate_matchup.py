"""In-docker league recalibration: CardPolicy vs CardPolicy across archetype matchups. Compare the
anchor deck's win rate vs the field to our REAL ladder matchup table — does a stronger policy make
the sim oracle track reality (esp. the inverted Dragapult cell 0.627 sim vs 0.32 real)?

  python /app/tools/imitation/validate_matchup.py f00 f02 f01 f04 f10 --games 24
  (anchor = first slug; rest = opponents)
"""
import sys, argparse
sys.path.insert(0, "/app"); sys.path.insert(0, "/app/autoresearch")
from eval import play_one
from tools.imitation.gameplay_policy import CardPolicy


def deck_of(slug):
    return [int(x) for x in open(f"/app/autoresearch/decks/field/{slug}.csv") if x.strip().isdigit()][:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="+")
    ap.add_argument("--games", type=int, default=24)
    a = ap.parse_args()
    anchor = a.slugs[0]
    da = deck_of(anchor); pa = CardPolicy(da)
    print(f"anchor {anchor} (CardPolicy) vs field, {a.games} games each", flush=True)
    for opp in a.slugs[1:]:
        db = deck_of(opp); pb = CardPolicy(db)
        ag = lambda obs: pa.act(obs); bg = lambda obs: pb.act(obs)
        w = n = 0
        for g in range(a.games):
            if g % 2 == 0:
                r = play_one(da, db, ag, bg); win = 1.0 if r == 0 else 0.0
            else:
                r = play_one(db, da, bg, ag); win = 1.0 if r == 1 else 0.0
            if r == 2:
                w += 0.5; n += 1
            elif r in (0, 1):
                w += win; n += 1
        print(f"  {anchor} vs {opp}: {w/n:.3f}  ({w:.1f}/{n})", flush=True)


if __name__ == "__main__":
    main()
