"""Extract deck lists + archetypes from Kaggle episode replay JSONs (json/*.json).
Replays downloaded from leaderboard episode URLs. Reveals what top teams actually play.
  python tools/replay_decks.py
"""
import json, csv, collections, glob, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
names = {int(r['cardId']): r['name'] for r in csv.DictReader(open(f"{ROOT}/autoresearch/cards_full.csv"))}
ARCH = {743: "Alakazam", 678: "MegaLucario", 345: "Crustle"}

def deck_of(steps, p):
    for s in steps:
        a = s[p].get("action")
        if isinstance(a, list) and len(a) == 60 and all(isinstance(x, int) for x in a):
            return a
    return None

def main():
    arch = collections.Counter()
    for f in sorted(glob.glob(f"{ROOT}/json/*.json")):
        d = json.load(open(f)); teams = d["info"].get("TeamNames", ["?", "?"])
        for p in (0, 1):
            dk = deck_of(d["steps"], p)
            if not dk: continue
            a = next((nm for cid, nm in ARCH.items() if cid in set(dk)), "OTHER")
            arch[a] += 1
            print(f"{teams[p][:22]:22} {a}")
    print("\narchetype freq:", dict(arch.most_common()))

if __name__ == "__main__":
    main()
