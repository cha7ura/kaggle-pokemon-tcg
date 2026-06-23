"""Extract the REAL ladder field from replay JSONs: every distinct 60-card deck + how often it
appears (frequency = games it shows up in). This is the opponent distribution for the better local
oracle (replaces the 4-deck gauntlet). Writes each distinct deck to decks/field/f<NN>.csv and a
manifest decks/field/weights.json = [{slug, count, archetype, sample_team}].

  python tools/extract_field.py            # all json/
  python tools/extract_field.py 20         # keep only decks seen >= 20x (trim the long tail)
"""
import json, csv, glob, os, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELD = f"{ROOT}/autoresearch/decks/field"
names = {int(r['cardId']): r['name'] for r in csv.DictReader(open(f"{ROOT}/autoresearch/cards_full.csv"))}
# archetype tag by signature card id present in the deck
ARCH = {743: "Alakazam", 678: "MegaLucario", 1031: "MegaStarmie", 345: "Crustle",
        # Trevenant / Bellibolt signatures (best-effort; OTHER if none match)
        }
# add Trevenant + Bellibolt if their ids resolve by name
for cid, nm in names.items():
    if nm == "Hop's Trevenant": ARCH.setdefault(cid, "Trevenant")
    if nm == "Dragapult ex": ARCH.setdefault(cid, "Dragapult")
    if "Bellibolt" in nm: ARCH.setdefault(cid, "Bellibolt")
    if "Cinderace" in nm: ARCH.setdefault(cid, "MegaStarmie")  # keidroid combo marker


def deck_of(steps, p):
    for s in steps:
        a = s[p].get("action")
        if isinstance(a, list) and len(a) == 60 and all(isinstance(x, int) for x in a):
            return a
    return None


def arch_of(deck):
    s = set(deck)
    for cid, nm in ARCH.items():
        if cid in s: return nm
    return "OTHER"


def main():
    min_count = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    os.makedirs(FIELD, exist_ok=True)
    seen = collections.Counter()        # canonical-deck-tuple -> count
    sample_team = {}
    for f in glob.glob(f"{ROOT}/json/*.json"):
        try: d = json.load(open(f))
        except: continue
        teams = d.get("info", {}).get("TeamNames", ["?", "?"])
        for p in (0, 1):
            dk = deck_of(d.get("steps", []), p)
            if not dk: continue
            key = tuple(sorted(dk))         # canonical: order-independent
            seen[key] += 1
            sample_team.setdefault(key, teams[p][:24])
    kept = [(k, c) for k, c in seen.most_common() if c >= min_count]
    manifest = []
    for i, (key, c) in enumerate(kept):
        deck = list(key)
        slug = f"f{i:02d}"
        open(f"{FIELD}/{slug}.csv", "w").write("\n".join(map(str, deck)) + "\n")
        manifest.append({"slug": slug, "count": c, "archetype": arch_of(deck),
                         "team": sample_team[key]})
    json.dump(manifest, open(f"{FIELD}/weights.json", "w"), indent=0)
    tot = sum(c for _, c in kept)
    print(f"{len(seen)} distinct decks; kept {len(kept)} (>= {min_count}x) = {tot} deck-appearances")
    byarch = collections.Counter()
    for m in manifest: byarch[m["archetype"]] += m["count"]
    print("field by archetype (weighted):", dict(byarch.most_common()))
    print("top decks:")
    for m in manifest[:12]:
        print(f"  {m['slug']} x{m['count']:4d} {m['archetype']:12} {m['team']}")


if __name__ == "__main__":
    main()
