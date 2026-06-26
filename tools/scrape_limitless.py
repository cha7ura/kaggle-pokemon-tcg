"""Harvest real tournament decklists from play.limitlesstcg.com and map them to our engine card_ids.
This is the proven ladder lever (copy real top decks). Server-rendered HTML -> urllib + regex, cheap.

  python tools/scrape_limitless.py <tournament_id> [topN]    # default top 20 players

Writes autoresearch/decks/limitless/<tid>_<rank>_<slug>.csv (60 card_ids) + a manifest. Decks that
don't fully map (60 cards, all ids known) are skipped + reported.
"""
import sys, os, re, json, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "autoresearch", "decks", "limitless")
UA = {"User-Agent": "Mozilla/5.0 (research; polite)"}
ENERGY_ALIAS = {"Grass": "Basic {G} Energy", "Fire": "Basic {R} Energy", "Water": "Basic {W} Energy",
                "Lightning": "Basic {L} Energy", "Psychic": "Basic {P} Energy",
                "Fighting": "Basic {F} Energy", "Darkness": "Basic {D} Energy",
                "Metal": "Basic {M} Energy", "Dragon": "Basic {N} Energy", "Fairy": "Basic {Y} Energy"}
# set/num from the href (always present); count+name from the link text (Pokemon append "(SET-NUM)")
CARD_RE = re.compile(r'<a href="https://limitlesstcg\.com/cards/([A-Za-z]+)/(\w+)"[^>]*>(\d+) (.+?)</a>')
_SUFFIX = re.compile(r"\s*\([A-Za-z]+-\w+\)\s*$")


def _get(u):
    return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=25).read().decode("utf-8", "ignore")


def _byname():
    cards = json.load(open(os.path.join(ROOT, "autoresearch", "data", "cards.json")))
    d = {}
    for r in cards:
        d.setdefault(r["name"].replace("’", "'"), r["card_id"])
    return d


def resolve(name, byname):
    name = name.replace("&#039;", "'").replace("&amp;", "&").replace("’", "'").strip()
    if name in byname:
        return byname[name]
    if name.endswith("Energy"):                      # basic energy: "Fire Energy" -> "Basic {R} Energy"
        t = name.replace("Energy", "").strip()
        eng = ENERGY_ALIAS.get(t)
        if eng and eng in byname:
            return byname[eng]
    return None


def players(tid):
    h = _get(f"https://play.limitlesstcg.com/tournament/{tid}/standings")
    return list(dict.fromkeys(re.findall(r"/tournament/" + tid + r"/player/([^/\"]+)/decklist", h)))


def deck_of(tid, slug, byname):
    h = _get(f"https://play.limitlesstcg.com/tournament/{tid}/player/{slug}/decklist")
    deck, missing = [], []
    for st, num, cnt, nametext in CARD_RE.findall(h):
        name = _SUFFIX.sub("", nametext)
        cid = resolve(name, byname)
        if cid is None:
            missing.append(name)
        else:
            deck += [cid] * int(cnt)
    return deck, missing


def main():
    tid = sys.argv[1] if len(sys.argv) > 1 else "6a0c908513f957d6d4b4b729"
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    os.makedirs(OUT, exist_ok=True)
    byname = _byname()
    pls = players(tid)[:topn]
    print(f"{len(pls)} players (top {topn}) from tournament {tid}", flush=True)
    manifest = []
    for rank, slug in enumerate(pls, 1):
        try:
            deck, missing = deck_of(tid, slug, byname)
        except Exception as e:
            print(f"  #{rank} {slug}: ERR {e}"); continue
        ok = len(deck) == 60 and not missing
        status = "OK" if ok else f"SKIP ({len(deck)} cards, miss={missing[:3]})"
        if ok:
            fn = f"{tid[:8]}_{rank:02d}_{slug}.csv"
            open(os.path.join(OUT, fn), "w").write("\n".join(map(str, deck)) + "\n")
            manifest.append({"rank": rank, "slug": slug, "file": fn})
        print(f"  #{rank} {slug}: {status}", flush=True)
    json.dump(manifest, open(os.path.join(OUT, f"{tid[:8]}_manifest.json"), "w"), indent=2)
    print(f"saved {len(manifest)} full decks -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
