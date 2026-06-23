"""Card text lookup — join cards_full.csv (stats) with EN_Card_Data.csv (moves/effects/abilities).
Stops us flying blind on mechanics. Data is LOCAL (no web/OCR needed for covered cards).

  python tools/card_text.py 879            # by cardId
  python tools/card_text.py "Hop's Trevenant"
  python tools/card_text.py --deck alakazam_top   # full text for every card in a deck
  python tools/card_text.py --missing decks/dragapult.csv  # list deck cards lacking effect text
"""
import csv, os, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = {int(r["cardId"]): r for r in csv.DictReader(open(f"{ROOT}/autoresearch/cards_full.csv"))}
NAME2ID = {}
for cid, r in FULL.items(): NAME2ID.setdefault(r["name"], cid)
EN = list(csv.DictReader(open(f"{ROOT}/data/EN_Card_Data.csv")))
EN_BY_NAME = collections.defaultdict(list)
for r in EN: EN_BY_NAME[r.get("Card Name")].append(r)
# OCR fallback (full pool, from the PDF) — used when EN_Card_Data lacks effect text
OCR = {}
_ocr_path = f"{ROOT}/data/card_ocr.csv"
if os.path.exists(_ocr_path):
    OCR = {int(r["cardId"]): r["ocr_text"] for r in csv.DictReader(open(_ocr_path))}


def text_of(cid):
    r = FULL.get(cid)
    if not r: return f"{cid}: unknown"
    out = [f"{cid} {r['name']} [{r['cardType']}] HP{r['hp'] or '-'} {r['type'] or ''} "
           f"weak={r['weakness'] or '-'}"]
    rows = EN_BY_NAME.get(r["name"], [])
    if not rows or not any(e.get("Effect Explanation") not in ("", "n/a", None) for e in rows):
        if OCR.get(cid):
            out.append("   [OCR fallback]:")
            out.append("   " + OCR[cid].replace("\n", " ").strip()[:600])
            return "\n".join(out)
        out.append("   (no EN effect text)")
    for e in rows:
        mv = e.get("Move Name", ""); ef = e.get("Effect Explanation", ""); rule = e.get("Rule", "")
        cat = e.get("Category", "")
        if mv and mv not in ("", "n/a"):
            out.append(f"   [{cat}] {mv} cost={e.get('Cost','')} dmg={e.get('Damage','')}")
        if ef and ef not in ("", "n/a"): out.append(f"      → {ef}")
        if rule and rule not in ("", "n/a"): out.append(f"      rule: {rule}")
    return "\n".join(out)


def has_effect(name):
    return any(e.get("Effect Explanation") not in ("", "n/a", None) for e in EN_BY_NAME.get(name, []))


def main():
    a = sys.argv[1:]
    if a and a[0] == "--deck":
        deck = [int(x) for x in open(f"{ROOT}/autoresearch/decks/{a[1]}.csv") if x.strip()]
        for cid in sorted(set(deck)): print(text_of(cid))
    elif a and a[0] == "--missing":
        deck = [int(x) for x in open(f"{ROOT}/autoresearch/decks/{a[1]}.csv") if x.strip()]
        miss = sorted({c for c in deck if not has_effect(FULL.get(c, {}).get("name", "")) and FULL.get(c, {}).get("cardType") != "ENERGY"})
        for c in miss: print(f"MISSING: {text_of(c)}")
        print(f"\n{len(miss)} distinct non-energy cards lack effect text")
    else:
        key = " ".join(a)
        cid = int(key) if key.isdigit() else NAME2ID.get(key)
        print(text_of(cid) if cid else f"not found: {key}")


if __name__ == "__main__":
    main()
