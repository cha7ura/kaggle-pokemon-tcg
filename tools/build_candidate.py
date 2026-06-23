"""Build a coherent deck for any wincon Pokemon (by cardId), following its evolution chain however
many stages, + Rare Candy + standard trainer shell + type-matched energy, capped legal. For testing
external-meta candidates (Dragapult, Zoroark, Slowking, Hydrapple...) through the real-field oracle.

  python tools/build_candidate.py <cardId> <slug>
"""
import csv, os, sys, re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECKS = f"{ROOT}/autoresearch/decks"
rows = list(csv.DictReader(open(f"{ROOT}/autoresearch/cards_full.csv")))
R = {int(r["cardId"]): r for r in rows}
name_id = {}
for r in rows:
    name_id.setdefault(r["name"], int(r["cardId"]))

EN = list(csv.DictReader(open(f"{ROOT}/data/EN_Card_Data.csv")))
prev, costs = {}, {}
for e in EN:
    nm = e.get("Card Name")
    if not nm: continue
    pv = e.get("Previous stage")
    if pv and pv != "n/a": prev[nm] = pv
    dmg = e.get("Damage", ""); cost = e.get("Cost", "")
    try: dv = abs(int(re.sub(r"[^0-9-]", "", dmg))) if dmg not in ("", "n/a") else 0
    except: dv = 0
    if dv >= 80 and cost: costs.setdefault(nm, cost)

SYM = {"F": 6, "M": 8, "R": 4, "W": 3, "G": 2, "L": 7, "P": 5, "D": 9, "C": 6}
TYPE_EN = {"FIGHTING": 6, "METAL": 8, "FIRE": 4, "WATER": 3, "GRASS": 2, "LIGHTNING": 7,
           "PSYCHIC": 5, "DARKNESS": 9, "DRAGON": 6, "COLORLESS": 6}
# Poffin, UltraBall, Boss, NightStretcher, PokePad, Hilda, Dawn, Dunsparce, RareCandy
SHELL = [(1086, 4), (1121, 4), (1182, 3), (1097, 2), (1152, 2), (1225, 3), (1231, 3),
         (206, 4), (1079, 4)]


def chain(name):
    """basic..wincon ids following prev; longest available."""
    seq = [name]
    while prev.get(seq[0]): seq.insert(0, prev[seq[0]])
    ids = [name_id.get(n) for n in seq]
    return [i for i in ids if i]  # drop any unresolved stage


def energy_for(name, typ):
    c = costs.get(name, "")
    ids = [SYM[s] for s in re.findall(r"\{([FMRWGLPDC])\}", c) if s in SYM and s != "C"]
    ids = list(dict.fromkeys(ids))
    return ids or [TYPE_EN.get(typ, 6)]


def build(cid):
    r = R[cid]; line = chain(r["name"])
    if not line: return None
    # counts: more basics, fewer top stages
    if len(line) >= 3:   deck = [line[0]]*4 + [line[1]]*2 + [line[2]]*3 + [line[-1]]*1*(len(line)>3)
    elif len(line) == 2: deck = [line[0]]*4 + [line[1]]*3
    else:                deck = [line[0]]*4
    deck = [c for c in deck if c]
    if len(line) >= 3: deck += [1079]*4   # rare candy only for stage2 lines
    for c, n in SHELL:
        if c == 1079 and len(line) < 3: continue
        deck += [c]*n
    ens = energy_for(r["name"], r.get("type", ""))
    i = 0
    while len(deck) < 60: deck.append(ens[i % len(ens)]); i += 1
    cnt = Counter(deck); fixed = []
    for c, n in cnt.items():
        fixed += [c]*(n if c in (2,3,4,5,6,7,8,9) else min(n, 4))
    while len(fixed) < 60: fixed.append(ens[0])
    return fixed[:60]


def main():
    cid = int(sys.argv[1]); slug = sys.argv[2]
    deck = build(cid)
    if not deck: print(f"FAIL build {R[cid]['name']}"); return
    open(f"{DECKS}/{slug}.csv", "w").write("\n".join(map(str, deck)) + "\n")
    print(f"{R[cid]['name']} -> decks/{slug}.csv  line={chain(R[cid]['name'])}  "
          f"energy={energy_for(R[cid]['name'], R[cid].get('type',''))}")


if __name__ == "__main__":
    main()
