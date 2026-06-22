"""Role-based full-pool deck builder + wide sweep. For each candidate stage2 wincon across the
WHOLE 1267-card pool, build a coherent deck: evolution line (atomic) + Rare Candy + energy
auto-matched to the wincon's attack cost + a standard role-shell (draw/search/gust/recovery from
role categories). Gauntlet vs meta field. Surfaces which of the 116 wincons beat the field.

  python tools/build_pool_deck.py <topN> <games>
"""
import csv, os, sys, subprocess, re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECKS = f"{ROOT}/autoresearch/decks"
rows = list(csv.DictReader(open(f"{ROOT}/autoresearch/cards_full.csv")))
R = {int(r["cardId"]): r for r in rows}
name_id = {}
for r in rows:
    name_id.setdefault(r["name"], int(r["cardId"]))
def F(r, k, d=0.0):
    try: return float(r[k]) if r[k] not in ("", "None") else d
    except: return d
def B(r, k): return r.get(k) == "True"

# EN card data: previous-stage chain + attack costs
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
    if dv >= 100 and cost:                       # remember the cost of a real attack
        costs.setdefault(nm, cost)

# energy symbol -> basic energy card id
SYM = {"F": 6, "M": 8, "R": 4, "W": 3, "G": 2, "L": 7, "P": 5, "D": 9, "C": 6}
def energy_for(name):
    c = costs.get(name, "")
    syms = re.findall(r"\{([FMRWGLPDC])\}", c) or re.findall(r"●", c) and ["C"] or ["C"]
    ids = [SYM.get(s, 6) for s in syms if s != "C"]
    return list(dict.fromkeys(ids)) or [6]

# role shell (standard staples that exist in engine) — draw/search, gust, recovery, candy
SHELL = [(1086, 4), (1121, 4), (1182, 3), (1097, 2), (1152, 2), (1225, 3), (1231, 3),
         (206, 4), (1079, 4)]   # Poffin, UltraBall, Boss, NightStretcher, PokePad, Hilda, Dawn, Dunsparce, RareCandy


def line_of(s2name):
    mid = prev.get(s2name); basic = prev.get(mid) if mid else None
    if not (mid and basic): return None
    ids = [name_id.get(basic), name_id.get(mid), name_id.get(s2name)]
    return ids if all(ids) else None


def build(s2id):
    r = R[s2id]; line = line_of(r["name"])
    if not line: return None
    b, m, s2 = line
    deck = [b]*4 + [m]*2 + [s2]*3
    for cid, n in SHELL: deck += [cid]*n
    ens = energy_for(r["name"]);
    i = 0
    while len(deck) < 60:
        deck.append(ens[i % len(ens)]); i += 1
    # legal: cap non-energy at 4
    cnt = Counter(deck); fixed = []
    for c, n in cnt.items():
        fixed += [c]*(n if c in (2,3,4,5,6,7,8,9) else min(n,4))
    while len(fixed) < 60: fixed.append(ens[0])
    return fixed[:60]


META = ["alakazam_top", "trevenant", "crustle", "lucario_meta"]
FIELD_W = {"alakazam_top": 0.40, "lucario_meta": 0.33, "trevenant": 0.25, "crustle": 0.02}
FRAGILITY = 0.35
from concurrent.futures import ThreadPoolExecutor


def _one(slug, opp, games):
    s = subprocess.run(
        ["docker","run","--rm","--platform","linux/amd64","-v",f"{ROOT}:/app",
         "-w","/app/autoresearch","-e","PYTHONPATH=/app/sdk","python:3.11-slim",
         "python","eval.py","--challenger","agent_typh.py","--champion","agent_typh.py",
         "--deck",f"decks/{slug}.csv","--deck-champion",f"decks/{opp}.csv","--games",str(games)],
        capture_output=True,text=True,timeout=900).stdout
    for ln in s.splitlines():
        if '"score"' in ln:
            try: return float(ln.split(":")[1].strip().rstrip(","))
            except: pass
    return 0.0


def gauntlet(slug, games, pool):
    futs = {opp: pool.submit(_one, slug, opp, games) for opp in META}
    return [futs[opp].result() for opp in META]


def fitness(scores):
    wsum = sum(FIELD_W.values())
    fw = sum(FIELD_W[o]*s for o,s in zip(META,scores))/wsum
    return fw - FRAGILITY*max(0.0, 0.5-min(scores))


def main():
    topN = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    allow_ex = "--ex" in sys.argv
    # candidate wincons: stage2 dmg>=150, buildable line; non-ex by default, +ex if --ex (broaden space)
    cands = []
    for r in rows:
        if r["cardType"] != "POKEMON" or not B(r, "stage2"): continue
        if F(r, "best_dmg") < 150: continue
        if (B(r, "ex") or B(r, "megaEx")) and not allow_ex: continue
        if not line_of(r["name"]): continue
        prize = 3 if B(r,"megaEx") else 2 if B(r,"ex") else 1
        score = F(r,"best_dmg")/30 + F(r,"dmg_per_energy")/15 + F(r,"hp")/120 - F(r,"min_cost_for_best",9)*0.4 - (prize-1)*1.2
        cands.append((round(score,2), int(r["cardId"]), r["name"]))
    cands.sort(reverse=True)
    cands = cands[:topN]
    print(f"LARGE role-pool sweep: {len(cands)} wincons x {META} (field-weighted+fragility), "
          f"{workers} parallel workers, {games} games\n", flush=True)
    pool = ThreadPoolExecutor(max_workers=workers)
    res = []
    for sc, cid, nm in cands:
        deck = build(cid)
        if not deck: continue
        slug = "pool_" + re.sub(r"[^a-z0-9]","",nm.lower())
        open(f"{DECKS}/{slug}.csv","w").write("\n".join(map(str,deck))+"\n")
        scores = gauntlet(slug, games, pool); fit = fitness(scores)
        res.append((fit, nm, scores))
        print(f"  {nm:22} fit={fit:.3f}  {dict(zip(META,[round(s,2) for s in scores]))}", flush=True)
    res.sort(reverse=True)
    print("\n=== RANKED by robust fitness ===", flush=True)
    for fit, nm, sc in res[:15]:
        print(f"  {fit:.3f}  {nm}  {dict(zip(META,[round(s,2) for s in sc]))}", flush=True)


if __name__ == "__main__":
    main()
