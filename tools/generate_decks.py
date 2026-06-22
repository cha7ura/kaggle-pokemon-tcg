"""Generative deck search: for each top UNUSED non-ex wincon, auto-build a coherent deck
(evolution line + Rare Candy + draw engine + reusable trainers + type-matched energy), then
gauntlet vs the meta field on the engine (typhlosion pilot both sides -> isolates DECK). Rank by
field win-rate. Local = filter; ladder = judge. Run on host (calls docker per matchup).
"""
import csv, json, glob, os, subprocess, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(f"{ROOT}/autoresearch/cards_full.csv")))
R = {int(r["cardId"]): r for r in rows}
def F(r, k, d=0.0):
    try: return float(r[k]) if r[k] not in ("", "None") else d
    except: return d
def B(r, k): return r.get(k) == "True"

# evolution line via cards_full name match (basic -> stage1 -> stage2 share the line's mons by name family)
EN = list(csv.DictReader(open(f"{ROOT}/data/EN_Card_Data.csv")))
prev = {}  # name -> previous-stage name
for e in EN:
    nm = e.get("Card Name"); pv = e.get("Previous stage")
    if nm and pv and pv != "n/a": prev[nm] = pv
name_ids = {}
for r in rows:
    name_ids.setdefault(r["name"], []).append((int(r["cardId"]), r))

# energy basic id by mon type
TYPE_ENERGY = {"FIGHTING": 6, "METAL": 8, "FIRE": 4, "WATER": 3, "GRASS": 2, "LIGHTNING": 7,
               "PSYCHIC": 5, "DARKNESS": 9, "DRAGON": 6, "COLORLESS": 6}

# reusable engine (draw + search + gust + recovery) — archetype-agnostic
ENGINE = [(1086, 4), (1121, 4), (1182, 3), (1097, 2), (1152, 2), (1225, 3), (1231, 3), (206, 4)]

META = ["alakazam_top", "trevenant", "crustle", "lucario_meta"]


def meta_cards(thresh=30):
    played = Counter()
    for fn in glob.glob(f"{ROOT}/json/*.json"):
        try: d = json.load(open(fn))
        except: continue
        seen = set()
        for s in d["steps"]:
            for p in (0, 1):
                a = s[p].get("action")
                if isinstance(a, list) and len(a) == 60 and p not in seen and all(isinstance(x, int) for x in a):
                    for c in set(a): played[c] += 1
                    seen.add(p)
            if len(seen) == 2: break
    return {c for c, n in played.items() if n >= thresh}


def line_of(stage2_name):
    """return [basic_id, mid_id, stage2_id] following Previous stage chain; None if incomplete."""
    mid = prev.get(stage2_name); basic = prev.get(mid) if mid else None
    if not (mid and basic): return None
    def pick(nm):
        c = name_ids.get(nm)
        return c[0][0] if c else None
    s2 = pick(stage2_name); m = pick(mid); b = pick(basic)
    return [b, m, s2] if all(x is not None for x in (b, m, s2)) else None


def build_deck(stage2_id):
    r = R[stage2_id]; line = line_of(r["name"])
    if not line: return None
    b, m, s2 = line
    deck = []
    deck += [b] * 4 + [m] * 2 + [s2] * 3 + [1079] * 4   # line + rare candy
    for cid, n in ENGINE: deck += [cid] * n
    en = TYPE_ENERGY.get(r["type"], 6)
    while len(deck) < 60: deck.append(en)
    return deck[:60]


def gauntlet(deck_path, games=40):
    scores = []
    for opp in META:
        out = subprocess.run(
            ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{ROOT}:/app",
             "-w", "/app/autoresearch", "-e", "PYTHONPATH=/app/sdk", "python:3.11-slim",
             "python", "eval.py", "--challenger", "agent_typh.py", "--champion", "agent_typh.py",
             "--deck", deck_path, "--deck-champion", f"decks/{opp}.csv", "--games", str(games)],
            capture_output=True, text=True, timeout=600).stdout
        sc = None
        for ln in out.splitlines():
            if '"score"' in ln:
                try: sc = float(ln.split(":")[1].strip().rstrip(",")); break
                except: pass
        scores.append(sc if sc is not None else 0.0)
    return scores


def main():
    meta = meta_cards()
    cands = []
    for r in rows:
        if r["cardType"] != "POKEMON" or not B(r, "stage2"): continue
        cid = int(r["cardId"]); dmg = F(r, "best_dmg"); cost = F(r, "min_cost_for_best", 9)
        if dmg < 150 or cid in meta: continue
        prize = 3 if B(r, "megaEx") else 2 if B(r, "ex") else 1
        if prize > 1: continue
        score = dmg / 30 + F(r, "dmg_per_energy") / 15 + F(r, "hp") / 120 - cost * 0.4
        cands.append((round(score, 2), cid, r["name"]))
    cands.sort(reverse=True)
    top = cands[: int(sys.argv[1]) if len(sys.argv) > 1 else 8]
    print(f"sweeping {len(top)} unused non-ex wincons vs meta field {META}\n", flush=True)
    results = []
    for sc, cid, nm in top:
        deck = build_deck(cid)
        if not deck:
            print(f"  {nm:22} SKIP (incomplete line)", flush=True); continue
        slug = nm.lower().replace(" ", "_").replace("'", "").replace("’", "")
        path = f"decks/gen_{slug}.csv"
        open(f"{ROOT}/autoresearch/{path}", "w").write("\n".join(map(str, deck)) + "\n")
        scores = gauntlet(path)
        avg = sum(scores) / len(scores)
        results.append((avg, nm, scores))
        print(f"  {nm:22} field={avg:.3f}  {dict(zip(META, [round(s,2) for s in scores]))}", flush=True)
    results.sort(reverse=True)
    print("\n=== RANKED by field win-rate ===", flush=True)
    for avg, nm, scores in results:
        print(f"  {avg:.3f}  {nm}", flush=True)


if __name__ == "__main__":
    main()
