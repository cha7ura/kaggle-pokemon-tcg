"""Decode the sample deck and scout buildable attackers from the card pool."""
import csv
from collections import defaultdict

CSV = "data/EN_Card_Data.csv"
SAMPLE = [1158, 721, 722, 723, 1145, 1205, 1227, 1235, 3]

ENERGY = {"{C}": "C", "{R}": "R", "{W}": "W", "{G}": "G", "{L}": "L",
          "{P}": "P", "{F}": "F", "{D}": "D", "{M}": "M", "{N}": "N"}


def load():
    with open(CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    cards = {}
    for r in rows:
        cid = int(r["Card ID"])
        r = {k: (v or "").strip() for k, v in r.items()}
        if cid not in cards:
            cards[cid] = {**r, "moves": []}
        if r.get("Move Name"):
            cards[cid]["moves"].append((r["Move Name"], r.get("Cost", ""), r.get("Damage", "")))
    return cards


def costlen(cost):
    return cost.count("●") + sum(cost.count(s) for s in ENERGY)


def main():
    cards = load()
    print("=== SAMPLE DECK ===")
    for cid in SAMPLE:
        c = cards.get(cid)
        if not c:
            print(f"  {cid}: (not in CSV)"); continue
        stage = c["Stage (Pokémon)/Type (Energy and Trainer)"]
        mv = "; ".join(f"{n}[{co}]={d}" for n, co, d in c["moves"][:2])
        print(f"  {cid:>4} {c['Card Name'][:28]:<28} {stage:<16} HP{c['HP']:<4} {c['Type']:<5} {mv}")

    print("\n=== BUILDABLE BASIC ATTACKERS (Basic, HP>=120, a <=2-energy attack >=90 dmg) ===")
    by_type = defaultdict(list)
    for cid, c in cards.items():
        stage = c["Stage (Pokémon)/Type (Energy and Trainer)"]
        if "Basic Pok" not in stage:
            continue
        try:
            hp = int(c["HP"])
        except ValueError:
            continue
        if hp < 120:
            continue
        for n, co, d in c["moves"]:
            dd = d.replace("+", "").replace("×", "").replace("x", "")
            try:
                dmg = int(dd)
            except ValueError:
                continue
            if costlen(co) <= 2 and dmg >= 90 and "×" not in d:
                by_type[c["Type"]].append((dmg, costlen(co), hp, c["Card Name"], n, co, c["Rule"]))
    for t, lst in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        print(f"\n  --- type {t} ({len(lst)}) ---")
        for dmg, cl, hp, name, mv, co, rule in sorted(lst, reverse=True)[:6]:
            print(f"    {dmg:>3} for {co:<6} HP{hp:<4} {name[:26]:<26} {rule}")


if __name__ == "__main__":
    main()
