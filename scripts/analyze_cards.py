"""Card-pool analysis for the PTCG AI Battle Challenge.

The CSV has ONE ROW PER MOVE, so a Pokémon with two attacks spans two rows.
We reconstruct per-card records (keyed by Card ID), attaching the list of moves,
then profile the pool for deck-construction signal.
"""
import csv
from collections import Counter, defaultdict

CSV = "data/EN_Card_Data.csv"

ENERGY = {  # PTCG energy symbol -> readable
    "{C}": "Colorless", "{R}": "Fire", "{W}": "Water", "{G}": "Grass",
    "{L}": "Lightning", "{P}": "Psychic", "{F}": "Fighting", "{D}": "Darkness",
    "{M}": "Metal", "{N}": "Dragon", "{Y}": "Fairy",
}


def load():
    with open(CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    cards = {}
    for r in rows:
        cid = r["Card ID"].strip()
        r = {k: (v or "").strip() for k, v in r.items()}
        if cid not in cards:
            cards[cid] = {**r, "moves": []}
        mv = r.get("Move Name", "")
        if mv:
            cards[cid]["moves"].append({
                "name": mv, "cost": r.get("Cost", ""),
                "damage": r.get("Damage", ""), "effect": r.get("Effect Explanation", ""),
            })
    return list(cards.values())


def stage_class(c):
    s = c["Stage (Pokémon)/Type (Energy and Trainer)"]
    return s if s else "n/a"


def is_pokemon(c):
    return "Pokémon" in stage_class(c) or c["HP"] not in ("", "n/a")


def num(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


def main():
    cards = load()
    print(f"UNIQUE CARDS: {len(cards)}\n")

    # --- broad split ---
    stages = Counter(stage_class(c) for c in cards)
    print("=== Stage / Type split ===")
    for k, v in stages.most_common():
        print(f"  {v:>4}  {k}")

    pokemon = [c for c in cards if is_pokemon(c)]
    trainers = [c for c in cards if "Trainer" in stage_class(c) and not is_pokemon(c)]
    energies = [c for c in cards if "Energy" in stage_class(c)]
    print(f"\nPokémon: {len(pokemon)}   Trainer-ish: {len(trainers)}   Energy: {len(energies)}")

    # --- Pokémon type distribution ---
    print("\n=== Pokémon type (primary energy) ===")
    types = Counter(ENERGY.get(c["Type"], c["Type"] or "?") for c in pokemon)
    for k, v in types.most_common():
        print(f"  {v:>4}  {k}")

    # --- HP curve ---
    hps = sorted(h for c in pokemon if (h := num(c["HP"])) is not None)
    if hps:
        import statistics as st
        print(f"\n=== HP curve (n={len(hps)}) ===")
        print(f"  min {hps[0]}  p25 {hps[len(hps)//4]}  median {st.median(hps)}"
              f"  p75 {hps[3*len(hps)//4]}  max {hps[-1]}")
        print(f"  high-HP walls (>=300): {sum(1 for h in hps if h >= 300)}")

    # --- ex / special rule cards (key deck centerpieces) ---
    print("\n=== Rule / mechanic tags ===")
    rules = Counter(c["Rule"] for c in cards if c["Rule"] and c["Rule"] != "n/a")
    for k, v in rules.most_common(15):
        print(f"  {v:>4}  {k}")

    # --- coin-flip dependence (variance signal for a search agent) ---
    flip = [c for c in pokemon if any("flip" in m["effect"].lower() for m in c["moves"])]
    print(f"\n=== Variance: Pokémon with coin-flip attacks: {len(flip)}/{len(pokemon)} ===")

    # --- highest raw-damage attackers (flat damage only) ---
    print("\n=== Top flat-damage attacks ===")
    hits = []
    for c in pokemon:
        for m in c["moves"]:
            d = num(m["damage"].replace("+", "").replace("×", "").replace("x", ""))
            if d is not None and "×" not in m["damage"]:
                cost = sum(m["cost"].count(s) for s in "●") or len(m["cost"])
                hits.append((d, cost, c["Card Name"], m["name"], m["cost"]))
    for d, cost, name, mv, costs in sorted(hits, reverse=True)[:12]:
        eff = f"{d/cost:.0f}/E" if cost else "?"
        print(f"  {d:>4}  cost={costs or '-':<6} {eff:>6}  {name} — {mv}")

    # --- HP-efficient basics (deck enablers) ---
    print("\n=== Beefy Basic Pokémon (potential ex centerpieces, HP>=200) ===")
    basics = [c for c in pokemon if "Basic" in stage_class(c) and (num(c["HP"]) or 0) >= 200]
    for c in sorted(basics, key=lambda c: -(num(c["HP"]) or 0))[:12]:
        print(f"  HP{c['HP']:>4}  {ENERGY.get(c['Type'], c['Type']):<10} {c['Card Name']}  ({c['Rule'] or '-'})")


if __name__ == "__main__":
    main()
