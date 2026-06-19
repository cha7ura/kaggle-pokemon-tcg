"""Scout the pool for strong NON-EX attackers (not walled by Crustle).

Ranks non-ex Pokemon by their best attack's damage and damage-per-energy, using
the engine's authoritative card + attack data. Favours Basics (consistency) and
flags weakness/HP so we can design a coherent anti-meta deck.
"""
from cg.api import all_card_data, all_attack, EnergyType, CardType

ENAME = {int(e): e.name[:4] for e in EnergyType}


def main():
    cards = all_card_data()
    atk = {a.attackId: a for a in all_attack()}

    rows = []
    for c in cards:
        if int(c.cardType) != int(CardType.POKEMON):
            continue
        if c.ex or c.megaEx:                      # NON-EX only (Crustle can't wall it)
            continue
        best = None
        for aid in c.attacks:
            a = atk.get(aid)
            if a is None or a.damage <= 0:
                continue
            cost = max(1, len(a.energies))
            key = (a.damage, a.damage / cost)
            if best is None or key > best[0]:
                best = (key, a, cost)
        if best is None:
            continue
        (_, a, cost) = best
        stage = "B" if c.basic else "S1" if c.stage1 else "S2" if c.stage2 else "?"
        rows.append({
            "id": c.cardId, "name": c.name, "stage": stage, "hp": c.hp,
            "type": ENAME.get(int(c.energyType), "?"),
            "weak": ENAME.get(int(c.weakness), "-") if c.weakness is not None else "-",
            "dmg": a.damage, "cost": cost, "dpe": a.damage / cost, "atk": a.name,
        })

    print("=== TOP BASIC non-ex attackers (HP>=110, dmg>=100), by damage/energy ===")
    basics = [r for r in rows if r["stage"] == "B" and r["hp"] >= 110 and r["dmg"] >= 100]
    for r in sorted(basics, key=lambda r: (-r["dpe"], -r["dmg"]))[:18]:
        print(f"  {r['id']:>4} HP{r['hp']:<4} {r['type']:<4} w:{r['weak']:<4} "
              f"{r['dmg']:>3}/{r['cost']}E ={r['dpe']:>5.0f}/E  {r['name'][:24]:<24} ({r['atk'][:18]})")

    print("\n=== TOP overall non-ex by raw damage (any stage) ===")
    for r in sorted(rows, key=lambda r: -r["dmg"])[:12]:
        print(f"  {r['id']:>4} {r['stage']:<2} HP{r['hp']:<4} {r['type']:<4} "
              f"{r['dmg']:>3}/{r['cost']}E  {r['name'][:26]:<26} ({r['atk'][:16]})")


if __name__ == "__main__":
    main()
