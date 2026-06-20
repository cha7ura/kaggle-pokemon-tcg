"""Dump the full card pool (cards joined with their attacks) to CSV + flag meta-relevant attackers.

Goal-aligned: the ladder ceiling is Crustle (~50% of field, immune to ex/megaEx damage). So the
useful signal is NON-EX attackers with high damage at low energy cost (they beat the Crustle wall).
"""
import csv
from cg.api import all_card_data, all_attack, CardType, EnergyType

ATK = {a.attackId: a for a in all_attack()}
ETYPE = {int(e): e.name for e in EnergyType}


def attack_summary(c):
    out = []
    for aid in c.attacks:
        a = ATK.get(aid)
        if a:
            cost = "+".join(ETYPE.get(int(e), str(e))[:3] for e in a.energies) or "-"
            out.append(f"{a.name}({a.damage}/{len(a.energies)}E:{cost})")
    return " | ".join(out)


def best_attack(c):
    dmgs = [(ATK[a].damage, len(ATK[a].energies)) for a in c.attacks if a in ATK]
    if not dmgs:
        return 0, 99
    bd = max(d for d, _ in dmgs)
    mc = min(e for d, e in dmgs if d == bd)
    return bd, mc


def main():
    cards = all_card_data()
    rows = []
    for c in cards:
        bd, mc = best_attack(c)
        rows.append({
            "cardId": c.cardId, "name": c.name, "cardType": CardType(c.cardType).name,
            "hp": c.hp, "type": ETYPE.get(int(c.energyType), ""),
            "weakness": ETYPE.get(int(c.weakness), "") if c.weakness is not None else "",
            "resistance": ETYPE.get(int(c.resistance), "") if c.resistance is not None else "",
            "retreat": c.retreatCost, "basic": c.basic, "stage1": c.stage1, "stage2": c.stage2,
            "ex": c.ex, "megaEx": c.megaEx, "aceSpec": c.aceSpec,
            "n_attacks": len(c.attacks), "best_dmg": bd, "min_cost_for_best": mc,
            "dmg_per_energy": round(bd / mc, 1) if mc and mc < 99 else 0,
            "attacks": attack_summary(c),
        })
    cols = list(rows[0].keys())
    with open("cards_full.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"wrote cards_full.csv: {len(rows)} cards", flush=True)

    # Crustle-beating candidates: NON-ex Pokemon, high damage at low cost
    pk = [r for r in rows if r["cardType"] == "POKEMON" and not r["ex"] and not r["megaEx"] and r["best_dmg"] > 0]
    print("\n=== TOP NON-EX ATTACKERS by damage (beat Crustle; ex/megaEx are walled) ===", flush=True)
    for r in sorted(pk, key=lambda r: -r["best_dmg"])[:20]:
        print(f"  {r['name'][:24]:24} id={r['cardId']:5} {r['type'][:8]:8} HP{r['hp']:>3} "
              f"dmg={r['best_dmg']:>3}/{r['min_cost_for_best']}E (dpe={r['dmg_per_energy']}) "
              f"{'basic' if r['basic'] else 'st1' if r['stage1'] else 'st2'}", flush=True)
    print("\n=== BEST NON-EX dmg-per-energy (>=1E, dmg>=60) ===", flush=True)
    eff = [r for r in pk if r["min_cost_for_best"] >= 1 and r["best_dmg"] >= 60]
    for r in sorted(eff, key=lambda r: -r["dmg_per_energy"])[:15]:
        print(f"  {r['name'][:24]:24} id={r['cardId']:5} dmg={r['best_dmg']}/{r['min_cost_for_best']}E "
              f"dpe={r['dmg_per_energy']} HP{r['hp']} {r['type'][:8]}", flush=True)


if __name__ == "__main__":
    main()
