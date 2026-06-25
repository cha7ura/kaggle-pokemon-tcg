"""Dump the FULL card pool from the cg engine to a schema-conformant JSON (docs/card-schema.md).
Engine spine = all 1267 cards with abilities (CardData.skills) + attacks (Attack.text).

Run in docker (engine is linux/amd64):
  docker run --rm --platform linux/amd64 -v REPO:/app -w /app/autoresearch \
    -e PYTHONPATH=/app/sdk python:3.11-slim python /app/tools/build_card_text.py

Writes /app/autoresearch/data/cards_engine.json and prints a coverage report that answers:
does the engine carry abilities? does it carry Trainer effect text? (→ is the wiki load-bearing?)
"""
import json, os
from cg.api import all_card_data, all_attack, CardType, EnergyType

ETYPE = {e.value: e.name for e in EnergyType}
CTYPE = {c.value: c.name for c in CardType}


def etypes(lst):
    return [ETYPE.get(int(e), str(e)) for e in (lst or [])]


def prize_value(c):
    return 3 if c.megaEx else 2 if c.ex else 1


def stage(c):
    return "STAGE2" if c.stage2 else "STAGE1" if c.stage1 else "BASIC" if c.basic else None


def main():
    cards = all_card_data()
    atk = {a.attackId: a for a in all_attack()}
    out = []
    for c in cards:
        ct = CTYPE.get(int(c.cardType), str(c.cardType))
        is_pokemon = ct == "POKEMON"
        is_trainer = ct in ("ITEM", "SUPPORTER", "STADIUM", "TOOL")
        # abilities: CardData.skills = list[Skill(name,text)] — present for ALL card types per engine
        abilities = [{"name": s.name, "text": s.text} for s in (c.skills or [])]
        attacks = []
        for aid in (c.attacks or []):
            a = atk.get(aid)
            if not a:
                continue
            attacks.append({"attack_id": aid, "name": a.name,
                            "energy_cost": etypes(a.energies), "damage_base": a.damage,
                            "damage_text": a.text})
        rec = {
            "card_id": c.cardId, "name": c.name, "card_type": ct,
            "ace_spec": bool(c.aceSpec),
            "hp": c.hp if is_pokemon else None,
            "energy_type": ETYPE.get(int(c.energyType)) if c.energyType is not None else None,
            "weakness": ETYPE.get(int(c.weakness)) if c.weakness is not None else None,
            "resistance": ETYPE.get(int(c.resistance)) if c.resistance is not None else None,
            "retreat_cost": c.retreatCost if is_pokemon else None,
            "stage": stage(c) if is_pokemon else None,
            "evolves_from": c.evolvesFrom,
            "is_ex": bool(c.ex), "is_mega_ex": bool(c.megaEx), "is_tera": bool(c.tera),
            "prize_value": prize_value(c) if is_pokemon else None,
            "abilities": abilities if is_pokemon else None,
            "attacks": attacks if is_pokemon else None,
            "trainer_subtype": ct if is_trainer else None,
            # engine may stash trainer/energy effect text in skills; capture raw for the report
            "effect_text": " ".join(s.text for s in (c.skills or [])) if is_trainer else None,
            "source": {"engine": True, "wiki": False, "validated": False},
        }
        out.append(rec)

    os.makedirs("data", exist_ok=True)
    json.dump(out, open("data/cards_engine.json", "w"), ensure_ascii=False, indent=0)

    # coverage report — the open question
    n = len(out)
    by_type = {}
    for r in out:
        by_type[r["card_type"]] = by_type.get(r["card_type"], 0) + 1
    pk = [r for r in out if r["card_type"] == "POKEMON"]
    pk_with_ability = sum(1 for r in pk if r["abilities"])
    tr = [r for r in out if r["trainer_subtype"]]
    tr_with_text = sum(1 for r in tr if (r["effect_text"] or "").strip())
    print(f"cards={n} by_type={by_type}")
    print(f"pokemon={len(pk)} with_ability={pk_with_ability}")
    print(f"trainers={len(tr)} with_engine_effect_text={tr_with_text}  "
          f"({'engine HAS trainer text' if tr_with_text else 'WIKI LOAD-BEARING for trainers'})")
    # show a couple of samples
    for name in ("Teal Mask Ogerpon ex", "Lillie's Determination", "Rare Candy"):
        m = next((r for r in out if r["name"] == name), None)
        if m:
            print(f"  [{m['card_id']}] {name}: abilities={m['abilities']} effect_text={m['effect_text']!r}")


if __name__ == "__main__":
    main()
