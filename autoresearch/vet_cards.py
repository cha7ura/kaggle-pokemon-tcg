"""Vet candidate Crustle-counter attackers: print attack TEXT (the condition behind high dmg/E),
evolution line, and energy type. Raw damage means nothing if the attack has a killer drawback.
"""
from cg.api import all_card_data, all_attack, CardType, EnergyType

CD = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: a for a in all_attack()}
ETYPE = {int(e): e.name for e in EnergyType}
BYNAME = {}
for c in CD.values():
    BYNAME.setdefault(c.name, c.cardId)

# candidate non-ex attackers from the dmg/energy screen
CANDS = [699, 797, 995, 129, 884, 224, 115, 1047, 636, 91, 495]  # Tinkaton, Ceruledge, Haxorus, Decidueye, Medicham, Annihilape, Conkeldurr, Probopass, Steven's Claydol, Rillaboom, Chandelure


def line_for(c):
    # find pre-evolution chain by evolvesFrom names
    chain = [c.name]
    cur = c
    seen = 0
    while cur and getattr(cur, "evolvesFrom", None) and seen < 3:
        pre = BYNAME.get(cur.evolvesFrom)
        chain.insert(0, cur.evolvesFrom + (f"({pre})" if pre else "(NOT IN POOL)"))
        cur = CD.get(pre); seen += 1
    return " -> ".join(chain)


for cid in CANDS:
    c = CD.get(cid)
    if not c:
        print(f"{cid}: not in pool"); continue
    stage = "basic" if c.basic else "stage1" if c.stage1 else "stage2" if c.stage2 else "?"
    print(f"\n### {c.name} (id {cid}) {ETYPE.get(int(c.energyType),'')} HP{c.hp} {stage} retreat{c.retreatCost}")
    print(f"    line: {line_for(c)}")
    for aid in c.attacks:
        a = ATK.get(aid)
        if a:
            cost = "+".join(ETYPE.get(int(e), str(e))[:3] for e in a.energies) or "free"
            print(f"    [{a.damage:>3} / {len(a.energies)}E {cost}] {a.name}: {a.text.strip()[:160]}")
