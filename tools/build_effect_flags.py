"""Classify the engine card text (data/cards_engine.json) into structured effect_flags + per-attack
scaling_basis (schema in docs/card-schema.md). Pure stdlib; no docker. Crude-but-auditable keyword
rules — unmatched cards keep empty flags + flag_needs_review.

  python tools/build_effect_flags.py        # writes data/cards.json + data/cards_effects.csv + self-check

Outputs:
  data/cards.json         — full schema records + card-level effect_flags + per-attack scaling_basis
  data/cards_effects.csv  — flat one-hot the feature layer consumes
"""
import json, csv, os, re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "autoresearch", "data")

FLAGS = ["accelerate_energy", "draw", "search", "gust", "heal", "spread_damage", "inflict_status",
         "switch", "move_energy", "disrupt_hand", "recover_from_discard", "setup_attack", "protect",
         "damage_boost", "retreat_reduce", "devolve", "item_lock", "deck_refresh", "prize_manipulate",
         "conditional_activation"]

# each rule: flag -> regex (searched on lowercased, nbsp-normalized text). Order-independent (multi-label).
RULES = {
    "search":              r"search your deck",
    "draw":                r"draw (a|\d+|that|cards|your)",
    "accelerate_energy":   r"attach (a |an |any |\d+ |up to )?.*?energy card.*?(from your hand|from your deck|from your discard|to (this|1 of|your))",
    "recover_from_discard":r"from your discard pile.*(into your hand|to your hand|attach|onto)",
    "deck_refresh":        r"shuffle your hand into your deck",
    "gust":                r"(opponent'?s benched pok[eé]mon to the active|switch in 1 of your opponent)",
    "switch":              r"switch your active pok[eé]mon with",
    "heal":                r"(heal \d+|remove \d+ damage counter)",
    "spread_damage":       r"damage counters? on .*(each|benched|opponent'?s pok[eé]mon)",
    "inflict_status":      r"\b(asleep|poisoned|burned|paralyzed|confused)\b",
    "move_energy":         r"move (a|an|\d+|any|that|all).*energy",
    "disrupt_hand":        r"opponent.{0,40}(shuffle|discard|reveal|put).{0,30}hand",
    "devolve":             r"devolve",
    "item_lock":           r"can'?t play .*item",
    "prize_manipulate":    r"prize card",
    "retreat_reduce":      r"(retreat cost.*(less|reduce|\-)|(less|reduce).*retreat)",
    "protect":             r"(prevent all damage|isn'?t affected by|no damage .*done to this|can'?t be (knocked|affected))",
    "damage_boost":        r"\d+ more damage",
    "conditional_activation": r"you can use this (card|attack) only if|only if you go second",
}
# scaling_basis on an attack's damage_text (lowercased)
SCALE = [
    ("per_energy_both", r"for each .*energy attached to both"),
    ("per_energy_self", r"for each .*energy attached to this"),
    ("per_hand_card",   r"for each card in your hand"),
    ("on_ko_last_turn", r"knocked out .*(last turn|during your opponent)"),
    ("on_opp_prizes",   r"opponent('s)? has.{0,20}prize card|opponent doesn'?t have exactly"),
    ("on_my_prizes",    r"you have (exactly |\d|more|fewer).{0,20}prize card"),
    ("per_bench",       r"for each .*benched"),
    ("per_damaged",     r"for each .*damage counter"),
    ("coinflip",        r"flip .*coin"),
]


def norm(s):
    return ((s or "").replace("\xa0", " ").replace("\n", " ")
            .replace("’", "'").replace("‘", "'").lower())


def classify_text(text):
    t = norm(text)
    return {f for f, rx in RULES.items() if re.search(rx, t)}


def scaling_of(damage_text):
    t = norm(damage_text)
    for name, rx in SCALE:
        if re.search(rx, t):
            return name
    if re.search(r"\d+\+|more damage|×|x \d", t):
        return "other"
    return "none"


def card_flags(rec):
    flags = set()
    texts = []
    for a in (rec.get("abilities") or []):
        texts.append(a.get("text", ""))
    if rec.get("effect_text"):
        texts.append(rec["effect_text"])
    has_setup = False
    for atk in (rec.get("attacks") or []):
        texts.append(atk.get("damage_text", ""))
        atk["scaling_basis"] = scaling_of(atk.get("damage_text", ""))
        if atk.get("damage_base", 0) == 0 and (atk.get("damage_text") or "").strip():
            has_setup = True
    for t in texts:
        flags |= classify_text(t)
    if has_setup:
        flags.add("setup_attack")
    return flags, has_setup


def main():
    # Read cards.json if present (preserves wiki_url/set/rulings added by fetch_card_wiki merge),
    # else the engine base. build_effect_flags is the FINAL pipeline step → adds flags + scaling
    # on top of whatever enrichment already happened, without clobbering it.
    base = os.path.join(DATA, "cards.json")
    if not os.path.exists(base):
        base = os.path.join(DATA, "cards_engine.json")
    cards = json.load(open(base))
    for r in cards:
        flags, has_setup = card_flags(r)
        r["effect_flags"] = sorted(flags)
        r["has_setup_attack"] = has_setup
        r["flag_needs_review"] = (not flags) and (
            bool(r.get("abilities")) or bool(r.get("effect_text")) or
            any((a.get("damage_text") or "").strip() for a in (r.get("attacks") or [])))
    json.dump(cards, open(os.path.join(DATA, "cards.json"), "w"), ensure_ascii=False, indent=0)

    with open(os.path.join(DATA, "cards_effects.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card_id", "name", "card_type", "has_setup_attack"] + FLAGS)
        for r in sorted(cards, key=lambda x: x["card_id"]):
            fl = set(r["effect_flags"])
            w.writerow([r["card_id"], r["name"], r["card_type"], int(r["has_setup_attack"])]
                       + [int(f in fl) for f in FLAGS])

    # coverage + self-check
    n = len(cards)
    flagged = sum(1 for r in cards if r["effect_flags"])
    review = sum(1 for r in cards if r["flag_needs_review"])
    print(f"cards={n} flagged={flagged} need_review={review}")
    per = {f: sum(1 for r in cards if f in r["effect_flags"]) for f in FLAGS}
    print("by_flag:", {k: v for k, v in sorted(per.items(), key=lambda x: -x[1])})

    byname = {r["name"].replace("’", "'"): r for r in cards}

    def has(name, flag):
        r = byname.get(name)
        return r and flag in r["effect_flags"]

    checks = [
        ("Boss's Orders", "gust"), ("Lillie's Determination", "draw"),
        ("Lillie's Determination", "deck_refresh"), ("Ultra Ball", "search"),
        ("Teal Mask Ogerpon ex", "accelerate_energy"), ("Teal Mask Ogerpon ex", "draw"),
        ("Switch", "switch"), ("Night Stretcher", "recover_from_discard"),
        ("Dudunsparce", "draw"),
    ]
    for name, flag in checks:
        ok = has(name, flag)
        print(f"  [{'ok' if ok else 'MISS'}] {name} -> {flag}")
        assert ok, f"self-check failed: {name} should have {flag}"
    # Ultra Ball must NOT be disrupt_hand (its discard-2 is a cost, not opponent disruption)
    assert not has("Ultra Ball", "disrupt_hand"), "Ultra Ball false-positive disrupt_hand"
    print("self-check PASSED")


if __name__ == "__main__":
    main()
