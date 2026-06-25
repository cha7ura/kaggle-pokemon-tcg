"""Static per-card features + evolution graph for the policy OPTION vector. Reads the card KB
(data/cards.json, built by build_effect_flags). Stdlib only; identical offline + in-sim.

  card_static(card_id) -> dict of numeric features (type, cost, dmg, stage, ex, effect_flags...)
  card_vector(card_id) -> flat list[float] (stable order = FEATURE_NAMES)
  evolution_graph()    -> {name: {stage, pre, nexts}}  for can-evolve / line reasoning
"""
import os, json, functools

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(HERE, "autoresearch", "data")

EFFECT_FLAGS = ["accelerate_energy", "draw", "search", "gust", "heal", "spread_damage",
                "inflict_status", "switch", "move_energy", "disrupt_hand", "recover_from_discard",
                "setup_attack", "protect", "damage_boost", "retreat_reduce", "devolve", "item_lock",
                "deck_refresh", "prize_manipulate", "conditional_activation"]
CARD_TYPES = ["POKEMON", "ITEM", "SUPPORTER", "STADIUM", "TOOL", "BASIC_ENERGY", "SPECIAL_ENERGY"]
ENERGY = ["COLORLESS", "GRASS", "FIRE", "WATER", "LIGHTNING", "PSYCHIC", "FIGHTING", "DARKNESS",
          "METAL", "DRAGON", "RAINBOW", "TEAM_ROCKET"]


@functools.lru_cache(maxsize=1)
def _cards():
    for fn in ("cards.json", "cards_engine.json"):
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            return {r["card_id"]: r for r in json.load(open(p))}
    return {}


def _stage_ord(r):
    return {"BASIC": 0, "STAGE1": 1, "STAGE2": 2}.get(r.get("stage"), 0)


def card_static(card_id):
    r = _cards().get(card_id)
    if not r:
        return {"unknown": 1.0}
    attacks = r.get("attacks") or []
    best = max((a.get("damage_base", 0) or 0 for a in attacks), default=0)
    flags = set(r.get("effect_flags") or [])
    d = {
        "card_type_idx": float(CARD_TYPES.index(r["card_type"]) if r["card_type"] in CARD_TYPES else -1),
        "energy_type_idx": float(ENERGY.index(r["energy_type"]) if r.get("energy_type") in ENERGY else -1),
        "hp": float(r.get("hp") or 0),
        "retreat_cost": float(r.get("retreat_cost") or 0),
        "stage": float(_stage_ord(r)),
        "is_ex": float(bool(r.get("is_ex"))),
        "is_mega_ex": float(bool(r.get("is_mega_ex"))),
        "is_tera": float(bool(r.get("is_tera"))),
        "prize_value": float(r.get("prize_value") or 0),
        "ace_spec": float(bool(r.get("ace_spec"))),
        "best_dmg": float(best),
        "n_attacks": float(len(attacks)),
        "n_abilities": float(len(r.get("abilities") or [])),
        "has_setup_attack": float(bool(r.get("has_setup_attack"))),
    }
    for f in EFFECT_FLAGS:
        d[f"flag_{f}"] = 1.0 if f in flags else 0.0
    return d


FEATURE_NAMES = (["card_type_idx", "energy_type_idx", "hp", "retreat_cost", "stage", "is_ex",
                  "is_mega_ex", "is_tera", "prize_value", "ace_spec", "best_dmg", "n_attacks",
                  "n_abilities", "has_setup_attack"] + [f"flag_{f}" for f in EFFECT_FLAGS])


def card_vector(card_id):
    d = card_static(card_id)
    return [d.get(k, 0.0) for k in FEATURE_NAMES]


@functools.lru_cache(maxsize=1)
def evolution_graph():
    """{name: {stage, pre, nexts}} keyed by Pokemon name (evolves_from is a name)."""
    g = {}
    for r in _cards().values():
        if r["card_type"] != "POKEMON":
            continue
        nm = r["name"]
        g.setdefault(nm, {"stage": _stage_ord(r), "pre": r.get("evolves_from"), "nexts": set()})
    for nm, node in list(g.items()):
        pre = node["pre"]
        if pre and pre in g:
            g[pre]["nexts"].add(nm)
    return g


def _selfcheck():
    cards = _cards()
    assert cards, "card KB not found (run build_card_text / build_effect_flags first)"
    byname = {r["name"].replace("’", "'"): r for r in cards.values()}
    alakazam = byname["Alakazam"]["card_id"]
    d = card_static(alakazam)
    assert d["stage"] == 2.0, d
    assert len(card_vector(alakazam)) == len(FEATURE_NAMES)
    # effect flag wired through
    boss = byname["Boss's Orders"]["card_id"]
    assert card_static(boss)["flag_gust"] == 1.0
    # evolution chain
    g = evolution_graph()
    assert g["Kadabra"]["pre"] == "Abra"
    assert "Kadabra" in g["Abra"]["nexts"]
    assert g["Alakazam"]["stage"] == 2
    print(f"card_features.py self-check PASSED ({len(FEATURE_NAMES)} static dims, "
          f"{len(g)} pokemon in evo graph)")


if __name__ == "__main__":
    _selfcheck()
