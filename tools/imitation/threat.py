"""Mechanics-correct damage / threat estimator over the RAW obs dict (same code offline + in-sim).
Stdlib only. Reads the card KB (data/cards.json) for attacks, cost, scaling, weakness/resistance.

Estimates (good-enough features, not the engine's exact resolution):
  our_max_damage_vs_each(obs, seat)  -> {(area,index): dmg}   damage we can deal to each opp Pokemon
  opp_max_damage_to_active(obs, seat)-> int                   worst hit the opp can land on our Active
  is_lethal_on(hp, dmg) / opp_lethal_next(obs, seat) -> bool

SV rules: weakness x2, resistance -30, Active only; KO when dmg >= hp. Colorless = wildcard energy.
"""
import os, re, json, functools

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(HERE, "autoresearch", "data")

# EnergyType name <-> value (matches cg.api.EnergyType)
ENAME = ["COLORLESS", "GRASS", "FIRE", "WATER", "LIGHTNING", "PSYCHIC", "FIGHTING",
         "DARKNESS", "METAL", "DRAGON", "RAINBOW", "TEAM_ROCKET"]
EVAL = {n: i for i, n in enumerate(ENAME)}
COLORLESS = 0


@functools.lru_cache(maxsize=1)
def _cards():
    for fn in ("cards.json", "cards_engine.json"):
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            return {r["card_id"]: r for r in json.load(open(p))}
    return {}


def attack_ready(attached, cost_names):
    """attached = list[int EnergyType] on the Pokemon; cost_names = list[str] from the KB.
    Colorless paid last from any leftover. RAINBOW/TEAM_ROCKET attached count as wildcard."""
    from collections import Counter
    pool = Counter(attached)
    wild = pool.pop(EVAL["RAINBOW"], 0) + pool.pop(EVAL["TEAM_ROCKET"], 0)
    need_colorless = 0
    for c in cost_names:
        v = EVAL.get(c, COLORLESS)
        if v == COLORLESS:
            need_colorless += 1
            continue
        if pool.get(v, 0) > 0:
            pool[v] -= 1
        elif wild > 0:
            wild -= 1
        else:
            return False
    return sum(pool.values()) + wild >= need_colorless


_PER = re.compile(r"(\d+)\s*(?:more\s+)?damage\s+for each", re.I)
_TIMES = re.compile(r"(\d+)\s*damage\s+times", re.I)
_PLUS = re.compile(r"does\s+(\d+)\s+more damage", re.I)


def _scaling_amount(text):
    """Per-unit damage increment parsed from the attack text (the count comes from live state)."""
    for rx in (_PER, _TIMES, _PLUS):
        m = rx.search(text or "")
        if m:
            return int(m.group(1))
    return 0


def _count_for(basis, ctx):
    return {
        "per_energy_self": ctx.get("att_energy", 0),
        "per_energy_both": ctx.get("att_energy", 0) + ctx.get("def_energy", 0),
        "per_hand_card": ctx.get("hand", 0),
        "per_bench": ctx.get("bench", 0),
        "per_damaged": ctx.get("def_damage", 0) // 10,
    }.get(basis, 0)


def eval_attack_damage(atk, ctx):
    """Estimate an attack's damage given live ctx (before weakness/resistance)."""
    base = atk.get("damage_base", 0) or 0
    basis = atk.get("scaling_basis", "none")
    dmg = base
    per = _scaling_amount(atk.get("damage_text", ""))
    if basis in ("per_energy_self", "per_energy_both", "per_hand_card", "per_bench", "per_damaged"):
        dmg = base + per * _count_for(basis, ctx)
    elif basis == "on_ko_last_turn" and ctx.get("ko_last_turn"):
        dmg = base + (per or 0)
    elif basis == "on_opp_prizes" and ctx.get("opp_prizes", 9) <= 3:
        dmg = base + (per or 0)
    elif basis == "on_my_prizes" and ctx.get("my_prizes", 9) <= 3:
        dmg = base + (per or 0)
    elif basis == "coinflip":
        dmg = base + 0.5 * (per or 0)            # EV for "X more for each heads"-ish
    return dmg


def apply_weak_resist(dmg, attacker_card, defender_card, defender_is_active, kind="damage"):
    # counters / "not affected by effects" bypass the weakness/resistance pipeline (the damage-vs-
    # counters rule). Only true 'damage' to the Active doubles on weakness.
    if kind != "damage" or not defender_is_active or dmg <= 0:
        return dmg
    atype = (attacker_card or {}).get("energy_type")
    if defender_card and defender_card.get("weakness") and defender_card["weakness"] == atype:
        dmg *= 2
    if defender_card and defender_card.get("resistance") and defender_card["resistance"] == atype:
        dmg = max(0, dmg - 30)
    return dmg


def _ctx(att, deff, obs_seat_state, opp_state, hand, ko_last_turn, my_prizes, opp_prizes):
    return {
        "att_energy": len(att.get("energies") or []),
        "def_energy": len((deff or {}).get("energies") or []),
        "def_damage": (deff or {}).get("maxHp", 0) - (deff or {}).get("hp", 0) if deff else 0,
        "hand": hand, "bench": len(obs_seat_state.get("bench") or []),
        "ko_last_turn": ko_last_turn, "my_prizes": my_prizes, "opp_prizes": opp_prizes,
    }


def best_damage(att_pkmn, deff_pkmn, deff_is_active, ctx):
    """Max damage att_pkmn can deal to deff_pkmn this turn (must have energy for the attack)."""
    cards = _cards()
    ac = cards.get(att_pkmn.get("id"))
    dc = cards.get((deff_pkmn or {}).get("id"))
    if not ac or not ac.get("attacks"):
        return 0
    attached = att_pkmn.get("energies") or []
    best = 0
    for atk in ac["attacks"]:
        if not attack_ready(attached, atk.get("energy_cost") or []):
            continue
        if deff_is_active:
            d = eval_attack_damage(atk, ctx)
            d = apply_weak_resist(d, ac, dc, True, atk.get("damage_kind", "damage"))
        else:
            # vs a BENCH target: the only reach is bench-spread counters (Phantom Dive etc.),
            # all dumpable on one Pokemon for a KO. Counters bypass weakness.
            d = (atk.get("spread_amount", 0) or 0) * 10
        best = max(best, d)
    return best


def _players(obs, seat):
    cur = obs.get("current") or {}
    pl = cur.get("players") or [{}, {}]
    return pl[seat], pl[1 - seat], cur


def our_max_damage_vs_each(obs, seat, ko_last_turn=False):
    me, opp, cur = _players(obs, seat)
    my_prizes, opp_prizes = len(me.get("prize") or []), len(opp.get("prize") or [])
    hand = me.get("handCount", len(me.get("hand") or []))
    attackers = [a for a in (me.get("active") or []) if a] + [b for b in (me.get("bench") or []) if b]
    targets = [("ACTIVE", 0, (opp.get("active") or [None])[0])]
    targets += [("BENCH", i, b) for i, b in enumerate(opp.get("bench") or []) if b]
    out = {}
    for area, idx, tgt in targets:
        if not tgt:
            continue
        best = 0
        for att in attackers:
            ctx = _ctx(att, tgt, me, opp, hand, ko_last_turn, my_prizes, opp_prizes)
            best = max(best, best_damage(att, tgt, area == "ACTIVE", ctx))
        out[(area, idx)] = best
    return out


def opp_max_damage_to_active(obs, seat, ko_last_turn=False):
    """Worst hit the opponent can land on OUR active next turn (attached energy; weakness applies)."""
    me, opp, cur = _players(obs, seat)
    my_active = (me.get("active") or [None])[0]
    if not my_active:
        return 0
    my_prizes, opp_prizes = len(me.get("prize") or []), len(opp.get("prize") or [])
    opp_hand = opp.get("handCount", 0)
    attackers = [a for a in (opp.get("active") or []) if a] + [b for b in (opp.get("bench") or []) if b]
    best = 0
    for att in attackers:
        ctx = _ctx(att, my_active, opp, me, opp_hand, ko_last_turn, opp_prizes, my_prizes)
        best = max(best, best_damage(att, my_active, True, ctx))
    return best


def spread_available(obs, seat):
    """Max bench-spread counters (×10 damage) our ready attackers can place this turn."""
    cards = _cards()
    me, _, _ = _players(obs, seat)
    best = 0
    for att in [a for a in (me.get("active") or []) if a] + [b for b in (me.get("bench") or []) if b]:
        ac = cards.get(att.get("id"))
        if not ac:
            continue
        attached = att.get("energies") or []
        for atk in (ac.get("attacks") or []):
            if attack_ready(attached, atk.get("energy_cost") or []):
                best = max(best, (atk.get("spread_amount", 0) or 0) * 10)
    return best


def is_lethal_on(hp, dmg):
    return dmg >= hp


def opp_lethal_next(obs, seat, ko_last_turn=False):
    me, opp, cur = _players(obs, seat)
    my_active = (me.get("active") or [None])[0]
    if not my_active:
        return False
    hp_left = my_active.get("hp", 0)
    return is_lethal_on(hp_left, opp_max_damage_to_active(obs, seat, ko_last_turn))


def _selfcheck():
    # energy match: Colorless wildcard
    assert attack_ready([5], ["PSYCHIC"])                      # P vs {P}
    assert attack_ready([5, 5, 5], ["GRASS", "GRASS", "GRASS"]) is False  # P != G
    assert attack_ready([5, 1], ["PSYCHIC", "COLORLESS"])      # P + any -> {P}{C}
    assert attack_ready([10], ["PSYCHIC"])                     # RAINBOW = wildcard
    assert attack_ready([], ["COLORLESS"]) is False
    # scaling parse
    assert _scaling_amount("This attack does 20 damage for each card in your hand.") == 20
    assert _scaling_amount("does 90 more damage") == 90
    # eval: Powerful-Hand-like 20 x hand=7 -> 140
    atk = {"damage_base": 0, "scaling_basis": "per_hand_card",
           "damage_text": "This attack does 20 damage for each card in your hand."}
    assert eval_attack_damage(atk, {"hand": 7}) == 140
    # weakness x2 on Active
    ac = {"energy_type": "PSYCHIC"}; dc = {"weakness": "PSYCHIC"}
    assert apply_weak_resist(100, ac, dc, True) == 200
    assert apply_weak_resist(100, ac, dc, False) == 100        # bench: no weakness
    assert is_lethal_on(140, 140) and not is_lethal_on(141, 140)
    print("threat.py self-check PASSED")


if __name__ == "__main__":
    _selfcheck()
