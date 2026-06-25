"""Assemble ONE training/inference row = STATE features + OPTION features, from the RAW obs dict.
Ties together state (board), threat.py, deck_tracker.py, card_features.py. Stdlib only; identical
offline (replay obs) and in-sim (live obs).

  state_row(obs, seat, decklist, ko_last_turn) -> (vec, names)        # the situation
  option_row(obs, seat, option, decklist)      -> (vec, names)        # one candidate action
  row(obs, seat, option, decklist, ko_last_turn)-> (vec, names)       # state ++ option (full row)

`option` is one entry from obs['select']['option'] (a raw dict). `decklist` = our 60-card id list.
"""
from . import threat, deck_tracker, card_features

ENERGY_N = 12  # EnergyType count


def _p(obs, seat):
    cur = obs.get("current") or {}
    pl = cur.get("players") or [{}, {}]
    return pl[seat], pl[1 - seat], cur


def _active(player):
    a = player.get("active") or []
    return a[0] if a else None


# ---------------- STATE ----------------
STATE_NAMES = [
    # global / turn
    "turn", "turn_action_count", "going_second", "is_first_turn",
    "supporter_played", "stadium_played", "energy_attached", "retreated", "stadium_present",
    # my side
    "my_prize_left", "my_deck_count", "my_hand_count", "my_bench", "my_bench_free",
    "my_active_hp", "my_active_dmg", "my_active_energy", "my_active_ready",
    "my_asleep", "my_paralyzed", "my_confused", "my_poisoned", "my_burned",
    "my_bench_damage_total",
    # opp side
    "opp_prize_left", "opp_deck_count", "opp_hand_count", "opp_bench", "opp_bench_free",
    "opp_active_hp", "opp_active_dmg", "opp_active_energy", "opp_active_is_ex",
    "opp_bench_damage_total",
    # race / threat
    "prize_diff", "our_max_dmg_active", "our_best_dmg_any", "imminent_ko_count",
    "opp_max_dmg_to_active", "opp_lethal_next", "our_lethal_active",
    "bench_beats_active",   # a benched attacker out-damages the active -> promote/retreat to it
    "our_spread_available", # bench-spread counters we can place (Dragapult Phantom Dive engine)
    # history
    "ko_last_turn",
]


def _bench_dmg(player):
    tot = 0
    for b in (player.get("bench") or []):
        if b:
            tot += (b.get("maxHp", 0) - b.get("hp", 0))
    return tot


def state_row(obs, seat, decklist, ko_last_turn=False):
    me, opp, cur = _p(obs, seat)
    ma, oa = _active(me), _active(opp)
    first = cur.get("firstPlayer", -1)
    going_second = 1.0 if (first != -1 and first != seat) else 0.0
    turn = cur.get("turn", 0)
    dmg_each = threat.our_max_damage_vs_each(obs, seat, ko_last_turn)
    our_active_dmg = dmg_each.get(("ACTIVE", 0), 0)
    our_best = max(dmg_each.values(), default=0)
    imminent = 0
    for (area, idx), d in dmg_each.items():
        tgt = oa if area == "ACTIVE" else (opp.get("bench") or [])[idx] if idx < len(opp.get("bench") or []) else None
        if tgt and d >= tgt.get("hp", 1):
            imminent += 1
    opp_dmg = threat.opp_max_damage_to_active(obs, seat, ko_last_turn)
    cardsdb = card_features._cards()

    def is_ex(pk):
        r = cardsdb.get((pk or {}).get("id"))
        return 1.0 if (r and r.get("is_ex")) else 0.0

    v = {
        "turn": turn, "turn_action_count": cur.get("turnActionCount", 0),
        "going_second": going_second, "is_first_turn": 1.0 if turn <= 2 else 0.0,
        "supporter_played": 1.0 if cur.get("supporterPlayed") else 0.0,
        "stadium_played": 1.0 if cur.get("stadiumPlayed") else 0.0,
        "energy_attached": 1.0 if cur.get("energyAttached") else 0.0,
        "retreated": 1.0 if cur.get("retreated") else 0.0,
        "stadium_present": 1.0 if (cur.get("stadium") or []) else 0.0,
        "my_prize_left": len(me.get("prize") or []), "my_deck_count": me.get("deckCount", 0),
        "my_hand_count": me.get("handCount", len(me.get("hand") or [])),
        "my_bench": len(me.get("bench") or []),
        "my_bench_free": me.get("benchMax", 5) - len(me.get("bench") or []),
        "my_active_hp": (ma or {}).get("hp", 0),
        "my_active_dmg": (ma or {}).get("maxHp", 0) - (ma or {}).get("hp", 0) if ma else 0,
        "my_active_energy": len((ma or {}).get("energies") or []),
        "my_active_ready": 1.0 if (ma and our_active_dmg > 0) else 0.0,
        "my_asleep": 1.0 if me.get("asleep") else 0.0, "my_paralyzed": 1.0 if me.get("paralyzed") else 0.0,
        "my_confused": 1.0 if me.get("confused") else 0.0, "my_poisoned": 1.0 if me.get("poisoned") else 0.0,
        "my_burned": 1.0 if me.get("burned") else 0.0, "my_bench_damage_total": _bench_dmg(me),
        "opp_prize_left": len(opp.get("prize") or []), "opp_deck_count": opp.get("deckCount", 0),
        "opp_hand_count": opp.get("handCount", 0), "opp_bench": len(opp.get("bench") or []),
        "opp_bench_free": opp.get("benchMax", 5) - len(opp.get("bench") or []),
        "opp_active_hp": (oa or {}).get("hp", 0),
        "opp_active_dmg": (oa or {}).get("maxHp", 0) - (oa or {}).get("hp", 0) if oa else 0,
        "opp_active_energy": len((oa or {}).get("energies") or []), "opp_active_is_ex": is_ex(oa),
        "opp_bench_damage_total": _bench_dmg(opp),
        "prize_diff": len(opp.get("prize") or []) - len(me.get("prize") or []),
        "our_max_dmg_active": our_active_dmg, "our_best_dmg_any": our_best,
        "imminent_ko_count": imminent, "opp_max_dmg_to_active": opp_dmg,
        "opp_lethal_next": 1.0 if (ma and opp_dmg >= ma.get("hp", 1)) else 0.0,
        "our_lethal_active": 1.0 if (oa and our_active_dmg >= oa.get("hp", 1)) else 0.0,
        "bench_beats_active": 1.0 if our_best > our_active_dmg else 0.0,
        "our_spread_available": threat.spread_available(obs, seat),
        "ko_last_turn": 1.0 if ko_last_turn else 0.0,
    }
    return [float(v[k]) for k in STATE_NAMES], STATE_NAMES


# ---------------- OPTION ----------------
OPTION_META = ["context", "opt_type", "opt_area", "opt_index", "opt_number", "opt_count",
               "n_options", "min_count", "max_count", "remain_damage_counter", "remain_energy_cost",
               "target_hp_remaining", "creates_imminent_ko", "draw_prob_card", "needed_in_discard"]
OPTION_NAMES = OPTION_META + ["opt_" + n for n in card_features.FEATURE_NAMES]


def option_row(obs, seat, option, decklist):
    me, opp, cur = _p(obs, seat)
    sel = obs.get("select") or {}
    o = option or {}
    cid = o.get("cardId")
    # threat-relevant: if this option targets a Pokemon, its remaining hp + imminent-ko
    target_hp = 0.0
    creates_ko = 0.0
    if o.get("playerIndex") is not None and o.get("index") is not None:
        side = opp if o.get("playerIndex") != seat else me
        area = o.get("area")
        arr = side.get("bench") if area == 5 else side.get("active")
        i = o.get("index")
        tgt = (arr or [None])[i] if i is not None and i < len(arr or []) else None
        if tgt:
            target_hp = tgt.get("hp", 0)
    meta = {
        "context": sel.get("context", -1), "opt_type": o.get("type", -1),
        "opt_area": o.get("area", -1) if o.get("area") is not None else -1,
        "opt_index": o.get("index", -1) if o.get("index") is not None else -1,
        "opt_number": o.get("number") if o.get("number") is not None else -1,   # DRAW_COUNT etc.
        "opt_count": o.get("count") if o.get("count") is not None else -1,       # multi-energy/draw count
        "n_options": len(sel.get("option") or []), "min_count": sel.get("minCount", 0),
        "max_count": sel.get("maxCount", 0), "remain_damage_counter": sel.get("remainDamageCounter", 0),
        "remain_energy_cost": sel.get("remainEnergyCost", 0),
        "target_hp_remaining": target_hp, "creates_imminent_ko": creates_ko,
        "draw_prob_card": deck_tracker.draw_prob(obs, seat, decklist, cid, 1) if cid else 0.0,
        "needed_in_discard": 1.0 if (cid and deck_tracker.needed_in_discard(obs, seat, cid)) else 0.0,
    }
    vec = [float(meta[k]) for k in OPTION_META]
    vec += card_features.card_vector(cid) if cid else [0.0] * len(card_features.FEATURE_NAMES)
    return vec, OPTION_NAMES


def row(obs, seat, option, decklist, ko_last_turn=False):
    sv, sn = state_row(obs, seat, decklist, ko_last_turn)
    ov, on = option_row(obs, seat, option, decklist)
    return sv + ov, sn + on


ALL_NAMES = STATE_NAMES + OPTION_NAMES


def _selfcheck():
    # minimal obs: my Abra(741) active w/ 1 Psychic energy, opp Dreepy(119) active; a MAIN option to play a card
    decklist = [741] * 4 + [743] * 3 + [0] * 53
    obs = {"current": {"turn": 3, "turnActionCount": 0, "firstPlayer": 1, "supporterPlayed": False,
                       "stadiumPlayed": False, "energyAttached": False, "retreated": False, "stadium": [],
                       "players": [
                           {"active": [{"id": 741, "hp": 70, "maxHp": 70, "energies": [5],
                                        "energyCards": [], "tools": [], "preEvolution": []}],
                            "bench": [], "benchMax": 5, "deckCount": 40, "prize": [None] * 6,
                            "discard": [], "hand": [{"id": 743}], "handCount": 1},
                           {"active": [{"id": 119, "hp": 60, "maxHp": 60, "energies": [],
                                        "energyCards": [], "tools": [], "preEvolution": []}],
                            "bench": [], "benchMax": 5, "deckCount": 39, "prize": [None] * 6,
                            "handCount": 5}]},
           "select": {"context": 0, "minCount": 1, "maxCount": 1,
                      "remainDamageCounter": 0, "remainEnergyCost": 0,
                      "option": [{"type": 7, "cardId": 743, "index": 0, "playerIndex": 0}]}}
    opt = obs["select"]["option"][0]
    sv, sn = state_row(obs, 0, decklist)
    ov, on = option_row(obs, 0, opt, decklist)
    full, names = row(obs, 0, opt, decklist)
    assert len(sv) == len(STATE_NAMES) == len(sn)
    assert len(ov) == len(OPTION_NAMES) == len(on)
    assert len(full) == len(ALL_NAMES) == len(STATE_NAMES) + len(OPTION_NAMES)
    assert all(isinstance(x, float) for x in full)
    d = dict(zip(sn, sv))
    assert d["going_second"] == 1.0 and d["turn"] == 3.0          # seat 0, firstPlayer 1 -> we went second
    assert d["my_prize_left"] == 6 and d["opp_prize_left"] == 6 and d["prize_diff"] == 0
    print(f"featurize.py self-check PASSED (state={len(sn)} + option={len(on)} = {len(names)} dims)")


if __name__ == "__main__":
    _selfcheck()
