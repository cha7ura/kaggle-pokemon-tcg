# tools/imitation/features.py
"""Shared featurizer for the imitation league. Operates on the RAW observation dict so the
exact same code runs offline (replay obs) and in-sim (live obs). STDLIB ONLY."""

STATE_NAMES = [
    "turn", "turn_action_count", "supporter_played", "stadium_played",
    "my_active_hp", "my_active_maxhp", "my_active_dmg", "my_active_energy",
    "my_bench", "my_hand", "my_prize_left",
    "my_asleep", "my_confused", "my_paralyzed", "my_poisoned", "my_burned",
    "opp_active_hp", "opp_active_maxhp", "opp_active_dmg", "opp_active_energy",
    "opp_bench", "opp_hand", "opp_prize_left",
]
OPTION_NAMES = ["context", "opt_type", "opt_area", "opt_index", "n_options",
                "min_count", "max_count"]
STATE_DIM = len(STATE_NAMES)
OPTION_DIM = len(OPTION_NAMES)


def feature_names():
    return STATE_NAMES + OPTION_NAMES


def _active(player):
    a = player.get("active") or []
    return a[0] if a else None


def _prize_left(player):
    p = player.get("prize") or []
    return len(p)


def state_features(current, seat):
    players = current.get("players") or [{}, {}]
    me = players[seat] if seat < len(players) else {}
    opp = players[1 - seat] if (1 - seat) < len(players) else {}
    ma, oa = _active(me), _active(opp)

    def act_fields(a):
        if not a:
            return [0.0, 0.0, 0.0, 0.0]
        hp = float(a.get("hp", 0)); mhp = float(a.get("maxHp", 0))
        return [hp, mhp, mhp - hp, float(len(a.get("energies") or []))]

    f = [
        float(current.get("turn", 0)),
        float(current.get("turnActionCount", 0)),
        1.0 if current.get("supporterPlayed") else 0.0,
        1.0 if current.get("stadiumPlayed") else 0.0,
    ]
    f += act_fields(ma)
    f += [float(len(me.get("bench") or [])), float(me.get("handCount", len(me.get("hand") or []))),
          float(_prize_left(me))]
    f += [1.0 if me.get(k) else 0.0 for k in ("asleep", "confused", "paralyzed", "poisoned", "burned")]
    f += act_fields(oa)
    f += [float(len(opp.get("bench") or [])), float(opp.get("handCount", 0)),
          float(_prize_left(opp))]
    return f


def option_features(option, context, select):
    o = option or {}
    return [
        float(context if context is not None else -1),
        float(o.get("type", -1)),
        float(o.get("area", -1)),
        float(o.get("index", -1)),
        float(len(select.get("option") or [])),
        float(select.get("minCount", 0)),
        float(select.get("maxCount", 0)),
    ]
