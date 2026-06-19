"""Non-ex aggro policy (anti-Crustle): pilot a basic non-ex attacker deck.

Non-ex attackers are NOT walled by Crustle, so the plan is simple and robust:
bench attackers, load energy onto the active, KO with the best attack, drag
targets with Boss, and — crucially vs the Crustle mill — do NOT deck ourselves
out. Crash-safe. Deck-agnostic (works on any basic-attacker deck).
"""
import os

from cg.api import (AreaType, CardType, EnergyType, SelectContext, OptionType,
                    Pokemon, all_card_data, all_attack, to_observation_class)

DECK_PATH = os.environ.get("AGENT_DECK", "decks/nonex_v1.csv")
LOW_DECK_COUNT = 13          # vs the stall/mill: stop drawing well before we deck out
BOSS_ORDERS = 1182
SWITCH = 1123
DRAW_SUPPORTERS = {1192, 1227}   # Carmine, Lillie

_CARD = {c.cardId: c for c in all_card_data()}
_ATK = {a.attackId: a.damage for a in all_attack()}


def read_deck(path=None):
    for p in ([path] if path else []) + ["deck.csv", "/kaggle_simulations/agent/deck.csv", DECK_PATH]:
        if p and os.path.exists(p):
            with open(p) as f:
                return [int(line) for line in f if line.strip()][:60]
    raise FileNotFoundError("deck.csv not found")


_DECK = read_deck()


def _get(obs, area, index, pidx):
    try:
        ps = obs.current.players[pidx]
        return {AreaType.HAND: ps.hand, AreaType.DISCARD: ps.discard, AreaType.ACTIVE: ps.active,
                AreaType.BENCH: ps.bench, AreaType.PRIZE: ps.prize}[area][index]
    except Exception:
        return None


def _eff_damage(attack_id, attacker_id, defender):
    dmg = _ATK.get(attack_id, 0)
    a, d = _CARD.get(attacker_id), _CARD.get(defender.id)
    if a and d:
        if d.weakness is not None and int(d.weakness) == int(a.energyType):
            dmg *= 2
        elif d.resistance is not None and int(d.resistance) == int(a.energyType):
            dmg -= 30
    return max(0, dmg)


def _is_basic_pokemon(cid):
    c = _CARD.get(cid)
    return c is not None and int(c.cardType) == int(CardType.POKEMON) and c.basic


def _legal_fallback(sel):
    n = len(sel.option)
    return list(range(min(max(1, sel.minCount), n))) if n else [0]


def _score_main(obs):
    st, sel = obs.current, obs.select
    me = st.yourIndex
    my, op = st.players[me], st.players[1 - me]
    low_deck = getattr(my, "deckCount", 999) <= LOW_DECK_COUNT
    opp_active = op.active[0] if op.active else None
    my_active = my.active[0] if my.active else None

    scores = []
    for o in sel.option:
        s = 0.0
        t = int(o.type)
        if t == int(OptionType.ATTACK):
            s = 1000.0
            if opp_active is not None and my_active is not None:
                dmg = _eff_damage(o.attackId, my_active.id, opp_active)
                if dmg >= opp_active.hp:
                    s = 50000 if len(op.prize) <= 1 else 9000 + dmg   # KO (win if last prize)
                else:
                    s = 1000 + dmg
        elif t == int(OptionType.EVOLVE):
            s = 9000
        elif t == int(OptionType.ABILITY):
            s = 8000
        elif t == int(OptionType.ATTACH):
            pk = _get(obs, o.inPlayArea, o.inPlayIndex, me)
            s = 8000 + (200 if (pk and o.inPlayArea == AreaType.ACTIVE) else 0)
        elif t == int(OptionType.PLAY):
            card = _get(obs, AreaType.HAND, o.index, me)
            if card is None:
                scores.append(0); continue
            cid = card.id
            if _is_basic_pokemon(cid):
                s = 7000
            elif cid == BOSS_ORDERS:
                s = 6000 if op.bench else -1                 # drag a benched target to KO
            elif cid in DRAW_SUPPORTERS:
                s = -1 if low_deck else 5000                 # deck-out guard
            elif cid == SWITCH:
                s = -1
            else:
                s = 4000
        elif t == int(OptionType.RETREAT):
            s = -1
        elif t == int(OptionType.END):
            s = -10
        scores.append(s)
    return scores


def _score_context(obs):
    st, sel = obs.current, obs.select
    me = st.yourIndex
    my = st.players[me]
    field = {}
    for p in (my.active + list(my.bench)):
        if p:
            field[p.id] = field.get(p.id, 0) + 1
    scores = []
    for o in sel.option:
        s = 0.0
        if int(o.type) == int(OptionType.NUMBER):
            s = o.number or 0
        elif int(o.type) == int(OptionType.YES):
            s = 1
        else:
            card = _get(obs, o.area, o.index, getattr(o, "playerIndex", me))
            if card is not None and isinstance(card, Pokemon):
                if o.area == AreaType.ACTIVE or getattr(o, "playerIndex", me) == me:
                    s = 50 + len(card.energies) - 20 * field.get(card.id, 0)  # spread attackers
            else:
                s = 1
        scores.append(s)
    return scores


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return _DECK
        sel = obs.select
        scores = _score_main(obs) if int(sel.type) == 0 else _score_context(obs)
        order = sorted(range(len(sel.option)), key=lambda i: scores[i], reverse=True)
        k = min(sel.maxCount, len(sel.option))
        k = max(k, min(max(1, sel.minCount), len(sel.option)))
        return order[:k] if order else _legal_fallback(sel)
    except Exception:
        try:
            return _legal_fallback(to_observation_class(obs_dict).select)
        except Exception:
            return [0]
