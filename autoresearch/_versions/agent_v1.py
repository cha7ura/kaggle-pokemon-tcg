"""Our strong, Crustle-aware Mega Lucario ex (Fighting) heuristic.

Informed by the day-1 meta (Crustle wall negates ex damage -> route through the
non-ex Hariyama), built as our own tunable policy and wrapped crash-safe so it
survives the validation mirror. Pairs with decks/lucario_meta.csv.

Design = context-scored greedy with an explicit "develop the board, then attack"
ordering and a few high-value pieces of intelligence:
  - bench the Riolu (Lucario) line first; evolve on curve
  - attach Fighting energy to the attacker that is closest to swinging
  - choose the attack that KOs / does the most effective damage (weakness x2),
    and NEVER swing an ex attacker into a Crustle wall (0 damage)
  - draw with Carmine/Lillie unless we'd deck ourselves out (deck-out guard)

Tunable knobs live in W (weights) so the autoresearch loop can mutate them.
"""
import os

from cg.api import (AreaType, CardType, EnergyType, SelectContext, OptionType,
                    Pokemon, all_card_data, all_attack, to_observation_class)

# ---- deck card IDs (the policy is deck-specific by necessity) ----
MAKUHITA, HARIYAMA, LUNATONE, SOLROCK, RIOLU, MEGA_LUCARIO = 673, 674, 675, 676, 677, 678
DUSK_BALL, SWITCH, PREMIUM_POWER, FIGHTING_GONG, POKE_PAD = 1102, 1123, 1141, 1142, 1152
HERO_CAPE, BOSS_ORDERS, CARMINE, LILLIE, GRAVITY_MTN = 1159, 1182, 1192, 1227, 1252
FIGHTING_ENERGY = 6
CRUSTLE = 345
LOW_DECK_COUNT = 8

DECK_PATH = os.environ.get("AGENT_DECK", "decks/lucario_meta.csv")

# Tunable weights (the autoresearch loop mutates these).
W = {
    "ability": 30000, "play_pokemon": 20000, "evolve": 9000, "attach": 8000,
    "draw": 3000, "boss": 3200, "switch": 6000, "attack": 1000,
    "ko_win": 50000, "crustle_whiff": -10000, "retreat": 2000, "end": -1,
    "energy_need": 100,
}

_CARD = {c.cardId: c for c in all_card_data()}
_ATK = {a.attackId: a.damage for a in all_attack()}


def read_deck(path=None):
    # Works both as our eval policy (decks/lucario_meta.csv) and as the Kaggle
    # submission main.py (deck.csv at top level / /kaggle_simulations/agent/).
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
        try:
            if area == AreaType.DECK:
                return obs.select.deck[index]
            if area == AreaType.STADIUM:
                return obs.current.stadium[index]
        except Exception:
            return None
    return None


def _prize_count(pk):
    d = _CARD.get(pk.id)
    return 3 if (d and d.megaEx) else 2 if (d and d.ex) else 1


def _eff_damage(base, attacker_id, defender):
    """base damage adjusted for weakness/resistance and the Crustle wall."""
    a, d = _CARD.get(attacker_id), _CARD.get(defender.id)
    dmg = base
    if d and a:
        if d.weakness is not None and int(d.weakness) == int(EnergyType.FIGHTING):
            dmg *= 2
        elif d.resistance is not None and int(d.resistance) == int(EnergyType.FIGHTING):
            dmg -= 30
        if defender.id == CRUSTLE and (a.ex or a.megaEx):   # wall negates ex damage
            dmg = 0
    return max(0, dmg)


def _is_main_attacker(cid):
    return cid in (RIOLU, MEGA_LUCARIO, MAKUHITA, HARIYAMA)


def _score_main(obs):
    """Score every option at a MAIN decision; develop first, attack last."""
    st, sel = obs.current, obs.select
    me = st.yourIndex
    my, op = st.players[me], st.players[1 - me]
    low_deck = getattr(my, "deckCount", 999) <= LOW_DECK_COUNT
    opp_active = op.active[0] if op.active else None

    field = {}
    for p in (my.active + list(my.bench)):
        if p:
            field[p.id] = field.get(p.id, 0) + 1

    scores = []
    for o in sel.option:
        s = 0.0
        t = int(o.type)
        if t == int(OptionType.ATTACK):
            s = W["attack"]
            if opp_active is not None:
                # find the attacking pokemon's effective damage for this attackId
                dmg = 0
                for p in (my.active + list(my.bench)):
                    if p and o.attackId in (_CARD.get(p.id).attacks if _CARD.get(p.id) else []):
                        dmg = _eff_damage(_ATK.get(o.attackId, 0), p.id, opp_active)
                        break
                if opp_active.id == CRUSTLE and dmg == 0:
                    s = W["crustle_whiff"]              # never whiff ex into the wall
                elif dmg >= opp_active.hp:
                    s = W["ko_win"] if len(op.prize) <= _prize_count(opp_active) else W["attack"] + 8000
                else:
                    s = W["attack"] + dmg
        elif t == int(OptionType.EVOLVE):
            s = W["evolve"]
        elif t == int(OptionType.ABILITY):
            card = _get(obs, o.area, o.index, me)
            s = -1 if (card and card.id == LUNATONE and low_deck) else W["ability"]
        elif t == int(OptionType.ATTACH):
            pk = _get(obs, o.inPlayArea, o.inPlayIndex, me)
            s = W["attach"]
            if pk is not None and _is_main_attacker(pk.id):
                s += W["energy_need"]
            card = _get(obs, AreaType.HAND, o.index, me)
            if card is not None and card.id == HERO_CAPE:
                s = 7000 + (200 if (pk and pk.id == MEGA_LUCARIO) else 0)
        elif t == int(OptionType.PLAY):
            card = _get(obs, AreaType.HAND, o.index, me)
            if card is None:
                scores.append(0); continue
            d = _CARD.get(card.id)
            if d and d.cardType == CardType.POKEMON:
                s = W["play_pokemon"]
                if card.id in (LUNATONE, SOLROCK) and field.get(card.id, 0) >= 1:
                    s = -1
                elif card.id == RIOLU and field.get(RIOLU, 0) + field.get(MEGA_LUCARIO, 0) >= 2:
                    s = -1
            elif card.id in (DUSK_BALL, POKE_PAD):
                s = 10000
            elif card.id == BOSS_ORDERS:
                s = W["boss"] if op.bench else -1
            elif card.id == SWITCH:
                s = -1
            elif card.id in (CARMINE, LILLIE):
                s = -1 if low_deck else W["draw"]
            elif card.id == GRAVITY_MTN:
                s = -1 if any(c.id == GRAVITY_MTN for c in st.stadium) else 5000
            else:
                s = 4000
        elif t == int(OptionType.RETREAT):
            s = -1
        elif t == int(OptionType.END):
            s = W["end"]
        scores.append(s)
    return scores


def _score_context(obs):
    """Score options for non-MAIN contexts (setup / search / discard / etc.)."""
    st, sel = obs.current, obs.select
    me = st.yourIndex
    ctx = sel.context
    my = st.players[me]
    field = {}
    hand = {}
    for p in (my.active + list(my.bench)):
        if p:
            field[p.id] = field.get(p.id, 0) + 1
    for c in (my.hand or []):
        hand[c.id] = hand.get(c.id, 0) + 1

    scores = []
    for o in sel.option:
        s = 0.0
        if int(o.type) == int(OptionType.NUMBER):
            s = o.number or 0
        elif int(o.type) == int(OptionType.YES):
            s = 1
        else:
            card = _get(obs, o.area, o.index, getattr(o, "playerIndex", me))
            if card is not None:
                cid = card.id
                if ctx in (SelectContext.SETUP_ACTIVE_POKEMON,):
                    s = {SOLROCK: 4, RIOLU: 3, MAKUHITA: 1}.get(cid, 0)
                elif ctx in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH):
                    if cid == RIOLU:
                        s = 120 - 25 * field.get(RIOLU, 0)
                    elif cid == SOLROCK:
                        s = 90 if field.get(SOLROCK, 0) == 0 else -1
                    elif cid == LUNATONE:
                        s = 80 if field.get(LUNATONE, 0) == 0 else -1
                    elif cid == MAKUHITA:
                        s = 65 if field.get(MAKUHITA, 0) == 0 else 10
                elif ctx == SelectContext.TO_HAND:          # search target
                    s = 200 - hand.get(cid, 0) * 100
                    if cid == RIOLU and field.get(RIOLU, 0) == 0:
                        s += 40
                elif ctx == SelectContext.DISCARD:
                    s = 70 if hand.get(cid, 0) >= 2 else 5
                    if cid in (RIOLU, MAKUHITA, BOSS_ORDERS, HERO_CAPE, MEGA_LUCARIO, HARIYAMA):
                        s = -40
                elif ctx in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
                    if isinstance(card, Pokemon):
                        s = (10000 + _prize_count(card) * 1000 - card.hp) \
                            if getattr(o, "playerIndex", me) != me else -card.hp
                else:
                    s = 1
        scores.append(s)
    return scores


def _legal_fallback(sel):
    n = len(sel.option)
    k = min(max(1, sel.minCount), n) if n else 0
    return list(range(k))


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
