"""Grimmsnarl ex (Dark) heuristic pilot — deck-aware policy conditioned on hand + board.

Game plan (from the deck's real mechanics, mined from replays):
  - Attacker: Marnie's Grimmsnarl ex (648, Stage2, 320hp, ex). Attack Shadow Bullet = 180 dmg for
    2 Dark energy (from card data: '180/2E:DAR+DAR'); replay steady-state also clusters at 2 energy
    while active (26190/28510 obs). Charge to 2, no more (over-attaching wastes tempo).
  - Line: Marnie's Impidimp (646 basic) -> Morgrem (647 s1) -> Grimmsnarl ex (648 s2).
    Rare Candy (1079) skips Morgrem: Impidimp -> Grimmsnarl ex directly.
  - Support: Munkidori (112, basic Psychic) ability moves damage counters (snipe / preserve).
    Froslass (104, s1 from Snorunt 860) chip line. Spikemuth Gym (1259) stadium.
  - Trainers: Buddy-Buddy Poffin (1086) fetch basics; Night Stretcher (1097) recover;
    Boss's Orders (1182) drag target; Poke Pad (1152) draw; Lillie's Determination (1227),
    Team Rocket's Petrel (1219), Dawn (1231) supporters; Unfair Stamp (1080) disrupt.

Crash-safe wrapper (same scaffold as agent_lucario): never throws, always returns legal indices.
Pairs with decks/grimmsnarl.csv.
"""
import os
from cg.api import (AreaType, CardType, EnergyType, SelectContext, OptionType,
                    Pokemon, all_card_data, to_observation_class)

# ---- deck card IDs ----
IMPIDIMP, MORGREM, GRIMM_EX = 646, 647, 648
MUNKIDORI, SNORUNT, FROSLASS = 112, 860, 104  # Snorunt=860 basic, Froslass=104 stage1
DARK_ENERGY = 7
POKE_PAD, BUDDY_POFFIN, RARE_CANDY, NIGHT_STRETCHER = 1152, 1086, 1079, 1097
LILLIE_DET, PETREL, SPIKEMUTH, BOSS_ORDERS = 1227, 1219, 1259, 1182
HANDHELD_FAN, UNFAIR_STAMP, DAWN = 1161, 1080, 1231
ATTACK_COST = 2  # Shadow Bullet costs 2 Dark (card data 180/2E:DAR+DAR)

LOW_DECK_COUNT = 8

DECK_PATH = os.environ.get("AGENT_DECK", "decks/grimmsnarl.csv")
def read_deck(path=None):
    for p in ([path] if path else []) + ["deck.csv", "/kaggle_simulations/agent/deck.csv", DECK_PATH]:
        if p and os.path.exists(p):
            with open(p) as f:
                return [int(line) for line in f if line.strip()][:60]
    raise FileNotFoundError("deck.csv not found")
_DECK = read_deck()

_ALLC = {c.cardId: c for c in all_card_data()}
_ATKDMG = {a.attackId: a.damage for a in __import__('cg.api', fromlist=['all_attack']).all_attack()}
_ATKCOST = {a.attackId: len(a.energies or []) for a in __import__('cg.api', fromlist=['all_attack']).all_attack()}

def _eff_dmg(base, attacker_id, defender):
    a, d = _ALLC.get(attacker_id), _ALLC.get(getattr(defender, 'id', None))
    dmg = base
    if a and d and d.weakness is not None and a.energyType is not None and int(d.weakness) == int(a.energyType):
        dmg *= 2
    if a and d and d.resistance is not None and a.energyType is not None and int(d.resistance) == int(a.energyType):
        dmg -= 30
    return max(0, dmg)

def _best_affordable_dmg(attacker_pk, defender_pk):
    """Max damage attacker can deal to defender using attacks it can currently afford
    (energy_have >= attack cost). Returns (dmg, energy_short_for_best) where energy_short
    is how many more energy needed for the biggest attack if not yet affordable."""
    cd = _ALLC.get(attacker_pk.id)
    if not cd:
        return 0, 99
    have = len(attacker_pk.energies or [])
    best_afford = 0
    for aid in (cd.attacks or []):
        cost = _ATKCOST.get(aid, 99)
        base = _ATKDMG.get(aid, 0)
        dmg = _eff_dmg(base, attacker_pk.id, defender_pk)
        if have >= cost:
            best_afford = max(best_afford, dmg)
    return best_afford, 0

def _opp_threat(op, my_active):
    """Does opp's active KO our active THIS turn (already can attack)? one-attach/turn => their
    energy grows <=1/turn, so 'imminent' = can afford now."""
    oa = op.active[0] if op.active else None
    if not (oa and my_active):
        return {'can_attack': False, 'will_ko': False, 'turns': 99}
    cd = _ALLC.get(oa.id)
    if not cd or not (cd.attacks):
        return {'can_attack': False, 'will_ko': False, 'turns': 99}
    have = len(oa.energies or [])
    min_cost = min((_ATKCOST.get(aid, 99) for aid in cd.attacks), default=99)
    can_attack = have >= min_cost
    dmg, _ = _best_affordable_dmg(oa, my_active)
    will_ko = can_attack and dmg >= getattr(my_active, 'hp', 999)
    return {'can_attack': can_attack, 'will_ko': will_ko, 'turns': max(0, min_cost - have)}

W = {
    "ability": 12000, "evolve": 9000, "play_pokemon": 7000, "attach": 6000,
    "draw": 3000, "boss": 3200, "stadium": 2500, "attack": 1000,
    "ko_win": 50000, "retreat": 500, "end": -1,
}

def set_weights(overrides):
    """Update the weight table in place (used by the tuning harness)."""
    if overrides:
        W.update({k: float(v) for k, v in overrides.items() if k in W})

# optional env-based override for one-shot runs
_wjson = os.environ.get("AGENT_W_JSON")
if _wjson and os.path.exists(_wjson):
    try:
        import json as _json
        set_weights(_json.load(open(_wjson)))
    except Exception:
        pass

# --- opponent archetype inference (line-based, from revealed cards) + matchup posture ---
# Grimmsnarl's recent matchup win-rates (from RECENT_META_MATRIX): >50 = favorable, <50 = race harder
_GRIMM_MATCHUP = {
    "HopTrev": 93, "MegaStarmie": 75, "OTHER": 72, "ToolboxEx": 68, "Archaludon": 60,
    "Alakazam": 57, "Grimmsnarl": 50, "SolrockLun": 41, "Chandelure": 37,
}
def _classify_opp(op):
    """Infer opponent archetype from any revealed cards (active+bench+discard). None if too few."""
    ids = set()
    for z in ((op.active or []) + list(op.bench or []) + list(op.discard or [])):
        cid = getattr(z, "id", None)
        if cid is not None:
            ids.add(cid)
    if len(ids) < 2:
        return None
    if 646 in ids or 648 in ids or 647 in ids: return "Grimmsnarl"
    if 878 in ids or 879 in ids: return "HopTrev"
    if 675 in ids or 676 in ids: return "SolrockLun"
    if 97 in ids or 98 in ids: return "Chandelure"
    if 1031 in ids: return "MegaStarmie"
    if 190 in ids or 169 in ids: return "Archaludon"
    if 743 in ids or 742 in ids: return "Alakazam"
    if 756 in ids or 272 in ids or 184 in ids: return "ToolboxEx"
    return "OTHER"

def _posture(op):
    """Return 'race' (bad matchup -> attack sooner, take risks) or 'develop' (favorable -> build)."""
    arch = _classify_opp(op)
    if arch is None:
        return "develop", None
    wr = _GRIMM_MATCHUP.get(arch, 50)
    return ("race" if wr < 48 else "develop"), arch

def _get(obs, area, index, pidx):
    try:
        ps = obs.current.players[pidx]
        return {AreaType.HAND: ps.hand, AreaType.DISCARD: ps.discard, AreaType.ACTIVE: ps.active,
                AreaType.BENCH: ps.bench, AreaType.PRIZE: ps.prize}[area][index]
    except Exception:
        try:
            if area == AreaType.DECK: return obs.select.deck[index]
            if area == AreaType.STADIUM: return obs.current.stadium[index]
        except Exception:
            return None
    return None

def _prize_count(pk):
    d = _ALLC.get(pk.id)
    return 3 if (d and getattr(d, "megaEx", False)) else 2 if (d and getattr(d, "ex", False)) else 1

def _is_attacker(cid):
    return cid in (GRIMM_EX, FROSLASS)

def _energy_ct(pk):
    return len(pk.energies or []) if pk else 0

def _score_main(obs):
    st, sel = obs.current, obs.select
    me = st.yourIndex
    my, op = st.players[me], st.players[1 - me]
    low_deck = getattr(my, "deckCount", 999) <= LOW_DECK_COUNT
    opp_active = op.active[0] if op.active else None
    # board census
    field = {}
    for p in (my.active + list(my.bench)):
        if p: field[p.id] = field.get(p.id, 0) + 1
    has_grimm_ready = any(p and p.id == GRIMM_EX and _energy_ct(p) >= ATTACK_COST
                          for p in (my.active + list(my.bench)))
    my_active = my.active[0] if my.active else None
    hand_ids = {c.id for c in (my.hand or [])}
    threat = _opp_threat(op, my_active)
    posture, _opp_arch = _posture(op)
    # a benched Pokemon that would SURVIVE the incoming hit (best wall), least prize value first
    def _survivor_bench():
        best = None
        for i, b in enumerate(my.bench):
            if not b:
                continue
            dmg, _ = _best_affordable_dmg(op.active[0], b) if op.active else (0, 0)
            survives = dmg < getattr(b, 'hp', 0)
            if survives:
                # prefer low-prize, already-charged
                key = (_prize_count(b), -_energy_ct(b))
                if best is None or key < best[0]:
                    best = (key, i)
        return best[1] if best else None
    scores = []
    for o in sel.option:
        s = 0.0
        t = int(o.type)
        if t == int(OptionType.ATTACK):
            s = W["attack"] + (2500 if posture == "race" else 0)
            if opp_active is not None:
                # KO check: is opp active HP <= our damage? we approximate high dmg from GrimmEx.
                # Grimmsnarl ex hits hard; if opp low HP, prioritize KO.
                if getattr(opp_active, "hp", 999) <= 130:
                    s = W["ko_win"] if len(op.prize) <= _prize_count(opp_active) else W["attack"] + 8000
                else:
                    s = W["attack"] + (999 - getattr(opp_active, "hp", 0))
        elif t == int(OptionType.ABILITY):
            # Munkidori damage-move: valuable when opp has a KO-able target (move dmg to finish)
            card = _get(obs, o.area, o.index, me)
            s = W["ability"]
            if card and card.id == MUNKIDORI:
                # move damage onto opp active if it sets up a KO; else lower priority
                s = W["ability"] + (2000 if opp_active and getattr(opp_active, "hp", 999) <= 90 else 0)
        elif t == int(OptionType.EVOLVE):
            s = W["evolve"]
            # evolving toward Grimmsnarl ex is the whole plan -> top of develop order
            card = _get(obs, o.area, o.index, me)
            if card and card.id in (MORGREM, GRIMM_EX):
                s = W["evolve"] + 500
        elif t == int(OptionType.ATTACH):
            pk = _get(obs, o.inPlayArea, o.inPlayIndex, me)
            s = W["attach"]
            if pk is not None and _is_attacker(pk.id):
                if _energy_ct(pk) < ATTACK_COST:
                    s += 2000            # charge an attacker to 2
                else:
                    s -= 3000            # already ready -> don't over-attach, start a 2nd
        elif t == int(OptionType.PLAY):
            card = _get(obs, AreaType.HAND, o.index, me)
            if card is None:
                scores.append(0); continue
            d = _ALLC.get(card.id)
            cid = card.id
            if d and d.cardType == CardType.POKEMON:
                s = W["play_pokemon"]
                # bench basics we need: Impidimp (attacker line) + Munkidori (support)
                if cid == IMPIDIMP:
                    s = W["play_pokemon"] + (300 if field.get(IMPIDIMP, 0) + field.get(MORGREM,0) + field.get(GRIMM_EX,0) < 2 else -50)
                elif cid == MUNKIDORI:
                    s = W["play_pokemon"] + (150 if field.get(MUNKIDORI, 0) == 0 else -50)
                elif cid == SNORUNT:
                    s = W["play_pokemon"] - 20
            elif cid == RARE_CANDY:
                # skip Morgrem: only if Grimmsnarl ex is in hand AND an Impidimp is in play
                s = 11000 if (field.get(IMPIDIMP, 0) >= 1 and GRIMM_EX in hand_ids) else -1
            elif cid == BUDDY_POFFIN:
                s = 10500            # fetch basics -> highest early
            elif cid == NIGHT_STRETCHER:
                s = 4000
            elif cid == POKE_PAD:
                s = 9000             # draw item
            elif cid == BOSS_ORDERS:
                s = W["boss"] if op.bench else -1
            elif cid == SPIKEMUTH:
                s = W["stadium"] if not any(c.id == SPIKEMUTH for c in st.stadium) else -1
            elif cid in (LILLIE_DET, PETREL, DAWN):
                s = -1 if low_deck else W["draw"]
            elif cid == UNFAIR_STAMP:
                s = 2000
            else:
                s = 4000
        elif t == int(OptionType.RETREAT):
            s = W["retreat"]
            # if our active is about to be KO'd and we have a bench survivor, retreat to deny the KO
            if threat["will_ko"] and _survivor_bench() is not None and my_active is not None:
                # only worth it if the active is valuable (an ex we'd lose 2-3 prizes on)
                if _prize_count(my_active) >= 2:
                    s = 15000
        elif t == int(OptionType.END):
            s = W["end"]
        scores.append(s)
    return scores

def _score_context(obs):
    st, sel = obs.current, obs.select
    me = st.yourIndex
    ctx = sel.context
    my = st.players[me]; op = st.players[1 - me]
    field = {}; hand = {}
    for p in (my.active + list(my.bench)):
        if p: field[p.id] = field.get(p.id, 0) + 1
    for c in (my.hand or []): hand[c.id] = hand.get(c.id, 0) + 1
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
                    # open with Impidimp (our attacker line); Munkidori/Snorunt otherwise
                    s = {IMPIDIMP: 5, MUNKIDORI: 2, SNORUNT: 1}.get(cid, 0)
                elif ctx in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH):
                    if cid == IMPIDIMP:
                        s = 120 - 25 * field.get(IMPIDIMP, 0)
                    elif cid == MUNKIDORI:
                        s = 90 if field.get(MUNKIDORI, 0) == 0 else 10
                    elif cid == SNORUNT:
                        s = 60 if field.get(SNORUNT, 0) == 0 else 5
                    else:
                        s = 20
                elif ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                    if getattr(o, "playerIndex", me) == me and isinstance(card, Pokemon):
                        en = _energy_ct(card)
                        s = (200 + en) if cid == GRIMM_EX else (50 + en)
                elif ctx == SelectContext.TO_HAND:              # search target
                    # prioritize the evolve line + energy we lack
                    pri = {GRIMM_EX: 250, IMPIDIMP: 200, MORGREM: 180, RARE_CANDY: 170,
                           MUNKIDORI: 120, DARK_ENERGY: 110, BUDDY_POFFIN: 90}.get(cid, 60)
                    s = pri - hand.get(cid, 0) * 40
                elif ctx == SelectContext.DISCARD:
                    # pitch duplicates / least-useful; protect the plan pieces
                    s = 70 if hand.get(cid, 0) >= 2 else 5
                    if cid in (GRIMM_EX, IMPIDIMP, MORGREM, RARE_CANDY, MUNKIDORI, BOSS_ORDERS):
                        s = -40
                elif ctx in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
                    # Munkidori: put damage on opp's highest-prize KO-able target
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

def select_indices(obs):
    sel = obs.select
    scores = _score_main(obs) if int(sel.type) == 0 else _score_context(obs)
    order = sorted(range(len(sel.option)), key=lambda i: scores[i], reverse=True)
    k = min(sel.maxCount, len(sel.option))
    k = max(k, min(max(1, sel.minCount), len(sel.option)))
    return order[:k] if order else _legal_fallback(sel)

def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return _DECK
        return select_indices(obs)
    except Exception:
        try:
            return _legal_fallback(to_observation_class(obs_dict).select)
        except Exception:
            return [0]
