"""Hop's Trevenant pilot (v12) — the new #1 (Debauchery, 1339) archetype. General develop->attack
rule agent tuned for the Trevenant line: evolve Phantump->Trevenant, load {P}/Mist energy on the
attacker, play draw/search supporters+items, Boss the key target, attack with best affordable
damage; almost never pass-with-resources or retreat. Robust: legal fallback, never crashes.
"""
import os, json
from cg.api import to_observation_class, OptionType, SelectType, all_card_data, all_attack

CD = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: a for a in all_attack()}

PHANTUMP, TREVENANT, CRAMORANT, SNORLAX = 878, 879, 311, 304
ATTACKERS = {TREVENANT, CRAMORANT, SNORLAX}
MIST, PSY = 11, 19
CHOICE_BAND = 1171
BOSS = 1182
# option type ints
NUM, YES, NO, CARD, TOOL, ECARD, ENERGY, PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END = \
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14


def _card_at(my, idx):
    return my.hand[idx] if (my.hand and idx is not None and 0 <= idx < len(my.hand) and my.hand[idx]) else None


def _energy_on(mon):
    return len(mon.energies) if mon else 0


def _affordable_attacks(mon):
    """attacks on `mon` whose energy cost <= attached energy count (cost-count heuristic)."""
    if not mon: return []
    c = CD.get(mon.id)
    if not c: return []
    have = _energy_on(mon)
    out = []
    for aid in c.attacks:
        a = ATK.get(aid)
        if a and len(a.energies or []) <= have:
            out.append((a.damage, aid))
    return out


def _score(o, opt):
    """higher = better. General develop->attack priority, Trevenant-aware."""
    st = o.current; me = st.yourIndex; my = st.players[me]
    t = int(opt.type); idx = getattr(opt, "index", None)
    card = _card_at(my, idx)
    cid = card.id if card else None
    active = my.active[0] if my.active and my.active[0] else None

    if t == ABILITY:
        return 95                                   # free value first
    if t == EVOLVE:
        return 90 if cid in (TREVENANT, CRAMORANT, SNORLAX) else 80
    if t == ATTACH:
        # energy onto the active attacker = top priority for setup
        if active and active.id in ATTACKERS:
            return 88
        return 70
    if t == TOOL:
        return 60                                   # Hop's Choice Band on attacker
    if t == PLAY:
        c = CD.get(cid)
        if c and int(c.cardType) != 0:              # trainer (supporter/item): draw/search
            return 75
        return 55                                   # basic Pokemon to bench (need attackers)
    if t == ENERGY:
        return 86
    if t == ATTACK:
        aid = getattr(opt, "attackId", None); a = ATK.get(aid)
        dmg = a.damage if a else 0
        # only value attacking once it's a real attack; bigger damage better
        return 40 + min(dmg, 300) / 10              # 40..70
    if t == RETREAT:
        return 5
    if t == END:
        return 1
    if t in (YES,):
        return 50
    if t == NO:
        return 30
    return 45                                       # CARD/target/number picks: neutral-ish


def _pick_target(o, sel):
    """for target/card selects (Boss gust, attack target, attach target): sensible default."""
    opts = sel.option
    # prefer opponent active or the highest-value enemy; else first
    return [0]


def agent(obs_dict):
    try:
        o = to_observation_class(obs_dict)
        if o.select is None:
            return _DECK
        sel = o.select
        n = len(sel.option)
        if n == 0:
            return []
        mn = max(1, sel.minCount); mx = min(sel.maxCount, n)
        if int(sel.type) == int(SelectType.MAIN):
            scores = [_score(o, opt) for opt in sel.option]
            order = sorted(range(n), key=lambda i: scores[i], reverse=True)
            k = max(mn, min(mx, 1))
            return order[:k]
        # non-MAIN selects: take the best-scored too (covers target/evolve/energy contexts)
        scores = [_score(o, opt) for opt in sel.option]
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        return order[:max(mn, min(mx, 1))]
    except Exception:
        try:
            sel = to_observation_class(obs_dict).select
            return list(range(min(max(1, sel.minCount), len(sel.option))))
        except Exception:
            return [0]


def _read_deck():
    for p in ("deck.csv", "/kaggle_simulations/agent/deck.csv",
              os.path.join(os.path.dirname(__file__), "decks/trevenant.csv")):
        if os.path.exists(p):
            return [int(x) for x in open(p) if x.strip()][:60]
    return []
_DECK = _read_deck()
