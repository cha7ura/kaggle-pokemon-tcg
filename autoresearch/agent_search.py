"""Search agent v1: pick each turn's actions by greedy turn-rollout + board eval.

For a MAIN single-choice decision, try every non-END opening action, then roll the
rest of MY turn out greedily inside the forward model (search_begin/search_step),
and evaluate the END-OF-TURN board. Pick the opening action with the best outcome.
Rolling out avoids the exp001 "attack too early" trap: attaching energy scores well
because the rollout then attacks with it.

Non-MAIN / multi-select decisions fall back to the develop-first priority greedy.
"""
import os

from cg.api import (to_observation_class, OptionType, SelectType,
                    search_begin, search_step, search_end)

DECK_PATH = os.environ.get("AGENT_DECK", "decks/champion.csv")
BASIC_POKEMON_ID = 721
ENERGY_ID = 3
ROLLOUT_CAP = 24

PRIORITY = {
    int(OptionType.EVOLVE): 0, int(OptionType.ABILITY): 1,
    int(OptionType.ATTACH): 2, int(OptionType.ENERGY): 2, int(OptionType.ENERGY_CARD): 2,
    int(OptionType.PLAY): 3, int(OptionType.TOOL_CARD): 4, int(OptionType.CARD): 5,
    int(OptionType.YES): 6, int(OptionType.NUMBER): 6, int(OptionType.SPECIAL_CONDITION): 6,
    int(OptionType.ATTACK): 7, int(OptionType.NO): 8, int(OptionType.DISCARD): 9,
    int(OptionType.RETREAT): 10, int(OptionType.END): 11,
}


def read_deck(path=DECK_PATH):
    with open(path) as f:
        return [int(line) for line in f if line.strip()][:60]


_DECK = read_deck()


def _greedy_indices(sel):
    opts = sel.option
    order = sorted(range(len(opts)), key=lambda i: PRIORITY.get(int(opts[i].type), 6))
    k = min(max(sel.minCount, 1), sel.maxCount) if sel.maxCount >= 1 else sel.minCount
    chosen = order[:k]
    if len(chosen) < sel.minCount:
        chosen = order[:sel.minCount]
    return chosen


def _determinize(state):
    me = state.yourIndex
    my, opp = state.players[me], state.players[1 - me]
    opp_active = [BASIC_POKEMON_ID] if (opp.active and opp.active[0] is None) else []
    return (_DECK[:], [ENERGY_ID] * len(my.prize), _DECK[:],
            [ENERGY_ID] * len(opp.prize), [ENERGY_ID] * opp.handCount, opp_active)


def _board_eval(state, me):
    if state is None:
        return -1e9
    if state.result == me:
        return 1e9
    if state.result == (1 - me):
        return -1e9
    if state.result == 2:
        return 0.0
    my, opp = state.players[me], state.players[1 - me]

    def presence(p):
        mons = ([p.active[0]] if (p.active and p.active[0]) else []) + [b for b in p.bench if b]
        return len(mons), sum(m.hp for m in mons)

    mn, mhp = presence(my)
    on, ohp = presence(opp)
    # prizes: I win when MY prize pile empties -> fewer of my prizes left is better
    return (100.0 * (len(opp.prize) - len(my.prize))
            + 10.0 * (mn - on)
            + (mhp - ohp) / 10.0)


def _rollout_value(obs, first_action, me):
    """Apply first_action, then greedily finish my turn, and eval the result."""
    ss = search_begin(obs, *_determinize(obs.current))
    sid = ss.searchId
    try:
        ss = search_step(sid, first_action)
        for _ in range(ROLLOUT_CAP):
            cur = ss.observation.current
            if cur is None or cur.result != -1 or cur.yourIndex != me:
                break
            sel = ss.observation.select
            if sel is None:
                break
            ss = search_step(sid, _greedy_indices(sel))
        return _board_eval(ss.observation.current, me)
    finally:
        search_end()


def choose(obs):
    sel = obs.select
    opts = sel.option
    # only search clean single-choice MAIN decisions; else develop-greedy
    if obs.current is None or sel.maxCount != 1 or sel.minCount != 1 \
            or int(sel.type) != int(SelectType.MAIN):
        return _greedy_indices(sel)

    me = obs.current.yourIndex
    candidates = [i for i in range(len(opts)) if int(opts[i].type) != int(OptionType.END)]
    if not candidates:
        return _greedy_indices(sel)

    best_i, best_v = None, -1e18
    for i in candidates:
        try:
            v = _rollout_value(obs, [i], me)
        except Exception:
            v = -1e17
        if v > best_v:
            best_v, best_i = v, i
    return [best_i] if best_i is not None else _greedy_indices(sel)


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck()
    return choose(obs)
