"""Inference-time ISMCTS over the engine forward model, heuristic as rollout policy.

Fixes the two faults of the rejected agent_search (see log): (a) one search_begin per simulation
with UCB allocating sims to promising root actions, (b) rollouts play BOTH seats with the strong
Lucario heuristic to a depth (not my-turn-only), so the value reflects the opponent's reply.
Only engages at single-choice MAIN decisions; everything else defers to the heuristic.
"""
import math
from cg.api import (to_observation_class, OptionType, SelectType,
                    search_begin, search_step, search_end)
import agent_lucario as H

ENERGY_ID = 3
BASIC = 721


def _determinize(state):
    me = state.yourIndex
    my, opp = state.players[me], state.players[1 - me]
    oa = [BASIC] if (opp.active and opp.active[0] is None) else []
    return (H._DECK[:], [ENERGY_ID] * len(my.prize), H._DECK[:],
            [ENERGY_ID] * len(opp.prize), [ENERGY_ID] * opp.handCount, oa)


def _value(state, me):
    if state is None:
        return 0.0
    if state.result == me:
        return 1.0
    if state.result == (1 - me):
        return -1.0
    if state.result == 2:
        return 0.0
    my, op = state.players[me], state.players[1 - me]
    return 0.1 * (len(op.prize) - len(my.prize))   # prize lead, lightly weighted


def _rollout(obs, first_action, me, depth):
    ss = search_begin(obs, *_determinize(obs.current))
    sid = ss.searchId
    try:
        ss = search_step(sid, first_action)
        for _ in range(depth):
            cur = ss.observation.current
            if cur is None or cur.result != -1:
                break
            sel = ss.observation.select
            if sel is None:
                break
            ss = search_step(sid, H.select_indices(ss.observation))
        return _value(ss.observation.current, me)
    finally:
        search_end()


def _ismcts(obs, sims=48, depth=12):
    sel = obs.select
    me = obs.current.yourIndex
    cands = [i for i in range(len(sel.option)) if int(sel.option[i].type) != int(OptionType.END)]
    if len(cands) <= 1:
        return H.select_indices(obs)
    N = {i: 0 for i in cands}
    Q = {i: 0.0 for i in cands}
    for t in range(sims):
        pick = None
        for i in cands:
            if N[i] == 0:
                pick = i
                break
        if pick is None:
            pick = max(cands, key=lambda i: Q[i] / N[i] + 1.4 * math.sqrt(math.log(t + 1) / N[i]))
        try:
            v = _rollout(obs, [pick], me, depth)
        except Exception:
            v = -1.0
        N[pick] += 1
        Q[pick] += v
    return [max(cands, key=lambda i: (Q[i] / N[i]) if N[i] else -1e9)]


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return H._DECK
        sel = obs.select
        if int(sel.type) == int(SelectType.MAIN) and sel.maxCount == 1 and sel.minCount == 1:
            return _ismcts(obs)
        return H.select_indices(obs)
    except Exception:
        try:
            return H._legal_fallback(to_observation_class(obs_dict).select)
        except Exception:
            return [0]
