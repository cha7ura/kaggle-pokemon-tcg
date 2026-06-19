"""Lucario heuristic + forward-model search (our edge: the LB960 author shipped
search OFF; our full-signature search_begin works).

At a MAIN decision we take the heuristic's top-K opening moves, and for each we
roll the rest of the turn out using the SAME heuristic inside the forward model,
then score the end-of-turn board with a Lucario-aware eval. We commit to the
opening move whose heuristic-played turn ends best. Everything is crash-safe:
any search error falls back to the pure heuristic.
"""
import os

from cg.api import (to_observation_class, OptionType, SelectContext,
                    search_begin, search_step, search_end)
import agent_lucario as base

TOP_K = 4          # how many heuristic-ranked opening moves to roll out
ROLLOUT_CAP = 30
BASIC_POKEMON_ID = base.RIOLU
ENERGY_ID = base.FIGHTING_ENERGY


def _determinize(state):
    me = state.yourIndex
    my, opp = state.players[me], state.players[1 - me]
    opp_active = [BASIC_POKEMON_ID] if (opp.active and opp.active[0] is None) else []
    return (base._DECK[:], [ENERGY_ID] * len(my.prize), base._DECK[:],
            [ENERGY_ID] * len(opp.prize), [ENERGY_ID] * opp.handCount, opp_active)


def _eval(state, me):
    if state is None:
        return -1e9
    if state.result == me:
        return 1e9
    if state.result == (1 - me):
        return -1e9
    if state.result == 2:
        return 0.0
    my, opp = state.players[me], state.players[1 - me]
    v = 1000.0 * (len(opp.prize) - len(my.prize))          # prize race dominates
    for p in ([my.active[0]] if (my.active and my.active[0]) else []) + [b for b in my.bench if b]:
        v += len(p.energies) * 8.0
        if p.id in (base.MEGA_LUCARIO, base.HARIYAMA):
            v += 20.0                                       # reward a set-up attacker
    if opp.active and opp.active[0]:
        v -= opp.active[0].hp / 8.0                         # pressure on their active
    if my.active and my.active[0]:
        v += my.active[0].hp / 20.0
    return v


def _rollout(sbi_obs, me, first):
    ss = search_begin(sbi_obs, *_determinize(sbi_obs.current))
    sid = ss.searchId
    try:
        ss = search_step(sid, first)
        for _ in range(ROLLOUT_CAP):
            cur = ss.observation.current
            if cur is None or cur.result != -1 or cur.yourIndex != me:
                break
            sel = ss.observation.select
            if sel is None:
                break
            ss = search_step(sid, base.select_indices(ss.observation))
        return _eval(ss.observation.current, me)
    finally:
        search_end()


def _search_plan(obs):
    sel = obs.select
    if obs.current is None or int(sel.type) != 0 or sel.maxCount != 1 or sel.minCount != 1:
        return None
    scores = base._score_main(obs)
    order = sorted(range(len(sel.option)), key=lambda i: scores[i], reverse=True)
    # candidates: heuristic's top-K, excluding clearly-bad (negative) and END
    cands = [i for i in order if scores[i] > 0
             and int(sel.option[i].type) != int(OptionType.END)][:TOP_K]
    if len(cands) < 2:
        return None                                         # nothing to decide -> heuristic
    me = obs.current.yourIndex
    best_i, best_v = cands[0], -1e18
    for i in cands:
        try:
            v = _rollout(obs, me, [i])
        except Exception:
            v = -1e17
        if v > best_v:
            best_v, best_i = v, i
    return [best_i]


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return base._DECK
        plan = _search_plan(obs)
        if plan is not None:
            return plan
        return base.select_indices(obs)
    except Exception:
        return base.agent(obs_dict)
