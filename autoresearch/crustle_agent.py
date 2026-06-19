"""Simple Crustle-wall opponent for meta-faithful benchmarking (from LB960).

Develop-first priority; pairs with decks/crustle.csv. Used only as a sparring
partner to measure whether our Lucario agent's Crustle-routing actually works.
"""
import random

from cg.api import to_observation_class, OptionType, SelectContext

_PRI = {int(OptionType.ATTACH): 1000, int(OptionType.EVOLVE): 800,
        int(OptionType.PLAY): 600, int(OptionType.ABILITY): 400,
        int(OptionType.ATTACK): 100, int(OptionType.RETREAT): -1}


def read_deck(path="decks/crustle.csv"):
    with open(path) as f:
        return [int(line) for line in f if line.strip()][:60]


_DECK = read_deck()


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return _DECK
    sel = obs.select
    if int(sel.type) == 0:  # MAIN
        scores = [_PRI.get(int(o.type), 0) for o in sel.option]
    else:
        scores = [1] * len(sel.option)
    order = sorted(range(len(sel.option)), key=lambda i: scores[i], reverse=True)
    k = min(sel.maxCount, len(sel.option))
    k = max(k, min(max(1, sel.minCount), len(sel.option)))
    return order[:k]
