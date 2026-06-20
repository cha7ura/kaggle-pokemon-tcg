"""External-sampling MCCFR on Kuhn (wiki/10-cfr-and-nash.md).

Cheaper than vanilla CFR: instead of traversing the whole tree, sample chance (the deal) and
the opponent's actions, and traverse only the traverser's own actions. Regrets get no reach
weighting (the sampling handles it). Same Nash, fewer touches per iteration — the pattern that
scales to big games. Single-process; parallelism (merge regret deltas across workers) is a
documented extension, omitted here because Kuhn converges in well under a second.
"""
import random

import numpy as np

from minizero.cfr.kuhn import ACTIONS, is_terminal, payoff, infoset_key

NA = len(ACTIONS)
_DEALS = [list(c) for c in __import__("itertools").permutations(range(3), 2)]


def _regret_match(regret):
    pos = np.maximum(regret, 0.0)
    s = pos.sum()
    return pos / s if s > 0 else np.full(NA, 1.0 / NA)


def _es(cards, history, traverser, regret_sum, strat_sum, rng):
    if is_terminal(history):
        u = payoff(cards, history)
        return u if traverser == 0 else -u
    player = len(history) % 2
    key = infoset_key(cards[player], history)
    regret_sum.setdefault(key, np.zeros(NA))
    strat_sum.setdefault(key, np.zeros(NA))
    strategy = _regret_match(regret_sum[key])

    if player == traverser:
        util = np.array([_es(cards, history + a, traverser, regret_sum, strat_sum, rng)
                         for a in ACTIONS])
        node = float(strategy @ util)
        regret_sum[key] += util - node          # external sampling: no reach weight
        return node
    # opponent node: accumulate average strategy, sample one action
    strat_sum[key] += strategy
    a = ACTIONS[int(rng.random() >= strategy[0])]  # sample p vs b from the distribution
    return _es(cards, history + a, traverser, regret_sum, strat_sum, rng)


def train_mccfr(iterations: int, seed: int = 0):
    rng = random.Random(seed)
    regret_sum, strat_sum = {}, {}
    for _ in range(iterations):
        cards = list(rng.choice(_DEALS))
        for traverser in (0, 1):
            _es(cards, "", traverser, regret_sum, strat_sum, rng)
    avg = {}
    for key, s in strat_sum.items():
        total = s.sum()
        avg[key] = s / total if total > 0 else np.full(NA, 1.0 / NA)
    return avg
