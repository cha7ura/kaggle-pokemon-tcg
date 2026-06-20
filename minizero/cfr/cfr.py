"""Vanilla CFR on Kuhn poker (wiki/10-cfr-and-nash.md).

Regret matching on cumulative counterfactual regrets; the AVERAGE strategy (strategy_sum,
normalized) is what converges to Nash — not the current strategy. That distinction is the
classic CFR gotcha, so train() returns the average.
"""
import itertools

import numpy as np

from minizero.cfr.kuhn import ACTIONS, is_terminal, payoff, infoset_key

NA = len(ACTIONS)


def _regret_match(regret):
    pos = np.maximum(regret, 0.0)
    s = pos.sum()
    return pos / s if s > 0 else np.full(NA, 1.0 / NA)


def _cfr(cards, history, p0, p1, regret_sum, strat_sum):
    if is_terminal(history):
        u = payoff(cards, history)
        return u if (len(history) % 2 == 0) else -u  # value to the player about to act's parent
    player = len(history) % 2
    key = infoset_key(cards[player], history)
    regret_sum.setdefault(key, np.zeros(NA))
    strat_sum.setdefault(key, np.zeros(NA))

    strategy = _regret_match(regret_sum[key])
    reach = p0 if player == 0 else p1
    strat_sum[key] += reach * strategy

    util = np.zeros(NA)
    for a, act in enumerate(ACTIONS):
        if player == 0:
            util[a] = -_cfr(cards, history + act, p0 * strategy[a], p1, regret_sum, strat_sum)
        else:
            util[a] = -_cfr(cards, history + act, p0, p1 * strategy[a], regret_sum, strat_sum)
    node_util = float(strategy @ util)

    cf_reach = p1 if player == 0 else p0
    regret_sum[key] += cf_reach * (util - node_util)
    return node_util


def train(iterations: int):
    """Run CFR; return the average strategy {infoset_key: np.array([p_pass, p_bet])}."""
    regret_sum, strat_sum = {}, {}
    deals = [list(c) for c in itertools.permutations(range(3), 2)]
    for _ in range(iterations):
        for cards in deals:
            _cfr(cards, "", 1.0, 1.0, regret_sum, strat_sum)
    avg = {}
    for key, s in strat_sum.items():
        total = s.sum()
        avg[key] = s / total if total > 0 else np.full(NA, 1.0 / NA)
    return avg
