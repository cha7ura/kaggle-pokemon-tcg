"""Tabular R-NaD (Regularized Nash Dynamics) on Kuhn poker (wiki/12-deepnash-rnad.md).

DeepNash's idea, shrunk to a toy: plain self-play in a zero-sum game *cycles* around Nash. R-NaD
adds a regularization that pulls the policy toward a reference policy, which damps the cycling so
the dynamics CONVERGE; you then move the reference toward the current policy and repeat. The
LAST iterate converges (no averaging — unlike CFR, see wiki/10), which is exactly why a trained
R-NaD net ships as a plain forward pass.

Validated on Kuhn because we already have a trusted exploitability oracle there
(minizero/cfr/exploit.py). Leduc is the spec's eventual target; the algorithm is identical.

Update (regularized mirror ascent on counterfactual values):
    pi <- softmax( log pi + lr * ( q  -  eta * (log pi - log pi_ref) ) )
Outer loop: pi_ref <- pi every `inner` steps.
"""
import itertools

import numpy as np

from minizero.cfr.kuhn import ACTIONS, is_terminal, payoff, infoset_key

NA = len(ACTIONS)
_DEALS = [list(c) for c in itertools.permutations(range(3), 2)]
_EPS = 1e-12


def _all_infosets():
    keys = []
    for c in range(3):
        for h in ("", "pb"):            # player 0 acts
            keys.append(infoset_key(c, h))
        for h in ("p", "b"):            # player 1 acts
            keys.append(infoset_key(c, h))
    return keys


def cf_action_values(policy):
    """Counterfactual action values q[I][a] under `policy` (current player's perspective)."""
    q = {}

    def rec(cards, history, p0, p1):
        if is_terminal(history):
            u = payoff(cards, history)
            return u if (len(history) % 2 == 0) else -u
        player = len(history) % 2
        key = infoset_key(cards[player], history)
        strat = policy[key]
        util = np.zeros(NA)
        for a, act in enumerate(ACTIONS):
            if player == 0:
                util[a] = -rec(cards, history + act, p0 * strat[a], p1)
            else:
                util[a] = -rec(cards, history + act, p0, p1 * strat[a])
        cf_reach = p1 if player == 0 else p0
        q.setdefault(key, np.zeros(NA))
        q[key] += cf_reach * util
        return float(strat @ util)

    for cards in _DEALS:
        rec(cards, "", 1.0, 1.0)
    return q


def _softmax(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def train_rnad(outer=120, inner=60, lr=0.5, eta=0.5):
    # eta must be strong enough (and `inner` long enough) for the inner loop to reach the
    # regularized fixed point before the reference updates — weak regularization never converges.
    keys = _all_infosets()
    pi = {k: np.full(NA, 1.0 / NA) for k in keys}
    for _ in range(outer):
        ref = {k: v.copy() for k, v in pi.items()}
        log_ref = {k: np.log(v + _EPS) for k, v in ref.items()}
        for _ in range(inner):
            q = cf_action_values(pi)
            for k in keys:
                log_pi = np.log(pi[k] + _EPS)
                logits = log_pi + lr * (q[k] - eta * (log_pi - log_ref[k]))
                pi[k] = _softmax(logits)
    return pi
