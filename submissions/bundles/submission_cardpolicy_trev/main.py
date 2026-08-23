"""Kaggle agent: Trevenant (proven 928 deck) piloted by the learned CardPolicy (XGBoost over
mechanics+trajectory+setup features). Validated vs the typh baseline: Trevenant 0.625, Alakazam 0.708.

Contract: agent(obs) -> list[int]; deck phase -> 60 card ids; never crash (legal fallback).
Robust to exec() where __file__ is undefined (the v16/v18 crash cause) and to the Kaggle path.
"""
import os, sys

# locate the agent dir (where deck.csv + cardpol/ live) without relying on __file__
_cands = ["/kaggle_simulations/agent", os.getcwd()]
try:
    _cands.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
AGENT_DIR = next((d for d in _cands if os.path.exists(os.path.join(d, "deck.csv"))), _cands[-1])
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

with open(os.path.join(AGENT_DIR, "deck.csv")) as f:
    DECK = [int(x) for x in f if x.strip().lstrip("-").isdigit()][:60]

_POL = None
try:
    from cardpol.gameplay_policy import CardPolicy
    _POL = CardPolicy(DECK)
except Exception:
    _POL = None     # fall back to safe legal moves if the model/deps fail to load


def _fallback(obs):
    sel = (obs or {}).get("select")
    if not sel:
        return list(DECK)
    opts = sel.get("option") or []
    for i, o in enumerate(opts):
        if (o or {}).get("type") == 14:        # END / pass
            return [i]
    mn = max(1, sel.get("minCount", 1) or 1)
    return list(range(min(mn, len(opts)))) if opts else []


def agent(obs):
    try:
        if _POL is None:
            return _fallback(obs)
        return _POL.act(obs)
    except Exception:
        return _fallback(obs)
