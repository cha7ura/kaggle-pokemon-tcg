"""Generic imitation pilot: pilots any deck via a learned per-deck policy tree. STDLIB ONLY.
Reuses v19 hardening: raw-dict deck-phase check, try/except legal fallback, NO __file__.

Deck via env IMIT_DECK (else deck.csv, else /kaggle_simulations/agent/deck.csv).
Policy via env IMIT_POLICY (else policy.json, else /kaggle_simulations/agent/policy.json).
Packaging note: for a real submission, features.py/policy.py are inlined beside this file
(out of scope here); locally it imports them so there is one shared featurizer (parity)."""
import os, json

from tools.imitation.features import state_features, option_features
from tools.imitation.policy import pick


def _path(env, name):
    p = os.environ.get(env)
    if p:
        return p
    if os.path.exists(name):
        return name
    return f"/kaggle_simulations/agent/{name}"


def _read_deck():
    try:
        with open(_path("IMIT_DECK", "deck.csv")) as f:
            deck = []
            for line in f:
                line = line.strip()
                if line.isdigit():          # skip headers / non-int lines (manifest etc.)
                    deck.append(int(line))
            return deck[:60]
    except Exception:
        return []


def _read_policy():
    try:
        return json.load(open(_path("IMIT_POLICY", "policy.json")))
    except Exception:
        return None


_DECK = _read_deck()
_POLICY = _read_policy()


def _legal_fallback(obs):
    try:
        sel = (obs or {}).get("select") or {}
        n = len(sel.get("option") or [])
        return list(range(min(max(0, sel.get("minCount", 0)), n)))
    except Exception:
        return []


def agent(obs):
    # deck phase: check the RAW dict before touching anything else
    try:
        if not isinstance(obs, dict) or obs.get("select") is None:
            return _DECK
    except Exception:
        return _DECK
    try:
        sel = obs["select"]
        opts = sel.get("option") or []
        if not opts:
            return []
        if _POLICY is None:
            return _legal_fallback(obs)
        cur = obs.get("current") or {}
        seat = cur.get("yourIndex", 0)
        ctx = sel.get("context")
        state = state_features(cur, seat)
        ovecs = [option_features(o, ctx, sel) for o in opts]
        return pick(_POLICY, ovecs, state, sel.get("minCount", 1), sel.get("maxCount", 1))
    except Exception:
        return _legal_fallback(obs)
