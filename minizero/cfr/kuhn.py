"""Kuhn poker: the canonical tiny imperfect-information game (see wiki/10-cfr-and-nash.md).

3-card deck J<Q<K = 0<1<2, one card each, ante 1. Actions: 'p' (pass/check/fold) and
'b' (bet/call). Payoff is from player 0's perspective.
"""
ACTIONS = ("p", "b")
TERMINAL = {"pp", "bp", "bb", "pbp", "pbb"}


def is_terminal(history: str) -> bool:
    return history in TERMINAL


def payoff(cards, history: str) -> int:
    """Payoff to player 0 at a terminal history."""
    p0_high = cards[0] > cards[1]
    if history == "bp":          # p0 bet, p1 folded
        return 1
    if history == "pbp":         # p0 checked, p1 bet, p0 folded
        return -1
    if history == "pp":          # both checked -> showdown for the antes
        return 1 if p0_high else -1
    if history in ("bb", "pbb"):  # a bet was called -> showdown for the bigger pot
        return 2 if p0_high else -2
    raise ValueError(f"not terminal: {history!r}")


def infoset_key(card: int, history: str) -> str:
    return f"{card}{history}"
