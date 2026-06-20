"""CFR on Kuhn poker must converge to Nash: exploitability -> ~0, value -> -1/18."""
from minizero.cfr.kuhn import payoff, TERMINAL
from minizero.cfr.cfr import train
from minizero.cfr.exploit import exploitability


def test_kuhn_payoff_basics():
    # player 0 holds K(2) vs J(0); both pass -> p0 wins ante
    assert payoff([2, 0], "pp") == 1
    assert payoff([0, 2], "pp") == -1
    assert payoff([0, 2], "bp") == 1      # p1 folded after p0 bet -> p0 wins
    assert payoff([2, 0], "bb") == 2      # double bet showdown, p0 higher
    assert "pbp" in TERMINAL and "pb" not in TERMINAL


def test_cfr_converges_to_nash():
    avg = train(30000)
    e = exploitability(avg)
    assert e < 1e-2, f"exploitability {e} not converged"


def test_exploitability_decreases():
    early = exploitability(train(200))
    late = exploitability(train(30000))
    assert late < early
