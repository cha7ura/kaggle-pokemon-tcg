"""Tabular R-NaD must reach Nash on Kuhn: LAST-iterate exploitability -> ~0 (wiki/12)."""
from minizero.rnad.rnad import train_rnad
from minizero.cfr.exploit import exploitability


def test_rnad_last_iterate_converges():
    pi = train_rnad()
    # R-NaD converges in the LAST iterate (no averaging), unlike CFR.
    assert exploitability(pi) < 5e-3, "R-NaD last iterate did not reach Nash"


def test_rnad_beats_uniform():
    import numpy as np
    from minizero.rnad.rnad import _all_infosets, NA
    uniform = {k: np.full(NA, 1.0 / NA) for k in _all_infosets()}
    assert exploitability(train_rnad()) < exploitability(uniform)
