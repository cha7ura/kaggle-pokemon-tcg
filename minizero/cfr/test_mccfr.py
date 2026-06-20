"""External-sampling MCCFR must reach the same Nash on Kuhn (wiki/10)."""
from minizero.cfr.mccfr import train_mccfr
from minizero.cfr.exploit import exploitability


def test_mccfr_converges():
    avg = train_mccfr(200000, seed=0)
    assert exploitability(avg) < 2e-2, "MCCFR did not converge near Nash"


def test_mccfr_reproducible():
    a = train_mccfr(5000, seed=1)
    b = train_mccfr(5000, seed=1)
    assert all((a[k] == b[k]).all() for k in a), "same seed must give same result"
