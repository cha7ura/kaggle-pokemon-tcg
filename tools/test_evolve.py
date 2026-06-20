"""evolve.py keep/revert logic, verified without Docker via an injected fake scorer."""
import os
import tempfile

from tools.evolve import gate, evolve


def test_gate():
    assert gate({"wilson_lb": 0.50}, lb_min=0.48)
    assert not gate({"wilson_lb": 0.45}, lb_min=0.48)


def _files():
    d = tempfile.mkdtemp()
    cand = os.path.join(d, "candidate.py"); champ = os.path.join(d, "champion.py")
    log = os.path.join(d, "experiments.md")
    open(cand, "w").write("CANDIDATE")
    open(champ, "w").write("OLD_CHAMPION")
    open(log, "w").write("# log\n")
    return cand, champ, log


def test_evolve_promotes_on_pass():
    cand, champ, log = _files()
    kept, _ = evolve(cand, champ, log, note="t", games=10,
                     runner=lambda c, ch, g: {"wilson_lb": 0.55, "games": g})
    assert kept
    assert open(champ).read() == "CANDIDATE"          # champion replaced
    assert "PROMOTE" in open(log).read()              # logged


def test_evolve_reverts_on_fail():
    cand, champ, log = _files()
    kept, _ = evolve(cand, champ, log, note="t", games=10,
                     runner=lambda c, ch, g: {"wilson_lb": 0.40, "games": g})
    assert not kept
    assert open(champ).read() == "OLD_CHAMPION"        # champion untouched
    assert "PROMOTE" not in open(log).read()
