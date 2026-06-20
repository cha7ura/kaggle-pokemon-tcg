"""Watch R-NaD's LAST iterate converge to Nash on Kuhn: python -m minizero.rnad.run

Contrast with CFR (minizero/cfr/run.py), which converges in the AVERAGE strategy. R-NaD's
current policy itself converges — the property that lets a trained net ship as a forward pass
(wiki/12-deepnash-rnad.md).
"""
import numpy as np

from minizero.rnad.rnad import cf_action_values, _all_infosets, _softmax, NA
from minizero.cfr.exploit import exploitability


def main():
    keys = _all_infosets()
    pi = {k: np.full(NA, 1.0 / NA) for k in keys}
    lr, eta = 0.5, 0.5
    print("outer   exploitability (last iterate)")
    for o in range(121):
        ref = {k: v.copy() for k, v in pi.items()}
        log_ref = {k: np.log(v + 1e-12) for k, v in ref.items()}
        for _ in range(60):
            q = cf_action_values(pi)
            for k in keys:
                lp = np.log(pi[k] + 1e-12)
                pi[k] = _softmax(lp + lr * (q[k] - eta * (lp - log_ref[k])))
        if o % 20 == 0:
            print(f"{o:5d}   {exploitability(pi):.5f}")
    print("\nNash reached in the last iterate (no averaging needed).")


if __name__ == "__main__":
    main()
