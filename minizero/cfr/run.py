"""Watch CFR converge to Nash on Kuhn poker: python -m minizero.cfr.run

Prints the exploitability curve (-> 0) and the learned average strategy. See
wiki/10-cfr-and-nash.md for the theory.
"""
from minizero.cfr.cfr import train
from minizero.cfr.exploit import exploitability


def main():
    print("iters   exploitability")
    for it in (10, 100, 1000, 10000, 30000):
        avg = train(it)
        print(f"{it:6d}   {exploitability(avg):.5f}")
    print("\nlearned average strategy (pass / bet):")
    for k in sorted(avg):
        print(f"  {k:5s}  pass={avg[k][0]:.3f}  bet={avg[k][1]:.3f}")
    print("\nKuhn Nash check: player-0 game value should be ~ -1/18 = -0.0556")


if __name__ == "__main__":
    main()
