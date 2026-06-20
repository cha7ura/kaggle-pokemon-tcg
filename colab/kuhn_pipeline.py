"""Task-6 shippable-pipeline proof, run on Colab T4 (any torch env works).

Proves the deployment path end-to-end on a verifiable toy:
  GPU-train a policy net (torch) -> export weights.npz -> pure-NumPy forward -> round-trip match.
Target policy = Kuhn Nash (from CFR), so the net reproduces a KNOWN-GOOD strategy, not noise.

Verified on Tesla T4: round-trip max|torch-numpy| = 5.96e-08, net-vs-Nash = 0.003.
The cg-scale version swaps the Kuhn target for R-NaD self-play on the cg engine (deferred — needs
the engine uploaded to Colab + a feature encoder). The ship mechanism is minizero/rnad/numpy_net.py.
"""
import itertools
import numpy as np

# --- 1. Kuhn Nash via CFR (the training target) ---
ACTIONS = ("p", "b"); TERMINAL = {"pp", "bp", "bb", "pbp", "pbb"}


def payoff(c, h):
    hi = c[0] > c[1]
    if h == "bp": return 1
    if h == "pbp": return -1
    if h == "pp": return 1 if hi else -1
    if h in ("bb", "pbb"): return 2 if hi else -2


def _rm(r):
    p = np.maximum(r, 0); s = p.sum(); return p / s if s > 0 else np.full(2, .5)


def _cfr(cards, h, p0, p1, R, S):
    if h in TERMINAL:
        u = payoff(cards, h); return u if len(h) % 2 == 0 else -u
    pl = len(h) % 2; k = f"{cards[pl]}{h}"; R.setdefault(k, np.zeros(2)); S.setdefault(k, np.zeros(2))
    strat = _rm(R[k]); S[k] += (p0 if pl == 0 else p1) * strat
    util = np.zeros(2)
    for a, act in enumerate(ACTIONS):
        util[a] = (-_cfr(cards, h + act, p0 * strat[a], p1, R, S) if pl == 0
                   else -_cfr(cards, h + act, p0, p1 * strat[a], R, S))
    R[k] += (p1 if pl == 0 else p0) * (util - strat @ util)
    return float(strat @ util)


def kuhn_nash(iters=30000):
    R, S = {}, {}
    deals = [list(c) for c in itertools.permutations(range(3), 2)]
    for _ in range(iters):
        for d in deals:
            _cfr(d, "", 1, 1, R, S)
    return {k: S[k] / S[k].sum() for k in S}


# --- 2/3. train torch net to the Nash target, export, verify numpy round-trip ---
def main(use_wandb=False):
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(game="kuhn", lr=0.01, hidden=16, steps=3000, target="cfr_nash")
    run = None
    if use_wandb:
        import wandb
        run = wandb.init(project="ptcg-rnad", name="kuhn-pipeline", config=cfg)
    nash = kuhn_nash(); keys = sorted(nash)
    X = torch.eye(len(keys), device=dev)
    Y = torch.tensor(np.stack([nash[k] for k in keys]), dtype=torch.float32, device=dev)

    net = nn.Sequential(nn.Linear(len(keys), cfg["hidden"]), nn.ReLU(), nn.Linear(cfg["hidden"], 2)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    for step in range(cfg["steps"]):
        opt.zero_grad()
        loss = -(Y * torch.log_softmax(net(X), 1)).sum(1).mean()
        loss.backward(); opt.step()
        if run and step % 100 == 0:
            run.log({"step": step, "ce_loss": loss.item()})

    W = {f"fc{i//2+1}.{p}": net[i].__getattr__(p).detach().cpu().numpy()
         for i in (0, 2) for p in ("weight", "bias")}
    np.savez("weights.npz", **W)

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from minizero.rnad.numpy_net import forward
    np_p = forward(np.eye(len(keys), dtype=np.float32), W)
    tp = torch.softmax(net(X), 1).detach().cpu().numpy()
    rt = np.abs(np_p - tp).max()
    print(f"device={dev}  round-trip max|torch-numpy|={rt:.2e}  net-vs-Nash={np.abs(np_p - Y.cpu().numpy()).max():.3f}")
    assert rt < 1e-5, "round-trip mismatch"
    if run:
        import wandb
        run.log({"roundtrip_max_diff": float(rt)})
        art = wandb.Artifact("kuhn-weights", type="model", metadata={"roundtrip": float(rt)})
        art.add_file("weights.npz"); run.log_artifact(art); run.finish()
    print("OK: GPU-train -> weights.npz -> numpy forward verified")


if __name__ == "__main__":
    main()
