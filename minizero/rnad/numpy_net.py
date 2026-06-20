"""Pure-NumPy forward pass for an R-NaD policy net — the SHIPPABLE inference path (wiki/12, wiki/17).

Train the net on GPU (torch), export the weights to weights.npz, then run inference here with
zero heavy deps: a 2-layer MLP is a few np.dot calls. No torch on the Kaggle ladder. The
torch->numpy round-trip is verified on GPU (max|diff| ~6e-8 on Kuhn — see colab/kuhn_pipeline.py);
this module is what an agent_net.py would call once a cg-scale net is trained.

weights.npz layout (torch Linear state_dict): fc1.weight (h,in) fc1.bias (h,) fc2.weight (out,h) fc2.bias (out,).
"""
import numpy as np


def load(path):
    return {k: v for k, v in np.load(path).items()}


def forward(x, W):
    """x: (..., in) features -> (..., out) action probabilities. Same math as the torch net."""
    h = np.maximum(x @ W["fc1.weight"].T + W["fc1.bias"], 0.0)          # ReLU
    z = h @ W["fc2.weight"].T + W["fc2.bias"]
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)                                  # softmax


def pick(probs, legal_mask):
    """Mask illegal actions, renormalize, return the argmax legal action index (crash-safe pick)."""
    p = probs * legal_mask
    return int(p.argmax()) if p.sum() > 0 else int(np.argmax(legal_mask))


if __name__ == "__main__":
    # self-check: random net -> valid probability simplex + masking works
    rng = np.random.default_rng(0)
    W = {"fc1.weight": rng.standard_normal((8, 5)), "fc1.bias": rng.standard_normal(8),
         "fc2.weight": rng.standard_normal((3, 8)), "fc2.bias": rng.standard_normal(3)}
    p = forward(rng.standard_normal((4, 5)), W)
    assert p.shape == (4, 3) and np.allclose(p.sum(1), 1.0), "softmax broken"
    assert pick(p[0], np.array([0, 1, 0])) == 1, "mask/pick broken"
    print("numpy_net OK: probabilities sum to 1, masking picks a legal action")
