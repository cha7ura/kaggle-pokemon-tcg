"""Compact forest model: pack a sklearn RandomForest into flat NumPy arrays, save compressed (.npz).
~5-30x smaller than JSON (numbers as binary, not text) and faster to load. Same walker offline + in-sim.
stdlib + numpy (numpy ships in the Kaggle runtime; no sklearn at inference).
"""
import numpy as np


def pack(rf, names):
    L = R = F = T = V = None
    Ls, Rs, Fs, Ts, Vs, sizes = [], [], [], [], [], []
    for est in rf.estimators_:
        t = est.tree_
        val = t.value.reshape(t.value.shape[0], -1)
        p1 = val[:, 1] / val.sum(axis=1).clip(min=1)
        Ls.append(t.children_left); Rs.append(t.children_right); Fs.append(t.feature)
        Ts.append(t.threshold); Vs.append(p1); sizes.append(len(t.children_left))
    return {
        "L": np.concatenate(Ls).astype(np.int32), "R": np.concatenate(Rs).astype(np.int32),
        "F": np.concatenate(Fs).astype(np.int16), "T": np.concatenate(Ts).astype(np.float32),
        "V": np.concatenate(Vs).astype(np.float32), "sizes": np.asarray(sizes, np.int32),
        "names": np.asarray(list(names)),
    }


def save(path, packed):
    np.savez_compressed(path, **packed)


def load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _offsets(sizes):
    return np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)


def score(p, X):
    """Mean leaf P(class1) over the forest. X: (n, d). Children indices are tree-local."""
    L, R, F, T, V, sizes = p["L"], p["R"], p["F"], p["T"], p["V"], p["sizes"]
    off = _offsets(sizes)
    nt = len(sizes)
    X = np.asarray(X, dtype=np.float32)
    out = np.zeros(len(X), dtype=np.float64)
    for ti in range(nt):
        o = int(off[ti])
        l, r, f, th, v = L[o:], R[o:], F[o:], T[o:], V[o:]
        for i in range(len(X)):
            xi = X[i]; n = 0
            while l[n] != -1:
                n = l[n] if xi[f[n]] <= th[n] else r[n]
            out[i] += v[n]
    return out / nt


def names(p):
    return [str(x) for x in p["names"]]
