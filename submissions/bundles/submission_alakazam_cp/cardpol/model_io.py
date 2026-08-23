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


def pack_xgb(clf, names):
    """Pack an XGBClassifier (binary:logistic) into the same flat arrays. Walked additively at
    inference: total = sum of leaf values over trees (sigmoid is monotonic -> argmax-equivalent, so
    we skip it). xgboost split rule: x[f] < threshold -> 'yes' (left)."""
    import json as _json
    dumps = clf.get_booster().get_dump(dump_format="json")
    Ls, Rs, Fs, Ts, Vs, sizes = [], [], [], [], [], []
    for d in dumps:
        root = _json.loads(d)
        l, r, f, t, v = [], [], [], [], []

        def add(node):
            i = len(l); l.append(-1); r.append(-1); f.append(-1); t.append(0.0); v.append(0.0)
            if "leaf" in node:
                v[i] = float(node["leaf"])
            else:
                f[i] = int(str(node["split"])[1:]); t[i] = float(node["split_condition"])
                kids = {c["nodeid"]: c for c in node["children"]}
                l[i] = add(kids[node["yes"]]); r[i] = add(kids[node["no"]])
            return i
        add(root)
        Ls.append(l); Rs.append(r); Fs.append(f); Ts.append(t); Vs.append(v); sizes.append(len(l))
    return {
        "L": np.concatenate([np.asarray(x, np.int32) for x in Ls]),
        "R": np.concatenate([np.asarray(x, np.int32) for x in Rs]),
        "F": np.concatenate([np.asarray(x, np.int16) for x in Fs]),
        "T": np.concatenate([np.asarray(x, np.float32) for x in Ts]),
        "V": np.concatenate([np.asarray(x, np.float32) for x in Vs]),
        "sizes": np.asarray(sizes, np.int32), "names": np.asarray(list(names)),
        "boosted": np.asarray([1], np.int8),
    }


def save(path, packed):
    np.savez_compressed(path, **packed)


def load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _offsets(sizes):
    return np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)


def score(p, X):
    """Aggregate leaf scores. RF: mean P(class1), split x[f] <= th. XGBoost (boosted): SUM of leaf
    margins, split x[f] < th (sigmoid skipped — monotonic, argmax-equivalent). Both rank correctly."""
    L, R, F, T, V, sizes = p["L"], p["R"], p["F"], p["T"], p["V"], p["sizes"]
    boosted = bool(int(p["boosted"][0])) if "boosted" in p else False
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
                go_left = xi[f[n]] < th[n] if boosted else xi[f[n]] <= th[n]
                n = l[n] if go_left else r[n]
            out[i] += v[n]
    return out if boosted else out / nt


def names(p):
    return [str(x) for x in p["names"]]
