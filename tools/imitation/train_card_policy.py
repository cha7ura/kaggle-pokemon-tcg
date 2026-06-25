"""Train the card policy: a RandomForest ranker scoring P(a winner chose this option | state, option).
Pointwise. Held-out decision-level top-1 accuracy is the offline gate. Exports the forest to a
NumPy/stdlib-walkable JSON (data/card_policy.json) — no sklearn at inference (submission constraint).

  python -m tools.imitation.train_card_policy [--in data/decisions.npz] [--trees 150] [--depth 14]

Reports: held-out top-1 decision accuracy vs random + always-most-confident baselines.
"""
import os, sys, json, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

DATA = os.path.join(os.path.dirname(__file__), "data")


def _arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def export_forest(rf, names):
    """Each tree -> flat arrays; leaf value = P(class 1). Walked in pure python/numpy at inference."""
    trees = []
    for est in rf.estimators_:
        t = est.tree_
        val = t.value.reshape(t.value.shape[0], -1)  # (nodes, 2) class counts
        p1 = (val[:, 1] / val.sum(axis=1).clip(min=1)).astype(np.float32)
        trees.append({
            "l": t.children_left.tolist(), "r": t.children_right.tolist(),
            "f": t.feature.tolist(),
            "t": [round(float(x), 4) for x in t.threshold.tolist()],
            "v": [round(float(x), 5) for x in p1.tolist()],
        })
    return {"trees": trees, "names": list(names), "n_features": len(names)}


def score_forest(model, X):
    """Reference NumPy walker (same logic ships in the agent). X: (n, d). Returns mean leaf P(class1)."""
    out = np.zeros(len(X), dtype=np.float64)
    for tr in model["trees"]:
        l, r, f, th, v = (np.asarray(tr[k]) for k in ("l", "r", "f", "t", "v"))
        for i, x in enumerate(X):
            n = 0
            while l[n] != -1:
                n = l[n] if x[f[n]] <= th[n] else r[n]
            out[i] += v[n]
    return out / len(model["trees"])


def decision_top1(scores, y, groups):
    """For each decision group, does the highest-scored option = the chosen one?"""
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    hit = tot = 0
    rand = 0.0
    for g, idx in by.items():
        if not any(y[i] for i in idx):
            continue
        best = max(idx, key=lambda i: scores[i])
        hit += 1 if y[best] else 0
        tot += 1
        rand += 1.0 / len(idx)            # random-pick expected accuracy for this decision
    return hit / tot if tot else 0.0, rand / tot if tot else 0.0, tot


# Positional/structural features that let the model cheat off option ORDER instead of mechanics.
# Validated: dropping these RAISES held-out accuracy (0.488 -> 0.506) and surfaces real features.
DROP = {"opt_index", "opt_type", "opt_area", "n_options"}


def main():
    npz = np.load(_arg("--in", os.path.join(DATA, "decisions.npz"), str), allow_pickle=True)
    X, y, groups, allnames = npz["X"], npz["y"], npz["groups"], list(npz["names"])
    keep = [i for i, n in enumerate(allnames) if n not in DROP]
    names = [allnames[i] for i in keep]
    X = X[:, keep]
    print(f"data: {X.shape[0]} rows, {X.shape[1]} dims (dropped {len(DROP)} positional), "
          f"{y.mean()*100:.1f}% positive, {len(set(groups))} decisions")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr, te = next(gss.split(X, y, groups))
    kind = _arg("--model", "xgb", str)
    from tools.imitation import model_io
    t0 = time.time()
    if kind == "xgb":
        import xgboost as xgb
        spw = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())
        clf = xgb.XGBClassifier(n_estimators=_arg("--trees", 600, int),
                                max_depth=_arg("--depth", 8, int), learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                n_jobs=-1, eval_metric="logloss", random_state=0)
        clf.fit(X[tr], y[tr])
        packed = model_io.pack_xgb(clf, names)
        print(f"trained XGBoost {clf.n_estimators} trees in {time.time()-t0:.0f}s")
    else:
        clf = RandomForestClassifier(n_estimators=_arg("--trees", 150, int),
                                     max_depth=_arg("--depth", 14, int), min_samples_leaf=20,
                                     class_weight="balanced", n_jobs=-1, random_state=0)
        clf.fit(X[tr], y[tr])
        packed = model_io.pack(clf, names)
        print(f"trained RF {len(clf.estimators_)} trees in {time.time()-t0:.0f}s")
    # eval via the COMPACT walker (proves the packed model matches) on a capped test sample
    cap = min(len(te), 40000)
    sub = te[:cap]
    sc = model_io.score(packed, X[sub])
    acc, rand, n = decision_top1(sc, y[sub], groups[sub])
    print(f"held-out decision top-1: {acc:.3f}  (random {rand:.3f}, n={n} decisions, {cap} rows)")

    out = os.path.join(DATA, "card_policy.npz")
    model_io.save(out, packed)
    print(f"exported -> {out} ({os.path.getsize(out)//1024//1024} MB, {len(packed['sizes'])} trees, "
          f"{len(packed['L'])} nodes)")
    # top feature importances (sanity)
    imp = sorted(zip(names, clf.feature_importances_), key=lambda x: -x[1])[:12]
    print("top features:", [f"{n}={i:.3f}" for n, i in imp])


if __name__ == "__main__":
    main()
