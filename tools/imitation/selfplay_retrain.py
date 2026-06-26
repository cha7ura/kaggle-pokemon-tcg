"""Gated self-play retrain. The honest test: does adding self-play winner-rows improve prediction of
HELD-OUT REAL human winner decisions? Train two models on the SAME held-out real test split:
  baseline  = imitation-train only
  augmented = imitation-train + ALL self-play
If augmented's top-1 on the real held-out beats baseline -> self-play learned something general; save
it. Else discard (sim-overfit). This is the gate the old RL skipped.

  python -m tools.imitation.selfplay_retrain
"""
import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from tools.imitation.train_card_policy import decision_top1, DROP
from tools.imitation import model_io
import xgboost as xgb

DATA = os.path.join(os.path.dirname(__file__), "data")


def _load(fn):
    z = np.load(os.path.join(DATA, fn), allow_pickle=True)
    return z["X"], z["y"], z["groups"], list(z["names"])


def _keep_cols(names):
    return [i for i, n in enumerate(names) if n not in DROP]


def _train(X, y):
    spw = (y == 0).sum() / max(1, (y == 1).sum())
    clf = xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, scale_pos_weight=spw, n_jobs=-1,
                            eval_metric="logloss", random_state=0)
    clf.fit(X, y)
    return clf


def main():
    Xi, yi, gi, ni = _load("decisions.npz")
    Xs, ys, gs, ns = _load("selfplay.npz")
    assert ni == ns, "feature schema mismatch between imitation and selfplay"
    keep = _keep_cols(ni); names = [ni[i] for i in keep]
    Xi, Xs = Xi[:, keep], Xs[:, keep]
    print(f"imitation {Xi.shape}, selfplay {Xs.shape}", flush=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr, te = next(gss.split(Xi, yi, gi))
    Xte, yte, gte = Xi[te], yi[te], gi[te]           # held-out REAL human decisions = the gate

    base = _train(Xi[tr], yi[tr])
    aug = _train(np.vstack([Xi[tr], Xs]), np.concatenate([yi[tr], ys]))

    def top1(clf):
        p = clf.predict_proba(Xte)[:, 1]
        return decision_top1(p, yte, gte)[0]

    b, a = top1(base), top1(aug)
    print(f"held-out REAL top-1:  baseline {b:.4f}   +selfplay {a:.4f}   delta {a-b:+.4f}", flush=True)
    if a > b + 0.001:
        model_io.save(os.path.join(DATA, "card_policy.npz"), model_io.pack_xgb(aug, names))
        print(f"GATE PASSED -> saved augmented model (card_policy.npz)", flush=True)
    else:
        print("GATE FAILED -> self-play did not help real-decision prediction; keeping prior model", flush=True)


if __name__ == "__main__":
    main()
