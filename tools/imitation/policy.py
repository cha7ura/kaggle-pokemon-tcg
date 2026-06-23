"""Stdlib inference over a serialized sklearn decision tree. NO sklearn/numpy. The pilot imports
this; the trainer imports it for a parity check."""


def score(tree, x):
    i = 0
    feat = tree["feature"]; thr = tree["threshold"]
    left = tree["left"]; right = tree["right"]; val = tree["value"]
    while left[i] != -1:
        i = left[i] if x[feat[i]] <= thr[i] else right[i]
    return val[i]


def pick(tree, option_vecs, state, min_count, max_count):
    scored = sorted(range(len(option_vecs)),
                    key=lambda j: -score(tree, state + option_vecs[j]))
    k = max(1, min(max_count or 1, len(option_vecs)))
    k = max(k, min_count or 0)
    k = min(k, len(option_vecs))
    return scored[:k]
