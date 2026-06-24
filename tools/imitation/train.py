"""Per-deck imitation trainer. For each learnable decision, the chosen option is a positive example
(state+option, label 1) and the other options are negatives (label 0). A DecisionTree scores an
option; inference picks the argmax. Exports to the stdlib JSON tree shape and reports per-context
top-1 accuracy on a held-out split — the Stage-1 go/no-go signal. sklearn allowed HERE only."""
import json, os, glob, collections, random
from sklearn.tree import DecisionTreeClassifier
from tools.imitation.policy import score
from tools.imitation.extract_decisions import sig_to_fname  # deck sigs are too long to be filenames

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
POL = os.path.join(HERE, "policies")
FLOOR = 30


def export_tree(clf):
    t = clf.tree_
    classes = list(clf.classes_)
    pos = classes.index(1) if 1 in classes else len(classes) - 1
    value = []
    for i in range(t.node_count):
        counts = t.value[i][0]
        tot = counts.sum()
        value.append(float(counts[pos] / tot) if tot else 0.0)
    return {
        "feature": [int(f) for f in t.feature],
        "threshold": [float(x) for x in t.threshold],
        "left": [int(x) for x in t.children_left],
        "right": [int(x) for x in t.children_right],
        "value": value,
    }


def _load(sig):
    recs = []
    p = os.path.join(DATA, f"{sig_to_fname(sig)}.jsonl")
    for line in open(p):
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def train_deck(sig, seed=0):
    recs = _load(sig)
    rnd = random.Random(seed)
    rnd.shuffle(recs)
    cut = int(len(recs) * 0.8)
    train, test = recs[:cut], recs[cut:]
    X, y = [], []
    for r in train:
        for j, opt in enumerate(r["options"]):
            X.append(r["state"] + opt)
            y.append(1 if j == r["chosen"] else 0)
    clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                 class_weight="balanced", random_state=seed).fit(X, y)
    tree = export_tree(clf)
    # held-out top-1 accuracy + baseline mean(1/n)
    hit = base = 0
    by_ctx = collections.defaultdict(lambda: [0, 0, 0.0])
    for r in test:
        scores = [score(tree, r["state"] + opt) for opt in r["options"]]
        pred = max(range(len(scores)), key=lambda j: scores[j])
        ok = 1 if pred == r["chosen"] else 0
        hit += ok; base += 1.0 / len(r["options"])
        c = by_ctx[r["context"]]; c[0] += ok; c[1] += 1; c[2] += 1.0 / len(r["options"])
    n = len(test) or 1
    return {"tree": tree, "n": len(recs), "accuracy": hit / n, "baseline": base / n,
            "by_context": {str(k): {"acc": v[0]/v[1], "base": v[2]/v[1], "n": v[1]}
                           for k, v in by_ctx.items()}}


def main():
    os.makedirs(POL, exist_ok=True)
    manifest = json.load(open(os.path.join(DATA, "manifest.json")))
    rows = []
    for m in manifest:
        if m["games"] < FLOOR:
            continue
        res = train_deck(m["deck_sig"])
        json.dump(res["tree"], open(os.path.join(POL, f"{m['file']}.json"), "w"))
        rows.append((m, res))
        print(f"  {m['games']:4}g acc={res['accuracy']:.3f} base={res['baseline']:.3f} "
              f"lift={res['accuracy']-res['baseline']:+.3f}  {m['archetype']:12} {m['player']}")
    if rows:
        avg = sum(r["accuracy"] for _, r in rows) / len(rows)
        avgb = sum(r["baseline"] for _, r in rows) / len(rows)
        print(f"\nGATE: mean top-1 acc={avg:.3f} vs baseline {avgb:.3f} (lift {avg-avgb:+.3f})")
        print("Dominant-context detail (ctx 0 and 7):")
        for m, r in rows[:6]:
            for c in ("0", "7"):
                if c in r["by_context"]:
                    d = r["by_context"][c]
                    print(f"  {m['player'][:18]:18} ctx{c}: acc={d['acc']:.3f} "
                          f"base={d['base']:.3f} n={d['n']}")


if __name__ == "__main__":
    main()
