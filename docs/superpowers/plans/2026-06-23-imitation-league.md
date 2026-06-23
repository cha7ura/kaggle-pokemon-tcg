# Imitation League Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Learn per-deck decision policies from real top-player replays and use them to pilot opponent decks, upgrading the offline oracle from a typh-piloted (Trevenant-blind) field to a realistic imitation field.

**Architecture:** A pure-Python featurizer (shared offline + in-sim) turns each replay decision into `state⊕option` vectors. A sklearn decision tree per deck scores options; the chosen option is the argmax. Trees are exported to JSON the stdlib agent walks at inference. A league harness pits (deck+policy) pilots against our decks via the existing docker `eval.py`.

**Tech Stack:** Python 3 stdlib (extractor, featurizer, pilot inference), scikit-learn (offline training only), existing `cg` engine + `eval.py` docker harness, replay JSONs in `json/`.

## Global Constraints

- The in-sim pilot (`autoresearch/agent_imitation.py`) and the featurizer's inference path MUST be **stdlib-only** — no sklearn/numpy at runtime. sklearn is allowed in offline trainer scripts only.
- The pilot MUST answer the deck request from the RAW dict before any conversion: `if not isinstance(obs_dict, dict) or obs_dict.get("select") is None: return _DECK`.
- The pilot MUST wrap the in-play path in try/except returning a legal fallback `list(range(min(minCount, len(option))))`.
- NO `__file__` anywhere in the pilot — Kaggle runs the agent via `exec(code_object, env)` where `__file__` is undefined. Resolve paths via env var → cwd-relative → `/kaggle_simulations/agent/`.
- The featurizer operates on the RAW observation dict (same object shape in replays and live obs) — one module, no environment branching.
- Deck signature canonical key = `tuple(sorted(deck))`, identical to `tools/extract_field.py`.
- Data floor for learning a policy = **30 games**. Decks below the floor use the typh fallback.
- All new tooling under `tools/imitation/`; tests under `tools/imitation/tests/`.

---

## File Structure

- `tools/imitation/features.py` — shared featurizer (state + option vectors), frozen feature order.
- `tools/imitation/extract_decisions.py` — replays → per-deck decision tables + manifest.
- `tools/imitation/train.py` — per-deck tree trainer + JSON export + accuracy report.
- `tools/imitation/policy.py` — stdlib tree-inference (loads exported JSON, scores a feature vector). Imported by both `train.py` (parity check) and the pilot.
- `autoresearch/agent_imitation.py` — generic stdlib pilot (deck + policy → option index).
- `tools/imitation/league.py` — (deck+policy) roster vs our decks via docker eval; field-weighted.
- `tools/imitation/tests/` — pytest tests for each unit.
- `tools/imitation/data/` — generated decision tables (gitignored).
- `tools/imitation/policies/` — generated policy JSONs (committed; small).

---

## STAGE 1 — Insight & gate (Tasks 1–4)

### Task 1: Featurizer

**Files:**
- Create: `tools/imitation/features.py`
- Test: `tools/imitation/tests/test_features.py`

**Interfaces:**
- Produces: `state_features(current: dict, seat: int) -> list[float]`,
  `option_features(option: dict, context: int, select: dict) -> list[float]`,
  `STATE_DIM: int`, `OPTION_DIM: int`, `feature_names() -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tools/imitation/tests/test_features.py
import json, glob
from tools.imitation.features import state_features, option_features, STATE_DIM, OPTION_DIM

def _a_decision():
    for f in sorted(glob.glob("json/*.json"))[:50]:
        d = json.load(open(f))
        for s in d["steps"]:
            for seat in (0, 1):
                obs = s[seat].get("observation", {})
                sel = obs.get("select")
                cur = obs.get("current")
                act = s[seat].get("action")
                if (sel and cur and isinstance(act, list) and sel.get("option")
                        and len(sel["option"]) >= 2
                        and all(isinstance(x, int) and x < len(sel["option"]) for x in act)):
                    return cur, sel, seat
    raise RuntimeError("no decision found")

def test_state_features_shape_and_finite():
    cur, sel, seat = _a_decision()
    v = state_features(cur, seat)
    assert len(v) == STATE_DIM
    assert all(isinstance(x, (int, float)) for x in v)

def test_option_features_shape():
    cur, sel, seat = _a_decision()
    v = option_features(sel["option"][0], sel.get("context"), sel)
    assert len(v) == OPTION_DIM

def test_state_features_handles_empty_active():
    # a player with empty active list must not crash, returns zeros for active fields
    cur = {"turn": 1, "yourIndex": 0, "supporterPlayed": False, "stadiumPlayed": False,
           "turnActionCount": 0,
           "players": [{"active": [], "bench": [], "hand": [], "prize": [None]*6,
                        "handCount": 0, "asleep": False, "confused": False,
                        "paralyzed": False, "poisoned": False, "burned": False},
                       {"active": [], "bench": [], "hand": [], "prize": [None]*6,
                        "handCount": 0, "asleep": False, "confused": False,
                        "paralyzed": False, "poisoned": False, "burned": False}]}
    v = state_features(cur, 0)
    assert len(v) == STATE_DIM
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chaturaattidiya/Documents/Github/kaggle/pokemon && python -m pytest tools/imitation/tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: tools.imitation.features`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/imitation/features.py
"""Shared featurizer for the imitation league. Operates on the RAW observation dict so the
exact same code runs offline (replay obs) and in-sim (live obs). STDLIB ONLY."""

STATE_NAMES = [
    "turn", "turn_action_count", "supporter_played", "stadium_played",
    "my_active_hp", "my_active_maxhp", "my_active_dmg", "my_active_energy",
    "my_bench", "my_hand", "my_prize_left",
    "my_asleep", "my_confused", "my_paralyzed", "my_poisoned", "my_burned",
    "opp_active_hp", "opp_active_maxhp", "opp_active_dmg", "opp_active_energy",
    "opp_bench", "opp_hand", "opp_prize_left",
]
OPTION_NAMES = ["context", "opt_type", "opt_area", "opt_index", "n_options",
                "min_count", "max_count"]
STATE_DIM = len(STATE_NAMES)
OPTION_DIM = len(OPTION_NAMES)


def feature_names():
    return STATE_NAMES + OPTION_NAMES


def _active(player):
    a = player.get("active") or []
    return a[0] if a else None


def _prize_left(player):
    p = player.get("prize") or []
    return len(p)


def state_features(current, seat):
    players = current.get("players") or [{}, {}]
    me = players[seat] if seat < len(players) else {}
    opp = players[1 - seat] if (1 - seat) < len(players) else {}
    ma, oa = _active(me), _active(opp)

    def act_fields(a):
        if not a:
            return [0.0, 0.0, 0.0, 0.0]
        hp = float(a.get("hp", 0)); mhp = float(a.get("maxHp", 0))
        return [hp, mhp, mhp - hp, float(len(a.get("energies") or []))]

    f = [
        float(current.get("turn", 0)),
        float(current.get("turnActionCount", 0)),
        1.0 if current.get("supporterPlayed") else 0.0,
        1.0 if current.get("stadiumPlayed") else 0.0,
    ]
    f += act_fields(ma)
    f += [float(len(me.get("bench") or [])), float(me.get("handCount", len(me.get("hand") or []))),
          float(_prize_left(me))]
    f += [1.0 if me.get(k) else 0.0 for k in ("asleep", "confused", "paralyzed", "poisoned", "burned")]
    f += act_fields(oa)
    f += [float(len(opp.get("bench") or [])), float(opp.get("handCount", 0)),
          float(_prize_left(opp))]
    return f


def option_features(option, context, select):
    o = option or {}
    return [
        float(context if context is not None else -1),
        float(o.get("type", -1)),
        float(o.get("area", -1)),
        float(o.get("index", -1)),
        float(len(select.get("option") or [])),
        float(select.get("minCount", 0)),
        float(select.get("maxCount", 0)),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/chaturaattidiya/Documents/Github/kaggle/pokemon && python -m pytest tools/imitation/tests/test_features.py -v`
Expected: PASS (3 tests). Add `tools/imitation/__init__.py` and `tools/imitation/tests/__init__.py` (empty) if import fails; ensure repo root on path via `python -m pytest` from repo root.

- [ ] **Step 5: Commit**

```bash
git add tools/imitation/features.py tools/imitation/tests/test_features.py tools/imitation/__init__.py tools/imitation/tests/__init__.py
git commit -m "feat(imitation): shared stdlib featurizer (state + option vectors)"
```

---

### Task 2: Decision extractor

**Files:**
- Create: `tools/imitation/extract_decisions.py`
- Test: `tools/imitation/tests/test_extract.py`

**Interfaces:**
- Consumes: `features.state_features`, `features.option_features`.
- Produces: `iter_decisions(game: dict) -> Iterator[dict]` yielding
  `{deck_sig, player, reward, context, state, options, chosen}` where `state` is a list[float],
  `options` is list[list[float]], `chosen` is int (index into options); and a `main()` that writes
  `tools/imitation/data/<deck_sig>.jsonl` + `tools/imitation/data/manifest.json`
  (`[{deck_sig, player, archetype, games, decisions}]`).

- [ ] **Step 1: Write the failing test**

```python
# tools/imitation/tests/test_extract.py
import json, glob
from tools.imitation.extract_decisions import iter_decisions, deck_sig_of

def test_iter_decisions_yields_valid_records():
    f = sorted(glob.glob("json/*.json"))[0]
    g = json.load(open(f))
    recs = list(iter_decisions(g))
    assert recs, "expected at least one decision"
    r = recs[0]
    assert set(r) >= {"deck_sig", "player", "reward", "context", "state", "options", "chosen"}
    assert 0 <= r["chosen"] < len(r["options"])
    assert len(r["options"]) >= 2
    # every option vector same width
    assert len({len(o) for o in r["options"]}) == 1

def test_deck_sig_stable():
    a = deck_sig_of([3, 1, 2, 1])
    b = deck_sig_of([1, 1, 2, 3])
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chaturaattidiya/Documents/Github/kaggle/pokemon && python -m pytest tools/imitation/tests/test_extract.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/imitation/extract_decisions.py
"""Replays -> per-deck decision tables. A decision is learnable when select has >=2 options and the
action is a list of small indices into option (the 60-card deck action is naturally excluded since
its entries are card ids >> n_options). STDLIB ONLY."""
import json, glob, os, sys, collections
from tools.imitation.features import state_features, option_features

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(os.path.dirname(__file__), "data")
import csv as _csv
_names = {int(r["cardId"]): r["name"]
          for r in _csv.DictReader(open(f"{ROOT}/autoresearch/cards_full.csv"))}


def deck_sig_of(deck):
    return "_".join(str(x) for x in sorted(deck))


def _deck_of(steps, seat):
    for s in steps:
        a = s[seat].get("action")
        if isinstance(a, list) and len(a) == 60 and all(isinstance(x, int) for x in a):
            return a
    return None


def _archetype(deck):
    ids = set(deck)
    for cid in ids:
        nm = _names.get(cid, "")
        if nm == "Alakazam Powerful Hand" or cid == 743: return "Alakazam"
        if nm == "Hop's Trevenant": return "Trevenant"
        if nm == "Dragapult ex": return "Dragapult"
        if cid == 678: return "MegaLucario"
        if cid == 1031: return "MegaStarmie"
    return "OTHER"


def iter_decisions(game):
    steps = game.get("steps") or []
    info = game.get("info") or {}
    names = info.get("TeamNames") or [None, None]
    rewards = game.get("rewards") or [0, 0]
    decks = {seat: _deck_of(steps, seat) for seat in (0, 1)}
    for s in steps:
        for seat in (0, 1):
            obs = s[seat].get("observation", {})
            sel = obs.get("select"); cur = obs.get("current")
            act = s[seat].get("action")
            if not (sel and cur and isinstance(act, list)):
                continue
            opts = sel.get("option") or []
            if len(opts) < 2:
                continue
            if not all(isinstance(x, int) and 0 <= x < len(opts) for x in act):
                continue
            if decks[seat] is None:
                continue
            ctx = sel.get("context")
            yield {
                "deck_sig": deck_sig_of(decks[seat]),
                "player": names[seat],
                "reward": rewards[seat] if seat < len(rewards) else 0,
                "context": ctx,
                "state": state_features(cur, seat),
                "options": [option_features(o, ctx, sel) for o in opts],
                "chosen": act[0],
            }


def main():
    os.makedirs(DATA, exist_ok=True)
    files = {}
    meta = collections.defaultdict(lambda: {"player": None, "games": set(), "decisions": 0,
                                            "deck": None})
    for path in glob.glob(f"{ROOT}/json/*.json"):
        try:
            g = json.load(open(path))
        except Exception:
            continue
        gid = g.get("info", {}).get("EpisodeId", path)
        seen_sig = set()
        for r in iter_decisions(g):
            sig = r["deck_sig"]
            if sig not in files:
                files[sig] = open(os.path.join(DATA, f"{sig}.jsonl"), "a")
            files[sig].write(json.dumps(r) + "\n")
            m = meta[sig]; m["player"] = r["player"]; m["decisions"] += 1
            m["games"].add(gid); seen_sig.add(sig)
    for fh in files.values():
        fh.close()
    manifest = []
    for sig, m in meta.items():
        deck = [int(x) for x in sig.split("_")]
        manifest.append({"deck_sig": sig, "player": m["player"],
                         "archetype": _archetype(deck),
                         "games": len(m["games"]), "decisions": m["decisions"]})
    manifest.sort(key=lambda x: -x["games"])
    json.dump(manifest, open(os.path.join(DATA, "manifest.json"), "w"), indent=2)
    print(f"decks={len(manifest)} "
          f">=30g={sum(1 for x in manifest if x['games']>=30)}")
    for x in manifest[:12]:
        print(f"  {x['games']:4}g {x['decisions']:6}d  {x['archetype']:12} {x['player']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, then run the extractor for real**

Run: `cd /Users/chaturaattidiya/Documents/Github/kaggle/pokemon && python -m pytest tools/imitation/tests/test_extract.py -v`
Expected: PASS (2 tests).
Then: `rm -f tools/imitation/data/*.jsonl; python -m tools.imitation.extract_decisions`
Expected: prints `decks=... >=30g=...` and a top-12 list with the heavy hitters (Debauchery, Kadoraba, foo_foo, keidroid). `>=30g` should be ~15–20.

- [ ] **Step 5: Commit**

```bash
echo "tools/imitation/data/" >> .gitignore
git add tools/imitation/extract_decisions.py tools/imitation/tests/test_extract.py .gitignore
git commit -m "feat(imitation): replay decision extractor + per-deck tables/manifest"
```

---

### Task 3: Stdlib tree inference

**Files:**
- Create: `tools/imitation/policy.py`
- Test: `tools/imitation/tests/test_policy.py`

**Interfaces:**
- Produces: `score(tree: dict, x: list[float]) -> float` (walks a serialized sklearn tree, returns
  P(chosen) at the leaf), `pick(tree: dict, option_vecs: list[list[float]], state: list[float],
  min_count: int, max_count: int) -> list[int]` (argmax over options, returns chosen index list).
- Serialized tree dict shape (produced by Task 4):
  `{"feature":[int], "threshold":[float], "left":[int], "right":[int], "value":[float]}`
  where a node is a leaf when `left[i] == -1`; `value[i]` is the positive-class probability.

- [ ] **Step 1: Write the failing test**

```python
# tools/imitation/tests/test_policy.py
from tools.imitation.policy import score, pick

# tiny hand-built tree: if x[0] <= 0.5 -> leaf value 0.1 else leaf value 0.9
TREE = {"feature": [0, -1, -1], "threshold": [0.5, 0.0, 0.0],
        "left": [1, -1, -1], "right": [2, -1, -1], "value": [0.0, 0.1, 0.9]}

def test_score_walks_tree():
    assert score(TREE, [0.0]) == 0.1
    assert score(TREE, [1.0]) == 0.9

def test_pick_argmax_respects_single_select():
    # option 1 has higher feature -> higher score -> chosen
    state = []
    opts = [[0.0], [1.0]]
    out = pick(TREE, opts, state, min_count=1, max_count=1)
    assert out == [1]

def test_pick_multi_select_returns_top_k():
    opts = [[1.0], [0.0], [1.0]]
    out = pick(TREE, opts, [], min_count=2, max_count=2)
    assert len(out) == 2 and set(out) <= {0, 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chaturaattidiya/Documents/Github/kaggle/pokemon && python -m pytest tools/imitation/tests/test_policy.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/imitation/policy.py
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
    return scored[:k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/chaturaattidiya/Documents/Github/kaggle/pokemon && python -m pytest tools/imitation/tests/test_policy.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/imitation/policy.py tools/imitation/tests/test_policy.py
git commit -m "feat(imitation): stdlib serialized-tree inference (score/pick)"
```

---

### Task 4: Trainer + accuracy report (STAGE-1 GATE)

**Files:**
- Create: `tools/imitation/train.py`
- Test: `tools/imitation/tests/test_train.py`

**Interfaces:**
- Consumes: decision tables from Task 2, `policy.score`, `features.STATE_DIM/OPTION_DIM`.
- Produces: `export_tree(clf) -> dict` (sklearn tree → the serialized dict shape of Task 3);
  `train_deck(sig) -> dict` (`{tree, accuracy, baseline, n, by_context}`); `main()` that trains all
  ≥30g decks, writes `tools/imitation/policies/<sig>.json`, and prints an accuracy table.

- [ ] **Step 1: Write the failing test**

```python
# tools/imitation/tests/test_train.py
from sklearn.tree import DecisionTreeClassifier
from tools.imitation.train import export_tree
from tools.imitation.policy import score

def test_export_matches_sklearn():
    X = [[0.0], [0.1], [0.9], [1.0]]
    y = [0, 0, 1, 1]
    clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    tree = export_tree(clf)
    for x in X:
        # exported P(class=1) must equal sklearn's predict_proba[:,1]
        p = clf.predict_proba([x])[0][1]
        assert abs(score(tree, x) - p) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chaturaattidiya/Documents/Github/kaggle/pokemon && python -m pytest tools/imitation/tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/imitation/train.py
"""Per-deck imitation trainer. For each learnable decision, the chosen option is a positive example
(state+option, label 1) and the other options are negatives (label 0). A DecisionTree scores an
option; inference picks the argmax. Exports to the stdlib JSON tree shape and reports per-context
top-1 accuracy on a held-out split — the Stage-1 go/no-go signal. sklearn allowed HERE only."""
import json, os, glob, collections, random
from sklearn.tree import DecisionTreeClassifier
from tools.imitation.policy import score

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
    p = os.path.join(DATA, f"{sig}.jsonl")
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
        json.dump(res["tree"], open(os.path.join(POL, f"{m['deck_sig']}.json"), "w"))
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
```

- [ ] **Step 4: Run test, then run the trainer (the GATE)**

Run: `cd /Users/chaturaattidiya/Documents/Github/kaggle/pokemon && python -m pytest tools/imitation/tests/test_train.py -v`
Expected: PASS.
Then: `python -m tools.imitation.train`
Expected: an accuracy table + a `GATE:` line. **Decision:** if mean top-1 accuracy clearly exceeds the baseline (target lift ≥ +0.10, and ctx-0 lift positive), proceed to Stage 2. If lift ≈ 0, STOP — write findings into the spec as "imitation near-random; salvage as insight," and do not build the pilot.

- [ ] **Step 5: Commit**

```bash
git add tools/imitation/train.py tools/imitation/tests/test_train.py tools/imitation/policies/
git commit -m "feat(imitation): per-deck tree trainer + accuracy gate report"
```

---

## STAGE 2 — Pilot + league (Tasks 5–6, only if gate passes)

### Task 5: Imitation pilot

**Files:**
- Create: `autoresearch/agent_imitation.py`
- Test: `tools/imitation/tests/test_pilot.py`

**Interfaces:**
- Consumes: `features.state_features/option_features`, `policy.pick`.
- Produces: `agent(obs_dict) -> list[int]`; deck via env `IMIT_DECK`/`deck.csv`; policy via env
  `IMIT_POLICY`/`policy.json`.

**NOTE on import constraint:** the pilot must be a single self-contained file when shipped (Kaggle copies one agent dir). For local league use it imports `tools.imitation.features`/`policy`; for a real submission those two modules are concatenated/copied beside it. The pilot keeps featurizer+policy logic importable but ALSO works if those funcs are defined inline — implement by importing with a try/except that falls back to inline copies. To stay DRY and avoid drift, import at module top; the submission-packaging step (out of scope) inlines them.

- [ ] **Step 1: Write the failing test**

```python
# tools/imitation/tests/test_pilot.py
import json, glob, importlib.util, os

def _load_agent():
    spec = importlib.util.spec_from_file_location(
        "agent_imitation", "autoresearch/agent_imitation.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def test_deck_phase_returns_60(tmp_path, monkeypatch):
    # any learned policy + the matching deck
    manifest = json.load(open("tools/imitation/data/manifest.json"))
    sig = next(m["deck_sig"] for m in manifest if m["games"] >= 30)
    deck = sig.split("_")
    deckcsv = tmp_path / "deck.csv"; deckcsv.write_text("\n".join(deck))
    monkeypatch.setenv("IMIT_DECK", str(deckcsv))
    monkeypatch.setenv("IMIT_POLICY", f"tools/imitation/policies/{sig}.json")
    mod = _load_agent()
    out = mod.agent({"select": None})
    assert isinstance(out, list) and len(out) == 60

def test_in_play_returns_legal_indices(tmp_path, monkeypatch):
    manifest = json.load(open("tools/imitation/data/manifest.json"))
    sig = next(m["deck_sig"] for m in manifest if m["games"] >= 30)
    deckcsv = tmp_path / "deck.csv"; deckcsv.write_text("\n".join(sig.split("_")))
    monkeypatch.setenv("IMIT_DECK", str(deckcsv))
    monkeypatch.setenv("IMIT_POLICY", f"tools/imitation/policies/{sig}.json")
    mod = _load_agent()
    # find a real decision obs
    for f in sorted(glob.glob("json/*.json"))[:10]:
        g = json.load(open(f))
        for s in g["steps"]:
            for seat in (0, 1):
                obs = s[seat].get("observation", {})
                sel = obs.get("select"); cur = obs.get("current")
                if sel and cur and sel.get("option") and len(sel["option"]) >= 2:
                    out = mod.agent({"select": sel, "current": cur})
                    n = len(sel["option"])
                    assert all(0 <= i < n for i in out)
                    assert len(out) >= sel.get("minCount", 1)
                    return
    raise RuntimeError("no decision found")

def test_malformed_obs_does_not_raise(monkeypatch):
    monkeypatch.setenv("IMIT_DECK", "tools/imitation/data/manifest.json")  # wrong file -> loader must cope
    mod = _load_agent()
    assert mod.agent("garbage") is not None
    assert mod.agent({"select": {"option": [], "minCount": 0}}) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chaturaattidiya/Documents/Github/kaggle/pokemon && python -m pytest tools/imitation/tests/test_pilot.py -v`
Expected: FAIL (`agent_imitation.py` does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# autoresearch/agent_imitation.py
"""Generic imitation pilot: pilots any deck via a learned per-deck policy tree. STDLIB ONLY.
Reuses the v19 hardening: raw-dict deck-phase, try/except legal fallback, NO __file__."""
import os, json

try:  # local league use
    from tools.imitation.features import state_features, option_features
    from tools.imitation.policy import pick
except Exception:  # shipped/inlined use — see packaging note in plan
    raise


def _read_deck():
    path = os.environ.get("IMIT_DECK")
    if not path:
        path = "deck.csv"
        if not os.path.exists(path):
            path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as f:
        return [int(line) for line in f if line.strip()][:60]


def _read_policy():
    path = os.environ.get("IMIT_POLICY")
    if not path:
        path = "policy.json"
        if not os.path.exists(path):
            path = "/kaggle_simulations/agent/policy.json"
    try:
        return json.load(open(path))
    except Exception:
        return None


_DECK = _read_deck()
_POLICY = _read_policy()


def _legal_fallback(obs_dict):
    try:
        sel = (obs_dict or {}).get("select") or {}
        n = len(sel.get("option") or [])
        return list(range(min(max(0, sel.get("minCount", 0)), n)))
    except Exception:
        return []


def agent(obs_dict):
    try:
        if not isinstance(obs_dict, dict) or obs_dict.get("select") is None:
            return _DECK
    except Exception:
        return _DECK
    try:
        sel = obs_dict["select"]
        opts = sel.get("option") or []
        if not opts:
            return []
        if _POLICY is None:
            return _legal_fallback(obs_dict)
        cur = obs_dict.get("current") or {}
        seat = cur.get("yourIndex", 0)
        ctx = sel.get("context")
        state = state_features(cur, seat)
        ovecs = [option_features(o, ctx, sel) for o in opts]
        return pick(_POLICY, ovecs, state,
                    sel.get("minCount", 1), sel.get("maxCount", 1))
    except Exception:
        return _legal_fallback(obs_dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/chaturaattidiya/Documents/Github/kaggle/pokemon && python -m pytest tools/imitation/tests/test_pilot.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add autoresearch/agent_imitation.py tools/imitation/tests/test_pilot.py
git commit -m "feat(imitation): generic stdlib imitation pilot (deck+policy -> option)"
```

---

### Task 6: League / oracle upgrade

**Files:**
- Create: `tools/imitation/league.py`
- Modify: none (reuses docker `eval.py`; reads `extract_field.py` weights)
- Test: `tools/imitation/tests/test_league.py`

**Interfaces:**
- Consumes: `tools/imitation/data/manifest.json`, `autoresearch/decks/field/weights.json`,
  `autoresearch/agent_imitation.py`, docker `eval.py`.
- Produces: `field_weighted(candidate_slug, pilot, games) -> dict`
  (`{field_weighted, raw_avg, by_archetype}`), and a `main()` that scores each of our contender
  decks against the imitation field and prints a ranked table. Opponent decks with a learned policy
  use `agent_imitation.py` + their policy; thin decks use `agent_typh.py`.

- [ ] **Step 1: Write the failing test**

```python
# tools/imitation/tests/test_league.py
from tools.imitation.league import opponent_pilot

def test_opponent_pilot_routing():
    # a deck_sig with a policy file -> imitation; else typh
    import os, glob
    pol = sorted(glob.glob("tools/imitation/policies/*.json"))
    assert pol, "need at least one trained policy (run Task 4)"
    sig = os.path.splitext(os.path.basename(pol[0]))[0]
    kind, _ = opponent_pilot(sig)
    assert kind == "imitation"
    kind2, _ = opponent_pilot("999999_does_not_exist")
    assert kind2 == "typh"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chaturaattidiya/Documents/Github/kaggle/pokemon && python -m pytest tools/imitation/tests/test_league.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# tools/imitation/league.py
"""Upgraded oracle: score our candidate decks vs the real ladder field, but pilot each opponent
deck with its LEARNED policy (imitation) instead of generic typh. Thin decks fall back to typh.
Reuses the docker eval harness and extract_field weights."""
import os, json, glob, subprocess, sys, statistics as st
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POL = os.path.join(os.path.dirname(__file__), "policies")
FIELD = json.load(open(f"{ROOT}/autoresearch/decks/field/weights.json"))


def opponent_pilot(deck_sig):
    p = os.path.join(POL, f"{deck_sig}.json")
    if os.path.exists(p):
        return "imitation", p
    return "typh", None


def _field_sig(slug):
    deck = [int(x) for x in open(f"{ROOT}/autoresearch/decks/field/{slug}.csv")]
    return "_".join(str(x) for x in sorted(deck))


def _run(cand_slug, pilot, dragdeck, opp_slug, games):
    sig = _field_sig(opp_slug)
    kind, polpath = opponent_pilot(sig)
    champ = "agent_imitation.py" if kind == "imitation" else "agent_typh.py"
    env = ["-e", "PYTHONPATH=/app/sdk:/app"]
    if dragdeck:
        env += ["-e", f"DRAG_DECK={dragdeck}"]
    if kind == "imitation":
        env += ["-e", f"IMIT_DECK=/app/autoresearch/decks/field/{opp_slug}.csv",
                "-e", f"IMIT_POLICY=/app/{os.path.relpath(polpath, ROOT)}"]
    out = subprocess.run(
        ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{ROOT}:/app",
         "-w", "/app/autoresearch"] + env + ["python:3.11-slim",
         "python", "eval.py", "--challenger", pilot, "--champion", champ,
         "--deck", f"decks/{cand_slug}.csv",
         "--deck-champion", f"decks/field/{opp_slug}.csv", "--games", str(games)],
        capture_output=True, text=True, timeout=1200).stdout
    for ln in out.splitlines():
        if '"score"' in ln:
            try:
                return float(ln.split(":")[1].strip().rstrip(","))
            except Exception:
                pass
    return None


def field_weighted(cand_slug, pilot, dragdeck="", games=40, workers=8):
    pool = ThreadPoolExecutor(max_workers=workers)
    futs = {m["slug"]: pool.submit(_run, cand_slug, pilot, dragdeck, m["slug"], games)
            for m in FIELD}
    num = den = 0.0; raw = []; per = {}
    for m in FIELD:
        sc = futs[m["slug"]].result()
        if sc is None:
            continue
        w, a = m["count"], m["archetype"]
        num += w * sc; den += w; raw.append(sc)
        per.setdefault(a, [0.0, 0]); per[a][0] += w * sc; per[a][1] += w
    return {"field_weighted": round(num / den, 3) if den else 0.0,
            "raw_avg": round(st.mean(raw), 3) if raw else 0.0,
            "by_archetype": {a: round(v[0]/v[1], 3) for a, v in per.items() if v[1]}}


CONTENDERS = [
    ("alakazam_top", "agent_alakazam.py", ""),
    ("dragapult", "agent_dragapult.py", "decks/dragapult.csv"),
]


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    res = {}
    for slug, pilot, dd in CONTENDERS:
        r = field_weighted(slug, pilot, dd, games)
        res[slug] = r
        print(f"  {r['field_weighted']:.3f}  {slug:16} raw={r['raw_avg']} {r['by_archetype']}")
    json.dump(res, open("/tmp/imitation_league.json", "w"), indent=2)
    print("\nValidation: compare Alakazam by_archetype['Trevenant'] vs the typh-oracle's 0.80 "
          "and the real 0.36 — it should drop toward 0.36 if imitation works.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, then a small live league**

Run: `cd /Users/chaturaattidiya/Documents/Github/kaggle/pokemon && python -m pytest tools/imitation/tests/test_league.py -v`
Expected: PASS.
Then (docker must be up): `python -m tools.imitation.league 20`
Expected: a ranked table; check Alakazam's `by_archetype['Trevenant']` — success = it moved down from the typh-oracle's ~0.80 toward the real 0.36. Record the number in the spec's Stage-2 validation note.

- [ ] **Step 5: Commit**

```bash
git add tools/imitation/league.py tools/imitation/tests/test_league.py
git commit -m "feat(imitation): league/oracle upgrade — opponents piloted by learned policies"
```

---

## Self-Review notes (addressed)

- **Spec coverage:** Component 1→Task 2, Component 2→Task 1, Component 3→Tasks 3+4, Component 4→Task 5,
  Component 5→Task 6. Staged gate = Task 4 Step 4. Featurizer parity risk = single shared module
  (Task 1) used by trainer and pilot. Trevenant-validation = Task 6 Step 4.
- **Packaging caveat:** shipping `agent_imitation.py` to Kaggle (inlining features/policy beside it)
  is explicitly out of scope per spec; the league uses it via local import with `PYTHONPATH=/app`.
- **Type consistency:** serialized tree dict shape (`feature/threshold/left/right/value`) defined in
  Task 3 interface and produced identically by `export_tree` in Task 4. `state⊕option` ordering is
  `state + option` everywhere (`train_deck`, `pick`, pilot).
