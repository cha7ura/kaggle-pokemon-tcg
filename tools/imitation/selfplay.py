"""Self-play data generation (docker). CardPolicy plays itself with TEMPERATURE exploration (sample,
not argmax) across the field decks; we keep the WINNER's decisions as new training rows. Exploration
is the improvement signal: winners that explored better-than-argmax lines pull the policy upward.
Pure argmax self-play would only reinforce the current policy (the old RL trap) — hence temperature.

  python tools/imitation/selfplay.py --games 6 --temp 1.0 --out data/selfplay.npz [--slugs f00 f01 ..]

Output NPZ (same schema as build_decisions): X, y, groups, names — concat with the imitation set + retrain.
Gate downstream: only keep the retrained model if it beats the typh baseline AND the prior model.
"""
import sys, os, json, math, random
import numpy as np
sys.path.insert(0, "/app"); sys.path.insert(0, "/app/autoresearch")
from eval import play_one
from tools.imitation import featurize, model_io

DATA = "/app/tools/imitation/data"
FIELD = "/app/autoresearch/decks/field"


def _deck(slug):
    return [int(x) for x in open(f"{FIELD}/{slug}.csv") if x.strip().isdigit()][:60]


class Recorder:
    """A sampling CardPolicy that logs its decisions (for later winner-filtering)."""
    def __init__(self, deck, packed, temp):
        self.deck, self.packed, self.names, self.temp = deck, packed, model_io.names(packed), temp
        self.log = []   # list of (state_vec, [opt_vecs], chosen_idx, group_step)
        self.step = 0

    def act(self, obs):
        sel = obs.get("select"); cur = obs.get("current")
        if not sel or not cur:
            return list(self.deck)
        opts = sel.get("option") or []
        n = len(opts)
        if n == 0:
            return []
        self.step += 1
        try:
            seat = cur.get("yourIndex", 0)
            sv, sn = featurize.state_row(obs, seat, self.deck)
            full = list(zip(sn, sv))
            ovecs, rows = [], []
            for opt in opts:
                ov, on = featurize.option_row(obs, seat, opt, self.deck)
                ovecs.append(ov)
                d = dict(full); d.update(zip(on, ov))
                rows.append([d.get(k, 0.0) for k in self.names])
            scores = model_io.score(self.packed, rows)
            # softmax sample with temperature (exploration)
            mx = max(scores)
            ws = [math.exp((s - mx) / max(1e-3, self.temp)) for s in scores]
            tot = sum(ws) or 1.0
            r = random.random() * tot
            idx, acc = 0, 0.0
            for i, w in enumerate(ws):
                acc += w
                if r <= acc:
                    idx = i; break
            self.log.append((list(sv), ovecs, idx, self.step))
            k = min(max(1, sel.get("minCount", 1) or 1), sel.get("maxCount", 1) or 1, n)
            # return sampled + fill to k with best others (legal)
            order = sorted(range(n), key=lambda i: scores[i], reverse=True)
            pick = [idx] + [i for i in order if i != idx]
            return pick[:k]
        except Exception:
            for i, o in enumerate(opts):
                if (o or {}).get("type") == 14:
                    return [i]
            return list(range(min(max(1, sel.get("minCount", 1) or 1), n)))

    def rows(self, gid):
        """Emit training rows (one per option, label=1 if chosen) for this game's decisions."""
        X, y, g = [], [], []
        for sv, ovecs, chosen, step in self.log:
            for oi, ov in enumerate(ovecs):
                X.append(sv + ov); y.append(1 if oi == chosen else 0); g.append(f"{gid}:{step}")
        return X, y, g


def main():
    games = int(sys.argv[sys.argv.index("--games") + 1]) if "--games" in sys.argv else 6
    temp = float(sys.argv[sys.argv.index("--temp") + 1]) if "--temp" in sys.argv else 1.0
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"{DATA}/selfplay.npz"
    if "--slugs" in sys.argv:
        slugs = sys.argv[sys.argv.index("--slugs") + 1:]
    else:
        slugs = ["f00", "f01", "f02", "f04", "f10"]   # Trev/Ala/Drag/Lucario/Starmie
    packed = model_io.load(f"{DATA}/card_policy.npz")
    X, y, G = [], [], []
    gid = 0
    pairs = [(a, b) for i, a in enumerate(slugs) for b in slugs[i:]]   # incl mirrors
    for a, b in pairs:
        da, db = _deck(a), _deck(b)
        for gi in range(games):
            ra = Recorder(da, packed, temp); rb = Recorder(db, packed, temp)
            if gi % 2 == 0:
                res = play_one(da, db, ra.act, rb.act); win = ra if res == 0 else rb if res == 1 else None
            else:
                res = play_one(db, da, rb.act, ra.act); win = rb if res == 0 else ra if res == 1 else None
            gid += 1
            if win is not None:
                xs, ys, gs = win.rows(gid)
                X += xs; y += ys; G += gs
            if gid % 10 == 0:
                print(f"  {gid} games, {len(X)} winner-rows", flush=True)
    if not X:
        print("no rows"); return
    names = featurize.STATE_NAMES + featurize.OPTION_NAMES
    np.savez_compressed(out, X=np.asarray(X, np.float32), y=np.asarray(y, np.int8),
                        groups=np.asarray(G), names=np.asarray(names))
    print(f"DONE selfplay: {gid} games -> {len(X)} winner-rows, {int(np.sum(y))} pos -> {out}", flush=True)


if __name__ == "__main__":
    main()
