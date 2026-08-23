"""Opponent hidden-state determinization via archetype priors (stdlib-only, offline-safe).

Fixes the prior forward-search failure: the old approach filled hidden opponent cards with a
single basic-energy id, producing implausible states (strategy fusion + nonlocality, wiki/17).
This module instead:
  1. infers the opponent archetype from revealed cards (hybrid: signature card, else a
     card-frequency fingerprint scored against per-archetype priors),
  2. samples the hidden portion of their deck from that archetype's empirical card distribution,
     consistent with what has already been revealed.

Priors are precomputed offline (arch_priors.json) from unique ladder decks per archetype, so
this ships without the replay DB. Validated held-out (determinize_eval.py / classify_eval.py):
  - archetype-ID: 20%->88% at 6 revealed cards (fingerprint vs single signature card)
  - hidden-deck recall: ~5% (old filler) -> 21-40% (this), improving as more cards are revealed.

API:
  d = Determinizer.load("arch_priors.json")
  arch = d.infer_archetype(revealed_ids)                 # str archetype or best guess
  hidden = d.sample_hidden(revealed_ids, hidden_count, arch=None, rng=None)  # list[int] card ids
"""
import json, math, random

# signature cards: (predicate on id/name) -> archetype. Unambiguous when present.
def _sig_archetype(revealed_ids, names):
    s = set(revealed_ids)
    for cid in s:
        if "Bellibolt" in names.get(cid, ""): return "Bellibolt"
    for cid in s:
        nm = names.get(cid, "")
        if nm == "Alakazam Powerful Hand" or cid == 743: return "Alakazam"
        if nm == "Hop's Trevenant": return "Trevenant"
        if nm == "Dragapult ex": return "Dragapult"
        if cid == 678: return "MegaLucario"
        if cid == 1031: return "MegaStarmie"
        if cid == 345: return "Crustle"
    return None

FILLER = 1  # basic energy fallback when priors can't cover the count


class Determinizer:
    def __init__(self, priors, names=None):
        # priors[arch] = {n_decks, p_present:{cid:float}, mean_copies:{cid:float}}
        self.priors = priors
        self.names = names or {}
        self.arches = [a for a in priors if priors[a].get("n_decks", 0) > 0]
        tot = sum(priors[a]["n_decks"] for a in self.arches) or 1
        self.base = {a: priors[a]["n_decks"] / tot for a in self.arches}
        self.vocab = set()
        for a in self.arches:
            self.vocab |= set(int(c) for c in priors[a]["p_present"])

    @classmethod
    def load(cls, priors_path, cards_csv=None):
        priors_raw = json.load(open(priors_path))
        priors = {}
        for a, d in priors_raw.items():
            priors[a] = {
                "n_decks": d["n_decks"],
                "p_present": {int(k): float(v) for k, v in d["p_present"].items()},
                "mean_copies": {int(k): float(v) for k, v in d["mean_copies"].items()},
            }
        names = {}
        if cards_csv:
            import csv
            names = {int(r["cardId"]): r["name"] for r in csv.DictReader(open(cards_csv))}
        return cls(priors, names)

    def _p_present(self, a, cid):  # Laplace-smoothed presence prob
        p = self.priors[a]["p_present"].get(cid)
        if p is None:
            return 1.0 / (self.priors[a]["n_decks"] + 2)
        return p

    def _fingerprint(self, revealed_ids):
        rs = set(revealed_ids)
        best, bestlp = None, -1e18
        for a in self.arches:
            lp = math.log(self.base[a])
            for cid in rs:
                if cid in self.vocab:
                    lp += math.log(self._p_present(a, cid))
            if lp > bestlp:
                bestlp, best = lp, a
        return best

    def infer_archetype(self, revealed_ids):
        """Hybrid: unambiguous signature card wins, else card-frequency fingerprint."""
        sig = _sig_archetype(revealed_ids, self.names)
        return sig if sig is not None else self._fingerprint(revealed_ids)

    def sample_hidden(self, revealed_ids, hidden_count, arch=None, rng=None, stochastic=True):
        """Return `hidden_count` predicted opponent card ids for the unseen slots.
        stochastic=True: sample each remaining slot ~ mean_copies distribution (for ISMCTS
        re-determinization per simulation). stochastic=False: deterministic MAP fill (for a
        single best-guess forward pass)."""
        if arch is None:
            arch = self.infer_archetype(revealed_ids)
        if arch not in self.priors:
            return [FILLER] * hidden_count
        from collections import Counter
        mc = self.priors[arch]["mean_copies"]
        revealed_c = Counter(revealed_ids)
        # expected remaining copies per card = round(mean) - already revealed
        remaining = {}
        for cid, m in mc.items():
            rem = round(m) - revealed_c.get(cid, 0)
            if rem > 0:
                remaining[cid] = rem
        if not stochastic:
            pool = []
            for cid, rem in sorted(remaining.items(), key=lambda kv: -mc[kv[0]]):
                pool += [cid] * rem
            pred = pool[:hidden_count]
            return pred + [FILLER] * (hidden_count - len(pred))
        # stochastic: weighted sampling without replacement, weight = mean_copies
        rng = rng or random.Random()
        cids = list(remaining.keys())
        weights = {cid: mc[cid] for cid in cids}
        pred = []
        avail = dict(remaining)
        for _ in range(hidden_count):
            live = [c for c in cids if avail.get(c, 0) > 0]
            if not live:
                pred.append(FILLER); continue
            w = [weights[c] for c in live]
            pick = rng.choices(live, weights=w, k=1)[0]
            pred.append(pick); avail[pick] -= 1
        return pred


if __name__ == "__main__":
    import os
    root = os.path.dirname(os.path.abspath(__file__))
    d = Determinizer.load(os.path.join(root, "arch_priors.json"),
                          os.path.join(root, "autoresearch/cards_full.csv"))
    # smoke: give it a few Dragapult signature cards, ask for a hidden fill
    dreepy = [cid for cid, nm in d.names.items() if nm == "Dreepy"][:1]
    revealed = dreepy + [cid for cid, nm in d.names.items() if nm == "Dragapult ex"][:1]
    arch = d.infer_archetype(revealed)
    hid = d.sample_hidden(revealed, 20, rng=random.Random(0))
    from collections import Counter
    print("inferred:", arch)
    print("sampled hidden (top):", [(d.names.get(c, c), n) for c, n in Counter(hid).most_common(8)])
