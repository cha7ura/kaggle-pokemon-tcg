"""The gameplay policy: wraps the trained card-policy forest into an agent. Scores every legal option
with the model (name-aligned, so dropped features are handled), applies a lethal gate, and ALWAYS
returns a legal action (never crashes — a hard scoring constraint). Same module drives the league
pilot and the Kaggle submission.

  CardPolicy(deck, model_path).act(obs_dict) -> list[int]

Stdlib + numpy. featurize/threat/deck_tracker/card_features must be importable (bundled in submission).
"""
import os
from . import featurize, threat, model_io, forward_search

DATA = os.path.join(os.path.dirname(__file__), "data")


class CardPolicy:
    def __init__(self, deck, model=None):
        self.deck = list(deck)
        m = model or os.path.join(DATA, "card_policy.npz")
        self.packed = model_io.load(m) if isinstance(m, str) else m
        self.names = model_io.names(self.packed)

    def _vec(self, full, opt_pairs):
        d = dict(full)
        d.update(opt_pairs)
        return [d.get(n, 0.0) for n in self.names]

    # ---- the agent ----
    def act(self, obs):
        sel = obs.get("select")
        cur = obs.get("current")
        # deck-selection phase: return the 60-card deck
        if not sel or not cur:
            return list(self.deck)
        opts = sel.get("option") or []
        n = len(opts)
        if n == 0:
            return []
        min_c = max(1, sel.get("minCount", 1) or 1)
        max_c = sel.get("maxCount", min_c) or min_c
        try:
            seat = cur.get("yourIndex", 0)
            sv, sn = featurize.state_row(obs, seat, self.deck)
            full = list(zip(sn, sv))
            rows = []
            for opt in opts:
                ov, on = featurize.option_row(obs, seat, opt, self.deck)
                rows.append(self._vec(full, list(zip(on, ov))))
            scores = model_io.score(self.packed, rows)        # batch-score all options at once
            order = sorted(range(n), key=lambda i: scores[i], reverse=True)
            # forward-search gate: for ATTACK choices, the engine resolves EXACT damage (beats the
            # model + threat estimate). Prefer its pick when available (live obs + engine present).
            fs = forward_search.best_option(obs, seat, self.deck)
            if fs is not None and 0 <= fs < n:
                order = [fs] + [i for i in order if i != fs]
            else:
                # lethal gate (estimate-based): take an ATTACK that can KO the opp active
                lethal = self._lethal_index(obs, seat, opts)
                if lethal is not None and lethal in order:
                    order = [lethal] + [i for i in order if i != lethal]
            k = min(max(min_c, 1), max_c, n)
            return order[:k]
        except Exception:
            return self._fallback(sel)

    def _lethal_index(self, obs, seat, opts):
        try:
            cur = obs["current"]; opp = cur["players"][1 - seat]
            oa = (opp.get("active") or [None])[0]
            if not oa:
                return None
            dmg = threat.our_max_damage_vs_each(obs, seat).get(("ACTIVE", 0), 0)
            if dmg < oa.get("hp", 1):
                return None
            for i, o in enumerate(opts):
                if o.get("type") == 13:           # OptionType.ATTACK
                    return i
        except Exception:
            pass
        return None

    @staticmethod
    def _fallback(sel):
        """Never crash: return a legal selection. Prefer END (pass) at MAIN, else the minimum legal set."""
        opts = sel.get("option") or []
        min_c = max(1, sel.get("minCount", 1) or 1)
        for i, o in enumerate(opts):
            if o.get("type") == 14:               # OptionType.END
                return [i]
        return list(range(min(min_c, len(opts)))) if opts else []


def load_default(deck):
    return CardPolicy(deck)
