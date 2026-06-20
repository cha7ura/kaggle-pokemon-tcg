# 17 — Imperfect information on the TCG (the bridge)

← [16 — PTCG-Bench](16-ptcg-bench.md) · index → [00](00-index.md)

This is the page the whole wiki builds toward. You've seen the methods
([CFR](10-cfr-and-nash.md), [Player of Games](11-player-of-games.md),
[DeepNash](12-deepnash-rnad.md), [AlphaStar](13-alphastar-league.md)). Now the honest question:
**what actually ports to *our* agent** — offline, stdlib-only, no GPU and no network at inference,
on a hidden-information, stochastic card game with a huge variable action space?

## What does NOT port (and why)

- **Player of Games' GT-CFR search at inference** — re-solving a subgame every move is impractical on
  the ladder (no per-move search budget; stdlib only). And full CFR needs the info-set tree, which is
  astronomically large for the TCG. PoG is the *right* idea, wrong deployment for us
  ([11](11-player-of-games.md)).
- **DeepNash / AlphaStar at their scale** — both used large compute (Stratego's 10^535 tree; an
  agent league). One Colab GPU can't reproduce that ([12](12-deepnash-rnad.md), [13](13-alphastar-league.md)).
- **Tabular CFR on the full game** — the foundation ([10](10-cfr-and-nash.md)) is exact but only
  enumerable on *toy* games (Kuhn, Leduc). Not the real TCG.
- **MCTS/AlphaZero directly** — unsound for imperfect information (next section).

## The trap to avoid: strategy fusion

The tempting shortcut is *determinization* (a.k.a. perfect-information Monte Carlo): guess the hidden
cards, solve the resulting perfect-info game, average over guesses. The engine even makes this easy
via `search_begin` + `_determinize` ([01](01-game-and-engine.md)). But it has two named flaws
[ismcts2012]:

- **Strategy fusion** — the solver assumes it can make *different* decisions in states that are
  actually in the same info set (it "knows" which determinization it's in). You can't — you only see
  the info set. This silently overestimates your control.
- **Nonlocality** — some sampled determinizations are vanishingly unlikely because a smart opponent
  would steer away from them, yet they still skew the average.

This is the theory behind a result the project already hit empirically: hand-tuned forward search
*lost* to the heuristic, twice ([03](03-autoresearch-loop.md)).

## What DOES port

- **Information-Set MCTS (ISMCTS)** — instead of many determinization trees, build **one tree whose
  nodes are info sets**, re-determinizing per simulation. It pools statistics in a single tree
  (better budget use) and reduces strategy fusion [ismcts2012]. Honest caveat: the win over plain
  determinized search is **domain-dependent** — in games with huge branching it's only on par
  [ismcts2012], and the TCG's action space is large. So ISMCTS is the *practical bridge* on the real
  `cg` sim, worth trying, not a guaranteed win. (minizero `cg/`, Plan 2.)
- **Tabular CFR / R-NaD on a toy** — Kuhn/Leduc are small enough to run CFR ([10](10-cfr-and-nash.md))
  and tabular R-NaD ([12](12-deepnash-rnad.md)) to convergence in pure NumPy, *watching exploitability
  → 0*. This is the genuine, verifiable member of the family — the teaching core and the correctness
  oracle for the net.
- **The regret / equilibrium *intuition*** — even the hand heuristic benefits: don't be exploitable,
  don't tune against a single opponent ([13](13-alphastar-league.md) → the Dragapult benchmark
  alongside the mirror).

## The one shippable neural path

The deployment constraint picks the method. **R-NaD is search-free at inference**
([12](12-deepnash-rnad.md)) — the trained policy net just emits an action distribution. So:

```
train R-NaD on GPU (torch, Colab)  ──>  export weights.npz  ──>  agent_net.py: NumPy forward pass
                                                                  (mask illegal actions, pick)
```

No torch, no search, no forward model on the ladder — just `np.dot`s over the encoded observation.
That is the *only* method in this lineage that both (a) genuinely seeks Nash and (b) fits an offline
stdlib agent. It's why the project chose R-NaD over Player of Games, and it feeds the same
[challenger gate](04-the-challenger-framework.md) as everything else.

## The honest bottom line

Today, the **hand-tuned heuristic (820.7) is the right answer** — it encodes real domain knowledge
cheaply and ships trivially. The imperfect-information theory tells us *where* a learned agent could
eventually beat it (an unexploitable, opponent-aware policy) and *which* method can actually ship
(R-NaD → NumPy). PTCG-Bench warns the learned route is unstable ([16](16-ptcg-bench.md)), so we
pursue it as a **gated challenger**, never as a bet against what already works.

> Takeaway: don't port PoG/DeepNash/AlphaStar wholesale. Port the *ideas* — ISMCTS + determinization
> on the real sim, tabular CFR/R-NaD on a toy to learn the math, and one shippable neural path
> (R-NaD trained on GPU, run as a NumPy forward pass) — all behind the same Wilson-LB gate.
