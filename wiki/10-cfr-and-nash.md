# 10 — CFR and Nash equilibria (the foundation)

← [04 — challenger framework](04-the-challenger-framework.md) · next → [11 — Player of Games](11-player-of-games.md)

Every imperfect-information method in this wiki (Player of Games, DeepNash) is, underneath,
an answer to one question: **how do you find an unexploitable strategy in a game with hidden
information?** The classical answer is **Counterfactual Regret Minimization (CFR)**. This page is
the foundation; read it before 11–13.

## Why "best move" isn't a thing here

In chess (perfect information), there's a best move in every position. In poker or the Pokémon TCG,
there isn't — if you *always* do X, the opponent learns to punish X. The right object is a
**mixed strategy** (a probability distribution over actions) that no opponent can exploit: a
**Nash equilibrium**. The yardstick is **exploitability** — how much a worst-case opponent beats
you. Nash = exploitability 0.

## Information sets

You don't know the true state, only your **information set** (`I`) — the set of states consistent
with everything you've observed. In the TCG, "my hand + the visible board" is one info set; the
opponent's hand and deck order make many true states indistinguishable to you. CFR reasons over
info sets, not states.

## Regret matching: the whole trick in one rule

CFR plays itself many times. At each info set it tracks, per action, a **cumulative counterfactual
regret** — roughly "how much better off I'd have been had I always played this action." The next
strategy is **regret matching**: play each action in proportion to its *positive* cumulative
regret, and uniformly when none is positive [cfr2013][labmlcfr]:

```
R⁺(I,a) = max(R(I,a), 0)
σ(I,a) = R⁺(I,a) / Σ_a' R⁺(I,a')      if the sum > 0
       = 1/|A(I)|                      otherwise (uniform)
```

The counterfactual value/regret definitions [cfr2013]:

```
v_i(σ, h) = Σ_{z⊒h} π^σ_{-i}(h) · π^σ(h,z) · u_i(z)     # value weighted by reach prob of OTHERS
r(h,a)    = v_i(σ_{I→a}, h) − v_i(σ, h)                  # regret of not playing a
R_i^T(I,a)= Σ_t r_i^t(I,a)                               # accumulate over iterations
```

The `π_{-i}` weighting ("counterfactual" = *as if* you'd aimed to reach `I`, times the opponents'
actual reach probability) is what makes the regrets decompose correctly across the tree.

## The one thing everyone gets wrong

> It is **the average strategy over all iterations**, not the final/current strategy, that
> converges to a Nash equilibrium. [cfr2013]

The current strategy keeps moving; the *time-average* settles. Concretely, when every player's
average regret is below ε, the average profile is a **2ε-Nash equilibrium** [labmlcfr]. So a CFR
implementation must accumulate and return the average strategy — forgetting this is the classic bug.

## The variants you'd actually implement

- **Vanilla CFR** — full tree traversal each iteration. Exact, but cost scales with the game tree.
- **MCCFR (Monte Carlo CFR)** — *sample* the tree (e.g. chance sampling, external sampling) and
  estimate regrets. The canonical teaching demo runs MCCFR on **Kuhn poker** [labmlcfr]. This is
  the ~200-line NumPy artifact minizero builds (Plan 2).
- **CFR+** — regret-matching⁺ (floors regrets at 0 every step) + linear averaging; "typically an
  order of magnitude or more faster" than CFR [cfrplus2014]. The same CFR+ machinery powers Player
  of Games' inner loop ([11](11-player-of-games.md)).

## Why this matters for our project

The autoresearch log already discovered, the hard way, that hand-tuned forward search *lost* to the
heuristic and "needs a learned value net" ([03](03-autoresearch-loop.md), SEARCH-v1). CFR is the
principled alternative: instead of guessing values, minimize regret to provably approach Nash. The
honest catch — full CFR needs the whole info-set tree, which for the TCG is astronomical. That gap
(toy CFR you can verify vs. a real game you can't enumerate) is exactly what
[11](11-player-of-games.md), [12](12-deepnash-rnad.md), and the bridge
[17](17-imperfect-info-on-tcg.md) are about.

> Takeaway: find a strategy nobody can exploit (Nash); reach it by minimizing counterfactual
> regret via regret matching; return the **average** strategy. Everything else is scaling this.
