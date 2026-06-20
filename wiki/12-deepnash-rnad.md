# 12 — DeepNash and R-NaD (search-free equilibrium)

← [11 — Player of Games](11-player-of-games.md) · next → [13 — AlphaStar and the league](13-alphastar-league.md)

DeepNash (Perolat et al. 2022) masters Stratego — and it is **the method this project ships toward**,
because of one property: it needs **no search at decision time** [deepnash2022]. That makes a
trained DeepNash-style net deployable as a plain forward pass on the offline Kaggle ladder, which
Player of Games ([11](11-player-of-games.md)) cannot be.

## The dumb version, and why it cycles

The dumb idea: "just do self-play RL until it's good." The problem in a zero-sum hidden-info game is
that naive self-play **cycles** — A beats B beats C beats A, forever, never settling (the same
non-transitivity AlphaStar fights, [13](13-alphastar-league.md)). You orbit the equilibrium but
never land on it.

## R-NaD: reshape the dynamics so they converge

DeepNash's core is **Regularized Nash Dynamics (R-NaD)** [deepnash2022]:

> R-NaD converges to an approximate Nash equilibrium, instead of "cycling" around it, by directly
> modifying the underlying multi-agent learning dynamics.

The trick (conceptually): add a **regularization** that pulls the learning dynamics toward a
reference policy, which damps the cycling and gives provable convergence; then iterate, moving the
reference toward the current policy. You get Nash-seeking behaviour out of plain policy-gradient-style
updates — no regret tables over info sets, no tree solve. It is:

- **Model-free** — learns from played transitions, no explicit forward model needed for the update.
- **Search-free** — at play time the **policy network just outputs an action distribution**; you
  sample a (legal) action. No MCTS, no CFR re-solve.

## Why "no search" is the headline for us

Contrast the two DeepMind imperfect-info approaches:

| | Player of Games | DeepNash (R-NaD) |
|---|---|---|
| Equilibrium via | CFR **search** + value net | **R-NaD** training dynamics |
| At inference | re-solves a subgame each move | one network forward pass |
| Shippable offline/stdlib? | no (per-move search) | **yes** (just `np.dot`s) |

For the Kaggle agent — offline, stdlib-only, no per-move search budget
([01](01-game-and-engine.md)) — search-free inference is the whole ballgame. Train the net heavy
(GPU/torch), export weights, and inference is a NumPy forward pass in `agent_net.py`. That pipeline
is the project's neural challenger ([04](04-the-challenger-framework.md), [17](17-imperfect-info-on-tcg.md)).

## Scale and the honest caveat

Stratego's game tree is on the order of **10^535 nodes** [deepnash2022] — vastly larger than chess
or Go — and DeepNash reached expert human level (top-3 on Gravon) purely from self-play. Impressive,
but it used large-scale compute. On one Colab GPU, a from-scratch R-NaD net on the TCG may **not**
beat the 820.7 hand-tuned heuristic. That's an expected, documented outcome, not a failure — the
challenger framework gates it ([04](04-the-challenger-framework.md)), and PTCG-Bench independently
found agent self-improvement on this game is unstable ([16](16-ptcg-bench.md)).

## Plan 2 staging

To de-risk: implement R-NaD **tabularly on Leduc poker first** (pure NumPy, watch exploitability →
0 — the correctness oracle), *then* reuse the same update rule with a torch net on the cg sim. If
Leduc converges, the algorithm is right and only the environment is left to debug.

> Takeaway: R-NaD reaches Nash by regularizing the learning dynamics so they stop cycling — giving
> a strong policy whose **inference is just a forward pass**. That single property is why DeepNash,
> not Player of Games, is what we try to ship.
