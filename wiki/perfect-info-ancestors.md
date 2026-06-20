# Perfect-information ancestors (AlphaGo → AlphaZero → MuZero)

← [index](00-index.md) · related → [11 — Player of Games](11-player-of-games.md), [17 — imperfect info on the TCG](17-imperfect-info-on-tcg.md)

Short page on purpose. The famous DeepMind game agents — **AlphaGo, AlphaZero, MuZero** — are the
*ancestors* of the methods this project actually needs, but they solve an easier problem: **perfect
information**. Understanding what they assume is the fastest way to see why the TCG needs something
else.

## The lineage in one line each

- **AlphaGo (2016)** — beat Go via MCTS guided by policy/value nets, bootstrapped from human games.
- **AlphaZero (2017)** — dropped human data: pure self-play, one network (policy + value), MCTS as a
  policy-improvement operator. Mastered Go, chess, shogi from scratch.
- **MuZero (2019)** — dropped the *rules*: learns a latent dynamics model and plans inside it, so it
  needs no given simulator. Still perfect-information at heart.

## The assumption that breaks here

All three lean on **MCTS**, and MCTS assumes a **single known state** at every node and (mostly)
deterministic transitions. The Pokémon TCG violates both:

- **Hidden information** — you don't know the opponent's hand or deck order, so a tree node is an
  *info set*, not a state. MCTS-on-the-true-state isn't available, and MCTS-on-sampled-states is
  **not sound** for imperfect information [pog2021] (it commits *strategy fusion* — see
  [17](17-imperfect-info-on-tcg.md)).
- **Stochasticity** — coin-flip attacks and shuffles are genuine chance nodes.

That's the whole reason the wiki's spine is the *imperfect-information* line —
[CFR](10-cfr-and-nash.md), [Player of Games](11-player-of-games.md) (CFR-search, the sound
generalization of AlphaZero), and [DeepNash](12-deepnash-rnad.md) (Nash without any search) —
rather than AlphaZero itself.

## What still transfers

The *shape* survives: self-play, a single network with value + policy heads, and "search/solve as
policy improvement." Player of Games is explicitly AlphaZero's structure with a sound
imperfect-information solver swapped in for MCTS [pog2021]. So these ancestors are worth knowing —
just don't try to run AlphaZero directly on a card game.

> Takeaway: AlphaGo/Zero/MuZero are perfect-information MCTS systems. Their assumptions
> (known state, determinism) fail on the TCG, which is why we climb the imperfect-information branch
> instead.
