# 11 — Player of Games (search-based equilibrium)

← [10 — CFR and Nash](10-cfr-and-nash.md) · next → [12 — DeepNash / R-NaD](12-deepnash-rnad.md)

Player of Games (Schmid et al. 2021; published as *Student of Games*, SOG) is the cleanest answer
to "what is AlphaZero for hidden-information games?" [pog2021]. If you understood
[10 — CFR](10-cfr-and-nash.md), this is CFR + search + a neural net, glued into one self-play
learner that works for **both** perfect and imperfect information.

## The dumb version, and why it's wrong

The dumb idea: "use MCTS like AlphaZero, just sample the hidden cards." But:

> Unlike MCTS, which is **not sound for imperfect information games**, SOG's search is based on
> counterfactual regret minimization and is sound for both. [pog2021]

MCTS assumes a single true state at each node. In a hidden-info game your decision at an info set
must be consistent across all states in it (you can't "know" which one you're in). MCTS-on-samples
violates that (see *strategy fusion*, [17](17-imperfect-info-on-tcg.md)). SOG fixes it by making
the *search itself* a CFR solve over public belief states.

## GT-CFR: search that grows its own tree

The search is **Growing-Tree CFR (GT-CFR)** — an anytime local search that alternates two phases
[pog2021]:

1. **Regret-update phase** — run several iterations of **public-tree CFR** on the current tree
   (using regret-matching⁺ and linearly-weighted averaging, i.e. CFR+ from [10](10-cfr-and-nash.md)).
2. **Expansion phase** — grow the tree by adding new public states via simulation-based expansion
   trajectories, producing a larger tree for the next iteration.

So instead of MCTS's "select → expand → rollout → backup," GT-CFR does "solve the current subgame
with CFR → expand it → solve again." It builds the tree non-uniformly toward the relevant parts,
the same spirit as AlphaZero's selective search, but the inner solver is regret minimization.

## The CVPN: a value net over beliefs, not states

At the leaves of the public tree, GT-CFR queries a **counterfactual value-and-policy network
(CVPN)** [pog2021]:

```
f_θ(β) = (v, p)      where β = (s_pub, r)
```

`β` is a **public belief state**: the public information `s_pub` plus `r`, the *ranges* (belief
distributions over each player's possible private states). The network returns counterfactual
**values** `v` and a **policy** `p` for the subgame rooted at that belief. This is the key
generalization of AlphaZero's value/policy net: it's defined over *beliefs*, because in
hidden-info games "what's this position worth?" only has an answer once you fix who-believes-what.

## Soundness and how it relates to AlphaZero

SOG is **provably sound**: it converges to perfect play (an approximate Nash equilibrium) as
computation and network capacity grow, by re-solving subgames to stay consistent during online
play, and it shows low exploitability in small games where exploitability is computable [pog2021].

> The main difference from AlphaZero is that the search and self-play training in SOG are also
> sound for imperfect-information games. [pog2021]

Even the compute notation mirrors AlphaZero: `SOG(s,c)` echoes simulation counts (e.g.
`SOG(8000,10)` = 800 GT-CFR iterations) [pog2021]. It validated across chess, Go, heads-up no-limit
poker, and Scotland Yard — one algorithm, four very different games.

## Compute reality (why we don't reproduce it)

Training used **3500 concurrent actors, each on a TPUv4, for 800k steps** [pog2021]. That is far
outside one Colab GPU. More importantly for *us*, SOG runs **CFR search at inference** — every move
re-solves a subgame. Shippable inference on the Kaggle ladder (offline, stdlib, no per-move search
budget) is impractical. That's precisely why the project's neural arm chooses DeepNash's R-NaD
instead — **search-free at inference** ([12](12-deepnash-rnad.md), [17](17-imperfect-info-on-tcg.md)).

> Takeaway: PoG/SOG = CFR-based growing-tree search + a belief-conditioned value/policy net. It's
> the *correct* generalization of AlphaZero to hidden info — but its search-at-inference and TPU
> scale make it the wrong thing to ship here. We borrow the ideas, not the architecture.
