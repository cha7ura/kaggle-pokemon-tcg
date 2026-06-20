# 13 — AlphaStar and the league

← [12 — DeepNash / R-NaD](12-deepnash-rnad.md) · next → [14 — AlphaEvolve](14-alphaevolve.md)

AlphaStar (Vinyals et al. 2019) reached Grandmaster in StarCraft II — an imperfect-information game
with a vast action space [alphastar2019]. The transferable lesson here isn't the StarCraft
machinery; it's **how it stops self-play from chasing its own tail**.

## The problem: non-transitive strategy cycles

Strategies in real games are often **non-transitive** — rock-paper-scissors at the strategic level:
build order A beats B, B beats C, C beats A. Plain self-play "train against your latest self" walks
in circles and **forgets** how to beat strategies it already mastered. (This is the same cycling
DeepNash's R-NaD attacks from the dynamics side, [12](12-deepnash-rnad.md).)

## The fix: the League

AlphaStar's central idea is **League training** [alphastar2019]:

> A central idea ... extends the notion of fictitious self-play to a group of agents — the League.

Instead of one agent vs. its latest copy, you maintain a **population** and train against a
distribution over the whole population (and its past versions). Training against many opponents at
once is what prevents both cycling and forgetting.

> ⚠️ Nuance worth getting right: it is **not** plain "fictitious self-play against a uniform mixture
> of all previous strategies" — that specific framing was refuted in research review. AlphaStar uses
> the *League* with **prioritized** matchmaking (you play opponents you're currently losing to more
> often). The population idea is the load-bearing part; "uniform mixture" is the wrong detail.

## Exploiters: agents whose job is to find your weakness

The League isn't homogeneous [alphastar2019]:

> The system employed both **primary agents** seeking victory and **exploiter agents** designed to
> expose weaknesses rather than maximize win rates.

Exploiters don't try to be good in general — they hunt for a specific hole in a main agent, which
then has to patch it. It's adversarial pressure baked into training: a built-in red team.

## What transfers to our project

We can't run an AlphaStar-scale league. But the *insight* directly shapes the challenger arm
([04](04-the-challenger-framework.md)):

- **Don't tune only against the mirror.** The autoresearch log already learned this empirically:
  mirror win-rate badly over-estimated ladder rank ([03](03-autoresearch-loop.md), LADDER REALITY).
  Tuning against one opponent is exactly the cycling/over-fit trap the League avoids. The gate
  already adds a **Dragapult** benchmark alongside the mirror for this reason.
- **Robustness over peak.** A policy that beats a *diverse* set of opponents is what climbs a
  diverse ladder — the League's whole thesis.
- **Exploiters as a future tool.** A cheap version: periodically evolve an opponent that targets the
  champion's weaknesses (a `tools/evolve.py` mode, [14](14-alphaevolve.md)), then harden against it.

> Takeaway: single-opponent self-play cycles and forgets. AlphaStar's League trains a prioritized
> population with dedicated exploiters. For us that means: evaluate against a diverse field, not just
> the mirror — which the eval gate already started doing.
