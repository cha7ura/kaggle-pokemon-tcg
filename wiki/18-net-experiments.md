# 18 — Neural net experiments (the honest record)

← [17 — imperfect info on the TCG](17-imperfect-info-on-tcg.md) · related → [12](12-deepnash-rnad.md), [16](16-ptcg-bench.md)

This page records the actual cg-scale neural experiments (run on Colab T4, tracked in wandb
`inc-llm/ptcg-rnad`). Per the autoresearch ethic ([03](03-autoresearch-loop.md)), the rejections
are evidence — and here they converge on one conclusion: **a from-scratch net did not beat the
820.7 hand-tuned heuristic.** That is a result, not a failure.

## The pipeline (all of it works)

1. **Engine on Colab** — `libcg.so` (gitignored, 1.3MB) shipped to Colab as a **wandb artifact**
   (`ptcg-engine`); engine loads on Colab (linux/amd64), 1267 cards. Transport solved without
   GitHub/Drive auth (wandb authed both sides).
2. **Self-play harness** — the engine has *no built-in opponent*; we supply both seats
   (`battle_start/select/finish`).
3. **Encoder + net** — `Net3` action-scoring policy: card-id embeddings + per-card static
   features (type, weakness, ex/megaEx, retreat/attack cost) + energy-type multi-hot + statuses;
   per-option features (type, target card, attack cost) → one logit per legal option (AlphaStar-
   style pointer head, [13](13-alphastar-league.md)).
4. **wandb** — every run logs curves + configs + weight artifacts (the neural ledger,
   [14](14-alphaevolve.md)).
5. **Shippable path proven** — torch net → `weights.npz` → NumPy forward, round-trip 5.96e-08
   ([17](17-imperfect-info-on-tcg.md), `minizero/rnad/numpy_net.py`).

## The experiments and what each proved

wandb project: **https://wandb.ai/inc-llm/ptcg-rnad** (team `inc-llm`).

| Run | Result | Lesson | wandb |
|---|---|---|---|
| REINFORCE from scratch (thin encoder) | winrate flat ~0.5 | sparse ±1 RL can't discover play from nothing | [znvvw6dq](https://wandb.ai/inc-llm/ptcg-rnad/runs/znvvw6dq) |
| REINFORCE + baseline (rich-ish) | flat ~0.5 | variance reduction alone doesn't help | [cd04tm7b](https://wandb.ai/inc-llm/ptcg-rnad/runs/cd04tm7b) |
| BC of heuristic (thin encoder) | imitation 0.78, **0.75 vs random, 0.10 vs heuristic** | BC learns *real* play; but imitation ≠ winning | [erlab5vl](https://wandb.ai/inc-llm/ptcg-rnad/runs/erlab5vl) |
| BC bigger net + more data (batched) | imitation **plateaus ~0.76** | net size / data is not the ceiling | [l4k4jevx](https://wandb.ai/inc-llm/ptcg-rnad/runs/l4k4jevx) |
| Rich encoder + **BCE** multi-hot | plays *worse* (0.33 vs random) | the objective matters; BCE hurt vs CE | [7xj2006f](https://wandb.ai/inc-llm/ptcg-rnad/runs/7xj2006f) |
| Rich encoder + **CE** warm-start | **0.7–0.83 vs random**, 0.15 vs heuristic | CE+rich = best warm-start | [o3gt9rbj](https://wandb.ai/inc-llm/ptcg-rnad/runs/o3gt9rbj) |
| **RL fine-tune** (PG + prize-shaping, vs heuristic) | **no climb** (~0.1–0.3, reward ~−0.9) | high-variance PG vs a strong opponent gives no gradient | [ig3kg72g](https://wandb.ai/inc-llm/ptcg-rnad/runs/ig3kg72g) |

Toy proof-of-pipeline (Kuhn, GPU→NumPy round-trip): [iqkaos2f](https://wandb.ai/inc-llm/ptcg-rnad/runs/iqkaos2f).
Engine artifact upload: [5k8j5478](https://wandb.ai/inc-llm/ptcg-rnad/runs/5k8j5478).
Model artifacts: `bc-net`, `bc-net-big`, `bc-net-rich`, `warm-ce`, `rl-net`, `ptcg-engine` (in the project's Artifacts tab).

## Why it hit a wall (honest diagnosis)

- **Imitation ≠ winning.** 0.75 per-move imitation still loses, because errors **compound** over a
  long game and BC weights every decision equally, not by leverage.
- **RL needs a usable gradient.** Policy-gradient against a *strong* fixed opponent wins so rarely
  that there's almost no signal to learn from; it stays flat. (DeepNash's R-NaD addresses this with
  regularized self-play dynamics, but at a compute scale far beyond one notebook —
  [12](12-deepnash-rnad.md).)
- **PTCG-Bench predicted exactly this** — LLM/agent self-improvement on this game is unstable
  ([16](16-ptcg-bench.md)). Our results are an independent confirmation.

## What would be needed to go further (not done)

- True **R-NaD self-play** (net vs net) with the regularized dynamics + reference updates, run for
  far longer than a notebook session (DeepNash scale).
- A **value head** + advantage estimation (lower-variance than REINFORCE), or AlphaStar-style
  league + exploiters ([13](13-alphastar-league.md)).
- Richer state still (full energy/ability/trainer semantics), and curriculum (beat random →
  weak heuristic → full heuristic).

## Decision

**The heuristic (`agent_lucario`, ladder 820.7) remains the competition agent.** The challenger
gate ([04](04-the-challenger-framework.md)) did its job: no neural challenger cleared it, so nothing
shipped. The net pipeline stands as a complete, reusable research artifact (encoder, self-play
harness, BC, RL, wandb tracking, NumPy-ship path). The highest-EV competitive lever remains
heuristic evolution (`tools/evolve.py`, [14](14-alphaevolve.md)), not the net.

> Takeaway: we built the whole DeepMind-style neural pipeline and ran it honestly. It learns real
> play (BC) but does not surpass a year of hand-tuning — the predicted imperfect-info wall. Banked
> as research; the heuristic stays champion.
