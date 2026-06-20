# 04 — The challenger framework

← [03 — the autoresearch loop](03-autoresearch-loop.md) · next → [10 — CFR and Nash](10-cfr-and-nash.md)

This page documents the **contract** that lets the project chase ambitious ideas (a neural net,
an auto-evolver) without ever putting the working submission at risk. The producers it describes
(`tools/evolve.py`, `minizero/rnad`) are **Plan 2** — not built yet. This is the rulebook they
will play by.

## The problem it solves

Right now the live submission is a hand-tuned heuristic scoring **820.7** on the ladder
([03](03-autoresearch-loop.md)). Plan 2 introduces two much riskier ways to make an agent:

- an **evolved heuristic** (AlphaEvolve-style code mutation, [14](14-alphaevolve.md)), and
- a **trained neural net** (DeepNash-style R-NaD, [12](12-deepnash-rnad.md)).

Both *might* be worse than 820.7. A from-scratch net on one GPU may simply not beat a year of
hand-tuning — and that's a perfectly likely outcome (PTCG-Bench found agent self-improvement on
this exact game is unstable, [16](16-ptcg-bench.md)). So the rule is simple:

> **A challenger replaces the champion only if it beats the frozen champion through the *same*
> `eval.py` Wilson-LB gate.** Until then, the heuristic stays live.

Ambition in the lab; discipline at the submission. Nothing ships on a hope.

## The frozen champion

`autoresearch/champion_lucario.py` is a **frozen copy** of the current best policy (v4/v5). It is
the benchmark opponent in the gauntlet and it does not change while a challenger is being
evaluated. Freezing the benchmark is what makes scores across experiments comparable — same
yardstick every time (the `eval.py` design note, `eval.py:1`).

## The gate (unchanged from the loop)

The exact mechanism from [03](03-autoresearch-loop.md), reused verbatim:

- Seat-swapped, same-deck gauntlet (`eval.py:71`).
- Score = Wilson **lower bound** (`eval.py:92`), draws = 0.5.
- Current bar: **mirror Wilson-LB ≥ 0.48 AND Dragapult ≥ baseline** (`log/experiments.md`,
  `POLICY-DEEPEN` line).
- Verdict is one line: `PROMOTE if lb > promote_lb else REJECT` (`eval.py:135`).

A neural challenger faces the identical bar a heuristic tweak does. No special pleading for the
shiny approach.

## The two challenger producers (Plan 2)

```
                          tools/evolve.py  ──>  evolved heuristic  ──┐
                                                                     ├─>  eval.py gate  ──>  champion?
   minizero/rnad (GPU) ──> weights.npz ──> agent_net.py (NumPy) ────┘
```

1. **`tools/evolve.py`** — automates the propose-step of the loop: Claude proposes a mutation to
   `agent_lucario.py`, `eval.py` scores it against the gate, auto-keep on PROMOTE, auto-append to
   `experiments.md`. The autoresearch loop ([03](03-autoresearch-loop.md)) with the human taken out
   of the inner cycle. See [14 — AlphaEvolve](14-alphaevolve.md).

2. **`minizero/rnad` → `agent_net.py`** — the trained net. Heavy training runs on a GPU (torch),
   then the network is exported to a plain `weights.npz` and **inference is a pure-NumPy forward
   pass** in `agent_net.py`. That keeps the shipped agent offline/stdlib — no torch on the ladder
   (the whole reason R-NaD, which is *search-free at inference*, is the chosen method;
   [12](12-deepnash-rnad.md) and [17](17-imperfect-info-on-tcg.md)).

Both feed the **same** `eval.py`, log to the **same** `experiments.md`, and the live submission
changes **only** when one of them wins. That single invariant is the framework.

> Takeaway: freeze the champion, make every challenger (heuristic or neural) clear the identical
> Wilson-LB gate, and only then swap. It's how the project can attempt DeepMind-scale ideas while
> a 820.7 agent stays safely on the ladder.
