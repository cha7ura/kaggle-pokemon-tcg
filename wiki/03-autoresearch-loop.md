# 03 — The autoresearch loop

← [02 — the Lucario policy](02-policy-lucario.md) · next → [04 — the challenger framework](04-the-challenger-framework.md)

This is the *method* behind the project — the thing that turned a dumb greedy agent into an
820-on-the-ladder one. It's adapted from Karpathy's `autoresearch` (`autoresearch/program.md:3`):
**edit one file, measure one number, keep or revert.** Boring on purpose. Boring is what compounds.

## The loop, in five lines

From `program.md:8`:

1. Form **one** hypothesis ("ranking attacks by damage-to-KO beats flat priority").
2. Edit **one file** (`agent_lucario.py`). Never touch `eval.py`, `baselines.py`, `sdk/`.
3. Run the seat-swapped gauntlet (`run.sh`).
4. Read the JSON verdict. **Keep iff `PROMOTE`** (Wilson lower bound clears the gate). Else revert.
5. On PROMOTE: copy to the champion, append to the log, next hypothesis.

The discipline is the whole point: one variable at a time, a fixed yardstick, and a written record
of *every* attempt — including the failures.

## Why a Wilson lower bound, not win-rate

The naive version: "I changed the agent, it won 51 of 100, ship it." That's how you ship noise.
The engine is **unseeded** (`program.md:29`) — every game is genuinely random — so a raw win-rate
is a point estimate with a fat confidence interval around it.

The fix lives in `eval.py:92`, `wilson_lb` — the **lower** end of a Wilson score interval:

```python
def wilson_lb(wins, n, z=1.96):
    p = wins / n
    denom  = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z*sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0.0, center - margin)
```

You don't promote on the *measured* win-rate; you promote on the win-rate you're **95% confident
you're at least at**. A noisy 51% over 100 games has a lower bound well under 0.5 → it does **not**
promote. To clear the bar you need either a big edge or a big `N`. This is the single most
important idea in the project: *measure against the bound, not the mean.*

The keep/revert decision is one line (`eval.py:135`):
```python
verdict = "PROMOTE" if lb > args.promote_lb else "REJECT"
```

## Why the gauntlet is seat-swapped, same-deck

`gauntlet` (`eval.py:71`) plays every matchup from **both seats** — challenger as player 0 on even
games, player 1 on odd (`:75`). First-player advantage in this game is real and large; swapping
seats cancels it (`program.md:24`). And both sides run the **same deck**, so you're measuring
**policy skill, not deck luck** (`program.md:25`). Draws count as 0.5 (`eval.py:133`) and a game
stuck past 10k steps is scored a draw (`eval.py:65`) — so an agent that stalls itself can't sneak a
promotion.

## How to read `log/experiments.md`

Each row is `id | hypothesis | diff | games | score | wilson_lb | verdict` (`program.md:62`). The
log is not a changelog — it's a **lab notebook**, and the rejects carry as much signal as the
keeps. A few real entries worth internalising:

- `001` eager-attack → **0.168**. Lesson: develop before attacking (now baked into `W`, see
  [02](02-policy-lucario.md)).
- The `DIAGNOSTIC` line: 81% of sample-deck games ended in board wipeout → the bottleneck was
  *deck quality*, not policy. Pivoted to a real meta deck. Lesson: don't micro-tune a policy when
  the deck is the constraint.
- `SEARCH-v1` / `LUCARIO-SEARCH` → both rejected. Hand-tuned forward search *lost* to the tuned
  heuristic, twice. Lesson written into the log: "Search needs a **learned value net**
  (AlphaZero), not hand eval." That lesson is the seed of Plan 2's neural arm
  (see [12 — DeepNash / R-NaD](12-deepnash-rnad.md)).
- `LADDER SCORES`: base 507.9 → v1 441 → v2 676 → v4 **820.7**. The mirror benchmark and the real
  ladder disagree (mirror over-estimates rank — you're only fighting yourself); the *ladder* is the
  ground truth, and iterations compounded on it.

The current gate (from the `POLICY-DEEPEN` line): **mirror Wilson-LB ≥ 0.48 AND Dragapult ≥
baseline.** task-4/5/6 all failed the mirror gate and were reverted; task-7 (energy-spread) cleared
it and became v5.

## This loop *is* a manual AlphaEvolve

Step back and look at the shape: a generator proposes a code mutation, an automated evaluator
scores it, you keep the winners and feed forward. That is exactly DeepMind's **AlphaEvolve** — with
a human (and Claude) as the mutation generator and `eval.py` as the evaluator. The only thing
missing is automation of the propose-step. That gap is the fast-follow `tools/evolve.py`, and the
full mapping is in [14 — AlphaEvolve](14-alphaevolve.md).

> Takeaway: one file, one number, keep-or-revert, log everything. The number is a Wilson *lower
> bound* over a seat-swapped same-deck gauntlet. The rejections are the evidence.
