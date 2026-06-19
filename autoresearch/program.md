# program.md — PTCG autoresearch loop

Adapted from Karpathy's `autoresearch`. You (the agent) iterate on a Pokémon TCG
policy. You edit **one file**, measure a **single comparable metric**, and keep
or discard the change. Repeat. Every experiment is logged — that log is also the
substrate for the Kaggle Strategy writeup ("what hypotheses were tested, and why").

## The loop

1. Form ONE hypothesis (e.g. "ranking attacks by damage-to-KO beats flat priority").
2. Edit **`agent.py` only**. Do not touch `eval.py`, `baselines.py`, or `sdk/`.
3. Run the gauntlet:
   ```
   ./run.sh --games 400                       # vs random baseline (the floor)
   ./run.sh --games 600 --champion champion_agent.py   # vs current champion
   ```
4. Read the JSON verdict. **Keep iff `verdict == "PROMOTE"`** (Wilson lower bound
   of win-rate > `--promote-lb`, default 0.5). Otherwise revert.
5. On PROMOTE: `cp agent.py champion_agent.py` (new champion), append to `log/`,
   start the next hypothesis.

## The metric (why it's trustworthy)

- Win-rate over a **seat-swapped** gauntlet (every matchup played from both
  seats) — cancels the first-player advantage, which is real and large.
- Same deck on both sides → measures **policy skill**, not deck luck.
- **Wilson lower bound**, not raw win-rate → a noisy 51% over 100 games does NOT
  promote; you need a signal that survives the confidence interval.
- The engine is unseeded, so the ONLY defense against noise is N. Use ≥400 games
  for a real decision, more when the edge is small. (≈106 games/s → 400 ≈ 4s.)

## Constraints

- Contract: `agent(obs_dict) -> list[int]`, indices into `obs.select.option`,
  length in `[minCount, maxCount]`, no duplicates. `select is None` → return the
  60-card deck.
- The engine only ever offers **legal** options — never validate legality.
- One global battle per process; don't try to thread it.
- Avoid infinite loops (an action the engine keeps re-offering). `eval.py` caps a
  game at 10k steps and scores a stuck game as a draw — a draw won't promote.

## Hypothesis backlog (rough priority order)

1. **Attack selection by value** — rank ATTACK options by expected damage vs the
   opponent active's HP (weakness ×2). KO > chip. Use `cg.api.all_attack()`.
2. **Energy efficiency** — attach energy toward the cheapest lethal attacker;
   don't over-commit to a Pokémon that gives up many prizes.
3. **Prize-trade awareness** — Mega-ex KO gives the opponent **3** prizes, ex = 2,
   most basics = 1 (`CardData.megaEx/ex`). Avoid bad trades; force good ones.
4. **Variance control** — decline coin-flip attacks when a flat-damage line of
   similar value exists (150/1088 Pokémon have flip attacks).
5. **Retreat / switch logic** — retreat a sleeping/paralyzed or near-dead active
   when the tempo cost is worth it.
6. **Bench development & evolution lines** — keep enough basics + evolve on curve.
7. **Search** — replace `choose()` with a determinized lookahead using
   `cg.api.search_begin(obs, your_deck, your_prize, opp_deck, opp_prize,
   opp_hand, opp_active)` as the forward model; sample opponent hands (IS-MCTS).

## Log discipline (feeds the Strategy writeup)

For each experiment append a line to `log/experiments.md`:
`<id> | hypothesis | diff summary | games | score | wilson_lb | verdict`
Keep the *why* — promoted and rejected alike. The rejections are evidence too.
