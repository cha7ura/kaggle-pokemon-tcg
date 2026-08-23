# Pilot validation result: deck-aware Grimmsnarl pilot vs generic heuristic

## Test (pilot_gate, real cg engine, Docker on Mac)
Same Grimmsnarl deck BOTH seats; only the pilot differs; seat alternates. 100 games.

    agent_grimmsnarl (deck-aware) vs agent_lucario (generic heuristic)

## Result: agent_grimmsnarl 38.1% (37/97 decisive) — LOSS. err 0. seat0 18/50, seat1 19/50.

The deck-aware pilot plays Grimmsnarl WORSE than the generic pilot. Clean result: zero crashes,
seat-balanced (not a first-player artifact). Draws 3.

## Why it lost (root cause)
1. CRUDE ATTACK SCORING (main cause). Generic pilot looks up REAL attack damage from engine tables
   (_ATK[attackId], weakness x2) and knows exactly which attack KOs. The Grimmsnarl pilot used a
   proxy (999 - opp_hp, fixed "hp<=130 -> KO"). It picks worse attacks and mis-times KOs.
2. WEAKER HAND-GUESSED WEIGHTS. Generic W (ability 30000, play 20000) gives a sharper
   develop-then-attack order than the flatter Grimmsnarl guesses (12000, 7000).
3. "RACE" POSTURE (+2500 attack) likely caused premature attacking in bad matchups.

## What this CONFIRMS (positive)
- Our SHIPPED Grimmsnarl submission uses agent_lucario (the generic pilot) = the WINNING pilot here.
  No regression risk; we already ship the better pilot.
- The gate works: it told us the truth cheaply (100 games) and stopped a bad pilot pre-submission.
- Reinforces the session-long lesson: the generic heuristic is genuinely strong; beating it needs
  REAL damage/KO computation, not hand-guessed proxies and weights.

## Options from here
A. DROP the deck-aware pilot. Keep shipping generic agent_lucario with Grimmsnarl. (Safe, done.)
B. FIX the attack scoring: give agent_grimmsnarl the same real-damage lookup the generic uses
   (it already has _ATKDMG/_ATKCOST tables from the threat module — wire them into ATTACK scoring),
   then re-run the gate. This addresses root cause #1 directly.
C. TUNE weights (tune_weights.py) — but tuning cannot fix the structural attack-scoring gap, so do
   B first, then optionally tune.

## CONFIRMATION RUN (tuned weights, 200 games)
- Weight-tuning (tune_weights.py, 12 gens x 8 pop @ 60 games) reported 52.5% -- but that was
  best-of-96 selection bias at high per-eval noise (+/-6-7 pts at 60 games).
- Re-gated the tuned weights at 200 games (AGENT_W_JSON=tuned_weights.json): **44.1%** (86/195),
  err 0, seat0 48/100 / seat1 38/100. The 52.5% did not hold -- true value ~44%.
- Tuning direction was interpretable (evolve up 9000->19900, attack down 1000->827 = develop harder,
  attack less) but insufficient: even tuned, deck-aware pilot LOSES to generic.

## FINAL VERDICT: DROP the deck-aware pilot. Ship generic agent_lucario (already shipped).
Tested this session, all beaten by the generic heuristic: learned clone 46%, search 11-26%,
deck-aware pilot 38% (raw) / 44% (tuned). The generic heuristic's real-damage/KO computation is the
edge; hand-authored deck logic + weight tuning could not match it. Policy is a settled dead end.
The ONLY proven lever remains DECK SELECTION (Grimmsnarl ship + slot-2 swap).

## Superseded recommendation (kept for record)
Do B (cheap, targeted at the actual root cause) then re-gate. If still <50%, drop it (option A) —
the generic pilot is our best and it's already shipped. Do NOT tune (C) until B closes the gap.
