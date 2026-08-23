# Gameplay walkthrough: what the engine shows us (both sides)

## Game: episode 82933532 — kazuki0123 (rank 9) beat tonakaiiii (rank 2)
Grimmsnarl mirror match, 206 steps. seat0=kazuki (WON, reward 1), seat1=tonakaiiii (reward -1).

## The observation every decision: {current, select}
current.players[yourIndex] = US (full info); current.players[1-yourIndex] = OPPONENT (partial).

### Every field, and who sees it
| Field | Us | Opponent |
|---|---|---|
| Active: id, hp/maxHp, energies[], tools[], status | FULL | FULL (their active fully visible) |
| Bench: per-Pokemon id, hp, energy count | FULL | ids + hp + energy count |
| Hand: actual card ids | FULL list | HIDDEN — only handCount |
| Discard: full ordered card-id list | FULL | FULL (public) |
| deckCount | yes | yes (count) |
| prizeCount | yes | yes (count; contents hidden) |
| turn, select-menu (legal options) | yes | n/a |

BOUNDARY: we see BOTH players' board completely (active full, bench ids/hp/energy, all discards).
HIDDEN: opponent's HAND contents and PRIZE contents — only counts.

## Three snapshots

### STEP 8, TURN 1 (setup) — trivial decision
US active: Marnie's Impidimp 70/70. Hand: Morgrem, Grimmsnarl ex, Rare Candy, Xerosic's Machinations
  (the whole plan visible: Rare Candy skips Morgrem -> straight to Grimmsnarl ex).
OPP active: Marnie's Impidimp 70/70. Hand: hidden (count 4).
DECISION: 1 option, type 14 (END) -> just pass. Openings are highly constrained.

### STEP 60, TURN 5 (the swing) — board tells the story
US: Grimmsnarl ex ACTIVE hp 320/320, energy [7,7] = 2 Dark = CHARGED (Shadow Bullet 180). Full bench.
OPP: ACTIVE = (none) — just got KO'd. Bench Froslass at hp 20 (badly hurt). Discard 9 (burned setup).
DECISION: context=7, 5 options area=6 (PRIZE pick after a knockout). We are ahead on board.

### STEP 200, TURN 11 (the kill) — full-information KO math
US: Grimmsnarl ex 320/320 charged [7,7], prizeCount=1 (one prize to win). Hand 8 (backup ex + draw).
OPP: active Grimmsnarl ex hp 130/320 (+Handheld Fan tool), prizeCount=2. Bench: backup ex 310hp.
DECISION: 9 options — 6x ATTACH energy (type 8), 1x ATTACK (type 13), 1x RETREAT (type 12), 1x END (type 14).
  kazuki CHOSE [2] = attach, setting up lethal. Their active at 130 hp is VISIBLE -> we KNOW
  Shadow Bullet (180) KOs it. Their charged backup (310hp) is VISIBLE. 2 prizes left is VISIBLE.

## Why this matters for our agent
Every input for correct KO/sequencing math is ON THE BOARD and visible: opp active hp, opp bench,
both discards, all counts. This is exactly why the COMPUTING heuristic (reads these fields, does the
KO math) beats every LEARNING approach (clone 46%, search 11-26%, hash-map 0.3%) — those try to
recall/recognize states that never recur, while the heuristic just reads the visible board and
computes the answer. The information needed to play well is present; the challenge is USING it, and
a hand-written heuristic that reads it directly is the proven winner.
