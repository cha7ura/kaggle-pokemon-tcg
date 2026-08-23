# Policy comparison — our agent vs top submissions

## Method
Unit of analysis = **submission** = (team, fixed deck_sig, episode-window): games sharing both deck
and policy. Ranked 1,101 submissions (>=30 games) by win-rate; extracted decision-context
distributions from decoded replays. Compared our `cha7ura` games to top submissions on the SAME deck.

## Key finding 1 — Trevenant is elite; our PILOT is the problem
Top HopTrevenant submissions hit 72-82% win-rate across many independent pilots (Satoshi Haimura 82%/45g,
THIRD PTCG Club 77%/35g, charmq 74%/136g). Our cha7ura Trevenant pilot: **38%** (6-10). Same deck, ~40-pt gap.
=> Trevenant is not a bad deck; our policy for it is bad.

## Key finding 2 — the gap is NOT in decision MIX
Decision-context distribution, our agent vs the 82% expert, on Trevenant:
| context      | Satoshi (82%) | charmq (74%) | our cha7ura (38%) |
|--------------|---------------|--------------|-------------------|
| MAIN         | 56%           | 65%          | 61%               |
| TO_HAND draw | 27%           | 20%          | 23%               |
| setup/bench  | ~10%          | ~10%         | ~14%              |
Our agent allocates decisions almost identically to the experts. The gap is in the QUALITY of the
specific choice within each decision (attack target, search target, exact attack timing), not in
what kind of decision we make. Decision-context counts cannot resolve this — it needs engine replay
of our agent in matched states.

## Key finding 3 — go-first policy is inconsistent (a cheap fix)
Top experts are near-deterministic on the opening choice:
- Satoshi Haimura, charmq, Debauchery, tonakaiiii, kazuki0123, yamy893: go FIRST ~100%
- Yushin Ito: go SECOND 100% (with the same Trevenant deck — a deliberate opposite line)
Our cha7ura: 5 first / 3 second — inconsistent. A committed opening policy is a concrete, shippable
correction.

## Coin toss / mulligan
Observable via SelectContext: IS_FIRST(41), MULLIGAN(42), COIN_HEAD(46). Go-first extracted above;
mulligan/coin are forced or near-universal in this data (little cross-team variance found).

## Honest limit
Pure-data extraction locates the gap (choice quality, not decision mix) but cannot pin exact misplays
without running our agent against the engine. That is the next lever if we pursue policy — but this
session already showed our search/value pilots lose to the heuristic (11-26%), so the realistic path is
distilling expert decision RULES into the rule-based pilot, not a learned clone.

## Key finding 4 — the expert policy is UNLEARNABLE with current features (decisive)
Trained a behavioral-clone on 5,457 decisions from WINNING games of the top Trevenant pilots
(Satoshi Haimura 82%, charmq 74%, THIRD PTCG Club). Held-out top-1 accuracy for predicting the
expert's own move:
- **expert-clone: 46.9%**
- always-pick-option-0 baseline: **50.8%**
- random: 22.0%
The clone is BELOW the trivial baseline. We cannot predict even an 82%-player's moves from the
available features (23 state + 7 option dims). By context: MAIN 39%, draw-search 65%.

### Why this is the decisive result
This reproduces the README's documented plateau ("imitation ~53% top-1, bottlenecked by FEATURES not
data") on clean expert-only data. The bottleneck is the FEATURE REPRESENTATION, upstream of any
learning algorithm:
- It explains why cloning fails: the features don't encode what drives the expert's choice (hidden
  hand, sequencing, threat/prize-race evaluation).
- It explains why search & value-net pilots lost (11-26%): you can't search/evaluate a decision you
  can't represent.
- => "Replicate a top player's policy" is NOT achievable with the current featurizer. It needs a
  richer state encoding (hand contents, board threat, energy-attachment state, prize race) — a
  substantial build the README already flagged as the plateau, with no guarantee.

## Verdict for the policy lever
The DECK lever is proven and shippable (Grimmsnarl 34.8%/61%). The POLICY lever is blocked at feature
representation — not at data, not at algorithm. Cloning, search, and value-net all hit the same wall.
The rational move: ship the deck; treat policy improvement as a feature-engineering research project,
not a quick win.

## Key finding 5 — richer features do NOT fix the clone (tested, decisive)
Built an enriched featurizer (features_rich.py: 30 state + 20 option dims) adding exactly the
information proposed: option ACTION SEMANTICS (play/attach/evolve/ability/discard/retreat/attack via
OptionType), resolved card identity (area+index -> card -> pokemon/trainer/energy, basic/stage2/ex, HP),
discard-pile counts, deck counts, prize race, threat/status state.
Retrained the expert-Trevenant clone on the same 5,457 winning-game decisions:
- **rich-feature clone: 45.6%** top-1  (base-feature clone: 46.9%; always-opt0 baseline: 50.8%)
Richer features did NOT help — still below the trivial baseline.

### Why (the real bottleneck)
The engine orders options sensibly, so "always pick option 0" already scores 50.8% (the obvious play).
The expert's EDGE is in the ~49% of decisions where they DEVIATE from the obvious option — and those
deviations depend on hidden hand contents, opponent reads, and multi-turn planning that NO static
per-decision feature vector captures. A tree over features can't represent sequential intent.

## FINAL verdict on the policy lever (tested 3 ways this session)
1. Learned search (ISMCTS): 11-17% vs heuristic
2. Value-net search: 19-26%
3. Behavioral clone of the 82% expert: 45.6-46.9% (below coin-flip baseline)
All hit the same wall: the decision quality separating a 38% pilot from an 82% pilot lives in
information + reasoning the offline pipeline cannot represent or search. This is NOT fixable by more
data, a better algorithm, OR richer static features — it is the sequential hidden-information nature
of the decision. => The DECK is the only proven lever. Ship deck; do not invest further in offline
policy learning without a fundamentally different (sequential, belief-state) approach.
