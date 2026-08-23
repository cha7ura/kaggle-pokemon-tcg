# Why alakazam_top scores 826.8 (our best) — and where the headroom is

## The scores
alakazam_top 826.8 | dragapult_v19 777.4 | grimmsnarl 449.5 (generic pilot, no Alakazam-quality logic)
Our active pair = alakazam_top + dragapult_v19. Grimmsnarl not in top-2.

## Why alakazam_top is nearly 2x Grimmsnarl: the pilot is PURPOSE-BUILT for the deck
main.py is 814 lines of Alakazam-SPECIFIC reasoning (vs the generic agent_lucario piloting Grimmsnarl):
- Names every card in the deck as a constant (Abra/Kadabra/Alakazam, Dudunsparce, Fezandipiti ex...).
- Models the WIN CONDITION exactly: Powerful Hand = 20 dmg x hand-size. So it computes a
  min/max DAMAGE RANGE for THIS turn by simulating every draw effect:
    - evolve Kadabra->Alakazam (net +2 hand), Rare Candy line, Dunsparce->Dudunsparce Run Away Draw,
      Fezandipiti Flip the Script (+3), each supporter (Hilda/Dawn/Boss), Enriching Energy (net +3: draw +4 - 1 to attach).
  -> max_damage = max_hand_size * 20, computed BEFORE acting.
- Real KO/target selection: op_active_hp, prize_count, Boss's Orders drag math, Enhanced Hammer
  (energy denial) needed-count. It knows exactly when it can KO and picks the sequence to get there.

This is precisely the "condition on cards in hand -> compute the exact play" quality the user asked
for, done RIGHT for one deck. It's why it wins: the pilot IS the deck's game plan in code.

## The contrast that explains the ladder result
- alakazam_top: deck + a pilot that models the deck's damage math -> 826.8
- grimmsnarl: strong deck + generic pilot that CANNOT model Shadow Bullet KO math (Lucario-hardcoded
  logic falls back to bare priorities) -> 449.5
=> DECK QUALITY x PILOT-FIT is the product that matters. A top deck with a mismatched pilot loses to
   a mid deck with a matched pilot. The recent-meta matrix measured HUMAN-piloted deck-vs-deck, which
   did not transfer because our bot doesn't pilot Grimmsnarl at that level.

## Headroom on alakazam_top (the promising thread)
1. It ALREADY has real damage/KO computation -> the representation is good (unlike our Grimm pilot).
   This is the deck to weight-tune / refine, because tuning on a working representation can help
   (whereas tuning our Grimm pilot could not fix its structural scoring gap).
2. Deck refinement: alakazam_top's list is proven; guided 1-2 card swaps (scored via deck_gate) are
   lower-risk than a new archetype the bot can't pilot.
3. The Alakazam pilot's logic could be a TEMPLATE: to make Grimmsnarl work we'd need to write
   Grimmsnarl's damage math (Shadow Bullet 180/2E, Munkidori move-damage, prize race) to the SAME
   depth as this 814-line Alakazam pilot — not the ~350-line proxy we built.

## Recommendation
- KEEP alakazam_top + dragapult_v19 (proven scorers). Do not swap in matrix-favored decks the bot
  can't pilot.
- Highest-value next work: refine the alakazam_top PAIRING (weight-tune its real-damage pilot, or
  guided deck tweaks) rather than chase new decks. Grimmsnarl only becomes viable with an
  Alakazam-depth pilot, which is a large build, not a quick fix.
