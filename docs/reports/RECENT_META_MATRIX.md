# Recent-meta matchup matrix (what beats what, current meta)

Decoded 3,000 recent games (episodes >82.1M, NOT in replay_field — the current Grimmsnarl meta).
Archetype = union of revealed cards per seat, classified on evolution LINE. Win% = row beats col.

## Recent field shares
Grimmsnarl 30.2% | Alakazam 22.2% | Archaludon 14.0% | OTHER 7.5% | SolrockLun 6.7% |
ToolboxEx 6.2% | MegaStarmie 4.8% | Chandelure 4.0% | HopTrev 3.5% | Dragapult 1.0%

## Field-weighted EWR (recent)
1. **Grimmsnarl 57.5%**  <- what we just shipped
2. SolrockLun 56.7%
3. ToolboxEx 56.4%
4. Dragapult 51.7%
5. Chandelure 50.7%
6. Archaludon 48.0%
7. OTHER 46.4%
8. Alakazam 42.2%  <- our slot 2 (alakazam_top)
9. MegaStarmie 39.8%
10. HopTrev 38.1%

## Matchup matrix (row beats col, %)
             Grimm  Alaka  Archa  OTHER  Solro  Toolb  MegaS  Chand  HopTr
Grimmsnarl     50     57     60     72     41     68     75     37     93
Alakazam       43     50     44     31     29     46     43     36     22
Archaludon     40     56     50     49     41     15     69     60     92
SolrockLun     59     71     59     65      -     11     50     53     16
ToolboxEx      32     54     85     40     89      -     89     73     76
Chandelure     63     64     40      9     47     27      -      -      -
HopTrev         7     78      8     73     84     24      -      -      -

## Key findings
1. **Grimmsnarl is the EWR leader (57.5%) and we shipped it — confirmed correct.**
   Beats HopTrev 93%, MegaStarmie 75%, OTHER 72%, ToolboxEx 68%, Archaludon 60%, Alakazam 57%.
   Only loses to SolrockLun (41%) and Chandelure (37%) — together ~11% of field.
2. **Our slot-2 Alakazam is WEAK in the recent meta (42.2% EWR, 8th).** It loses to Grimmsnarl (43),
   SolrockLun (29), OTHER (31), HopTrev (22). This is the slot to reconsider.
3. **SolrockLunatone (56.7% EWR, 6.7% share) is the standout sleeper** — beats Alakazam 71%, OTHER 65%,
   Grimmsnarl 59%, Archaludon 59%. It's the deck that BEATS Grimmsnarl. Only ToolboxEx (11) crushes it.
4. **ToolboxEx (56.4% EWR) is a hidden powerhouse** — beats Archaludon 85%, SolrockLun 89%, MegaStarmie
   89%, HopTrev 76%. Its weakness is Grimmsnarl (32) and OTHER (40).
5. **Rock-paper-scissors core:** Grimmsnarl > ToolboxEx > SolrockLun > Grimmsnarl. And SolrockLun is
   Grimmsnarl's main counter.

## Implication for our PAIR (two slots)
Current: Grimmsnarl (57.5%, great) + Alakazam (42.2%, weak).
Grimmsnarl's holes are SolrockLun + Chandelure. A better slot-2 would COVER those:
- SolrockLun (56.7% EWR) beats the field AND goes even vs its own counter is ToolboxEx only. But
  SolrockLun loses to ToolboxEx 11% — a shared-ish risk.
- ToolboxEx (56.4%) beats SolrockLun 89% and Chandelure 73% (Grimmsnarl's two counters!) — so
  **Grimmsnarl + ToolboxEx is a strong pair**: Grimmsnarl handles the field, ToolboxEx covers the
  SolrockLun/Chandelure hole that beats Grimmsnarl.

## Pair overlap check (shared holes)
Computed which opponents beat BOTH decks in a pair (both matchups <50%):
- **Grimmsnarl + ToolboxEx: NO shared holes** — no opponent beats both. Ideal coverage.
- **Grimmsnarl + SolrockLun: NO shared holes** — no opponent beats both. Ideal coverage.
Both candidate pairs are complementary. ToolboxEx directly covers Grimmsnarl's two counters
(SolrockLun 89%, Chandelure 73%), so Grimmsnarl + ToolboxEx is the tightest coverage.

=> Candidate next move: replace weak Alakazam slot (42.2% EWR) with ToolboxEx or SolrockLunatone.
