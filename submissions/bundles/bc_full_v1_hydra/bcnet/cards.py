"""Static card and attack feature tables, built from the SDK.

The id embeddings in the model learn *usage*; these tables carry *stats*, so a
card that appears twice in the corpus still gets an HP and a retreat cost. Built
from `all_card_data()` / `all_attack()` and never from the CSV, per `AGENTS.md` —
the engine references ids the CSV snapshot does not have.

Nothing here is committed. The tables are derived from competition-use-only card
data, so they are rebuilt from the SDK wherever they are needed: at training time
from `--sdk`, and at inference time from the `cg/` package the submission
bundles. Building takes milliseconds, so there is no reason to ship a copy.

Index convention, shared with `encoder.py` and relied on by every table here:

    0            PAD   — no card / no attack in this slot
    1            UNK   — an id outside the range this checkpoint was built with
    2 .. N+1     card id 1 .. N

`UNK` is the reason an id is never used as a raw index. `cg/api.py` says members
may be added during the competition, so a checkpoint trained today must not index
out of bounds against an SDK refreshed tomorrow.
"""

from __future__ import annotations

import numpy as np

PAD = 0
UNK = 1
FIRST_ID = 2

N_ENERGY_TYPES = 12  # EnergyType COLORLESS..TEAM_ROCKET
N_CARD_TYPES = 7  # CardType POKEMON..SPECIAL_ENERGY


def _one_hot(value, size: int) -> list[float]:
    out = [0.0] * size
    if value is not None and 0 <= int(value) < size:
        out[int(value)] = 1.0
    return out


class CardTables:
    """Feature matrices plus the id -> index maps that go with them."""

    def __init__(self, card_data, attacks):
        self.n_cards = max((c.cardId for c in card_data), default=0)
        self.n_attacks = max((a.attackId for a in attacks), default=0)
        self.attack_by_id = {a.attackId: a for a in attacks}
        self.card_by_id = {c.cardId: c for c in card_data}

        self.attack_feat = self._build_attack_feat()
        self.card_feat = self._build_card_feat()

    # -- index maps ---------------------------------------------------------

    def card_index(self, card_id) -> int:
        """Map a raw card id to an embedding row. Out-of-range becomes UNK, never a crash."""
        if card_id is None or card_id < 1:
            return PAD
        if card_id > self.n_cards:
            return UNK
        return FIRST_ID + int(card_id) - 1

    def attack_index(self, attack_id) -> int:
        if attack_id is None or attack_id < 1:
            return PAD
        if attack_id > self.n_attacks:
            return UNK
        return FIRST_ID + int(attack_id) - 1

    @property
    def card_rows(self) -> int:
        return FIRST_ID + self.n_cards

    @property
    def attack_rows(self) -> int:
        return FIRST_ID + self.n_attacks

    # -- feature matrices ---------------------------------------------------

    def _build_attack_feat(self) -> np.ndarray:
        width = 3 + N_ENERGY_TYPES
        feat = np.zeros((self.attack_rows, width), dtype=np.float32)
        for attack in self.attack_by_id.values():
            row = feat[self.attack_index(attack.attackId)]
            energies = list(attack.energies or [])
            row[0] = (attack.damage or 0) / 100.0
            row[1] = len(energies) / 5.0
            row[2] = 1.0 if attack.damage else 0.0
            for energy in energies:
                if 0 <= int(energy) < N_ENERGY_TYPES:
                    row[3 + int(energy)] += 1.0 / 5.0
        return feat

    def _build_card_feat(self) -> np.ndarray:
        width = N_CARD_TYPES + 3 * N_ENERGY_TYPES + 12
        feat = np.zeros((self.card_rows, width), dtype=np.float32)
        for card in self.card_by_id.values():
            attack_ids = list(card.attacks or [])
            costs = [len(self.attack_by_id[a].energies or []) for a in attack_ids if a in self.attack_by_id]
            damages = [self.attack_by_id[a].damage or 0 for a in attack_ids if a in self.attack_by_id]
            row = (
                _one_hot(card.cardType, N_CARD_TYPES)
                + _one_hot(card.energyType, N_ENERGY_TYPES)
                + _one_hot(card.weakness, N_ENERGY_TYPES)
                + _one_hot(card.resistance, N_ENERGY_TYPES)
                + [
                    (card.hp or 0) / 100.0,
                    (card.retreatCost or 0) / 4.0,
                    1.0 if card.basic else 0.0,
                    1.0 if card.stage1 else 0.0,
                    1.0 if card.stage2 else 0.0,
                    1.0 if card.ex else 0.0,
                    1.0 if card.megaEx else 0.0,
                    1.0 if card.tera else 0.0,
                    1.0 if card.aceSpec else 0.0,
                    1.0 if card.skills else 0.0,
                    max(damages, default=0) / 100.0,
                    min(costs, default=0) / 5.0,
                ]
            )
            feat[self.card_index(card.cardId)] = row
        # UNK gets the mean of the real rows: a card this checkpoint has never
        # seen is better described by an average card than by zeros.
        feat[UNK] = feat[FIRST_ID:].mean(axis=0)
        return feat


_TABLES: CardTables | None = None


def load_tables() -> CardTables:
    """Build once per process. Import of `cg` must already resolve."""
    global _TABLES
    if _TABLES is None:
        from cg.api import all_attack, all_card_data

        _TABLES = CardTables(all_card_data(), all_attack())
    return _TABLES
