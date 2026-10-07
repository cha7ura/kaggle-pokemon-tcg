"""The two frontends that feed `encoder.py`: parquet rows and a live observation.

Both produce the same plain dict, keyed by the `steps` column names, so the
encoder never learns which one it is looking at. Everything that could drift
between training and inference lives in this file and nowhere else.

Three derivations here are not guesses — each was checked against the corpus and
the check is worth repeating if the extractor is ever regenerated:

* `_stage` is `len(preEvolution)`, the evolution depth **in play**, not the
  card's printed stage. Deriving it from `CardData.stage1/stage2` disagrees with
  the corpus on 18% of Active rows, because Rare Candy puts a Stage 2 down over
  a Basic and leaves one card underneath, not two. Verified exact on
  2026-07-27: 0 mismatches over 715,592 Active and 2,694,899 Bench entries.
* An option belongs to the acting seat when `playerIndex` is null or equal to
  it. Verified on 2.43M option references with zero out-of-range resolutions.
* `looking` positions are preserved, including face-down entries. Option indices
  into `AreaType.LOOKING` are positional, so compacting the list would shift
  them.
"""

from __future__ import annotations

from typing import Any, Iterator

# Every `steps` column the encoder reads. Reading a narrow projection is what
# makes a per-row Python encoder affordable over 30M rows.
STEP_COLUMNS = [
    "episode_id", "date", "seat", "select_type", "context",
    "min_count", "max_count", "n_options", "action",
    "first_player", "turn", "turn_action", "remaining_overage",
    "remain_energy_cost", "remain_damage_counter",
    "supporter_played", "stadium_played", "energy_attached", "retreated",
    "stadium_id", "stadium_owner", "context_card_id", "effect_card_id",
    "option_types", "option_areas", "option_indices", "option_players",
    "option_inplay_area", "option_inplay_index", "option_attack_ids",
    # Schema-v2 option payload (docs/rl-data-spec.md §5). Read unconditionally,
    # like the log block: a projection that varies by build mode makes the two
    # frontends disagree about the row shape, which selftest.py checks key by key.
    # `option_tool_indices` and `option_serials` are deliberately absent — both
    # measured inert, see encoder.OPTION_SLOTS.
    "option_numbers", "option_counts", "option_energy_indices", "option_card_ids",
    "my_hand_ids", "select_deck_ids", "looking_ids",
    # The log block. Read whether or not the cache is built with --with-logs:
    # a narrower projection per build mode would make the two frontends disagree
    # about the row shape, which tools/rl/selftest.py checks key by key.
    "log_types", "log_players", "log_card_ids", "log_attack_ids",
    "log_from_areas", "log_to_areas", "log_target_card_ids",
    "log_values", "log_damage_flags",
]
for _prefix in ("me", "opp"):
    STEP_COLUMNS += [
        f"{_prefix}_deck_count", f"{_prefix}_hand_count", f"{_prefix}_prize_left",
        f"{_prefix}_discard_count", f"{_prefix}_bench_count", f"{_prefix}_bench_max",
        f"{_prefix}_poisoned", f"{_prefix}_burned", f"{_prefix}_asleep",
        f"{_prefix}_paralyzed", f"{_prefix}_confused",
        f"{_prefix}_active_id", f"{_prefix}_active_hp", f"{_prefix}_active_max_hp",
        f"{_prefix}_active_energy_types", f"{_prefix}_active_tool_ids",
        f"{_prefix}_active_stage", f"{_prefix}_active_appear_this_turn",
        f"{_prefix}_active_pre_evolution_ids",
        f"{_prefix}_bench_ids", f"{_prefix}_bench_hp", f"{_prefix}_bench_max_hp",
        f"{_prefix}_bench_energy_types", f"{_prefix}_bench_tool_ids",
        f"{_prefix}_bench_stage", f"{_prefix}_bench_appear_this_turn",
        f"{_prefix}_bench_pre_evolution_ids",
        f"{_prefix}_discard_ids",
    ]


def rows_from_parquet(path, batch_size: int = 8192, columns=None,
                      keep_keys=None, drop_forced: bool = False) -> Iterator[dict]:
    """Yield decision rows from one `steps.parquet`.

    `to_pylist()` on a record batch already produces exactly the dict shape the
    encoder wants, with Arrow nulls arriving as `None` — which is the whole
    reason the encoder treats `None` rather than `-1` as "absent".

    Filtering happens on the Arrow batch, before `to_pylist()`. That ordering is
    the difference between materialising 30.8M Python dicts and materialising
    the 3.5M that survive the archetype filter.

    `keep_keys` is a sorted int64 array of `episode_id * 2 + seat`.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches(batch_size=batch_size, columns=columns or STEP_COLUMNS):
        mask = None
        if keep_keys is not None:
            key = (batch.column("episode_id").to_numpy().astype(np.int64) * 2
                   + batch.column("seat").to_numpy().astype(np.int64))
            mask = np.isin(key, keep_keys)
        if drop_forced:
            n = batch.column("n_options").to_numpy()
            lo = batch.column("min_count").to_numpy()
            hi = batch.column("max_count").to_numpy()
            unforced = ~((n <= 1) | ((lo == hi) & (hi == n)))
            mask = unforced if mask is None else (mask & unforced)
        if mask is not None:
            if not mask.any():
                continue
            batch = batch.filter(pa.array(mask))
        yield from batch.to_pylist()


# ── live observation ─────────────────────────────────────────────────────────

def _ids(cards) -> list[int | None]:
    """Card ids with positions preserved; face-down entries become None."""
    return [None if card is None else card.id for card in (cards or [])]


def _pokemon_fields(row: dict, prefix: str, pokemon) -> None:
    if pokemon is None:
        for suffix in ("id", "hp", "max_hp", "energy", "stage"):
            row[f"{prefix}_{suffix}"] = None
        row[f"{prefix}_energy_types"] = []
        row[f"{prefix}_tool_ids"] = []
        row[f"{prefix}_pre_evolution_ids"] = []
        row[f"{prefix}_appear_this_turn"] = False
        return
    pre_evolution = list(pokemon.preEvolution or [])
    row[f"{prefix}_id"] = pokemon.id
    row[f"{prefix}_hp"] = pokemon.hp
    row[f"{prefix}_max_hp"] = pokemon.maxHp
    row[f"{prefix}_energy"] = len(pokemon.energies or [])
    row[f"{prefix}_energy_types"] = [int(e) for e in (pokemon.energies or [])]
    row[f"{prefix}_tool_ids"] = _ids(pokemon.tools)
    row[f"{prefix}_pre_evolution_ids"] = _ids(pre_evolution)
    row[f"{prefix}_stage"] = len(pre_evolution)
    row[f"{prefix}_appear_this_turn"] = bool(pokemon.appearThisTurn)


def row_from_observation(obs, remaining_overage=None, deck_ids=None) -> dict:
    """Build the encoder's row dict from a live `Observation`.

    Caller must have checked `obs.select is not None` and `obs.current is not
    None`; deck selection has no decision to encode.

    `deck_ids` is the acting seat's own 60-card list — the live agent reads it
    from `deck.csv`, the cache builder joins it from `decks.parquet` on the
    acting seat. Pass it only for a checkpoint trained with `--deck-tokens`; the
    encoder emits the deck block exactly when `my_deck_ids` is non-empty, so a
    non-deck-token model handed `None` sees the features it trained on.
    """
    state = obs.current
    select = obs.select
    seat = state.yourIndex
    me = state.players[seat]
    opp = state.players[1 - seat]

    row: dict[str, Any] = {
        "seat": seat,
        "select_type": int(select.type),
        "context": int(select.context),
        "min_count": select.minCount,
        "max_count": select.maxCount,
        "n_options": len(select.option),
        "first_player": state.firstPlayer,
        "turn": state.turn,
        "turn_action": state.turnActionCount,
        "remaining_overage": remaining_overage,
        "remain_energy_cost": select.remainEnergyCost,
        "remain_damage_counter": select.remainDamageCounter,
        "supporter_played": state.supporterPlayed,
        "stadium_played": state.stadiumPlayed,
        "energy_attached": state.energyAttached,
        "retreated": state.retreated,
        "stadium_id": state.stadium[0].id if state.stadium else None,
        "stadium_owner": state.stadium[0].playerIndex if state.stadium else None,
        "context_card_id": select.contextCard.id if select.contextCard else None,
        "effect_card_id": select.effect.id if select.effect else None,
        "my_hand_ids": _ids(me.hand),
        "my_deck_ids": [int(c) for c in (deck_ids or [])],
        "select_deck_ids": _ids(select.deck),
        "looking_ids": _ids(state.looking),
    }

    for prefix, player in (("me", me), ("opp", opp)):
        row[f"{prefix}_deck_count"] = player.deckCount
        row[f"{prefix}_hand_count"] = player.handCount
        row[f"{prefix}_prize_left"] = len(player.prize or [])
        row[f"{prefix}_discard_count"] = len(player.discard or [])
        row[f"{prefix}_bench_max"] = player.benchMax
        row[f"{prefix}_poisoned"] = player.poisoned
        row[f"{prefix}_burned"] = player.burned
        row[f"{prefix}_asleep"] = player.asleep
        row[f"{prefix}_paralyzed"] = player.paralyzed
        row[f"{prefix}_confused"] = player.confused
        row[f"{prefix}_discard_ids"] = _ids(player.discard)

        active = player.active[0] if player.active else None
        _pokemon_fields(row, f"{prefix}_active", active)

        bench = [p for p in (player.bench or [])]
        row[f"{prefix}_bench_count"] = len(bench)
        row[f"{prefix}_bench_ids"] = [p.id for p in bench]
        row[f"{prefix}_bench_hp"] = [p.hp for p in bench]
        row[f"{prefix}_bench_max_hp"] = [p.maxHp for p in bench]
        row[f"{prefix}_bench_energy_types"] = [[int(e) for e in (p.energies or [])] for p in bench]
        row[f"{prefix}_bench_tool_ids"] = [_ids(p.tools) for p in bench]
        row[f"{prefix}_bench_stage"] = [len(p.preEvolution or []) for p in bench]
        row[f"{prefix}_bench_appear_this_turn"] = [bool(p.appearThisTurn) for p in bench]
        row[f"{prefix}_bench_pre_evolution_ids"] = [_ids(p.preEvolution) for p in bench]

    row["option_types"] = [int(o.type) for o in select.option]
    row["option_areas"] = [None if o.area is None else int(o.area) for o in select.option]
    row["option_indices"] = [o.index for o in select.option]
    row["option_players"] = [o.playerIndex for o in select.option]
    row["option_inplay_area"] = [None if o.inPlayArea is None else int(o.inPlayArea) for o in select.option]
    row["option_inplay_index"] = [o.inPlayIndex for o in select.option]
    row["option_attack_ids"] = [o.attackId for o in select.option]
    # Schema-v2 payload. `Option` is a union, so these are None on every option
    # type that does not carry them — the same "absent" the corpus writes as -1.
    row["option_numbers"] = [o.number for o in select.option]
    row["option_counts"] = [o.count for o in select.option]
    row["option_energy_indices"] = [o.energyIndex for o in select.option]
    row["option_card_ids"] = [o.cardId for o in select.option]

    logs = list(obs.logs or [])
    row["log_types"] = [int(log.type) for log in logs]
    row["log_players"] = [log.playerIndex for log in logs]
    row["log_card_ids"] = [log.cardId for log in logs]
    row["log_attack_ids"] = [log.attackId for log in logs]
    row["log_from_areas"] = [None if log.fromArea is None else int(log.fromArea) for log in logs]
    row["log_to_areas"] = [None if log.toArea is None else int(log.toArea) for log in logs]
    row["log_target_card_ids"] = [log.cardIdTarget for log in logs]
    row["log_values"] = [log.value for log in logs]
    row["log_damage_flags"] = [bool(log.putDamageCounter) for log in logs]
    return row
