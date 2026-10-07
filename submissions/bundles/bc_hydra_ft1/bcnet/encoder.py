"""The one encoder. Training tensors and the live agent both go through here.

Two frontends feed it — `rows.py:rows_from_table` for parquet and
`rows.py:row_from_observation` for a live `Observation` — but there is exactly
one implementation of the features themselves. A second, "fast" vectorised copy
would be the obvious optimisation and is deliberately not written: a silent
train/inference feature skew is the single cheapest way to lose a training run,
and building the tensors is a one-off cost that parallelises across partitions.

Everything is categorical. Counts, HP and time are bucketed into embedding
indices rather than fed as scalars, as described in `docs/ARCHITECTURE.md`. That keeps the
whole dataset int16 and keeps normalisation out of the inference path, where a
mismatched mean would be invisible.

Layouts produced per decision row:

    globals   int16[N_GLOBAL]          one row, fixed width
    tokens    int16[n_tokens, N_TOKEN] ragged — only occupied entities are emitted
    options   int16[n_options, N_OPT]  ragged
    target    the chosen option indices

Index 0 of every categorical slot means "not applicable", so a slot that does
not apply to a token is not confusable with a real value. This matters most for
options: `option_indices`, `option_players` and `option_types` all take the
value 0 legitimately, and `docs/rl-data-spec.md` records the null counts.
"""

from __future__ import annotations

import numpy as np

from .cards import CardTables

# ── enum ranges, sized with headroom ─────────────────────────────────────────
# cg/api.py states members may be added during the competition, so every table
# is oversized and every lookup clamps. A new enum member must degrade to a
# shared bucket, never index out of bounds.
N_SELECT_TYPE = 24
N_CONTEXT = 64
N_OPTION_TYPE = 24
N_AREA = 20
N_ENERGY_TYPE = 16

# ── global slots ─────────────────────────────────────────────────────────────
GLOBAL_SLOTS = (
    ("select_type", N_SELECT_TYPE),
    ("context", N_CONTEXT),
    ("min_count", 14),
    ("max_count", 14),
    ("n_options", 16),
    ("turn", 16),
    ("turn_action", 16),
    ("first_player", 3),
    ("supporter_played", 2),
    ("stadium_played", 2),
    ("energy_attached", 2),
    ("retreated", 2),
    ("remain_energy_cost", 8),
    ("remain_damage_counter", 12),
    ("me_prize_left", 8),
    ("opp_prize_left", 8),
    ("me_deck_count", 16),
    ("opp_deck_count", 16),
    ("me_hand_count", 16),
    ("opp_hand_count", 16),
    ("me_discard_count", 16),
    ("opp_discard_count", 16),
    ("me_bench_count", 10),
    ("opp_bench_count", 10),
    ("me_bench_max", 10),
    ("opp_bench_max", 10),
    ("me_status", 32),
    ("opp_status", 32),
    ("remaining_overage", 14),
    ("stadium_owner", 3),
)
N_GLOBAL = len(GLOBAL_SLOTS)

# ── entity token slots ───────────────────────────────────────────────────────
TOK_ME_ACTIVE, TOK_OPP_ACTIVE = 1, 2
TOK_ME_BENCH, TOK_OPP_BENCH = 3, 4
TOK_HAND, TOK_STADIUM = 5, 6
TOK_CONTEXT_CARD, TOK_EFFECT_CARD = 7, 8
# The acting seat's own deck list (Architecture_A deck conditioning). One token
# per distinct card id, emitted only when the caller supplies the 60 — build
# time via `--deck-tokens`, inference via `deck.csv`. Its absence is a build
# mode, not a missing feature.
TOK_DECK = 9
N_TOK_TYPE = 12  # 1..9 used; headroom per the enum-growth rule

HAND_TOKEN_CAP = 16  # p99 hand size is 18; the tail is truncated, not dropped
BENCH_TOKEN_CAP = 10  # corpus max benchMax is 8; headroom for an effect that raises it
DECK_TOKEN_CAP = 40  # a 60-card list is 15-24 distinct ids; 40 clears the maximum

TOKEN_SLOTS = (
    ("tok_type", N_TOK_TYPE),
    ("card_id", 0),  # 0 = sized from CardTables at build time
    ("tool_id", 0),
    ("pre_evolution_id", 0),
    ("stage", 6),
    ("slot", BENCH_TOKEN_CAP + HAND_TOKEN_CAP + 2),
    ("hp", 24),
    ("hp_fraction", 14),
    ("energy_count", 12),
    ("energy_type", N_ENERGY_TYPE + 1),
    ("appear_this_turn", 3),
    # Copies of this card in the deck list. Only deck tokens set it; every other
    # token type leaves it 0, this encoder's "not applicable" everywhere. Cap 6
    # (size 7): the four-copy rule exempts basic energy, and corpus lists run up
    # to 9 of one. Appended, never inserted — earlier slots are indexed by number
    # (`TOKEN_SLOTS[5]`), so their positions must not move.
    ("copies", 7),
)
N_TOKEN = len(TOKEN_SLOTS)
TOKEN_CARD_SLOTS = (1, 2, 3)  # share one card-embedding table across these
TOKEN_COPIES_SLOT = N_TOKEN - 1

# ── option slots ─────────────────────────────────────────────────────────────
# Slots are appended, never reordered: a checkpoint carries these sizes and the
# fingerprint is over this file, so an insertion would silently re-point every
# embedding table in an older cache.
#
# `number`, `count` and `energy_index` come from the schema-v2 columns
# (`docs/rl-data-spec.md` §5). Sizes are the measured ranges over three days of
# the regenerated corpus, +2 for the reserved 0 and the clamp ceiling:
# number 0–13, count 1–3, energyIndex 0–11. Exact values rather than buckets —
# a COUNT select asks "how many?", and 2 versus 3 is the whole decision.
#
# Three of the eight v2 columns are deliberately NOT slots, all three measured
# on the full corpus rather than assumed:
#   * `option_tool_indices` is **0 on every option that carries it** — a
#     constant is not a feature.
#   * `option_card_ids` / `option_serials` appear only on SKILL, and every SKILL
#     row has minCount == maxCount == n_options, so 0 of 33,548,261 unforced rows
#     carry them. `option_card_ids` is still used, as a *source* rather than a
#     slot: `option_card_id()` prefers it over index resolution when populated.
OPTION_SLOTS = (
    ("option_type", N_OPTION_TYPE),
    ("area", N_AREA),
    ("index", 16),
    ("owner", 3),
    ("inplay_area", N_AREA),
    ("inplay_index", 12),
    ("attack_id", 0),  # sized from CardTables
    ("card_id", 0),
    ("inplay_card_id", 0),
    ("ordinal", 16),
    ("number", 16),
    ("count", 6),
    ("energy_index", 14),
    # Autoregressive decoding state: 0 before any pick is made, 1 still
    # available, 2 already taken this decision. The cache stores 1 for every
    # option; the decoder rewrites it per step on both frontends.
    ("picked", 3),
)
N_OPT = len(OPTION_SLOTS)
OPT_ATTACK_SLOT = 6
OPT_CARD_SLOT = 7
OPT_INPLAY_CARD_SLOT = 8
OPT_PICKED_SLOT = 13
PICK_AVAILABLE, PICK_TAKEN = 1, 2

# OptionType.PLAY carries an index and **no area** — the zone is implicitly the
# hand. Measured on 2026-07-27: 162,209 of 867,000 options are PLAY, every one of
# them with `area` null, which is 18.7% of all options and the most common MAIN
# action there is. Resolving them without this rule silently drops the identity
# of the card being played.
OPTION_PLAY = 7

# ── log slots ────────────────────────────────────────────────────────────────
# `obs.logs` is everything that happened since this seat's previous selection —
# what the opponent played, what attacked, what took damage. Eleven corpus
# columns that v1 did not use at all.
#
# Emitted only when the tensor cache was built with `--with-logs`, so the delta
# can be measured on its own. The manifest records which, and the model sizes
# itself from the manifest.
LOG_CAP = 24  # keeps every event on 95.5% of rows; the newest are kept
N_LOG_TYPE = 32  # LogType runs to 23; headroom per cg/api.py

LOG_SLOTS = (
    ("log_type", N_LOG_TYPE),
    ("owner", 3),
    ("card_id", 0),
    ("target_card_id", 0),
    ("attack_id", 0),
    ("from_area", N_AREA),
    ("to_area", N_AREA),
    ("value", 20),
    ("damage_flag", 3),
    ("recency", 16),
)
N_LOG = len(LOG_SLOTS)
LOG_CARD_SLOTS = (2, 3)
LOG_ATTACK_SLOT = 4

# HP changes are signed and reach -820. Buckets are asymmetric because damage is
# far more common and more finely graded than healing.
VALUE_EDGES = (-300, -200, -150, -120, -100, -80, -60, -40, -20, -10,
               -1, 0, 10, 20, 40, 80, 160)

# Areas an option index can be resolved to a card id in, using the rule verified
# in docs/rl-data-spec.md: the option belongs to the acting seat when
# playerIndex is null or equal to the acting seat.
AREA_DECK, AREA_HAND, AREA_DISCARD = 1, 2, 3
AREA_ACTIVE, AREA_BENCH, AREA_PRIZE = 4, 5, 6
AREA_STADIUM, AREA_LOOKING = 7, 12


def _clamp(value, size: int) -> int:
    """Fit a raw enum value into its table, reserving 0 for not-applicable."""
    if value is None:
        return 0
    value = int(value)
    if value < 0:
        return 0
    return min(value + 1, size - 1)


def _bucket(value, edges: tuple[int, ...]) -> int:
    """Bucket index in 1..len(edges)+1, or 0 when the value is absent.

    Edges are upper-inclusive bounds. Negative values land in the first bucket
    rather than in "absent" — HP genuinely goes negative in this corpus.
    """
    if value is None:
        return 0
    value = int(value)
    for i, edge in enumerate(edges):
        if value <= edge:
            return i + 1
    return len(edges) + 1


COUNT_EDGES = (0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 30, 45)
HP_EDGES = (0, 10, 30, 50, 70, 90, 110, 130, 160, 190, 220, 260, 300, 340, 380, 420)
FRACTION_EDGES = tuple(range(0, 100, 10))
TURN_EDGES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 32, 45)
ACTION_EDGES = (0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 30, 45)
OVERAGE_EDGES = (0, 30, 90, 150, 240, 330, 420, 480, 540, 570, 590)
SMALL_EDGES = tuple(range(0, 12))


def slot_sizes(tables: CardTables, with_logs: bool = False) -> dict[str, list[int]]:
    """Vocabulary size per slot, for building the embedding tables.

    The model reads this rather than hardcoding sizes, so adding a slot here is
    the only edit needed. A checkpoint records these sizes; loading one whose
    sizes differ is a hard error, not a silent reshape.
    """
    tokens = []
    for _, size in TOKEN_SLOTS:
        tokens.append(tables.card_rows if size == 0 else size)
    options = []
    for name, size in OPTION_SLOTS:
        if name == "attack_id":
            options.append(tables.attack_rows)
        elif name in ("card_id", "inplay_card_id"):
            options.append(tables.card_rows)
        else:
            options.append(size)
    sizes = {
        "globals": [size for _, size in GLOBAL_SLOTS],
        "tokens": tokens,
        "options": options,
    }
    if with_logs:
        sizes["logs"] = [
            tables.card_rows if name in ("card_id", "target_card_id")
            else tables.attack_rows if name == "attack_id"
            else size
            for name, size in LOG_SLOTS
        ]
    return sizes


def _status_mask(row, prefix: str) -> int:
    bits = 0
    for i, name in enumerate(("poisoned", "burned", "asleep", "paralyzed", "confused")):
        if row.get(f"{prefix}_{name}"):
            bits |= 1 << i
    return bits


def encode_globals(row: dict) -> np.ndarray:
    """Fixed-width global feature vector.

    `first_player` is made seat-relative here. The raw column is an absolute
    player index, so feeding it directly would teach the model that seat 0 is
    special when what matters is whether *this* seat moves first.
    """
    seat = int(row["seat"])
    first = row.get("first_player")
    if first is None or first < 0:
        first_rel = 0
    else:
        first_rel = 1 if int(first) == seat else 2
    owner = row.get("stadium_owner")
    if owner is None or owner < 0:
        stadium_owner = 0
    else:
        stadium_owner = 1 if int(owner) == seat else 2

    values = [
        _clamp(row["select_type"], N_SELECT_TYPE),
        _clamp(row["context"], N_CONTEXT),
        _bucket(row["min_count"], SMALL_EDGES),
        _bucket(row["max_count"], SMALL_EDGES),
        _bucket(row["n_options"], COUNT_EDGES),
        _bucket(row["turn"], TURN_EDGES),
        _bucket(row["turn_action"], ACTION_EDGES),
        first_rel,
        1 if row.get("supporter_played") else 0,
        1 if row.get("stadium_played") else 0,
        1 if row.get("energy_attached") else 0,
        1 if row.get("retreated") else 0,
        _bucket(row.get("remain_energy_cost"), SMALL_EDGES[:6]),
        _bucket(row.get("remain_damage_counter"), SMALL_EDGES[:10]),
        _bucket(row.get("me_prize_left"), SMALL_EDGES[:6]),
        _bucket(row.get("opp_prize_left"), SMALL_EDGES[:6]),
        _bucket(row.get("me_deck_count"), COUNT_EDGES),
        _bucket(row.get("opp_deck_count"), COUNT_EDGES),
        _bucket(row.get("me_hand_count"), COUNT_EDGES),
        _bucket(row.get("opp_hand_count"), COUNT_EDGES),
        _bucket(row.get("me_discard_count"), COUNT_EDGES),
        _bucket(row.get("opp_discard_count"), COUNT_EDGES),
        _bucket(row.get("me_bench_count"), SMALL_EDGES[:8]),
        _bucket(row.get("opp_bench_count"), SMALL_EDGES[:8]),
        _bucket(row.get("me_bench_max"), SMALL_EDGES[:8]),
        _bucket(row.get("opp_bench_max"), SMALL_EDGES[:8]),
        _status_mask(row, "me"),
        _status_mask(row, "opp"),
        _bucket(row.get("remaining_overage"), OVERAGE_EDGES),
        stadium_owner,
    ]
    out = np.asarray(values, dtype=np.int16)
    assert out.shape == (N_GLOBAL,), f"global width {out.shape} != {N_GLOBAL}"
    return out


def _pokemon_token(
    tables: CardTables,
    tok_type: int,
    slot: int,
    card_id,
    hp,
    max_hp,
    energy_types,
    tool_ids,
    stage,
    appear,
    pre_evolution_ids,
) -> list[int]:
    energy_types = list(energy_types or [])
    tool_ids = list(tool_ids or [])
    pre_evolution_ids = list(pre_evolution_ids or [])
    if hp is not None and max_hp:
        fraction = _bucket(max(0, int(hp)) * 100 // max(1, int(max_hp)), FRACTION_EDGES)
    else:
        fraction = 0
    return [
        tok_type,
        tables.card_index(card_id),
        tables.card_index(tool_ids[0] if tool_ids else None),
        tables.card_index(pre_evolution_ids[0] if pre_evolution_ids else None),
        _clamp(stage, 6),
        _clamp(slot, TOKEN_SLOTS[5][1]),
        _bucket(hp, HP_EDGES),
        fraction,
        _bucket(len(energy_types), SMALL_EDGES[:10]),
        _clamp(_dominant_energy(energy_types), N_ENERGY_TYPE + 1),
        2 if appear else 1,
        0,  # copies: only deck tokens set it
    ]


def _dominant_energy(energy_types) -> int | None:
    """The most common attached energy type, or None when nothing is attached.

    A full 12-way count vector would be better and is an R2 delta; this keeps the
    row int16 and still separates a Darkness-loaded attacker from a Water one.
    """
    if not energy_types:
        return None
    counts: dict[int, int] = {}
    for energy in energy_types:
        counts[int(energy)] = counts.get(int(energy), 0) + 1
    return max(counts, key=lambda k: (counts[k], -k))


def _card_token(tables: CardTables, tok_type: int, slot: int, card_id,
                copies: int = 0) -> list[int]:
    return [
        tok_type,
        tables.card_index(card_id),
        0,
        0,
        0,
        _clamp(slot, TOKEN_SLOTS[5][1]),
        0,
        0,
        0,
        0,
        0,
        _clamp(copies - 1, TOKEN_SLOTS[TOKEN_COPIES_SLOT][1]) if copies else 0,
    ]


def encode_tokens(row: dict, tables: CardTables) -> np.ndarray:
    """Ragged entity tokens: only what is actually on the board or in hand.

    Empty bench slots and an empty Active Spot emit no token at all rather than a
    padded one — an empty slot is already visible from `me_bench_count` in the
    globals, and shorter sequences are cheaper on 2 vCPUs.
    """
    tokens: list[list[int]] = []

    for tok_type, prefix in ((TOK_ME_ACTIVE, "me"), (TOK_OPP_ACTIVE, "opp")):
        if row.get(f"{prefix}_active_id") is not None:
            tokens.append(
                _pokemon_token(
                    tables,
                    tok_type,
                    0,
                    row[f"{prefix}_active_id"],
                    row.get(f"{prefix}_active_hp"),
                    row.get(f"{prefix}_active_max_hp"),
                    row.get(f"{prefix}_active_energy_types"),
                    row.get(f"{prefix}_active_tool_ids"),
                    row.get(f"{prefix}_active_stage"),
                    row.get(f"{prefix}_active_appear_this_turn"),
                    row.get(f"{prefix}_active_pre_evolution_ids"),
                )
            )

    for tok_type, prefix in ((TOK_ME_BENCH, "me"), (TOK_OPP_BENCH, "opp")):
        ids = list(row.get(f"{prefix}_bench_ids") or [])
        hps = list(row.get(f"{prefix}_bench_hp") or [])
        max_hps = list(row.get(f"{prefix}_bench_max_hp") or [])
        energies = list(row.get(f"{prefix}_bench_energy_types") or [])
        tools = list(row.get(f"{prefix}_bench_tool_ids") or [])
        stages = list(row.get(f"{prefix}_bench_stage") or [])
        appears = list(row.get(f"{prefix}_bench_appear_this_turn") or [])
        pre_evolutions = list(row.get(f"{prefix}_bench_pre_evolution_ids") or [])
        for slot, card_id in enumerate(ids[:BENCH_TOKEN_CAP]):
            tokens.append(
                _pokemon_token(
                    tables,
                    tok_type,
                    slot,
                    card_id,
                    hps[slot] if slot < len(hps) else None,
                    max_hps[slot] if slot < len(max_hps) else None,
                    energies[slot] if slot < len(energies) else None,
                    tools[slot] if slot < len(tools) else None,
                    stages[slot] if slot < len(stages) else None,
                    appears[slot] if slot < len(appears) else None,
                    pre_evolutions[slot] if slot < len(pre_evolutions) else None,
                )
            )

    hand = list(row.get("my_hand_ids") or [])
    for slot, card_id in enumerate(hand[:HAND_TOKEN_CAP]):
        tokens.append(_card_token(tables, TOK_HAND, BENCH_TOKEN_CAP + slot, card_id))

    # The acting seat's own 60 (Architecture_A deck conditioning): one token per
    # distinct card id with its copy count. The backbone already infers the deck
    # from the board; making it an input is what lets one checkpoint serve many
    # decks without averaging into the field's mean policy — the measured win for
    # low-seat archetypes. Present only when the caller supplied the list, so a
    # non-deck-token checkpoint is fed no deck tokens and sees exactly what it
    # trained on.
    deck = row.get("my_deck_ids")
    if deck:
        copies: dict[int, int] = {}
        for card_id in deck:
            if card_id is None:
                continue
            copies[int(card_id)] = copies.get(int(card_id), 0) + 1
        for card_id in sorted(copies)[:DECK_TOKEN_CAP]:
            tokens.append(_card_token(tables, TOK_DECK, 0, card_id, copies[card_id]))

    if row.get("stadium_id") is not None:
        tokens.append(_card_token(tables, TOK_STADIUM, 0, row["stadium_id"]))
    if row.get("context_card_id") is not None:
        tokens.append(_card_token(tables, TOK_CONTEXT_CARD, 0, row["context_card_id"]))
    if row.get("effect_card_id") is not None:
        tokens.append(_card_token(tables, TOK_EFFECT_CARD, 0, row["effect_card_id"]))

    if not tokens:
        # A row with nothing on the board (deck selection, first setup pick)
        # still needs one token so attention has a key to look at.
        tokens.append(_card_token(tables, TOK_HAND, 0, None))
    out = np.asarray(tokens, dtype=np.int16)
    assert out.shape[1] == N_TOKEN, f"token width {out.shape[1]} != {N_TOKEN}"
    return out


def option_card_id(row: dict, area, index, player, seat: int, option_type=None):
    """Resolve an option's referenced card id.

    Mirrors `card_at()` in the shipped heuristic agents and the rule verified in
    `docs/rl-data-spec.md`: an option belongs to the acting seat when
    `playerIndex` is null or equal to it. Verified against 2.43M option
    references on 2026-07-27 with zero out-of-range resolutions.

    `OptionType.PLAY` is the exception the shipped agents also special-case: it
    carries an index with no area, and the zone is the hand. Confirmed against
    the log stream — see `tools/rl/validate_resolution.py`.

    PRIZE resolves to nothing on purpose — prize cards are face down, so there is
    no id to encode and inventing one would leak information the agent cannot
    have at inference.
    """
    if area is None and option_type is not None and int(option_type) == OPTION_PLAY:
        area = AREA_HAND
    if area is None or index is None:
        return None
    area = int(area)
    index = int(index)
    mine = player is None or int(player) == seat
    if area == AREA_HAND:
        source = row.get("my_hand_ids")
    elif area == AREA_DECK:
        source = row.get("select_deck_ids")
    elif area == AREA_LOOKING:
        source = row.get("looking_ids")
    elif area == AREA_DISCARD:
        source = row.get("me_discard_ids") if mine else row.get("opp_discard_ids")
    elif area == AREA_BENCH:
        source = row.get("me_bench_ids") if mine else row.get("opp_bench_ids")
    elif area == AREA_ACTIVE:
        card_id = row.get("me_active_id") if mine else row.get("opp_active_id")
        return card_id if index == 0 else None
    elif area == AREA_STADIUM:
        return row.get("stadium_id") if index == 0 else None
    else:
        return None
    if not source or index >= len(source):
        return None
    return source[index]


def inplay_card_id(row: dict, area, index, seat: int):
    """Resolve the Pokémon an option acts *on*, from `inPlayArea`/`inPlayIndex`.

    `ATTACH` and `EVOLVE` are the only option types that carry these fields, and
    both act on the acting seat's own board — measured on 2026-07-27, 188,075
    ATTACH and 31,495 EVOLVE options, all with a null `playerIndex`. Without
    this the model sees which card is being attached but not to what, and
    This supports attach/evolve/retreat targeting across the bulk of MAIN decisions.
    """
    if area is None or index is None:
        return None
    area = int(area)
    index = int(index)
    if area == AREA_ACTIVE:
        return row.get("me_active_id") if index == 0 else None
    if area == AREA_BENCH:
        bench = row.get("me_bench_ids") or []
        return bench[index] if index < len(bench) else None
    return None


def encode_options(row: dict, tables: CardTables) -> np.ndarray:
    """One feature row per option, in the engine's order.

    `ordinal` carries that order deliberately. For COUNT and SKILL selects the
    shipped corpus has no field that distinguishes one option from another — the
    engine's `Option.number` / `Option.cardId` were not extracted — so position
    is the only signal available, and 477k rows depend on it. See
    `docs/rl-data-spec.md` §2 item 10.
    """
    seat = int(row["seat"])
    types = list(row["option_types"] or [])
    areas = list(row.get("option_areas") or [])
    indices = list(row.get("option_indices") or [])
    players = list(row.get("option_players") or [])
    inplay_areas = list(row.get("option_inplay_area") or [])
    inplay_indices = list(row.get("option_inplay_index") or [])
    attack_ids = list(row.get("option_attack_ids") or [])
    numbers = list(row.get("option_numbers") or [])
    counts = list(row.get("option_counts") or [])
    energy_indices = list(row.get("option_energy_indices") or [])
    card_ids = list(row.get("option_card_ids") or [])

    def at(seq, i):
        return seq[i] if i < len(seq) else None

    def v2(seq, i):
        """A schema-v2 column: -1 from parquet and None live both mean absent.

        A pre-v2 partition has no column at all, so the list is empty and every
        read returns None. That is the same "absent" the encoder already handles,
        which is why a v1 cache and a v2 cache differ in value but not in shape.
        """
        value = at(seq, i)
        return None if value is None or int(value) < 0 else int(value)

    out = np.zeros((len(types), N_OPT), dtype=np.int16)
    for i in range(len(types)):
        option_type = at(types, i)
        area, index, player = at(areas, i), at(indices, i), at(players, i)
        inplay_area, inplay_index = at(inplay_areas, i), at(inplay_indices, i)
        owner = 0
        if player is not None:
            owner = 1 if int(player) == seat else 2
        elif area is not None or (option_type is not None and int(option_type) == OPTION_PLAY):
            owner = 1  # null playerIndex means the acting seat's own zone
        out[i] = (
            _clamp(option_type, N_OPTION_TYPE),
            _clamp(area, N_AREA),
            _bucket(index, SMALL_EDGES[:8] + (10, 14, 20, 30)),
            owner,
            _clamp(inplay_area, N_AREA),
            _clamp(inplay_index, 12),
            tables.attack_index(at(attack_ids, i)),
            tables.card_index(
                v2(card_ids, i)          # SKILL states its own identity; prefer it
                or option_card_id(row, area, index, player, seat, option_type)
            ),
            tables.card_index(inplay_card_id(row, inplay_area, inplay_index, seat)),
            _bucket(i, SMALL_EDGES[:8] + (10, 14, 20, 30, 60)),
            _clamp(v2(numbers, i), 16),
            _clamp(v2(counts, i), 6),
            _clamp(v2(energy_indices, i), 14),
            PICK_AVAILABLE,
        )
    return out


def encode_logs(row: dict, tables: CardTables) -> np.ndarray:
    """Events since this seat's previous selection, newest last.

    Absent fields arrive as `-1` from the corpus and as `None` from a live
    `Log`; `_clamp` maps both to index 0, so the two frontends agree without
    either of them normalising first.

    Only the last `LOG_CAP` events are kept. Truncating the *old* end is the
    right way round — a 183-event burst is a chain of effects resolving, and the
    decision in front of the agent is about how it ended.
    """
    types = list(row.get("log_types") or [])
    if not types:
        return np.zeros((0, N_LOG), dtype=np.int16)

    seat = int(row["seat"])

    def col(name):
        return list(row.get(name) or [])

    players, card_ids = col("log_players"), col("log_card_ids")
    targets, attacks = col("log_target_card_ids"), col("log_attack_ids")
    from_areas, to_areas = col("log_from_areas"), col("log_to_areas")
    values, flags = col("log_values"), col("log_damage_flags")

    def at(seq, i):
        return seq[i] if i < len(seq) else None

    start = max(0, len(types) - LOG_CAP)
    out = np.zeros((len(types) - start, N_LOG), dtype=np.int16)
    for position, i in enumerate(range(start, len(types))):
        player = at(players, i)
        owner = 0
        if player is not None and int(player) >= 0:
            owner = 1 if int(player) == seat else 2
        value = at(values, i)
        out[position] = (
            _clamp(at(types, i), N_LOG_TYPE),
            owner,
            tables.card_index(_absent(at(card_ids, i))),
            tables.card_index(_absent(at(targets, i))),
            tables.attack_index(_absent(at(attacks, i))),
            _clamp(at(from_areas, i), N_AREA),
            _clamp(at(to_areas, i), N_AREA),
            # `-1` is the absent sentinel here, not a one-point HP change: real
            # HP changes are multiples of ten. Every other negative is damage.
            0 if value is None or int(value) == -1 else _bucket(int(value), VALUE_EDGES),
            2 if at(flags, i) else 1,
            # Distance from the decision, newest = 1. Order matters more than
            # position here: the model reads these as a set with a recency mark.
            _clamp(len(types) - 1 - i, 16),
        )
    return out


def _absent(value):
    """Corpus sentinel `-1` and live `None` both mean the field does not apply."""
    return None if value is None or value < 0 else value


def is_forced(row: dict) -> bool:
    """No real choice: the engine offered one option, or demanded all of them.

    Excluded from the training loss and from the reported accuracy. 8.62% of the
    corpus, and counting them inflates top-1 by roughly 8 points.
    """
    n = int(row["n_options"])
    return n <= 1 or (int(row["min_count"]) == int(row["max_count"]) == n)
