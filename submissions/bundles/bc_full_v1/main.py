"""Marnie's Grimmsnarl ex, piloted by a behaviour-cloned option scorer.

Trained on top-of-ladder replay decisions. The network scores each entry of
`obs.select.option` against
the encoded state; the guard shell around it is what keeps a bad forward pass
from costing a game.

The `bcnet/` package here is a copy produced by `tools/bc/export_agent.py`, not
a hand-written second implementation. Training and inference run the same
encoder, and `_package_fingerprint()` re-hashes the shipped copy at load so an
edited encoder disables inference instead of silently feeding the model features
it was not trained on.

Failure policy, in order of preference: answer forced selections without
inference; run the net; validate its output against the action contract; on any
doubt play a legal default. An exception scores -1, exactly what a loss scores,
so there is never a reason to raise.

`agent` is the last module-level callable in this file. `kaggle_environments`
ignores the name and takes the last one bound, so nothing may be defined below
it — not a helper, not an import, not a class.
"""

import hashlib
import math
import os
import random
import sys

# `kaggle_environments.get_last_callable` execs this file with a namespace that
# does NOT define `__file__`, so touching it unguarded is a NameError before the
# first card — an ERROR for both agents, which scores -1. Confirmed from a failed
# validation episode. Local harnesses may inject `__file__`, so the exported
# package must defend this path explicitly.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = ""


def _looks_like_us(directory):
    """Our own agent directory: it holds both our deck and our package."""
    return bool(directory) and os.path.isfile(os.path.join(directory, "deck.csv")) \
        and os.path.isdir(os.path.join(directory, "bcnet"))


if not _looks_like_us(_HERE):
    # `get_last_callable` appends main.py's directory to sys.path before exec'ing
    # it — that is why `from cg.api import ...` resolves at all — so the agent
    # directory is discoverable even with no `__file__` and a foreign cwd. The
    # two markers together keep this from latching onto another agent's folder
    # when two are loaded in one process, as tools/ab.py does.
    _HERE = ""
    for _candidate in list(sys.path) + ["/kaggle_simulations/agent"]:
        if _looks_like_us(_candidate):
            _HERE = _candidate
            break

for _path in (_HERE, "/kaggle_simulations/agent"):
    if _path and _path not in sys.path and os.path.isdir(_path):
        sys.path.insert(0, _path)

import torch

# The sandbox is 2 vCPU. Thread oversubscription is the one way a model that
# evaluates in a millisecond becomes slow enough to matter against the shared
# 600-second bank.
torch.set_num_threads(2)

from cg.api import to_observation_class

from bcnet.cards import load_tables
try:
    # Decoder-era encoders only. An agent wrapping a pre-decoder checkpoint
    # ships a bcnet without these; the decoder path is unreachable there
    # (_DECODER is False for any checkpoint that never trained a stop head),
    # so import failure must not kill the whole module.
    from bcnet.encoder import OPT_PICKED_SLOT, PICK_TAKEN
except ImportError:
    OPT_PICKED_SLOT = PICK_TAKEN = None
from bcnet.encoder import (
    encode_globals, encode_logs, encode_options,
    encode_tokens, is_forced,
)
from bcnet.model import OptionScorer
from bcnet.rows import row_from_observation

_MODEL = None
_TABLES = None
_DECK = None
_READY = False
_MULTI_PICK_RATIO = 0.5  # keep options within a factor of the best; see below
# Set from the checkpoint. A stop head that was never trained still emits a
# logit, and acting on it would be acting on noise, so the old threshold path
# stays in place for every checkpoint that did not learn one.
_DECODER = False
# Set from the checkpoint. Deck conditioning (Architecture_A): when true the agent
# feeds its own deck.csv as deck tokens so one checkpoint can pilot many decks.
# Handing deck tokens to a model that never trained on them shifts every row off
# distribution, so this stays a per-checkpoint switch, never assumed.
_DECK_TOKENS = False
# Set from the checkpoint. Whether the critic head was trained. Value-backed PUCT
# runs only when this is true; an untrained value head emits noise, and the PIMC
# logs are the warning — search that trusted a crude leaf regressed.
_VALUE = False


def _data_path(name):
    candidates = [name]
    if _HERE:
        candidates.append(os.path.join(_HERE, name))
    candidates.append(os.path.join("/kaggle_simulations/agent", name))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return name


def _read_deck():
    global _DECK
    if _DECK is None:
        with open(_data_path("deck.csv")) as handle:
            _DECK = [int(x) for x in handle.read().split() if x.strip()]
    return list(_DECK)


def _package_fingerprint():
    """Hash the encoder we are actually about to run.

    This must hash the *shipped* files, not read a number someone wrote down.
    The earlier version compared `fingerprint.json`'s stamp against the
    checkpoint's, and `export_agent.py` writes both from the same value — so the
    two always agreed and editing `bcnet/encoder.py` after export changed
    neither. The check passed by construction while the docstring above claimed
    it guarded against exactly that.

    Same files, same order, same digest as `build_dataset.encoder_fingerprint`,
    which is what the checkpoint recorded at training time. Comparing the two
    catches both directions: an edited encoder, and a checkpoint dropped in from
    a different one.
    """
    digest = hashlib.sha256()
    for name in ("encoder.py", "cards.py", "rows.py"):
        with open(_data_path(os.path.join("bcnet", name)), "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()[:16]


def _load():
    """Build the model once. Any failure leaves the agent on legal defaults."""
    global _MODEL, _TABLES, _READY, _DECODER, _DECK_TOKENS, _VALUE
    if _READY:
        return
    _READY = True
    try:
        checkpoint = torch.load(_data_path(os.path.join("bcnet", "bc_model.pt")),
                                map_location="cpu", weights_only=False)
        shipped = _package_fingerprint()
        if shipped != checkpoint["encoder_fingerprint"]:
            raise RuntimeError(
                f"encoder mismatch: bcnet/ hashes to {shipped}, checkpoint was trained "
                f"by {checkpoint['encoder_fingerprint']}"
            )
        _TABLES = load_tables()
        model = OptionScorer(
            checkpoint["slot_sizes"],
            torch.from_numpy(_TABLES.card_feat),
            torch.from_numpy(_TABLES.attack_feat),
            **checkpoint["model"],
        )
        # Checkpoints written before the static tables became non-persistent
        # buffers carry `card_feat`/`attack_feat` in their state_dict, and
        # `load_state_dict` rejects them as unexpected keys. Dropping them is
        # always correct: the constructor above already built both tables from
        # the SDK, and they are inputs, never learned. Without this an old
        # checkpoint disables the net and the agent plays legal defaults.
        state = {k: v for k, v in checkpoint["state_dict"].items()
                 if k not in ("card_feat", "attack_feat")}
        # strict=False: the value_head is always constructed but a pre-critic
        # checkpoint has no weights for it. Those keys land in `missing` and stay
        # at init — inert, because _VALUE stays false and nothing calls them. The
        # fingerprint gate already guarantees the encoder matches, so a genuine
        # architecture mismatch cannot hide behind this.
        result = model.load_state_dict(state, strict=False)
        unexpected = [k for k in getattr(result, "unexpected_keys", []) if k]
        # Only the critic head may be absent (a pre-value checkpoint). Anything
        # else missing means a truncated or wrong state_dict, which strict=False
        # would otherwise wave through into an agent that plays legal defaults.
        missing = [k for k in getattr(result, "missing_keys", [])
                   if not k.startswith("value_head")]
        if unexpected or missing:
            raise RuntimeError(f"checkpoint key mismatch: missing {missing[:4]} "
                               f"unexpected {unexpected[:4]}")
        model.eval()
        _MODEL = model
        _DECODER = bool(checkpoint.get("decoder", False))
        _DECK_TOKENS = bool(checkpoint.get("deck_tokens", False))
        _VALUE = bool(checkpoint.get("value", False))
    except Exception as exc:  # noqa: BLE001 - a load failure must not end the episode
        # Playing legal defaults for the rest of the game is the right response
        # and an invisible one. stderr reaches the Kaggle episode logs, so a
        # disabled net is at least diagnosable after the fact.
        # tools/smoke_archive.py catches it before the fact.
        print(f"bcnet disabled: {type(exc).__name__}: {exc}", file=sys.stderr)
        _MODEL = None


def _legal_default(select):
    """Cheapest action that satisfies the contract: the first `minCount` indices."""
    count = len(select.option)
    take = min(max(select.minCount, 1 if count else 0), select.maxCount, count)
    return list(range(take))


def _validate(picked, select):
    """Enforce the action contract. Returns a legal list, always."""
    count = len(select.option)
    seen = []
    for index in picked:
        if isinstance(index, int) and 0 <= index < count and index not in seen:
            seen.append(index)
    if len(seen) > select.maxCount:
        seen = seen[: select.maxCount]
    for index in range(count):
        if len(seen) >= select.minCount:
            break
        if index not in seen:
            seen.append(index)
    if not seen and count and select.minCount == 0 and select.maxCount > 0:
        # No demonstrator declined once in 30.8M corpus rows, so an empty pick
        # is never the cloned behaviour even where the engine permits it. The
        # maxCount guard is belt and braces: the corpus never shows maxCount 0,
        # and returning [0] against one would be the INVALID this shell exists
        # to prevent.
        seen = [0]
    return seen


def _score(obs, remaining_overage, model=None, decoder=None, deck_ids=None):
    """Score an option set with `model` (the own net by default).

    `model`/`decoder` are threaded so a rollout can run a *different* policy for
    the opponent seat than the one piloting our own — the whole point of the
    archetype opponent models. Both default to the live agent's own net and its
    stop-head flag, so the live decision path is unchanged.

    `deck_ids` is the acting seat's 60-card list for deck conditioning. Pass it
    only for a deck-token checkpoint and only the deck that policy should be
    conditioned on: our own list for our seat, the world's belief list for the
    opponent seat. `None` emits no deck tokens, which is what a non-deck-token
    checkpoint must see.
    """
    if model is None:
        model = _MODEL
    if decoder is None:
        decoder = _DECODER
    row = row_from_observation(obs, remaining_overage, deck_ids=deck_ids)
    globals_x = torch.from_numpy(encode_globals(row)).unsqueeze(0)
    tokens_x = torch.from_numpy(encode_tokens(row, _TABLES)).unsqueeze(0)
    options_x = torch.from_numpy(encode_options(row, _TABLES)).unsqueeze(0)
    token_mask = torch.ones(tokens_x.shape[:2], dtype=torch.bool)
    option_mask = torch.ones(options_x.shape[:2], dtype=torch.bool)
    logs_x = log_mask = None
    if model.log_emb is not None:
        # Only when the checkpoint was trained with a log stream. A model built
        # without one must not be handed logs, and one built with it must not be
        # starved of them — either way the features would differ from training.
        encoded = encode_logs(row, _TABLES)
        if encoded.shape[0] == 0:
            # 15.6% of decisions follow no events at all. Attention still needs a
            # key, so emit one fully-masked slot rather than an empty sequence.
            encoded = encoded[:0].reshape(0, encoded.shape[1])
            logs_x = torch.zeros((1, 1, encoded.shape[1]), dtype=torch.int16)
            log_mask = torch.zeros((1, 1), dtype=torch.bool)
        else:
            logs_x = torch.from_numpy(encoded).unsqueeze(0)
            log_mask = torch.ones(logs_x.shape[:2], dtype=torch.bool)
    def run(stop_legal=None):
        stop_mask = (None if stop_legal is None
                     else torch.tensor([bool(stop_legal)], dtype=torch.bool))
        with torch.no_grad():
            if stop_mask is None:
                # Positional-only call: a pre-decoder model.py has no stop_mask
                # parameter, and this main.py must be able to wrap one — the
                # pimc layer's whole point is composing with any checkpoint.
                logits = model(globals_x, tokens_x, token_mask, options_x,
                               option_mask, logs_x, log_mask)
            else:
                logits = model(globals_x, tokens_x, token_mask, options_x,
                               option_mask, logs_x, log_mask, stop_mask=stop_mask)
        return torch.softmax(logits[0].float(), dim=-1)

    if not decoder:
        return run()
    return run, options_x


def _decode(scorer, options_x, select):
    """Autoregressive multi-pick: score, take the best, mark it, score again.

    Replaces the threshold heuristic below for checkpoints trained with
    `--decoder`. The stop action is scored on the same scale as the options
    rather than compared against a ratio, and it is only offered once minCount
    is satisfied — so the contract is enforced by construction, not by
    `_validate` topping the list up in index order afterwards.
    """
    count = options_x.shape[1]
    if not count:
        return []
    picked: list[int] = []
    limit = min(select.maxCount, count)
    while len(picked) < limit:
        stop_legal = len(picked) >= select.minCount and len(picked) > 0
        probabilities = scorer(stop_legal)
        if stop_legal and int(torch.argmax(probabilities)) == count:
            break
        order = torch.argsort(probabilities[:count], descending=True).tolist()
        nxt = next((i for i in order if i not in picked), None)
        if nxt is None:
            break
        picked.append(nxt)
        options_x[0, nxt, OPT_PICKED_SLOT] = PICK_TAKEN
    return picked or [0]


def _pick_from(probabilities, select):
    """Turn per-option probabilities into an index list.

    The loss put uniform mass over every option the demonstrator chose, so a
    k-pick decision should come back as k options each near 1/k. Taking
    everything within half of the best probability recovers k without needing a
    separate count head, and 95.4% of corpus actions are single-pick anyway, so
    this path is the minority case. A learned count head is an R2 delta.
    """
    order = torch.argsort(probabilities, descending=True).tolist()
    if select.maxCount <= 1:
        return order[:1]
    # Seed with the k best, where k is the count the engine demands. Without
    # this the threshold below could return fewer than minCount and `_validate`
    # would top the list up in *index* order, which throws the model's ranking
    # away exactly where it is most needed: asked to pick 3 of 5 with the model
    # saying 5%/50%/20%/15%/10%, the old path returned indices [1, 0, 2] — the
    # best option, then the worst, then the second-best. Roughly one unforced
    # multi-pick decision in twenty.
    floor = min(max(select.minCount, 1), select.maxCount, len(order))
    picked = order[:floor]
    threshold = float(probabilities[order[0]]) * _MULTI_PICK_RATIO
    for index in order[floor:]:
        if len(picked) >= select.maxCount:
            break
        if float(probabilities[index]) < threshold:
            break
        picked.append(index)
    return picked


def _value(observation, model, deck_ids=None):
    """The critic's P(win) for the seat to act in `observation`, in (0, 1).

    Only the state stream is encoded — the critic scores the board, not a choice —
    so this is cheaper than a policy pass and reuses the model's state encoder.
    """
    row = row_from_observation(observation, 300.0, deck_ids=deck_ids)
    globals_x = torch.from_numpy(encode_globals(row)).unsqueeze(0)
    tokens_x = torch.from_numpy(encode_tokens(row, _TABLES)).unsqueeze(0)
    token_mask = torch.ones(tokens_x.shape[:2], dtype=torch.bool)
    logs_x = log_mask = None
    if model.log_emb is not None:
        encoded = encode_logs(row, _TABLES)
        if encoded.shape[0] == 0:
            logs_x = torch.zeros((1, 1, encoded.shape[1]), dtype=torch.int16)
            log_mask = torch.zeros((1, 1), dtype=torch.bool)
        else:
            logs_x = torch.from_numpy(encoded).unsqueeze(0)
            log_mask = torch.ones(logs_x.shape[:2], dtype=torch.bool)
    with torch.no_grad():
        return float(model.state_value(globals_x, tokens_x, token_mask, logs_x, log_mask))


# ─────────────────────────────────────────────────────────────────────────────
# Optional near-tie guard
#
# Not a search. The net proposes; the engine only breaks ties the net is not
# confident about. That distinction is the whole design: a hand value function
# that ranks *every* option measured 0.105 against the steel agent
# in campaign testing, while the same function used only to separate two
# options the net scores within a hair of each other never overrules a
# confident policy.
#
# Fires only when all of these hold, which is a small minority of decisions:
#   * SelectType is MAIN and maxCount is 1
#   * at least two options
#   * obs.search_begin_input is present (the engine will let us fork)
#   * the top-2 log-probability gap is under `margin`
#   * there is bank left above `reserve`
#
# Configured by bcnet/search.json, written by export_agent.py --search. A
# missing or malformed file disables it, so an agent exported before this
# existed behaves exactly as it did. It is OFF unless deliberately enabled,
# unless a controlled measurement explicitly enables it
# criterion in advance: lower bound vs plain BC <= 0.5 over 400 games -> drop it.
# ─────────────────────────────────────────────────────────────────────────────

import time as _time

_SEARCH = None
_SEARCH_API = None
_SEARCH_STATS = {"eligible": 0, "fired": 0, "changed": 0, "seconds": 0.0, "errors": 0,
                 "opp_worlds": 0}


def _search_config():
    """Read bcnet/search.json once. Anything unreadable means disabled."""
    global _SEARCH, _SEARCH_API
    if _SEARCH is not None:
        return _SEARCH
    _SEARCH = {"enabled": False}
    try:
        import json as _json
        with open(_data_path(os.path.join("bcnet", "search.json"))) as handle:
            cfg = _json.load(handle)
    except Exception:
        return _SEARCH
    if not cfg.get("enabled"):
        return _SEARCH
    try:
        from cg.api import SelectType, search_begin, search_end, search_release, search_step
    except Exception as exc:  # noqa: BLE001 - an SDK without the search API is not an error
        print(f"search guard disabled: {type(exc).__name__}: {exc}", file=sys.stderr)
        return _SEARCH
    _SEARCH_API = (search_begin, search_step, search_release, search_end)
    _SEARCH = {
        "enabled": True,
        # "guard" is the Plan C near-tie guard above. "pimc" is Architecture D
        # Optional determinized full-line rollouts with the net
        # playing both seats, aggregated over sampled worlds.
        "mode": str(cfg.get("mode", "guard")),
        # Resolved from the SDK enum, never persisted as an int: `cg/api.py`
        # says new members may be added during the competition.
        "main": int(SelectType.MAIN),
        "budget_fraction": float(cfg.get("budget_fraction", 0.02)),
        "budget_cap": float(cfg.get("budget_cap", 2.0)),
        "reserve": float(cfg.get("reserve", 120.0)),
        "candidates": int(cfg.get("candidates", 3)),
        "samples": int(cfg.get("samples", 2)),
        "margin": float(cfg.get("margin", 1.0)),
        "follow_forced": int(cfg.get("follow_forced", 8)),
        "worlds": int(cfg.get("worlds", 6)),
        "rollout_depth": int(cfg.get("rollout_depth", 240)),
        # "puct" (this file's value-backed search): PUCT allocation over the
        # prior's top-k, each simulation a short policy rollout truncated by the
        # critic V(s) instead of the material squash. c_puct is the exploration
        # constant; sims is the PUCT budget per world; leaf_depth is how far a
        # simulation walks before the critic evaluates (0 = evaluate the child
        # directly); min_sims is the floor below which the search defers to the
        # policy rather than act on a handful of visits.
        "c_puct": float(cfg.get("c_puct", 1.4)),
        "sims": int(cfg.get("sims", 24)),
        "leaf_depth": int(cfg.get("leaf_depth", 8)),
        "min_sims": int(cfg.get("min_sims", 8)),
        # puct only: overrule the policy's own pick only when the best candidate's
        # mean critic value beats it by at least this many P(win) points. We fire
        # on a policy near-tie, so the value edge must be decisive, not a hair
        # inside the simulation noise. 0.0 takes argmax-Q, an upper-bound setting.
        "overrule_margin": float(cfg.get("overrule_margin", 0.03)),
        "log": bool(cfg.get("log", False)),
    }
    return _SEARCH


def _basics():
    """Card ids in our own deck that can legally open. Cached."""
    global _BASICS
    if _BASICS is None:
        deck = _read_deck()
        table = getattr(_TABLES, "card_by_id", {}) or {}
        _BASICS = [c for c in deck if getattr(table.get(c), "basic", False)] or list(deck)
    return _BASICS


_BASICS = None


def _sample_pool(pool, count, rng):
    if count <= 0:
        return []
    if count <= len(pool):
        return rng.sample(pool, count)
    return [rng.choice(pool) for _ in range(count)]


def _determinize(observation, rng):
    """A legal hypothesis for every hidden zone search_begin demands.

    Each list must be at least as long as the true count or search_begin raises,
    and at setup opponent_deck must contain a Basic. Filling from our own list is
    a weak belief and a legal one; the guard only needs the fork to be valid.
    """
    state = observation.current
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    pool = _read_deck()

    opponent_active = []
    if opponent.active and opponent.active[0] is None:
        opponent_active = [rng.choice(_basics())]

    return {
        "your_deck": _sample_pool(pool, me.deckCount, rng),
        "your_prize": _sample_pool(pool, len(me.prize), rng),
        "opponent_deck": _sample_pool(pool, opponent.deckCount, rng),
        "opponent_prize": _sample_pool(pool, len(opponent.prize), rng),
        "opponent_hand": _sample_pool(pool, opponent.handCount, rng),
        "opponent_active": opponent_active,
    }


def _evaluate(observation, seat):
    """Value of a forked state from our seat. Prizes dominate by design.

    The runtime guard asks only for "does not lose material/prizes
    immediately", so this is deliberately crude — it exists to separate a play
    that knocks out our own Active from one that does not, never to rank a full
    option list.
    """
    state = observation.current
    if state is None:
        return 0.0
    if state.result == seat:
        return 1e9
    if state.result >= 0 and state.result != 2:
        return -1e9
    me = state.players[seat]
    opponent = state.players[1 - seat]
    value = (len(opponent.prize) - len(me.prize)) * 10000.0
    for pokemon in [p for p in (me.active + me.bench) if p]:
        value += pokemon.hp * 1.5 + len(pokemon.energies or []) * 120.0
    for pokemon in [p for p in (opponent.active + opponent.bench) if p]:
        value -= pokemon.hp * 1.5
    if not [p for p in me.active if p]:
        value -= 5000.0  # no Active is a loss on the next check
    value += me.handCount * 30.0
    if me.deckCount < 4:
        value -= (4 - me.deckCount) * 3000.0  # decking out loses outright
    return value


def _advance_forced(search_step, search_id, observation, seat, limit, seen):
    """Walk past selections the engine leaves no choice in. No inference.

    `seen` collects every search id this touches. `search_step` returns a
    SearchState with its own `searchId`, and nothing in `cg/api.py` promises it
    is the one passed in — so every id has to be recorded and released, not just
    the last. Searches run thousands of times per episode against a 12.2 GiB
    cap; leaking the intermediates is how that cap gets hit.
    """
    for _ in range(limit):
        state = observation.current
        if state is None or state.result >= 0:
            break
        select = observation.select
        if select is None or not select.option or state.yourIndex != seat:
            break
        shape = {"n_options": len(select.option),
                 "min_count": select.minCount, "max_count": select.maxCount}
        if not is_forced(shape):
            break
        if select.minCount == select.maxCount == len(select.option):
            picks = list(range(len(select.option)))
        else:
            picks = _legal_default(select)
        result = search_step(search_id, picks)
        observation, search_id = result.observation, result.searchId
        seen.add(search_id)
    return observation


# ─────────────────────────────────────────────────────────────────────────────
# Optional determinized rollout support
#
# Same firing surface and discipline as the guard, different evaluation: instead
# of one engine step scored by material, each candidate is played to the end of
# the game with the net choosing for BOTH seats, inside several sampled worlds.
# Verified against the live engine 2026-08-12: search_step costs ~0.5 ms, the
# returned observation's yourIndex flips to whichever seat must act, the same
# root id can be stepped once per candidate, and search_begin can be called once
# per world on one observation. `current.result` is the winner seat when the
# game ends (2 is a draw), -1 while it runs.
# ─────────────────────────────────────────────────────────────────────────────

_OPP_LISTS = None


def _opponent_lists():
    """Candidate opponent 60s from bcnet/opponent_decks.json, if shipped.

    The file is a list of {"name": str, "cards": [60 ints]} written by
    export_agent.py from decks/*.csv — corpus-mined lists already committed to
    the repo, no card database content. Returns `[(name, cards)]`: the name is
    what selects the archetype's opponent policy below, so it is kept, not
    dropped. Absent file means the fallback belief (our own list) is used, which
    is the guard's original behaviour.
    """
    global _OPP_LISTS
    if _OPP_LISTS is None:
        _OPP_LISTS = []
        try:
            import json as _json
            with open(_data_path(os.path.join("bcnet", "opponent_decks.json"))) as handle:
                data = _json.load(handle)
            _OPP_LISTS = [(d.get("name", ""), d["cards"])
                          for d in data if len(d.get("cards", [])) == 60]
        except Exception:
            _OPP_LISTS = []
    return _OPP_LISTS


# ─────────────────────────────────────────────────────────────────────────────
# Optional archetype-specific opponent policies
#
# The rollout used to run our *own* checkpoint for both seats — self-play. That
# is wrong for a specialist: our net is Marnie's Grimmsnarl ex, and asking it to
# pilot an Alakazam deck in a hypothesised world produces a line the real
# Alakazam pilot would never play, so every candidate is scored against a
# strawman opponent. This layer separates the two policies:
#
#     opponent belief  ->  hypothesised archetype  ->  its own checkpoint
#
# `bcnet/opponents/manifest.json` maps an archetype tag to a checkpoint file:
#
#     {"archetypes": {"grimmsnarl": "grimmsnarl.pt",
#                     "alakazam":   "generalist.pt",
#                     "dragapult":  "generalist.pt"},
#      "default": "generalist.pt"}
#
# A tag is matched as a substring of the belief's deck-list name, so
# "marnies-grimmsnarl-ex" and "dragapult-crushing-hammer" resolve without an
# exact-name table. Where a dedicated specialist has been trained it is named
# directly; where one has not, the entry points at the corpus-wide generalist,
# which pilots every archetype far more faithfully than our specialist does.
# Drop a real `alakazam.pt` in and repoint its entry to upgrade — no code change.
#
# Every checkpoint here must carry the same encoder fingerprint as the shipped
# bcnet/, checked at load exactly as the own model is, and its own decoder flag,
# so a decoder specialist and a non-decoder generalist can serve different seats
# in the same rollout. Anything missing or mismatched degrades to self-play,
# which is the previous behaviour, never an error.
# ─────────────────────────────────────────────────────────────────────────────

_OPP_MANIFEST = None
_OPP_MODELS = {}   # checkpoint filename -> (model, decoder) or None (load failed)


def _opponent_manifest():
    """Read bcnet/opponents/manifest.json once. Empty dict means: self-play."""
    global _OPP_MANIFEST
    if _OPP_MANIFEST is None:
        _OPP_MANIFEST = {}
        try:
            import json as _json
            with open(_data_path(os.path.join("bcnet", "opponents", "manifest.json"))) as handle:
                loaded = _json.load(handle)
            if isinstance(loaded, dict):
                _OPP_MANIFEST = loaded
        except Exception:
            _OPP_MANIFEST = {}
    return _OPP_MANIFEST


def _archetype_file(name):
    """Checkpoint filename for a belief's deck-list name, via the manifest.

    Substring match so deck-list variants ("dragapult-ex",
    "dragapult-crushing-hammer") share one archetype entry; the manifest's
    "default" catches everything else. None means no opponent model applies.
    """
    manifest = _opponent_manifest()
    archetypes = manifest.get("archetypes") or {}
    lowered = (name or "").lower()
    for tag, filename in archetypes.items():
        if tag and tag in lowered:
            return filename
    return manifest.get("default")


def _load_opponent_file(filename):
    """Lazy-load and cache one opponent checkpoint. None on any failure.

    Same construction and fingerprint gate as `_load`, so an opponent net with
    the wrong encoder is refused rather than fed features it never trained on.
    """
    if filename in _OPP_MODELS:
        return _OPP_MODELS[filename]
    record = None
    try:
        if _TABLES is None:
            raise RuntimeError("tables not built")
        checkpoint = torch.load(_data_path(os.path.join("bcnet", "opponents", filename)),
                                map_location="cpu", weights_only=False)
        shipped = _package_fingerprint()
        if shipped != checkpoint["encoder_fingerprint"]:
            raise RuntimeError(
                f"opponent encoder mismatch: bcnet/ hashes to {shipped}, "
                f"{filename} was trained by {checkpoint['encoder_fingerprint']}"
            )
        model = OptionScorer(
            checkpoint["slot_sizes"],
            torch.from_numpy(_TABLES.card_feat),
            torch.from_numpy(_TABLES.attack_feat),
            **checkpoint["model"],
        )
        state = {k: v for k, v in checkpoint["state_dict"].items()
                 if k not in ("card_feat", "attack_feat")}
        # strict=False for the same reason as the own model: an opponent policy
        # trained without a critic has no value_head weights, and it never needs
        # them — the tree's leaf value comes from our own critic.
        loaded = model.load_state_dict(state, strict=False)
        missing = [k for k in getattr(loaded, "missing_keys", [])
                   if not k.startswith("value_head")]
        if [k for k in getattr(loaded, "unexpected_keys", []) if k] or missing:
            raise RuntimeError("opponent checkpoint key mismatch")
        model.eval()
        record = (model, bool(checkpoint.get("decoder", False)),
                  bool(checkpoint.get("deck_tokens", False)))
    except Exception as exc:  # noqa: BLE001 - a bad opponent net degrades to self-play
        print(f"opponent model {filename} disabled: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        record = None
    _OPP_MODELS[filename] = record
    return record


def _opponent_policy(name):
    """(model, decoder, deck_tokens) for a hypothesised archetype, or a triple of
    None to fall the rollout back to self-play."""
    if _MODEL is None:
        return None, None, None
    filename = _archetype_file(name)
    if not filename:
        return None, None, None
    record = _load_opponent_file(filename)
    if record is None:
        return None, None, None
    return record


def _board_ids(player):
    """Every card id visibly owned by one seat: board stacks, tools, energy, discard.

    getattr throughout: the unit tests drive this with stub states, and a stub
    without a zone should count as an empty zone, not a crash into the fallback.
    """
    seen = []
    board = list(getattr(player, "active", None) or []) + list(getattr(player, "bench", None) or [])
    for pokemon in [p for p in board if p]:
        for card in [pokemon] + list(getattr(pokemon, "preEvolution", None) or []) \
                + list(getattr(pokemon, "energyCards", None) or []) \
                + list(getattr(pokemon, "tools", None) or []):
            card_id = getattr(card, "id", None)
            if card_id is not None:
                seen.append(card_id)
    for card in getattr(player, "discard", None) or []:
        card_id = getattr(card, "id", None)
        if card_id is not None:
            seen.append(card_id)
    return seen


def _pool_minus(full, seen):
    """Multiset difference: `full` with one copy removed per element of `seen`."""
    pool = list(full)
    for card in seen:
        try:
            pool.remove(card)
        except ValueError:
            pass
    return pool


def _draw(pool, count, fallback, rng):
    """`count` cards without replacement from pool, topped up from fallback."""
    if count <= 0:
        return []
    rng.shuffle(pool)
    taken = pool[:count]
    del pool[:len(taken)]
    while len(taken) < count:
        taken.append(rng.choice(fallback))
    return taken


def _determinize_pimc(observation, rng):
    """A sampled world plus the archetype it hypothesises for the opponent.

    Our own hidden zones are drawn from our actual list minus every card we can
    see we hold — hand, board, discard — so deck and prize samples are near the
    truth rather than 60 random copies. The opponent's are drawn from the best
    matching shipped candidate list (most revealed ids covered), minus what they
    have revealed; with no shipped lists this degrades to the guard's original
    own-list belief, which is weak but legal.

    Returns `(kwargs, archetype, opp_list)`: `kwargs` unpacks into `search_begin`,
    `archetype` is the matched deck-list name (or None) that selects which
    opponent policy pilots that seat, and `opp_list` is the belief's 60-card list
    that a deck-token opponent policy is conditioned on. The same belief that
    fills the hidden cards now also picks, and conditions, the hand that plays
    them.
    """
    state = observation.current
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]
    my_list = _read_deck()

    my_seen = _board_ids(me) + [c.id for c in (getattr(me, "hand", None) or [])
                                if getattr(c, "id", None) is not None]
    my_pool = _pool_minus(my_list, my_seen)

    opp_seen = _board_ids(opponent)
    best = None
    for name, cards in _opponent_lists():
        covered = len(opp_seen) - len(_pool_minus(list(opp_seen), cards))
        if best is None or covered > best[0]:
            best = (covered, name, cards)
    if best and best[0] > 0:
        archetype, opp_list = best[1], best[2]
    else:
        archetype, opp_list = None, my_list
    opp_pool = _pool_minus(opp_list, opp_seen)

    opponent_active = []
    if opponent.active and opponent.active[0] is None:
        table = getattr(_TABLES, "card_by_id", {}) or {}
        basics = [c for c in opp_pool if getattr(table.get(c), "basic", False)]
        opponent_active = [rng.choice(basics or opp_pool or my_list)]

    return {
        "your_deck": _draw(my_pool, me.deckCount, my_list, rng),
        "your_prize": _draw(my_pool, len(me.prize), my_list, rng),
        "opponent_deck": _draw(opp_pool, opponent.deckCount, opp_list, rng),
        "opponent_prize": _draw(opp_pool, len(opponent.prize), opp_list, rng),
        "opponent_hand": _draw(opp_pool, opponent.handCount, opp_list, rng),
        "opponent_active": opponent_active,
    }, archetype, opp_list


def _policy_pick(observation, model=None, decoder=None, deck_ids=None):
    """A policy's action for one seat's selection inside a rollout.

    The SearchState observation arrives from the acting seat's perspective, so
    the encoder serves whichever policy is handed in: our own net for our seat,
    an archetype-matched opponent net for the opponent's. With no arguments this
    runs the own net — the original self-play behaviour, still the fallback when
    no opponent model is shipped. `deck_ids` conditions a deck-token policy on the
    seat's own list. Forced selections skip inference; any failure falls back to
    the legal default, exactly as the live shell does.
    """
    if model is None:
        model = _MODEL
    if decoder is None:
        decoder = _DECODER
    select = observation.select
    shape = {"n_options": len(select.option),
             "min_count": select.minCount, "max_count": select.maxCount}
    if is_forced(shape):
        if select.minCount == select.maxCount == len(select.option):
            return list(range(len(select.option)))
        return _legal_default(select)
    try:
        # 300 s is a mid-range trained overage value; rollout states carry none.
        scored = _score(observation, 300.0, model, decoder, deck_ids)
        if decoder:
            scorer, options_x = scored
            return _validate(_decode(scorer, options_x, select), select)
        return _validate(_pick_from(scored, select), select)
    except Exception:
        return _legal_default(select)


def _rollout(search_step, search_id, observation, seat, deadline, seen, depth,
             opp_model=None, opp_decoder=None, own_deck_ids=None,
             opp_deck_ids=None, opp_deck_tokens=False):
    """Play one line to the end. Returns P(win)-ish.

    Our seat is piloted by our own net conditioned on our own deck; the opponent
    seat by `opp_model` — the archetype-matched policy for this world — conditioned
    on the belief's deck when that policy trained with deck tokens. When no
    opponent model is supplied both seats fall back to our own net (the original
    self-play). Which seat is acting is read from the SearchState's `yourIndex`,
    which flips to whoever must choose.

    Terminal: 1 win / 0 loss / 0.5 draw from `current.result`. Deadline or
    depth cap: the material evaluation squashed to (0, 1), so a truncated line
    still ranks above a lost one and below a won one.
    """
    for _ in range(depth):
        state = observation.current
        if state is None:
            break
        result = getattr(state, "result", -1)
        if result is not None and result >= 0:
            if result == seat:
                return 1.0
            if result == 2:
                return 0.5
            return 0.0
        select = observation.select
        if select is None or not select.option:
            break
        if _time.perf_counter() > deadline:
            break
        if opp_model is not None and getattr(state, "yourIndex", seat) != seat:
            picks = _policy_pick(observation, opp_model, opp_decoder,
                                 opp_deck_ids if opp_deck_tokens else None)
        else:
            picks = _policy_pick(observation, deck_ids=own_deck_ids)
        step = search_step(search_id, picks)
        observation, search_id = step.observation, step.searchId
        seen.add(search_id)
    return 1.0 / (1.0 + math.exp(-_evaluate(observation, seat) / 8000.0))


def _pimc(obs_dict, observation, probabilities, picked):
    """Architecture D's decision: mean rollout value across worlds per candidate.

    Common random numbers: every candidate is scored inside the same sampled
    worlds, so world luck cancels out of the comparison. A candidate only
    competes if it was scored in every completed world, and the net's own pick
    must be among them — the same fairness rules as the guard.
    """
    cfg = _search_config()
    select = observation.select
    if select.maxCount != 1 or len(select.option) < 2:
        return picked
    if getattr(observation, "search_begin_input", None) is None:
        return picked
    if int(select.type) != cfg["main"]:
        return picked

    ordered = torch.argsort(probabilities, descending=True).tolist()
    top, second = float(probabilities[ordered[0]]), float(probabilities[ordered[1]])
    if second <= 0.0 or top <= 0.0:
        return picked
    if math.log(top) - math.log(second) >= cfg["margin"]:
        return picked
    _SEARCH_STATS["eligible"] += 1

    overage = obs_dict.get("remainingOverageTime")
    remaining = 600.0 if overage is None else float(overage)
    if remaining <= cfg["reserve"]:
        return picked
    budget = min(cfg["budget_cap"], (remaining - cfg["reserve"]) * cfg["budget_fraction"])
    if budget <= 0.0:
        return picked

    search_begin, search_step, search_release, search_end = _SEARCH_API
    started = _time.perf_counter()
    deadline = started + budget
    seat = observation.current.yourIndex
    rng = random.Random(observation.current.turn * 2003 + len(select.option))
    candidates = ordered[: max(2, cfg["candidates"])]
    totals = {}
    # Our own list conditions our seat when this checkpoint trained with deck
    # tokens; None otherwise, which emits no deck tokens.
    own_deck_ids = _read_deck() if _DECK_TOKENS else None
    _SEARCH_STATS["fired"] += 1
    try:
        for _ in range(cfg["worlds"]):
            if _time.perf_counter() > deadline:
                break
            seen = set()
            world_scores = {}
            try:
                world, archetype, opp_list = _determinize_pimc(observation, rng)
                root = search_begin(observation, **world)
                seen.add(root.searchId)
                # The opponent seat is piloted by this world's archetype policy,
                # conditioned on the belief's deck; a triple of None falls the
                # rollout back to self-play.
                opp_model, opp_decoder, opp_deck_tokens = _opponent_policy(archetype)
                if opp_model is not None:
                    _SEARCH_STATS["opp_worlds"] += 1
                for candidate in candidates:
                    if _time.perf_counter() > deadline:
                        break
                    nxt = search_step(root.searchId, [candidate])
                    seen.add(nxt.searchId)
                    world_scores[candidate] = _rollout(
                        search_step, nxt.searchId, nxt.observation, seat,
                        deadline, seen, cfg["rollout_depth"], opp_model, opp_decoder,
                        own_deck_ids, opp_list, opp_deck_tokens)
            except Exception:
                _SEARCH_STATS["errors"] += 1
            finally:
                for sid in seen:
                    try:
                        search_release(sid)
                    except Exception:
                        pass
            # Keep only complete worlds: a world where the deadline cut the
            # candidate loop short would compare rolled-out candidates against
            # absent ones.
            if len(world_scores) == len(candidates):
                for candidate, value in world_scores.items():
                    totals.setdefault(candidate, []).append(value)
    except Exception:
        return picked
    finally:
        try:
            search_end()
        except Exception:
            pass
        _SEARCH_STATS["seconds"] += _time.perf_counter() - started

    scores = {i: sum(v) / len(v) for i, v in totals.items() if v}
    if len(scores) < 2 or picked[0] not in scores:
        return picked
    best = max(scores, key=lambda i: (scores[i], -ordered.index(i)))
    if best != picked[0]:
        _SEARCH_STATS["changed"] += 1
    if cfg["log"]:
        print(f"pimc turn={observation.current.turn} worlds={len(next(iter(totals.values())))} "
              f"net={picked[0]} pimc={best} scores="
              + ",".join(f"{i}:{scores[i]:.3f}" for i in sorted(scores))
              + f" {_time.perf_counter() - started:.3f}s stats={_SEARCH_STATS}",
              file=sys.stderr)
    return [best]


# ─────────────────────────────────────────────────────────────────────────────
# Optional value-backed tree-search support
#
# The PIMC logs said the leaf was the problem: at margin 0.50 the search spent
# 4 s / 8 worlds reaching a depth-200 rollout that then fell to the material
# squash — the value function measured 0.045 against the steel agent — and it
# regressed (930 vs 1002). PUCT keeps the determinization and opponent policies
# but changes two things: simulation budget is allocated by PUCT over the prior's
# top-k instead of split uniformly, and every simulation is a SHORT policy
# rollout truncated by the learned critic V(s), never the material read. Runs
# only when the checkpoint actually trained a critic (`_VALUE`); otherwise the
# caller falls back to the raw policy. guard/pimc are untouched.
# ─────────────────────────────────────────────────────────────────────────────


def _leaf_value(observation, root_seat, own_deck_ids):
    """Critic P(win) for `root_seat`, evaluated only on OUR own to-move board.

    The critic was trained on our seat's boards conditioned on our deck, so that is
    the only distribution it is calibrated for. `_value_rollout` walks to our next
    decision before calling this, so the common case is `acting == root_seat` and
    the critic scores our board with our deck — in distribution. If the leaf is
    terminal/decisionless, or we could not reach our own decision (deadline caught
    us mid opponent turn), fall back to the squashed material read, which is
    seat-consistent. We never score the opponent's board with our critic and negate
    it — that `1 - v`, no-deck path was off-distribution and is gone.
    """
    material = 1.0 / (1.0 + math.exp(-_evaluate(observation, root_seat) / 8000.0))
    state = getattr(observation, "current", None)
    if state is None or observation.select is None or not observation.select.option:
        return material
    if getattr(state, "yourIndex", root_seat) != root_seat:
        return material
    try:
        return _value(observation, _MODEL, own_deck_ids)
    except Exception:
        return material


# Extra plies `_value_rollout` may walk past its minimum depth to reach our own
# next decision, so the critic always scores our board, never the opponent's. A
# hard ceiling on top of the deadline; search_step is ~0.5 ms, so 64 is ~30 ms.
_LEAF_WALK_CAP = 64


def _value_rollout(search_step, search_id, observation, root_seat, deadline, seen, depth,
                   opp_model, opp_decoder, own_deck_ids, opp_deck_ids, opp_deck_tokens):
    """Play policy moves, then evaluate the leaf with the critic at OUR own node.

    Returns P(win) for `root_seat`. A terminal reached along the way scores the
    exact 1 / 0.5 / 0. Otherwise the walk continues until it is our turn again —
    `depth` is the *minimum* plies to walk, not a fixed leaf — so `_leaf_value`
    scores our own to-move board with our deck, never the opponent's board via a
    negated no-deck read. The steps needed to reach our node past `depth` are
    bounded by `_LEAF_WALK_CAP` and the deadline, keeping the cost bounded. Our
    seat is our net on our deck; the opponent seat is its archetype policy on the
    belief deck, or — with no opponent model — our net with NO deck tokens (never
    our own list piloting the opponent's hand).
    """
    for step_i in range(depth + _LEAF_WALK_CAP):
        state = observation.current
        if state is None:
            break
        result = getattr(state, "result", -1)
        if result is not None and result >= 0:
            if result == root_seat:
                return 1.0
            if result == 2:
                return 0.5
            return 0.0
        select = observation.select
        if select is None or not select.option:
            break
        acting = getattr(state, "yourIndex", root_seat)
        # Stop once the minimum depth is spent AND it is our turn, so the leaf is
        # always our own decision (terminals are handled above).
        if step_i >= depth and acting == root_seat:
            break
        if _time.perf_counter() > deadline:
            break
        if opp_model is not None and acting != root_seat:
            picks = _policy_pick(observation, opp_model, opp_decoder,
                                 opp_deck_ids if opp_deck_tokens else None)
        else:
            picks = _policy_pick(observation,
                                 deck_ids=(own_deck_ids if acting == root_seat else None))
        step = search_step(search_id, picks)
        observation, search_id = step.observation, step.searchId
        seen.add(search_id)
    return _leaf_value(observation, root_seat, own_deck_ids)


def _puct(obs_dict, observation, probabilities, picked):
    """PUCT over the prior's top-k with a learned-critic leaf. Never raises."""
    cfg = _search_config()
    select = observation.select
    if select.maxCount != 1 or len(select.option) < 2:
        return picked
    if getattr(observation, "search_begin_input", None) is None:
        return picked
    if int(select.type) != cfg["main"]:
        return picked
    if _MODEL is None or not _VALUE:
        # PUCT is only as good as its leaf, and without a trained critic the leaf
        # is noise. Defer to the policy rather than repeat the 0.50-config regress.
        return picked

    ordered = torch.argsort(probabilities, descending=True).tolist()
    top, second = float(probabilities[ordered[0]]), float(probabilities[ordered[1]])
    if second <= 0.0 or top <= 0.0:
        return picked
    if math.log(top) - math.log(second) >= cfg["margin"]:
        return picked
    _SEARCH_STATS["eligible"] += 1

    overage = obs_dict.get("remainingOverageTime")
    remaining = 600.0 if overage is None else float(overage)
    if remaining <= cfg["reserve"]:
        return picked
    budget = min(cfg["budget_cap"], (remaining - cfg["reserve"]) * cfg["budget_fraction"])
    if budget <= 0.0:
        return picked

    search_begin, search_step, search_release, search_end = _SEARCH_API
    started = _time.perf_counter()
    deadline = started + budget
    root_seat = observation.current.yourIndex
    own_deck_ids = _read_deck() if _DECK_TOKENS else None
    rng = random.Random(observation.current.turn * 7919 + len(select.option))
    candidates = ordered[: max(2, cfg["candidates"])]
    prior_raw = {c: max(float(probabilities[c]), 1e-9) for c in candidates}
    norm = sum(prior_raw.values()) or 1.0
    prior = {c: p / norm for c, p in prior_raw.items()}
    visits = {c: 0 for c in candidates}   # aggregated across worlds
    value_sum = {c: 0.0 for c in candidates}
    total_sims = 0
    _SEARCH_STATS["fired"] += 1
    try:
        for _ in range(cfg["worlds"]):
            if _time.perf_counter() > deadline:
                break
            seen = set()
            try:
                world, archetype, opp_list = _determinize_pimc(observation, rng)
                root = search_begin(observation, **world)
                seen.add(root.searchId)
                opp_model, opp_decoder, opp_deck_tokens = _opponent_policy(archetype)
                if opp_model is not None:
                    _SEARCH_STATS["opp_worlds"] += 1
                # Per-world PUCT statistics, folded into the global totals after.
                n = {c: 0 for c in candidates}
                w = {c: 0.0 for c in candidates}
                for _ in range(cfg["sims"]):
                    if _time.perf_counter() > deadline:
                        break
                    total_n = sum(n.values())
                    # PUCT: unvisited candidates score from an optimistic 0.5 prior
                    # so each is tried before any is exploited.
                    def _puct_score(c):
                        q = w[c] / n[c] if n[c] else 0.5
                        u = cfg["c_puct"] * prior[c] * math.sqrt(total_n + 1) / (1 + n[c])
                        return q + u
                    choice = max(candidates, key=_puct_score)
                    nxt = search_step(root.searchId, [choice])
                    seen.add(nxt.searchId)
                    v = _value_rollout(
                        search_step, nxt.searchId, nxt.observation, root_seat,
                        deadline, seen, cfg["leaf_depth"], opp_model, opp_decoder,
                        own_deck_ids, opp_list, opp_deck_tokens)
                    n[choice] += 1
                    w[choice] += v
                    total_sims += 1
                for c in candidates:
                    visits[c] += n[c]
                    value_sum[c] += w[c]
            except Exception:
                _SEARCH_STATS["errors"] += 1
            finally:
                for sid in seen:
                    try:
                        search_release(sid)
                    except Exception:
                        pass
    except Exception:
        return picked
    finally:
        try:
            search_end()
        except Exception:
            pass
        _SEARCH_STATS["seconds"] += _time.perf_counter() - started

    explored = {c: visits[c] for c in candidates if visits[c] > 0}
    # Not enough evidence, or the net's own pick was never tried: keep the policy.
    if total_sims < cfg["min_sims"] or len(explored) < 2 or picked[0] not in explored:
        return picked
    # Rank by mean critic value, not visit count. At tens of simulations the visit
    # count has not converged onto the value the way AlphaZero's hundreds do, so Q
    # is the better estimator here; visits only break ties.
    q = {c: value_sum[c] / visits[c] for c in explored}
    best = max(explored, key=lambda c: (q[c], visits[c], -ordered.index(c)))
    # We only fired on a policy near-tie, so overruling the net's own pick needs a
    # decisive value edge, not a hair inside the noise. Below the margin, keep it.
    if best != picked[0] and q[best] - q[picked[0]] < cfg["overrule_margin"]:
        best = picked[0]
    if best != picked[0]:
        _SEARCH_STATS["changed"] += 1
    if cfg["log"]:
        print(f"puct turn={observation.current.turn} sims={total_sims} "
              f"net={picked[0]} puct={best} "
              + ",".join(f"{c}:{visits[c]}/{q[c]:.2f}" for c in sorted(explored))
              + f" {_time.perf_counter() - started:.3f}s stats={_SEARCH_STATS}",
              file=sys.stderr)
    return [best]


def _guard(obs_dict, observation, probabilities, picked):
    """Return a possibly-different single pick. Never raises."""
    cfg = _search_config()
    if not cfg.get("enabled") or not picked:
        return picked
    select = observation.select
    if select.maxCount != 1 or len(select.option) < 2:
        return picked
    if getattr(observation, "search_begin_input", None) is None:
        return picked
    if int(select.type) != cfg["main"]:
        return picked

    ordered = torch.argsort(probabilities, descending=True).tolist()
    top, second = float(probabilities[ordered[0]]), float(probabilities[ordered[1]])
    if second <= 0.0 or top <= 0.0:
        return picked
    if math.log(top) - math.log(second) >= cfg["margin"]:
        return picked  # the net is confident; do not second-guess it
    _SEARCH_STATS["eligible"] += 1

    overage = obs_dict.get("remainingOverageTime")
    remaining = 600.0 if overage is None else float(overage)
    if remaining <= cfg["reserve"]:
        return picked
    budget = min(cfg["budget_cap"], (remaining - cfg["reserve"]) * cfg["budget_fraction"])
    if budget <= 0.0:
        return picked

    search_begin, search_step, search_release, search_end = _SEARCH_API
    started = _time.perf_counter()
    deadline = started + budget
    seat = observation.current.yourIndex
    rng = random.Random(observation.current.turn * 1009 + len(select.option))
    candidates = ordered[: max(2, cfg["candidates"])]
    totals = {}
    _SEARCH_STATS["fired"] += 1
    try:
        # One hypothesis per sample, every candidate scored against it — common
        # random numbers. Sampling per candidate instead would compare each
        # option against a *different* hidden-information draw, and at two
        # samples that noise is larger than the differences being measured; it
        # can flip the ranking on its own. It also cuts search_begin calls from
        # candidates x samples to samples, which matters because search_begin
        # consumes `Observation.search_begin_input` and the SDK is not in this
        # container to confirm how often that can be spent.
        for _ in range(cfg["samples"]):
            if _time.perf_counter() > deadline:
                break
            seen = set()
            try:
                root = search_begin(observation, **_determinize(observation, rng))
                seen.add(root.searchId)
                for candidate in candidates:
                    if _time.perf_counter() > deadline:
                        break
                    nxt = search_step(root.searchId, [candidate])
                    seen.add(nxt.searchId)
                    forked = _advance_forced(search_step, nxt.searchId, nxt.observation,
                                             seat, cfg["follow_forced"], seen)
                    totals.setdefault(candidate, []).append(_evaluate(forked, seat))
            except Exception:
                _SEARCH_STATS["errors"] += 1
            finally:
                for sid in seen:
                    try:
                        search_release(sid)
                    except Exception:
                        pass
    except Exception:
        return picked
    finally:
        try:
            search_end()
        except Exception:
            pass
        _SEARCH_STATS["seconds"] += _time.perf_counter() - started

    # Only compare candidates that were scored the same number of times, or the
    # mean of two samples competes against the mean of one.
    depth = max((len(v) for v in totals.values()), default=0)
    scores = {i: sum(v) / len(v) for i, v in totals.items() if len(v) == depth}
    # The net's own pick must have been evaluated. Without this, a deadline hit
    # on the first candidate would leave the guard choosing between two options
    # it looked at while the option it is overruling was never scored.
    if len(scores) < 2 or picked[0] not in scores:
        return picked
    best = max(scores, key=lambda i: (scores[i], -ordered.index(i)))
    if best != picked[0]:
        _SEARCH_STATS["changed"] += 1
    if cfg["log"]:
        print(f"guard turn={observation.current.turn} gap={math.log(top) - math.log(second):.3f} "
              f"net={picked[0]} guard={best} scores="
              + ",".join(f"{i}:{scores[i]:.0f}" for i in sorted(scores))
              + f" {_time.perf_counter() - started:.3f}s stats={_SEARCH_STATS}",
              file=sys.stderr)
    return [best]


def agent(obs_dict):
    try:
        observation = to_observation_class(obs_dict)
    except Exception:
        # Malformed observation. `[0]` would be INVALID against an empty option
        # list, which is the one shape where the only legal answer is `[]`.
        if obs_dict.get("select") is None:
            return _read_deck()
        return [0] if (obs_dict.get("select") or {}).get("option") else []

    if observation.select is None or observation.current is None:
        return _read_deck()

    select = observation.select
    if not select.option:
        return []

    row_shape = {"n_options": len(select.option),
                 "min_count": select.minCount, "max_count": select.maxCount}
    if is_forced(row_shape):
        if select.minCount == select.maxCount == len(select.option):
            return list(range(len(select.option)))
        return _legal_default(select)

    _load()
    if _MODEL is None:
        return _legal_default(select)

    try:
        # Every training row carries a real overage, 16 s to 599 s, so the
        # "absent" bucket has no trained embedding behind it. If a harness ever
        # omits the key, a full bank is the honest in-distribution stand-in.
        overage = obs_dict.get("remainingOverageTime")
        # Condition on our own deck when this checkpoint trained with deck tokens.
        deck_ids = _read_deck() if _DECK_TOKENS else None
        scored = _score(observation, 600.0 if overage is None else overage,
                        deck_ids=deck_ids)
        if _DECODER:
            scorer, options_x = scored
            probabilities = scorer(None)          # for the near-tie search guard
            picked = _decode(scorer, options_x, select)
        else:
            probabilities = scored
            picked = _pick_from(probabilities, select)
    except Exception:
        return _legal_default(select)

    try:
        # Outside the try above on purpose: a guard failure must cost the net's
        # pick, not fall all the way back to the first legal option.
        cfg = _search_config()
        if cfg.get("enabled") and picked:
            mode = cfg.get("mode")
            if mode == "puct":
                picked = _puct(obs_dict, observation, probabilities, picked)
            elif mode == "pimc":
                picked = _pimc(obs_dict, observation, probabilities, picked)
            else:
                picked = _guard(obs_dict, observation, probabilities, picked)
    except Exception as exc:  # noqa: BLE001 - the guard is optional, the game is not
        print(f"search guard error: {type(exc).__name__}: {exc}", file=sys.stderr)

    return _validate(picked, select)
