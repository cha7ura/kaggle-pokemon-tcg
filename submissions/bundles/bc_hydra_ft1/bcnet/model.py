"""The option scorer.

The engine's action is a list of indices into `obs.select.option`, and the
option set changes shape every decision, so a fixed action head cannot express
the policy. The network is a pointer: it encodes the state, then scores each
option against it.

Options attend to the state but **not to each other**, except through one
masked-mean summary token. That keeps scoring permutation-equivariant and, more
importantly, length-agnostic: the corpus tops out at 130 options but nothing in
the architecture assumes a bound, so an unusually wide select at inference is
scored rather than truncated.

Sizes come from the dataset manifest, never from constants here. A checkpoint
carries the manifest's slot sizes and refuses to load against different ones —
an embedding table that silently reshapes is a wrong model that still runs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import (
    LOG_ATTACK_SLOT, LOG_CARD_SLOTS, OPT_ATTACK_SLOT, OPT_CARD_SLOT,
    OPT_INPLAY_CARD_SLOT, TOKEN_CARD_SLOTS,
)

MASKED_LOGIT = -1e4  # finite, and far enough below any real logit to be inert


class SlotEmbedding(nn.Module):
    """Sum of one embedding per categorical slot, with optional shared tables.

    Summing rather than concatenating keeps the parameter count in the card
    tables where it belongs instead of in a wide projection.
    """

    def __init__(self, sizes: list[int], dim: int, shared: dict[int, nn.Embedding] | None = None):
        super().__init__()
        shared = shared or {}
        self.slots = nn.ModuleList()
        for i, size in enumerate(sizes):
            self.slots.append(shared[i] if i in shared else nn.Embedding(size, dim, padding_idx=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature tensors are int16 on disk to keep the cache small; embeddings
        # need int64, so the cast happens here rather than in three callers.
        out = None
        for i, table in enumerate(self.slots):
            piece = table(x[..., i].long())
            out = piece if out is None else out + piece
        return out


class OptionScorer(nn.Module):
    def __init__(
        self,
        slot_sizes: dict[str, list[int]],
        card_feat: torch.Tensor,
        attack_feat: torch.Tensor,
        dim: int = 128,
        layers: int = 3,
        heads: int = 4,
        ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.slot_sizes = slot_sizes
        self.dim = dim

        # Static card / attack stats. **Non-persistent** buffers: they move with
        # the module to a device but stay out of `state_dict`, so no checkpoint
        # carries them.
        #
        # They are derived from the competition card database — type, energy
        # type, weakness, resistance, HP, retreat cost, stage and rarity flags,
        # attack damage and cost, per card id. `AGENTS.md` forbids committing
        # that database, and a numeric reconstruction of its mechanical content
        # is the same thing by another name. `cards.py` says these tables are
        # rebuilt from the SDK wherever they are needed and never committed;
        # persistent buffers silently defeated that, because the shipped weights
        # are committed.
        #
        # Nothing needs them from the checkpoint: both the trainer and the agent
        # construct the model with tables from `load_tables()`, which reads the
        # SDK that is already on the path in the one case and bundled in the
        # archive in the other.
        self.register_buffer("card_feat", card_feat, persistent=False)
        self.register_buffer("attack_feat", attack_feat, persistent=False)
        self.card_proj = nn.Linear(card_feat.shape[1], dim, bias=False)
        self.attack_proj = nn.Linear(attack_feat.shape[1], dim, bias=False)

        card_table = nn.Embedding(slot_sizes["tokens"][TOKEN_CARD_SLOTS[0]], dim, padding_idx=0)
        self.token_emb = SlotEmbedding(
            slot_sizes["tokens"], dim, shared={i: card_table for i in TOKEN_CARD_SLOTS}
        )
        self.option_emb = SlotEmbedding(
            slot_sizes["options"], dim,
            shared={OPT_CARD_SLOT: card_table, OPT_INPLAY_CARD_SLOT: card_table},
        )
        # Present only when the cache was built with --with-logs. Sized from the
        # manifest, so a checkpoint trained without logs cannot be loaded into a
        # model that expects them, or the reverse.
        self.log_emb = (
            SlotEmbedding(slot_sizes["logs"], dim,
                          shared={i: card_table for i in LOG_CARD_SLOTS})
            if "logs" in slot_sizes else None
        )

        global_dim = 16
        self.global_emb = nn.ModuleList(
            [nn.Embedding(size, global_dim) for size in slot_sizes["globals"]]
        )
        self.global_proj = nn.Linear(global_dim * len(slot_sizes["globals"]), dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.state_norm = nn.LayerNorm(dim)

        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.option_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 4, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1)
        )
        # The stop action of the autoregressive decoder: a learned pseudo-option
        # scored by the same head, so "take another" and "that is enough" are
        # ranked on one scale instead of against a threshold. It is only ever in
        # play when the caller passes `stop_mask` — a single-pick decision never
        # sees it, and 95.4% of the corpus is single-pick.
        self.stop_option = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.stop_option, std=0.02)

        # The critic (DeNA CEDEC 2026): a scalar board evaluation = P(win) for the
        # acting seat, read off the state summary token. Always constructed so a
        # checkpoint's shape is fixed; only trained when the cache carries value
        # targets and only used at inference when the checkpoint records
        # `value=True`. A model wrapping a value-less checkpoint loads these with
        # `strict=False` and never calls them. This is the leaf evaluation the
        # PIMC logs showed the material squash was too crude to be: search that
        # went deep into that leaf regressed (0.50-margin config, 930 vs 1002).
        self.value_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1)
        )

    # -- pieces -------------------------------------------------------------

    def encode_logs(self, logs_x):
        logs = self.log_emb(logs_x)
        logs = logs + self.card_proj(self.card_feat[logs_x[..., LOG_CARD_SLOTS[0]].long()])
        return logs + self.attack_proj(self.attack_feat[logs_x[..., LOG_ATTACK_SLOT].long()])

    def encode_state(self, globals_x, tokens_x, token_mask, logs_x=None, log_mask=None):
        pieces = [table(globals_x[:, i].long()) for i, table in enumerate(self.global_emb)]
        global_token = self.global_proj(torch.cat(pieces, dim=-1)).unsqueeze(1)

        tokens = self.token_emb(tokens_x)
        tokens = tokens + self.card_proj(self.card_feat[tokens_x[..., TOKEN_CARD_SLOTS[0]].long()])

        pieces_seq = [global_token, tokens]
        pieces_pad = [torch.zeros(token_mask.shape[0], 1, dtype=torch.bool, device=token_mask.device),
                      ~token_mask]
        if self.log_emb is not None and logs_x is not None:
            pieces_seq.append(self.encode_logs(logs_x))
            pieces_pad.append(~log_mask)
        sequence = torch.cat(pieces_seq, dim=1)
        pad = torch.cat(pieces_pad, dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=pad)
        return self.state_norm(encoded), pad

    def encode_options(self, options_x):
        options = self.option_emb(options_x)
        options = options + self.card_proj(self.card_feat[options_x[..., OPT_CARD_SLOT].long()])
        options = options + self.card_proj(self.card_feat[options_x[..., OPT_INPLAY_CARD_SLOT].long()])
        options = options + self.attack_proj(self.attack_feat[options_x[..., OPT_ATTACK_SLOT].long()])
        return self.option_norm(options)

    def state_value(self, globals_x, tokens_x, token_mask, logs_x=None, log_mask=None):
        """P(win) for the acting seat on this board, in (0, 1).

        Options are not needed — the critic scores the board, not a choice — so
        this is cheaper than a policy pass and reuses the same state encoder.
        """
        encoded, _ = self.encode_state(globals_x, tokens_x, token_mask, logs_x, log_mask)
        return torch.sigmoid(self.value_head(encoded[:, 0]).squeeze(-1))

    def forward(self, globals_x, tokens_x, token_mask, options_x, option_mask,
                logs_x=None, log_mask=None, stop_mask=None, return_value=False):
        """Returns logits [B, O]; padded option slots get MASKED_LOGIT.

        With `stop_mask` — a bool per row saying whether stopping is legal right
        now — the returned logits are [B, O+1] and the last column is the stop
        action. Rows whose stop is illegal get MASKED_LOGIT there, so the same
        tensor covers "must take another" and "may stop".

        With `return_value` the state summary is scored by the critic and the
        method returns `(logits, value)`; the value reuses the state encoding the
        policy already computed, so the critic is nearly free at training time
        (DeNA Technique ②). The default single-tensor return keeps every existing
        caller — the agent's policy path, the decoder — untouched.
        """
        encoded, state_pad = self.encode_state(globals_x, tokens_x, token_mask, logs_x, log_mask)
        options = self.encode_options(options_x)

        # The pooled summary is over the *real* options only. Letting the stop
        # token into its own summary would make the decision depend on itself.
        weights = option_mask.unsqueeze(-1).to(options.dtype)
        pooled = (options * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True).clamp(min=1.0)

        if stop_mask is not None:
            options = torch.cat([options, self.stop_option.expand(options.shape[0], 1, -1)], dim=1)
            option_mask = torch.cat([option_mask, stop_mask.unsqueeze(1)], dim=1)

        attended, _ = self.cross(options, encoded, encoded, key_padding_mask=state_pad)

        context = encoded[:, :1].expand(-1, options.shape[1], -1)
        pooled = pooled.expand(-1, options.shape[1], -1)
        logits = self.head(torch.cat([options, attended, context, pooled], dim=-1)).squeeze(-1)
        # A large finite penalty, not -inf. `-inf` makes softmax and argmax
        # correct but poisons the loss: 0 * -inf is NaN, and torch.where hides
        # it in the forward pass while still handing NaN to the backward pass.
        logits = logits.masked_fill(~option_mask, MASKED_LOGIT)
        if return_value:
            value = torch.sigmoid(self.value_head(encoded[:, 0]).squeeze(-1))
            return logits, value
        return logits


def masked_cross_entropy(logits, target, option_mask, smoothing: float = 0.05, weight=None):
    """Cross-entropy against a distribution, not a class index.

    A multi-pick action is encoded as uniform mass over the chosen options, so
    the same loss covers both cases and the 4.6% of rows that pick more than one
    option are trained rather than dropped. Label smoothing is spread over the
    *valid* options only — spreading it over padding would push mass onto slots
    that can never be chosen.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    valid = option_mask.sum(-1, keepdim=True).clamp(min=1).to(logits.dtype)
    smoothed = (1.0 - smoothing) * target + smoothing * option_mask.to(logits.dtype) / valid
    # `smoothed` is exactly 0 on padded slots and `log_probs` there is finite
    # (see MASKED_LOGIT), so the product is 0 with a clean gradient.
    per_row = -(smoothed * log_probs).sum(-1)
    if weight is None:
        return per_row.mean()
    # Normalised by the weight sum, not the row count, so lowering the weight on
    # losing seats does not quietly lower the effective learning rate too.
    weight = weight.to(per_row.dtype)
    return (per_row * weight).sum() / weight.sum().clamp(min=1e-6)
