"""Memmapped reader over the tensor cache, plus the batch collate.

The cache is ragged, so a batch is padded to the widest row it happens to
contain rather than to the corpus maximum. Mean options per row is 7 and the
maximum is 130; padding globally would multiply the batch by ~18 for nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import torch.utils.data

from .encoder import OPT_PICKED_SLOT, PICK_TAKEN

ARRAYS = ("globals", "tokens", "tokens_off", "options", "options_off",
          "targets", "targets_off", "meta")
# Written since 2026-08-03; a cache built before then has neither. Absent
# weights mean every row counts 1.0, absent logs mean the model has no log
# stream — both are the pre-existing behaviour, so old caches still train.
OPTIONAL = ("weights", "logs", "logs_off", "values")


class Shard:
    def __init__(self, path: Path):
        self.path = path
        self.data = {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
        for name in OPTIONAL:
            file = path / f"{name}.npy"
            if file.exists():
                self.data[name] = np.load(file, mmap_mode="r")
        self.n = self.data["globals"].shape[0]
        self.has_logs = "logs" in self.data and "logs_off" in self.data

    def row(self, i: int) -> dict:
        d = self.data
        t0, t1 = int(d["tokens_off"][i]), int(d["tokens_off"][i + 1])
        o0, o1 = int(d["options_off"][i]), int(d["options_off"][i + 1])
        a0, a1 = int(d["targets_off"][i]), int(d["targets_off"][i + 1])
        row = {
            "globals": np.asarray(d["globals"][i]),
            "tokens": np.asarray(d["tokens"][t0:t1]),
            "options": np.asarray(d["options"][o0:o1]),
            "target": np.asarray(d["targets"][a0:a1]),
            "meta": np.asarray(d["meta"][i]),
            "weight": float(d["weights"][i]) if "weights" in d else 1.0,
        }
        if "values" in d:
            row["value_target"] = float(d["values"][i])
        if self.has_logs:
            l0, l1 = int(d["logs_off"][i]), int(d["logs_off"][i + 1])
            row["logs"] = np.asarray(d["logs"][l0:l1])
        return row


class TensorDataset(torch.utils.data.Dataset):
    """One row per decision, drawn from a set of date partitions."""

    def __init__(self, root: Path, dates: list[str]):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.shards = []
        for date in dates:
            path = self.root / f"date={date}"
            if not path.is_dir():
                raise SystemExit(f"missing partition {path}")
            shard = Shard(path)
            if shard.n:
                self.shards.append(shard)
        if not self.shards:
            raise SystemExit(f"no rows under {root} for {dates}")
        counts = np.array([s.n for s in self.shards], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(counts)])
        self.total = int(counts.sum())

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, i: int) -> dict:
        shard_index = int(np.searchsorted(self.starts, i, side="right") - 1)
        return self.shards[shard_index].row(i - int(self.starts[shard_index]))


def expand_decode_steps(batch: list[dict], rng: np.random.Generator) -> list[dict]:
    """Turn each multi-pick decision into the sequence of steps that produced it.

    A row whose `maxCount` is 1 is left exactly as it was — one step, no stop
    action, the path 95.4% of the corpus takes. Anything wider becomes:

      step j (j = 0 .. k-1)  j options already marked taken, target = uniform
                             over the k-j the demonstrator had not taken yet
      step k                 target = stop, emitted only when stopping was legal
                             (k >= minCount) and decoding would not have halted
                             on its own (k < maxCount)

    The target at each step is uniform over the *remaining* chosen options
    rather than the next one in `action` order, because that order is not
    recoverable. Measured over three days: 76.7% of unforced multi-pick actions
    are strictly ascending against 34.6% expected if the demonstrator's order
    had survived, so many agents submit a sorted list and the rest do not, and
    nothing in the row says which. Uniform is correct under both.

    The prefix that is marked taken is a fresh random subset each epoch, which
    is the same reason: no order to reproduce, so cover the orders.
    """
    out: list[dict] = []
    for item in batch:
        meta = item["meta"]
        max_count = int(meta[5]) if meta.shape[0] > 5 else 1
        min_count = int(meta[4]) if meta.shape[0] > 5 else 1
        chosen = np.unique(item["target"].astype(np.int64))
        if max_count <= 1 or chosen.size == 0:
            out.append(item)
            continue
        order = rng.permutation(chosen)
        for j in range(order.size):
            step = dict(item)
            if j:
                options = item["options"].copy()
                options[order[:j], OPT_PICKED_SLOT] = PICK_TAKEN
                step["options"] = options
            step["target"] = order[j:]
            # Stopping short of what the demonstrator took is a negative example,
            # and it only exists as a choice once minCount is satisfied. The
            # `max(min_count, 1)` matters: minCount 0 rows exist, but the agent
            # never offers stop at zero picks — no demonstrator ever declined and
            # the guard shell forbids an empty action — so offering it here would
            # train a mask the agent never presents.
            step["stop_legal"] = j >= max(min_count, 1)
            out.append(step)
        if order.size >= max(min_count, 1) and order.size < max_count:
            step = dict(item)
            options = item["options"].copy()
            options[order, OPT_PICKED_SLOT] = PICK_TAKEN
            step["options"] = options
            step["target"] = np.empty(0, dtype=np.int64)   # mass goes on stop
            step["stop_legal"] = True
            step["stop_target"] = True
            out.append(step)
    return out


class DecodeCollate:
    """Picklable collate for `--decoder`.

    Windows multiprocessing is **spawn**: the DataLoader pickles `collate_fn` to
    send it to each worker, and a closure defined inside `main()` dies there with
    `Can't get local object 'main.<locals>.collate_fn'`. A module-level class
    pickles fine. This matters because a Linux fork
    never reaches the code path.

    The generator is built lazily inside the worker, so it is absent at pickle
    time and each worker draws its own pick orders from a seed derived from the
    run seed and the worker id — reproducible, and not the same permutations in
    every worker.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._rng = None

    def __call__(self, batch: list[dict]) -> dict:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            self._rng = np.random.default_rng(self.seed + 1000 * (0 if info is None else info.id + 1))
        return collate(expand_decode_steps(batch, self._rng))


def collate(batch: list[dict]) -> dict:
    size = len(batch)
    max_tokens = max(item["tokens"].shape[0] for item in batch)
    with_logs = "logs" in batch[0]
    max_logs = max((item["logs"].shape[0] for item in batch), default=0) if with_logs else 0
    max_options = max(item["options"].shape[0] for item in batch)
    n_token = batch[0]["tokens"].shape[1]
    n_option = batch[0]["options"].shape[1]

    globals_x = np.zeros((size, batch[0]["globals"].shape[0]), dtype=np.int16)
    tokens_x = np.zeros((size, max_tokens, n_token), dtype=np.int16)
    options_x = np.zeros((size, max_options, n_option), dtype=np.int16)
    token_mask = np.zeros((size, max_tokens), dtype=bool)
    option_mask = np.zeros((size, max_options), dtype=bool)
    # One extra column for the stop action. It stays 0 unless a step targets
    # stop, so slicing it off reproduces the pre-decoder tensor exactly.
    target = np.zeros((size, max_options + 1), dtype=np.float32)
    meta = np.zeros((size, batch[0]["meta"].shape[0]), dtype=np.int16)
    single = np.zeros(size, dtype=bool)
    weight = np.ones(size, dtype=np.float32)
    # At least one log slot, so an all-empty batch still gives attention a key.
    logs_x = np.zeros((size, max(1, max_logs), batch[0]["logs"].shape[1]), dtype=np.int16) if with_logs else None
    log_mask = np.zeros((size, max(1, max_logs)), dtype=bool) if with_logs else None

    stop_mask = np.zeros(size, dtype=bool)
    with_value = "value_target" in batch[0]
    value_target = np.zeros(size, dtype=np.float32) if with_value else None

    for i, item in enumerate(batch):
        n_t, n_o = item["tokens"].shape[0], item["options"].shape[0]
        stop_mask[i] = bool(item.get("stop_legal", False))
        if with_value:
            value_target[i] = item.get("value_target", 0.0)
        globals_x[i] = item["globals"]
        tokens_x[i, :n_t] = item["tokens"]
        options_x[i, :n_o] = item["options"]
        token_mask[i, :n_t] = True
        option_mask[i, :n_o] = True
        # np.unique, not the raw list: a duplicated index would give that option
        # double mass and quietly change what the loss is asking for. The corpus
        # has none, but a regenerated corpus is not this corpus.
        chosen = np.unique(item["target"].astype(np.int64))
        if item.get("stop_target"):
            target[i, max_options] = 1.0
        else:
            target[i, chosen] = 1.0 / len(chosen)
        meta[i] = item["meta"]
        single[i] = len(chosen) == 1
        weight[i] = item.get("weight", 1.0)
        if with_logs:
            n_l = item["logs"].shape[0]
            logs_x[i, :n_l] = item["logs"]
            log_mask[i, :n_l] = True

    out = {
        "globals": torch.from_numpy(globals_x),
        "tokens": torch.from_numpy(tokens_x),
        "token_mask": torch.from_numpy(token_mask),
        "options": torch.from_numpy(options_x),
        "option_mask": torch.from_numpy(option_mask),
        "target": torch.from_numpy(target),
        "meta": torch.from_numpy(meta),
        "single": torch.from_numpy(single),
        "stop_mask": torch.from_numpy(stop_mask),
        "weight": torch.from_numpy(weight),
    }
    if with_logs:
        out["logs"] = torch.from_numpy(logs_x)
        out["log_mask"] = torch.from_numpy(log_mask)
    if with_value:
        out["value_target"] = torch.from_numpy(value_target)
    return out
