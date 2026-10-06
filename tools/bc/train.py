#!/usr/bin/env python3
"""Train the option scorer on the cached tensors.

    python3 tools/bc/train.py --data data/tensors/grimm-v1 --sdk <sdk-path> \
        --out runs/grimm-v1 --epochs 10

Reports accuracy overall and per `SelectType`, with MAIN broken out, because
MAIN is 56.5% of decisions and where games are decided. Forced rows were already
dropped at build time; if they are ever kept, they are excluded from the
reported numbers here too — counting them inflates top-1 by roughly 8 points and
would make gate G0 meaningless.

Accuracy is a diagnostic, never the shipping criterion. Checkpoints are chosen
by head-to-head games: near-identical
accuracies have differed by 30-80 ladder points in the field's own reports.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bc.build_dataset import encoder_fingerprint  # noqa: E402
from tools.bc.cards import load_tables  # noqa: E402
from tools.bc.dataset import DecodeCollate, TensorDataset, collate  # noqa: E402
from tools.bc.model import OptionScorer, masked_cross_entropy  # noqa: E402

SELECT_TYPE_NAMES = {
    0: "MAIN", 1: "CARD", 2: "ATTACHED_CARD", 3: "CARD_OR_ATTACHED_CARD",
    4: "ENERGY", 5: "SKILL", 6: "ATTACK", 7: "EVOLVE", 8: "COUNT",
    9: "YES_NO", 10: "SPECIAL_CONDITION",
}


def partition_dates(root: Path) -> list[str]:
    return sorted(p.name.replace("date=", "") for p in Path(root).glob("date=*") if p.is_dir())


def split_dates(root: Path, holdout: list[str]) -> tuple[list[str], list[str]]:
    """Temporal split. A random row split would leak: adjacent decisions in one
    game are near-duplicates, and the meta drifts week to week."""
    every = partition_dates(root)
    unknown = [d for d in holdout if d not in every]
    if unknown:
        raise SystemExit(f"holdout dates not in the cache: {unknown}")
    train = [d for d in every if d not in holdout]
    if not train:
        raise SystemExit("holdout consumed every partition")
    return train, list(holdout)


class Accumulator:
    """Top-1 counts, split by SelectType and by single- vs multi-pick.

    For a multi-pick row "correct" means the top-scoring option is one the
    demonstrator chose. That is a weaker claim than reproducing the whole set,
    so the two are never added together in the report.
    """

    def __init__(self):
        self.hits: dict[int, int] = {}
        self.total: dict[int, int] = {}
        self.single_hits = self.single_total = 0
        self.multi_hits = self.multi_total = 0
        self.loss_sum = 0.0
        self.loss_batches = 0
        self.value_loss_sum = 0.0
        self.value_batches = 0

    def update(self, logits, batch, target=None):
        if target is None:
            target = batch["target"][:, :logits.shape[1]]
        top1 = logits.argmax(dim=-1)
        chosen = target.gather(1, top1.unsqueeze(1)).squeeze(1) > 0
        select_type = batch["meta"][:, 0]
        single = batch["single"]
        for st in select_type.unique().tolist():
            m = select_type == st
            self.total[st] = self.total.get(st, 0) + int(m.sum())
            self.hits[st] = self.hits.get(st, 0) + int((chosen & m).sum())
        self.single_total += int(single.sum())
        self.single_hits += int((chosen & single).sum())
        self.multi_total += int((~single).sum())
        self.multi_hits += int((chosen & ~single).sum())

    def report(self) -> dict:
        total = sum(self.total.values())
        hits = sum(self.hits.values())
        return {
            "rows": total,
            "top1": hits / max(1, total),
            "main_rows": self.total.get(0, 0),
            "main_top1": self.hits.get(0, 0) / max(1, self.total.get(0, 0)),
            "single_top1": self.single_hits / max(1, self.single_total),
            "multi_top1_in_set": self.multi_hits / max(1, self.multi_total),
            "multi_rows": self.multi_total,
            "loss": self.loss_sum / max(1, self.loss_batches),
            "value_loss": self.value_loss_sum / self.value_batches if self.value_batches else None,
            "per_select_type": {
                SELECT_TYPE_NAMES.get(st, str(st)): {
                    "rows": self.total[st],
                    "top1": self.hits[st] / max(1, self.total[st]),
                }
                for st in sorted(self.total)
            },
        }


def run_epoch(model, loader, device, optimizer=None, scheduler=None, smoothing=0.05,
              amp=False, max_batches=0, log_every=200, decoder=False,
              value=False, value_coef=1.0):
    train = optimizer is not None
    model.train(train)
    stats = Accumulator()
    started = time.perf_counter()
    for step, batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        # Without the decoder the stop column is dead weight on every row, so it
        # is sliced off and the tensors are exactly what they were before it
        # existed. With it, the mask gains the stop column the model appends.
        stop_mask = batch.get("stop_mask") if decoder else None
        if stop_mask is None:
            target = batch["target"][:, :-1]
            loss_mask = batch["option_mask"]
        else:
            target = batch["target"]
            loss_mask = torch.cat([batch["option_mask"], stop_mask.unsqueeze(1)], dim=1)
        want_value = value and batch.get("value_target") is not None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(batch["globals"], batch["tokens"], batch["token_mask"],
                        batch["options"], batch["option_mask"],
                        batch.get("logs"), batch.get("log_mask"),
                        stop_mask=stop_mask, return_value=want_value)
            logits, value_pred = out if want_value else (out, None)
            loss = masked_cross_entropy(logits.float(), target,
                                        loss_mask, smoothing,
                                        batch.get("weight"))
            if want_value:
                # Plain BCE, deliberately NOT weighted by `weight`: that factor
                # downweights losing seats, and the critic needs the z=0 rows at
                # full strength or it learns to predict winning everywhere.
                value_loss = torch.nn.functional.binary_cross_entropy(
                    value_pred.float(), batch["value_target"].float())
                loss = loss + value_coef * value_loss
                stats.value_loss_sum += float(value_loss.detach())
                stats.value_batches += 1
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        with torch.no_grad():
            stats.update(logits.float().detach(), batch, target)
        stats.loss_sum += float(loss.detach())
        stats.loss_batches += 1
        if log_every and step % log_every == 0:
            rate = (step + 1) * logits.shape[0] / max(1e-6, time.perf_counter() - started)
            print(f"    step {step:>6}  loss {stats.loss_sum / stats.loss_batches:.4f}"
                  f"  {rate:,.0f} rows/s", flush=True)
    return stats.report()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sdk", type=str, default="")
    parser.add_argument("--holdout", type=str, default="",
                        help="comma-separated dates; default is the two latest partitions")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--smoothing", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=None,
                        help="force page-locked host buffers on (default: on for CUDA)")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false",
                        help="the first thing to turn off when host RAM runs out. Pinned "
                             "memory cannot be swapped or reclaimed, and there is one buffer "
                             "per in-flight batch")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="batches held per worker. 1 halves the collated batches resident "
                             "at any moment; only applies with --workers > 0")
    parser.add_argument("--decoder", action="store_true",
                        help="train the autoregressive multi-pick decoder: expand every "
                             "maxCount>1 decision into its pick sequence and learn a stop "
                             "action. Rows with maxCount 1 are untouched, which is 95.4%% "
                             "of the corpus.")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--value-coef", type=float, default=1.0,
                        help="weight on the critic BCE loss, added to the policy loss. Only "
                             "used when the cache was built with --value")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="cap batches per epoch; for smoke tests, not for a real run")
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-from", type=Path, default=None,
                        help="start from this checkpoint's weights instead of random init. "
                             "For fine-tuning a corpus-wide pretrain onto one deck: pretrain "
                             "with --deck-card -1, then fine-tune on the deck cache with a "
                             "lower --lr. The architecture flags must match the checkpoint")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.sdk:
        sys.path.insert(0, args.sdk)
    tables = load_tables()

    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest["encoder_fingerprint"] != encoder_fingerprint():
        raise SystemExit(
            "tensor cache was built by a different encoder "
            f"({manifest['encoder_fingerprint']} != {encoder_fingerprint()}). "
            "Rebuild with tools/bc/build_dataset.py."
        )

    every = partition_dates(args.data)
    holdout = [d for d in args.holdout.split(",") if d] or every[-2:]
    train_dates, val_dates = split_dates(args.data, holdout)
    value = bool(manifest.get("value", False))
    print(f"filter {manifest['filter']}  logs {manifest.get('with_logs', False)}"
          f"  value {value}")
    print(f"train partitions {len(train_dates)}  ({train_dates[0]} .. {train_dates[-1]})")
    print(f"holdout          {val_dates}")

    train_set = TensorDataset(args.data, train_dates)
    val_set = TensorDataset(args.data, val_dates)
    print(f"train rows {len(train_set):,}   holdout rows {len(val_set):,}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Host RAM, not VRAM, is what runs out on a large cache. Three knobs, all
    # exposed because the corpus-wide build makes the difference between a run
    # and an OOM:
    #
    #  * pin_memory allocates PAGE-LOCKED host memory the OS cannot swap or
    #    reclaim, one buffer per in-flight batch. It buys a faster host->device
    #    copy and is the first thing to drop under pressure.
    #  * prefetch_factor is batches held per worker; the default 2 means
    #    workers * 2 collated batches resident at all times.
    #  * persistent_workers keeps them alive across epochs. On Windows, workers
    #    are spawned rather than forked, so without this every epoch pays a
    #    fresh torch import per worker.
    #
    # The dataset itself is memmapped, so the cache is page cache rather than
    # heap — but `shuffle=True` touches it randomly, which pulls essentially the
    # whole file resident. `--since` on the build is the real lever there.
    pin = device.type == "cuda" if args.pin_memory is None else args.pin_memory
    # Module-level and picklable: Windows spawns its DataLoader workers.
    collate_fn = DecodeCollate(args.seed) if args.decoder else collate
    common = dict(batch_size=args.batch, collate_fn=collate_fn, num_workers=args.workers,
                  pin_memory=pin, drop_last=False)
    if args.workers > 0:
        common["persistent_workers"] = True
        common["prefetch_factor"] = args.prefetch_factor
    print(f"loader: workers {args.workers}  pin_memory {pin}  "
          f"prefetch {args.prefetch_factor if args.workers else '-'}  batch {args.batch}")
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **common)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **common)

    model = OptionScorer(
        manifest["slot_sizes"],
        torch.from_numpy(tables.card_feat),
        torch.from_numpy(tables.attack_feat),
        dim=args.dim, layers=args.layers, heads=args.heads, ff=args.ff, dropout=args.dropout,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"device {device}  parameters {params:,}")

    if args.init_from is not None:
        # Every one of these mismatches would otherwise load *something* and
        # train to a plausible-looking number against features the weights were
        # never fitted to. Each is a hard error, not a warning.
        prior = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if prior["encoder_fingerprint"] != manifest["encoder_fingerprint"]:
            raise SystemExit(
                f"--init-from was trained by encoder {prior['encoder_fingerprint']} and this "
                f"cache by {manifest['encoder_fingerprint']}. Rebuild one of them."
            )
        if prior["slot_sizes"] != manifest["slot_sizes"]:
            # The usual cause: one side built with --with-logs and the other
            # without. The log stream adds a slot group, so the shapes differ.
            raise SystemExit(
                "--init-from has different slot sizes from this cache. The most likely cause "
                "is --with-logs on one build and not the other; both must match."
            )
        shape = {"dim": args.dim, "layers": args.layers, "heads": args.heads,
                 "ff": args.ff, "dropout": args.dropout}
        if {k: v for k, v in prior["model"].items() if k != "dropout"} != \
                {k: v for k, v in shape.items() if k != "dropout"}:
            raise SystemExit(
                f"--init-from is {prior['model']} and this run is {shape}. Architecture flags "
                f"must match the checkpoint (dropout may differ)."
            )
        state = {k: v for k, v in prior["state_dict"].items()
                 if k not in ("card_feat", "attack_feat")}
        model.load_state_dict(state)
        print(f"initialised from {args.init_from}  "
              f"(pretrain filter {prior['filter']}, epoch {prior['epoch']}, "
              f"holdout MAIN {prior['val']['main_top1']:.4f})")
        if args.lr > 1e-4:
            print(f"  NOTE: --lr {args.lr:g} on a fine-tune. The pretrained weights are the "
                  f"asset; 3e-5 to 5e-5 is the usual range, and the default 3e-4 will move "
                  f"them a long way in the first epoch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = args.max_train_batches or math.ceil(len(train_set) / args.batch)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.min_lr
    )

    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nepoch {epoch}/{args.epochs}")
        train_report = run_epoch(model, train_loader, device, optimizer, scheduler,
                                 args.smoothing, args.amp, args.max_train_batches,
                                 decoder=args.decoder, value=value, value_coef=args.value_coef)
        with torch.no_grad():
            val_report = run_epoch(model, val_loader, device, None, None,
                                   args.smoothing, args.amp, args.max_eval_batches,
                                   log_every=0, decoder=args.decoder, value=value,
                                   value_coef=args.value_coef)
        if value and train_report.get("value_loss") is not None:
            print(f"  value loss train {train_report['value_loss']:.4f}"
                  f"  val {val_report['value_loss']:.4f}")
        print(f"  train loss {train_report['loss']:.4f}  top1 {train_report['top1']:.4f}"
              f"  MAIN {train_report['main_top1']:.4f}")
        print(f"  val   loss {val_report['loss']:.4f}  top1 {val_report['top1']:.4f}"
              f"  MAIN {val_report['main_top1']:.4f}  (MAIN rows {val_report['main_rows']:,})")
        # The decoder's own gate metric: it ships only
        # if multi-pick accuracy improves. It was in the report dict and nowhere on
        # screen, which made the gate a checkpoint-archaeology exercise.
        print(f"  val   multi-pick top1-in-set {val_report['multi_top1_in_set']:.4f}"
              f"  (multi rows {val_report['multi_rows']:,})"
              f"   single {val_report['single_top1']:.4f}")
        for name, entry in val_report["per_select_type"].items():
            print(f"      {name:<22} rows {entry['rows']:>8,}  top1 {entry['top1']:.4f}")
        history.append({"epoch": epoch, "train": train_report, "val": val_report})
        (args.out / "history.json").write_text(json.dumps(history, indent=1))

        # Selected on MAIN accuracy, which is a diagnostic. The real gate is tools/ab.py.
        if val_report["main_top1"] > best:
            best = val_report["main_top1"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "slot_sizes": manifest["slot_sizes"],
                    "encoder_fingerprint": manifest["encoder_fingerprint"],
                    "model": {"dim": args.dim, "layers": args.layers, "heads": args.heads,
                              "ff": args.ff, "dropout": args.dropout},
                    # The agent decodes autoregressively only if the checkpoint
                    # was trained that way. A stop head that never saw a gradient
                    # would still emit a logit, and it would be noise.
                    "decoder": bool(args.decoder),
                    # The agent feeds its deck.csv as deck tokens only if this
                    # cache carried them; handing deck tokens to a model that
                    # never trained on them shifts every row off distribution.
                    "deck_tokens": bool(manifest.get("deck_tokens", False)),
                    # The agent runs value-backed PUCT only if the critic head was
                    # actually trained; a value head that never saw a target emits
                    # a number that is noise, exactly what search must not trust.
                    "value": value,
                    "filter": manifest["filter"],
                    "val": val_report,
                    "epoch": epoch,
                    # So an exported agent states what it was fine-tuned from.
                    # Without it a fine-tuned checkpoint is indistinguishable
                    # from one trained on the deck cache alone.
                    "init_from": (str(args.init_from) if args.init_from else None),
                    "pretrain_filter": (prior["filter"] if args.init_from else None),
                },
                args.out / "best.pt",
            )
            print(f"  saved {args.out / 'best.pt'}  (MAIN {best:.4f})")

    gate = "PASS" if best >= 0.60 else "FAIL"
    print(f"\nG0 gate (MAIN top-1 >= 0.60 on the temporal holdout, forced rows excluded): "
          f"{best:.4f} {gate}")
    return 0 if best >= 0.60 else 1


if __name__ == "__main__":
    raise SystemExit(main())
