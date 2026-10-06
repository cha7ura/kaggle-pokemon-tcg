#!/usr/bin/env python3
"""Assemble a submittable agent directory from a trained checkpoint.

    python3 tools/bc/export_agent.py --checkpoint runs/grimm-v1/best.pt \
        --agent agents/dragapult-finalv2 --deck decks/dragapult-ex.csv

Writes a **complete** agent directory: `main.py`, `deck.csv`, and `bcnet/`. Point
`--agent` at a new path and it is submittable without any manual step.

The agent cannot `import tools.bc` — the archive contains only what is under the
agent directory — so the encoder is *copied* into `agents/<name>/bcnet/` rather
than duplicated by hand. That copy is the whole point of this script: a
hand-maintained second encoder is the cheapest way to ship a model whose
features do not match the ones it was trained on, and the failure is silent.

`main.py` is copied the same way, from `tools/bc/agent_main.py`. It used to be
hand-written per agent directory, which meant exporting to a *new* directory
produced an agent with no entry point at all, and meant every guard-shell fix had
to be applied by hand to each copy. Edit `tools/bc/agent_main.py`; never edit an
exported `main.py`.

`fingerprint.json` records which encoder built the checkpoint. Both `main.py` at
load and `tools/check_submission.py` before shipping **re-hash the copied
`bcnet/` files** and compare the result to that record, so editing an exported
encoder is caught instead of silently feeding the model skewed features. Do not
"optimise" either check into a comparison of two recorded numbers — that is what
it used to be, and it agreed by construction.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bc.build_dataset import encoder_fingerprint  # noqa: E402

COPIED = ("__init__.py", "cards.py", "encoder.py", "rows.py", "model.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--search", action="store_true",
                        help="enable the Plan C near-tie guard in the exported agent. Off by "
                             "default: enable only for a separately measured runtime variant; "
                             "the kill criterion at lower bound <= 0.5 vs plain BC over 400 games")
    parser.add_argument("--search-margin", type=float, default=1.0,
                        help="fire only when the top-2 log-probability gap is under this")
    parser.add_argument("--search-candidates", type=int, default=3)
    parser.add_argument("--search-samples", type=int, default=2)
    parser.add_argument("--search-budget-cap", type=float, default=2.0,
                        help="seconds per decision, hard ceiling")
    parser.add_argument("--search-mode", choices=("guard", "pimc", "puct"), default="guard",
                        help="guard = Plan C near-tie tiebreak; pimc = Architecture D "
                             "determinized rollouts; puct = value-backed PUCT (needs a "
                             "critic checkpoint, built with --value)")
    parser.add_argument("--search-worlds", type=int, default=6,
                        help="pimc/puct: sampled hidden-information worlds per decision")
    parser.add_argument("--search-rollout-depth", type=int, default=240,
                        help="pimc only: max engine selections per rollout line")
    parser.add_argument("--search-sims", type=int, default=24,
                        help="puct only: PUCT simulations per world")
    parser.add_argument("--search-leaf-depth", type=int, default=8,
                        help="puct only: policy moves before the critic evaluates a leaf")
    parser.add_argument("--search-c-puct", type=float, default=1.4,
                        help="puct only: exploration constant")
    parser.add_argument("--search-overrule-margin", type=float, default=0.03,
                        help="puct only: overrule the policy pick only when the best "
                             "candidate's mean critic value beats it by at least this many "
                             "P(win) points. 0.0 = always take argmax-Q (upper-bound setting)")
    parser.add_argument("--opponent-decks", type=Path, default=None,
                        help="directory of 60-card deck lists (decks/) bundled as the pimc "
                             "determinizer's opponent belief; omit to fall back to own-list")
    parser.add_argument("--opponent-model", action="append", default=[], metavar="TAG=CHECKPOINT",
                        help="archetype tag mapped to an opponent-policy checkpoint, e.g. "
                             "grimmsnarl=runs/grimm/best.pt. Repeatable. In a pimc rollout the "
                             "opponent seat is piloted by this net when the belief's deck-list "
                             "name contains TAG, instead of by our own specialist. Every "
                             "checkpoint must share the exported encoder's fingerprint.")
    parser.add_argument("--opponent-default", type=Path, default=None, metavar="CHECKPOINT",
                        help="opponent-policy checkpoint for any archetype without its own "
                             "--opponent-model (the manifest 'default'); typically the "
                             "corpus-wide generalist, which pilots every archetype")
    parser.add_argument("--search-log", action="store_true",
                        help="print one stderr line per guarded decision")
    args = parser.parse_args()

    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["encoder_fingerprint"] != encoder_fingerprint():
        raise SystemExit(
            "checkpoint was trained by a different encoder "
            f"({checkpoint['encoder_fingerprint']} != {encoder_fingerprint()}); retrain or "
            "check out the matching source"
        )

    deck = [int(line) for line in args.deck.read_text().split() if line.strip()]
    if len(deck) != 60:
        raise SystemExit(f"{args.deck} has {len(deck)} card ids, expected 60")

    package = args.agent / "bcnet"
    package.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    for name in COPIED:
        shutil.copy2(source / name, package / name)
    # Never ship the card-derived static tables, whatever the checkpoint carries.
    # A run trained before they became non-persistent buffers still has them in
    # its state_dict; they are rebuilt from the SDK at construction, so dropping
    # them is inert; the exported package rebuilds licensed tables at runtime.
    stripped = {k: v for k, v in checkpoint["state_dict"].items()
                if k not in ("card_feat", "attack_feat")}
    if len(stripped) != len(checkpoint["state_dict"]):
        print(f"  stripped card-derived tables from the checkpoint "
              f"({len(checkpoint['state_dict']) - len(stripped)} keys)")
    torch.save({**checkpoint, "state_dict": stripped}, package / "bc_model.pt")
    shutil.copy2(source / "agent_main.py", args.agent / "main.py")
    (args.agent / "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n")
    (package / "fingerprint.json").write_text(json.dumps({
        "encoder_fingerprint": checkpoint["encoder_fingerprint"],
        "model": checkpoint["model"],
        "filter": checkpoint["filter"],
        "val": checkpoint["val"],
        "epoch": checkpoint["epoch"],
    }, indent=1))
    # Written every time, enabled or not, so an agent directory always states
    # which policy it ships rather than leaving it to whether a file exists.
    (package / "search.json").write_text(json.dumps({
        "enabled": bool(args.search),
        "mode": args.search_mode,
        "margin": args.search_margin,
        "candidates": args.search_candidates,
        "samples": args.search_samples,
        "budget_fraction": 0.02 if args.search_mode == "guard" else 0.03,
        "budget_cap": args.search_budget_cap,
        "reserve": 120.0,
        "follow_forced": 8,
        "worlds": args.search_worlds,
        "rollout_depth": args.search_rollout_depth,
        "c_puct": args.search_c_puct,
        "sims": args.search_sims,
        "leaf_depth": args.search_leaf_depth,
        "min_sims": 8,
        "overrule_margin": args.search_overrule_margin,
        "log": bool(args.search_log),
    }, indent=1))
    if args.search and args.search_mode == "puct" and not checkpoint.get("value"):
        print("  WARNING: --search-mode puct but this checkpoint has no critic "
              "(value=False). The agent will fall back to the raw policy until you "
              "export a checkpoint trained with build_dataset --value.")

    if args.opponent_decks is not None:
        lists = []
        for path in sorted(args.opponent_decks.glob("*.csv")):
            cards = [int(x) for x in path.read_text().split() if x.strip()]
            if len(cards) == 60:
                lists.append({"name": path.stem, "cards": cards})
        (package / "opponent_decks.json").write_text(json.dumps(lists))
        print(f"  opponent belief     {len(lists)} candidate lists from {args.opponent_decks}")

    # Archetype-specific opponent policies. Each is stripped of the card-derived
    # tables and fingerprint-checked exactly like the own model, then dropped
    # under bcnet/opponents/ with a manifest the rollout reads. Checkpoints that
    # point at the same source file are written once and shared, so routing
    # three archetypes at one generalist costs one copy, not three.
    opp_specs = []
    for item in args.opponent_model:
        tag, sep, spec = item.partition("=")
        if not sep or not tag.strip() or not spec.strip():
            raise SystemExit(f"--opponent-model expects TAG=CHECKPOINT, got {item!r}")
        opp_specs.append((tag.strip(), Path(spec.strip())))

    if opp_specs or args.opponent_default is not None:
        opp_dir = package / "opponents"
        opp_dir.mkdir(parents=True, exist_ok=True)
        by_source: dict[str, str] = {}   # resolved source path -> written filename
        provenance: dict[str, dict] = {}

        def _place(checkpoint_path: Path, preferred: str) -> str:
            key = str(checkpoint_path.resolve())
            if key in by_source:
                return by_source[key]
            oc = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if oc["encoder_fingerprint"] != encoder_fingerprint():
                raise SystemExit(
                    f"opponent checkpoint {checkpoint_path} was trained by a different "
                    f"encoder ({oc['encoder_fingerprint']} != {encoder_fingerprint()})"
                )
            filename = preferred
            stripped = {k: v for k, v in oc["state_dict"].items()
                        if k not in ("card_feat", "attack_feat")}
            torch.save({**oc, "state_dict": stripped}, opp_dir / filename)
            by_source[key] = filename
            provenance[filename] = {
                "source": checkpoint_path.name,
                "deck_card": oc.get("filter", {}).get("deck_card"),
                "decoder": bool(oc.get("decoder", False)),
                "main_top1": oc.get("val", {}).get("main_top1"),
            }
            return filename

        default_file = None
        if args.opponent_default is not None:
            default_file = _place(args.opponent_default, "generalist.pt")

        archetypes = {}
        for tag, ckpt_path in opp_specs:
            archetypes[tag] = _place(ckpt_path, f"{tag}.pt")

        manifest = {"archetypes": archetypes}
        if default_file:
            manifest["default"] = default_file
        manifest["provenance"] = provenance
        (opp_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"  opponent policies   {len(provenance)} checkpoint(s), "
              f"{len(archetypes)} archetype(s) mapped"
              + (f" + default {default_file}" if default_file else ""))
        for tag, filename in archetypes.items():
            print(f"                        {tag:<12} -> {filename}")

    print(f"exported to {args.agent}")
    print(f"  near-tie guard      {'ENABLED' if args.search else 'off'}"
          + (f" (margin {args.search_margin:g}, {args.search_candidates} candidates x "
             f"{args.search_samples} samples, <= {args.search_budget_cap:g}s/decision)"
             if args.search else ""))
    print(f"  encoder fingerprint {checkpoint['encoder_fingerprint']}")
    print(f"  holdout MAIN top-1  {checkpoint['val']['main_top1']:.4f} "
          f"over {checkpoint['val']['main_rows']:,} rows")
    print("\nNext, in order:")
    print("  python3 tools/bc/selftest.py --sdk <sdk> --data <tensor-cache>")
    print("  python3 tools/bc/validate_resolution.py --day data/replays/date=2026-07-27")
    print(f"  PYTHONPATH=<sdk> python3 tools/ab.py {args.agent}/main.py "
          "--ref agents/bc-grimmsnarl-final --games 400")
    print(f"  python3 tools/build_submission.py {args.agent} --sdk <sdk>")
    print(f"  PYTHONPATH= python3 tools/smoke_archive.py {args.agent}/submission.tar.gz --games 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
