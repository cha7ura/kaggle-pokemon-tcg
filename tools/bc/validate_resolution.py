#!/usr/bin/env python3
"""Check option-to-card resolution against the corpus's own log stream.

    python3 tools/bc/validate_resolution.py --day data/replays/date=2026-07-27

`encoder.option_card_id` and `encoder.inplay_card_id` turn an option's
`(area, index)` and `(inPlayArea, inPlayIndex)` into card ids by indexing the
zone arrays. Nothing in the corpus states that those arrays are in the engine's
order — index bounds hold either way, so a reordered discard pile or hand would
resolve to the wrong card, silently, on the most common actions in the game.

The log block settles it. `obs.logs` covers everything since the acting seat's
previous selection, so the consequence of decision *k* appears in the logs of
decision *k+1*:

* choosing an `OptionType.PLAY` option produces `LogType.PLAY` carrying `cardId`
* choosing an `OptionType.ATTACH` option produces `LogType.ATTACH` carrying both
  `cardId` (what was attached) and `cardIdTarget` (what it went on)
* choosing an `OptionType.EVOLVE` option produces `LogType.EVOLVE`, same pair

So the resolution is checkable against the engine's own account of what
happened, on hundreds of thousands of real decisions per day.

The consequence does not always land in the *very next* selection: playing a
search Item opens a sub-selection whose logs hold only `LogType.SHUFFLE`, and
the `PLAY` entry arrives a step or two later. `--window` is how many following
selections may carry it; 3 is enough that the residual is noise rather than
structure.

Run it after any change to the resolution rules, and after any corpus
regeneration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bc.encoder import inplay_card_id, option_card_id  # noqa: E402

OPTION_PLAY, OPTION_ATTACH, OPTION_EVOLVE = 7, 8, 9
LOG_PLAY, LOG_ATTACH, LOG_EVOLVE = 10, 11, 12

COLUMNS = [
    "episode_id", "seat", "action_step", "action",
    "option_types", "option_areas", "option_indices", "option_players",
    "option_inplay_area", "option_inplay_index",
    "my_hand_ids", "select_deck_ids", "looking_ids",
    "me_discard_ids", "opp_discard_ids", "me_bench_ids", "opp_bench_ids",
    "me_active_id", "opp_active_id", "stadium_id",
    "log_types", "log_card_ids", "log_target_card_ids",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--day", type=Path, required=True, help="a data/replays/date=* directory")
    parser.add_argument("--window", type=int, default=3,
                        help="how many following selections may carry the consequence")
    parser.add_argument("--self-test-swap", action="store_true",
                        help="deliberately swap the attached card with its target before "
                             "checking. A working gate FAILS under this; the pre-2026-08-05 "
                             "version still reported 99.99%%, which is why the flag exists")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    table = pq.read_table(args.day / "steps.parquet", columns=COLUMNS)
    data = table.to_pydict()
    order = np.lexsort((
        np.asarray(data["action_step"]),
        np.asarray(data["seat"]),
        np.asarray(data["episode_id"]),
    ))

    counts = {name: [0, 0] for name in ("play", "attach_card", "attach_target",
                                        "evolve_card", "evolve_target")}
    previous_key = None
    # [name, log_type, expected card id, which log field must carry it,
    #  selections still allowed to carry it]
    #
    # `slot` is the fix for a check that was blind. It used to ask whether the
    # expected id appeared *anywhere* in the (cardId, cardIdTarget) pair, so an
    # encoder that swapped what was attached with what it was attached to still
    # scored 99.99%. The two fields answer different questions and are compared
    # separately now.
    pending: list[list] = []
    CARD, TARGET = 0, 1

    for position in order:
        key = (data["episode_id"][position], data["seat"][position])
        if previous_key != key:
            # The seat's run ended; anything still pending never showed up.
            for entry in pending:
                counts[entry[0]][0] += 1
            pending = []
        else:
            logs = list(zip(data["log_types"][position],
                            data["log_card_ids"][position],
                            data["log_target_card_ids"][position]))
            still: list[list] = []
            for entry in pending:
                name, log_type, expected, slot, left = entry
                seen = [(card, target) for kind, card, target in logs if kind == log_type]
                if any(pair[slot] == expected for pair in seen):
                    counts[name][0] += 1
                    counts[name][1] += 1
                elif left > 1:
                    entry[4] = left - 1
                    still.append(entry)
                else:
                    counts[name][0] += 1
            pending = still
        previous_key = key

        action = data["action"][position]
        if len(action) != 1:
            continue  # one action, one consequence — multi-pick muddies the join
        index = action[0]
        row = {name: data[name][position] for name in COLUMNS}
        option_type = data["option_types"][position][index]
        area = data["option_areas"][position][index]
        option_index = data["option_indices"][position][index]
        player = data["option_players"][position][index]
        inplay_area = data["option_inplay_area"][position][index]
        inplay_index = data["option_inplay_index"][position][index]
        seat = data["seat"][position]

        card = option_card_id(row, area, option_index, player, seat, option_type)
        target = inplay_card_id(row, inplay_area, inplay_index, seat)
        if args.self_test_swap:
            card, target = target, card
        if option_type == OPTION_PLAY and card is not None:
            pending.append(["play", LOG_PLAY, card, CARD, args.window])
        elif option_type == OPTION_ATTACH:
            if card is not None:
                pending.append(["attach_card", LOG_ATTACH, card, CARD, args.window])
            if target is not None:
                pending.append(["attach_target", LOG_ATTACH, target, TARGET, args.window])
        elif option_type == OPTION_EVOLVE:
            if card is not None:
                pending.append(["evolve_card", LOG_EVOLVE, card, CARD, args.window])
            if target is not None:
                pending.append(["evolve_target", LOG_EVOLVE, target, TARGET, args.window])

    # ATTACH and EVOLVE log their consequence immediately and completely, so
    # anything below 99.9% there is a resolution bug. PLAY does not: about 1% of
    # plays never produce a LogType.PLAY entry in any following window — widening
    # --window from 3 to 12 moves the rate only from 98.88% to 98.99%, so the
    # residual is a property of the log stream, not of the index rule. A wrong
    # index rule would land near 1/hand-size, not near 99%.
    thresholds = {"play": 0.98}
    print(f"{args.day.name}\n")
    failed = False
    for name, (checked, matched) in counts.items():
        rate = matched / checked if checked else 0.0
        floor = thresholds.get(name, 0.999)
        verdict = "OK  " if checked and rate >= floor else "FAIL"
        if checked and rate < floor:
            failed = True
        print(f"{verdict} {name:<15} checked {checked:>8,}  confirmed by the log stream "
              f"{matched:>8,}  ({rate:.4%})")
    if not any(checked for checked, _ in counts.values()):
        print("FAIL nothing was checked; the option-type constants are probably stale")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
