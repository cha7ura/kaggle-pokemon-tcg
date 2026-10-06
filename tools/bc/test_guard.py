#!/usr/bin/env python3
"""Exercise the Plan C near-tie guard without the SDK.

    python3 tools/bc/test_guard.py

The guard only fires inside a live engine, so the one thing that could not be
checked before shipping it was whether it fires *at all* and whether it lets go
when it should. This stubs `cg.api` with a fake search API and a fake board,
loads `tools/bc/agent_main.py` the way `kaggle_environments` does — no
`__file__`, foreign cwd — and drives the guard directly.

What it asserts, in order:

1. With no `search.json`, the guard is inert and returns the net's pick.
2. With `enabled: false`, same.
3. With `enabled: true` and a wide net margin, the guard declines (confident net).
4. With `enabled: true` and a near tie, the guard fires and prefers the option
   that does not lose a Pokemon, even though the net ranked it second.
5. Every `search_begin` is matched by a `search_release`, and `search_end` runs.
6. A search API that raises on every call costs the net's pick and nothing else.
7. The time budget is respected: with reserve above the remaining bank, no
   search happens.

None of this proves the guard helps. Runtime impact still requires a controlled gate
separately: lower bound vs plain BC <= 0.5 over 400 games and it is dropped.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# ── the fake engine ──────────────────────────────────────────────────────────

class Mon:
    def __init__(self, hp, energies=0):
        self.hp = hp
        self.energies = [0] * energies


class Player:
    def __init__(self, prize=2, deck=20, hand=5, active_hp=100, bench=()):
        self.prize = [None] * prize
        self.deckCount = deck
        self.handCount = hand
        self.active = [Mon(active_hp, 2)] if active_hp else [None]
        self.bench = [Mon(hp) for hp in bench]


class State:
    def __init__(self, your_index=0, turn=5, result=-1, players=None):
        self.yourIndex = your_index
        self.turn = turn
        self.result = result
        self.players = players or [Player(), Player()]


class Select:
    def __init__(self, n_options=3, select_type=0, min_count=1, max_count=1):
        self.option = [types.SimpleNamespace(type=0) for _ in range(n_options)]
        self.type = select_type
        self.context = 0
        self.minCount = min_count
        self.maxCount = max_count


class Obs:
    def __init__(self, select=None, current=None, search_input="x"):
        self.select = select if select is not None else Select()
        self.current = current if current is not None else State()
        self.search_begin_input = search_input
        self.logs = []


class FakeEngine:
    """Records calls so the test can assert on release/end discipline.

    `outcomes[i]` is the state reached by playing option `i`. Option 1 keeps our
    Active alive; option 0 kills it. A guard that works prefers 1.
    """

    def __init__(self, raise_on=None, single_use_begin=False):
        self.begun = []
        self.released = []
        self.ended = 0
        self.raise_on = raise_on or set()
        self._next = 1
        self.begin_calls = 0
        self.step_calls = 0
        self.hypotheses = []
        self.single_use_begin = single_use_begin

    def _state_for(self, option):
        if option == 1:
            return State(players=[Player(prize=1, active_hp=120), Player(prize=2)])
        return State(players=[Player(prize=2, active_hp=0), Player(prize=2)])

    def search_begin(self, observation, **kwargs):
        if "begin" in self.raise_on:
            raise RuntimeError("begin blew up")
        self.begin_calls += 1
        if self.single_use_begin and self.begin_calls > 1:
            # search_begin consumes Observation.search_begin_input. If that is once per
            # observation, this is what the engine does on the second call.
            raise ValueError("search_begin_input already consumed")
        self.hypotheses.append(tuple(sorted(kwargs["opponent_hand"])))
        for name in ("your_deck", "your_prize", "opponent_deck", "opponent_prize",
                     "opponent_hand", "opponent_active"):
            if name not in kwargs:
                raise AssertionError(f"search_begin missing {name}")
        counts = {
            "your_deck": observation.current.players[observation.current.yourIndex].deckCount,
            "opponent_deck": observation.current.players[1 - observation.current.yourIndex].deckCount,
            "opponent_hand": observation.current.players[1 - observation.current.yourIndex].handCount,
        }
        for name, need in counts.items():
            if len(kwargs[name]) < need:
                raise AssertionError(f"{name} has {len(kwargs[name])}, engine needs >= {need}")
        sid = self._next
        self._next += 1
        self.begun.append(sid)
        return types.SimpleNamespace(searchId=sid, observation=observation)

    def search_step(self, search_id, picks):
        if "step" in self.raise_on:
            raise RuntimeError("step blew up")
        self.step_calls += 1
        if search_id not in self.begun or search_id in self.released:
            raise ValueError("unknown or released search id")
        state = self._state_for(picks[0])
        # `cg/api.py` does not promise search_step returns the id it was given,
        # and archaludon-challenger's rollout reassigns from the returned state,
        # so a fresh id is the case that must not leak. Mint one every step.
        sid = self._next
        self._next += 1
        self.begun.append(sid)
        # No further selection for us: the guard should stop advancing here.
        return types.SimpleNamespace(searchId=sid, observation=Obs(select=None, current=state))

    def search_release(self, search_id):
        self.released.append(search_id)

    def search_end(self):
        self.ended += 1


def install_fake_cg(engine, select_main=0):
    cg = types.ModuleType("cg")
    api = types.ModuleType("cg.api")
    api.SelectType = types.SimpleNamespace(MAIN=select_main)
    api.to_observation_class = lambda d: d
    api.search_begin = engine.search_begin
    api.search_step = engine.search_step
    api.search_release = engine.search_release
    api.search_end = engine.search_end
    cg.api = api
    sys.modules["cg"] = cg
    sys.modules["cg.api"] = api


# ── loading agent_main.py the way the sandbox does ───────────────────────────

def load_agent_namespace(agent_dir: Path, search_config):
    for name in [m for m in list(sys.modules) if m == "bcnet" or m.startswith("bcnet.")]:
        del sys.modules[name]
    package = agent_dir / "bcnet"
    if search_config is None:
        (package / "search.json").unlink(missing_ok=True)
    else:
        (package / "search.json").write_text(json.dumps(search_config))
    cwd = os.getcwd()
    sys.path.insert(0, str(agent_dir))
    os.chdir(agent_dir)
    try:
        namespace = {"__name__": "__main__"}
        exec(compile((agent_dir / "main.py").read_text(), "main.py", "exec"), namespace)
        return namespace
    finally:
        os.chdir(cwd)
        sys.path.pop(0)


def build_agent_dir(tmp: Path) -> Path:
    """A minimal agent directory: the real main.py and bcnet sources, no weights."""
    agent = tmp / "agent"
    (agent / "bcnet").mkdir(parents=True)
    shutil.copy2(ROOT / "tools/bc/agent_main.py", agent / "main.py")
    for name in ("__init__.py", "cards.py", "encoder.py", "rows.py", "model.py"):
        shutil.copy2(ROOT / "tools/bc" / name, agent / "bcnet" / name)
    shutil.copy2(ROOT / "decks/marnies-grimmsnarl-ex.csv", agent / "deck.csv")
    return agent


# ── the checks ───────────────────────────────────────────────────────────────

BASE = {"enabled": True, "margin": 1.0, "candidates": 3, "samples": 2,
        "budget_fraction": 0.5, "budget_cap": 5.0, "reserve": 10.0,
        "follow_forced": 8, "log": False}

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'ok  ' if condition else 'FAIL'} {name}" + (f"  — {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


def run() -> int:
    import torch

    tmp = Path(tempfile.mkdtemp(prefix="guardtest-"))
    try:
        agent_dir = build_agent_dir(tmp)
        obs_dict = {"remainingOverageTime": 500.0}
        near_tie = torch.tensor([0.40, 0.38, 0.22])   # log gap 0.051, well under 1.0
        confident = torch.tensor([0.90, 0.06, 0.04])  # log gap 2.71

        # 1 — no search.json at all
        engine = FakeEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, None)
        out = ns["_guard"](obs_dict, Obs(), near_tie, [0])
        check("no search.json -> inert", out == [0] and not engine.begun)

        # 2 — present but disabled
        engine = FakeEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, {**BASE, "enabled": False})
        out = ns["_guard"](obs_dict, Obs(), near_tie, [0])
        check("enabled:false -> inert", out == [0] and not engine.begun)

        # 3 — enabled, but the net is confident
        engine = FakeEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, BASE)
        out = ns["_guard"](obs_dict, Obs(), confident, [0])
        check("confident net -> guard declines", out == [0] and not engine.begun,
              f"began {len(engine.begun)} searches")

        # 4 — enabled, near tie: the guard should overrule the net
        engine = FakeEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, BASE)
        out = ns["_guard"](obs_dict, Obs(), near_tie, [0])
        check("near tie -> guard fires and overrules", out == [1],
              f"returned {out}, began {len(engine.begun)}")

        # 5 — release/end discipline
        check("every search released", sorted(engine.released) == sorted(engine.begun),
              f"begun {engine.begun}, released {engine.released}")
        check("search_end called", engine.ended >= 1, f"ended {engine.ended}")
        check("stats recorded", ns["_SEARCH_STATS"]["fired"] == 1
              and ns["_SEARCH_STATS"]["changed"] == 1, str(ns["_SEARCH_STATS"]))

        # 6 — a search API that always raises
        for where in ("begin", "step"):
            engine = FakeEngine(raise_on={where})
            install_fake_cg(engine)
            ns = load_agent_namespace(agent_dir, BASE)
            out = ns["_guard"](obs_dict, Obs(), near_tie, [0])
            check(f"search_{where} raises -> net's pick survives", out == [0], f"returned {out}")
            check(f"search_{where} raises -> still released/ended",
                  sorted(engine.released) == sorted(engine.begun) and engine.ended >= 1)

        # 7 — no bank left
        engine = FakeEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, {**BASE, "reserve": 120.0})
        out = ns["_guard"]({"remainingOverageTime": 100.0}, Obs(), near_tie, [0])
        check("bank below reserve -> no search", out == [0] and not engine.begun)

        # 8 — non-MAIN select type is never guarded
        engine = FakeEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, BASE)
        out = ns["_guard"](obs_dict, Obs(select=Select(select_type=3)), near_tie, [0])
        check("non-MAIN select -> no search", out == [0] and not engine.begun)

        # 9 — multi-pick is never guarded (the guard returns one index)
        engine = FakeEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, BASE)
        out = ns["_guard"](obs_dict, Obs(select=Select(max_count=3)), near_tie, [0, 1])
        check("maxCount > 1 -> no search", out == [0, 1] and not engine.begun)

        # 10 — logging path runs without blowing up
        engine = FakeEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, {**BASE, "log": True})
        out = ns["_guard"](obs_dict, Obs(), near_tie, [0])
        check("log:true -> fires and logs", out == [1])

        # 11 — common random numbers: one hypothesis per sample, shared across
        # candidates. Scoring each candidate against its own draw compares means
        # over different hidden-information samples, which at 2 samples is
        # noisier than the difference being measured.
        engine = FakeEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, {**BASE, "candidates": 3, "samples": 2})
        ns["_guard"](obs_dict, Obs(), near_tie, [0])
        check("search_begin called once per sample, not per candidate",
              engine.begin_calls == 2, f"{engine.begin_calls} calls for 3 candidates x 2 samples")
        check("all candidates share each hypothesis",
              len(set(engine.hypotheses)) == engine.begin_calls,
              f"{len(set(engine.hypotheses))} distinct draws from {engine.begin_calls} begins")
        check("every candidate stepped from the shared root",
              engine.step_calls == 6, f"{engine.step_calls} steps")

        # 12 — search_begin usable only once per observation. If the SDK really
        # consumes search_begin_input, the guard must degrade to the net's pick,
        # not to a ranking built from a single sample of one candidate.
        engine = FakeEngine(single_use_begin=True)
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, {**BASE, "samples": 2})
        out = ns["_guard"](obs_dict, Obs(), near_tie, [0])
        check("single-use search_begin -> still scores from the first sample",
              out == [1] and engine.begin_calls == 2, f"returned {out}")
        check("single-use search_begin -> the failed sample is counted",
              ns["_SEARCH_STATS"]["errors"] == 1, str(ns["_SEARCH_STATS"]))

        # 13 — the net's own pick must have been scored before it is overruled
        engine = FakeEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, BASE)
        # A pick the net did not rank first is not in `ordered[:candidates]`
        # unless it was scored; feed a pick outside the evaluated set.
        out = ns["_guard"](obs_dict, Obs(select=Select(n_options=6)),
                           torch.tensor([0.19, 0.18, 0.17, 0.16, 0.15, 0.15]), [5])
        check("unscored net pick -> guard declines", out == [5], f"returned {out}")

        # ── Architecture D: the pimc mode ────────────────────────────────────
        PIMC = {**BASE, "mode": "pimc", "worlds": 4, "candidates": 3,
                "rollout_depth": 50, "margin": 2.5}

        class PimcEngine(FakeEngine):
            """Lines remember which root candidate opened them, and the second
            step of a line ends the game: candidate 1's lines are wins for seat
            0, every other candidate's are losses. A pimc that rolls lines out
            correctly must rank 1 first; one that reads only the first step's
            material cannot see it (both intermediate states are alive)."""
            def __init__(self, **kw):
                super().__init__(**kw)
                self.line = {}
            def search_step(self, search_id, picks):
                self.step_calls += 1
                if search_id not in self.begun or search_id in self.released:
                    raise ValueError("unknown or released search id")
                sid = self._next
                self._next += 1
                self.begun.append(sid)
                if search_id not in self.line:      # first step from a root
                    self.line[sid] = picks[0]
                    state = State(players=[Player(active_hp=100), Player(active_hp=100)])
                    return types.SimpleNamespace(searchId=sid, observation=Obs(current=state))
                winner = 0 if self.line[search_id] == 1 else 1
                self.line[sid] = self.line[search_id]
                state = State(result=winner)
                return types.SimpleNamespace(searchId=sid, observation=Obs(select=None, current=state))

        # 14 — fires on a near tie and prefers the line that wins the rollout
        engine = PimcEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, PIMC)
        out = ns["_pimc"](obs_dict, Obs(), near_tie, [0])
        check("pimc: near tie -> fires and overrules on rollout outcome", out == [1],
              f"returned {out}, began {len(engine.begun)}")
        check("pimc: one search_begin per world", engine.begin_calls == 4,
              f"{engine.begin_calls} begins for 4 worlds")
        check("pimc: every id released", sorted(engine.released) == sorted(engine.begun),
              f"begun {len(engine.begun)}, released {len(engine.released)}")
        check("pimc: search_end called", engine.ended >= 1)

        # 15 — a raising engine never costs the net's pick
        for where in ("begin", "step"):
            engine = FakeEngine(raise_on={where})
            install_fake_cg(engine)
            ns = load_agent_namespace(agent_dir, PIMC)
            out = ns["_pimc"](obs_dict, Obs(), near_tie, [0])
            check(f"pimc: search_{where} raises -> net's pick survives", out == [0],
                  f"returned {out}")
            check(f"pimc: search_{where} raises -> released/ended",
                  sorted(engine.released) == sorted(engine.begun) and engine.ended >= 1)

        # 16 — confident net, wrong select type, empty bank: all decline
        engine = FakeEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, {**PIMC, "reserve": 120.0})
        out_conf = ns["_pimc"](obs_dict, Obs(), confident, [0])
        out_type = ns["_pimc"](obs_dict, Obs(select=Select(select_type=3)), near_tie, [0])
        out_bank = ns["_pimc"]({"remainingOverageTime": 100.0}, Obs(), near_tie, [0])
        check("pimc: confident/non-MAIN/no-bank all decline",
              out_conf == [0] and out_type == [0] and out_bank == [0] and not engine.begun)

        # 17 — config dispatch: mode 'pimc' present but the guard entry point is
        # still callable and inert config-wise (mode is read at the call site)
        engine = FakeEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, PIMC)
        cfg = ns["_search_config"]()
        check("pimc: config carries mode/worlds/rollout_depth",
              cfg["mode"] == "pimc" and cfg["worlds"] == 4 and cfg["rollout_depth"] == 50)

        # 18 — archetype opponent-policy routing (bcnet/opponents/manifest.json).
        # Pure resolution logic, tested without weights: a manifest maps an
        # archetype tag (a substring of the belief's deck-list name) to a
        # checkpoint file, with a default for everything unmapped. This agent dir
        # ships no manifest, so it must resolve to self-play.
        ns = load_agent_namespace(agent_dir, PIMC)
        check("no manifest -> no opponent file (self-play)",
              ns["_archetype_file"]("marnies-grimmsnarl-ex") is None
              and ns["_opponent_policy"]("marnies-grimmsnarl-ex") == (None, None, None))
        ns["_OPP_MANIFEST"] = {
            "archetypes": {"grimmsnarl": "grimmsnarl.pt", "alakazam": "generalist.pt",
                           "dragapult": "generalist.pt"},
            "default": "generalist.pt"}
        check("routing: mirror deck name -> specialist",
              ns["_archetype_file"]("marnies-grimmsnarl-ex") == "grimmsnarl.pt")
        check("routing: deck-list variant matches its archetype substring",
              ns["_archetype_file"]("dragapult-crushing-hammer") == "generalist.pt")
        check("routing: unmapped archetype -> default",
              ns["_archetype_file"]("teal-mask-ogerpon-ex") == "generalist.pt")
        check("routing: no belief name -> default",
              ns["_archetype_file"](None) == "generalist.pt")
        check("opponent policy declines while the own model is unloaded",
              ns["_opponent_policy"]("marnies-grimmsnarl-ex") == (None, None, None))

        # 19 — value-backed PUCT (experiment 13). The fake agent has no weights,
        # so _MODEL/_VALUE are set by hand to drive the search; the PimcEngine's
        # second step ends the game, and candidate 1 wins for seat 0, so a PUCT
        # that rolls out and reads the outcome must prefer it.
        PUCT = {**BASE, "mode": "puct", "worlds": 3, "candidates": 3, "sims": 24,
                "leaf_depth": 4, "c_puct": 1.4, "min_sims": 4, "margin": 2.5}
        engine = FakeEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, PUCT)
        out = ns["_puct"](obs_dict, Obs(), near_tie, [0])
        check("puct: no critic -> defers to the policy", out == [0] and not engine.begun)

        engine = PimcEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, PUCT)
        ns["_VALUE"] = True
        ns["_MODEL"] = object()   # truthy; the rollout's policy falls back to legal
        out = ns["_puct"](obs_dict, Obs(), near_tie, [0])
        check("puct: fires and prefers the winning line", out == [1], f"returned {out}")
        check("puct: every id released", sorted(engine.released) == sorted(engine.begun),
              f"begun {len(engine.begun)}, released {len(engine.released)}")
        check("puct: search_end called", engine.ended >= 1)

        for where in ("begin", "step"):
            engine = FakeEngine(raise_on={where})
            install_fake_cg(engine)
            ns = load_agent_namespace(agent_dir, PUCT)
            ns["_VALUE"] = True
            ns["_MODEL"] = object()
            out = ns["_puct"](obs_dict, Obs(), near_tie, [0])
            check(f"puct: search_{where} raises -> net's pick survives", out == [0], f"returned {out}")
            check(f"puct: search_{where} raises -> released/ended",
                  sorted(engine.released) == sorted(engine.begun) and engine.ended >= 1)

        engine = FakeEngine()
        install_fake_cg(engine)
        ns = load_agent_namespace(agent_dir, {**PUCT, "reserve": 120.0})
        ns["_VALUE"] = True
        ns["_MODEL"] = object()
        oc = ns["_puct"](obs_dict, Obs(), confident, [0])
        ot = ns["_puct"](obs_dict, Obs(select=Select(select_type=3)), near_tie, [0])
        ob = ns["_puct"]({"remainingOverageTime": 100.0}, Obs(), near_tie, [0])
        check("puct: confident/non-MAIN/no-bank all decline",
              oc == [0] and ot == [0] and ob == [0] and not engine.begun)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.modules.pop("cg", None)
        sys.modules.pop("cg.api", None)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("all guard checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
