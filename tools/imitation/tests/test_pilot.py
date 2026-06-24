import json, importlib.util
from tools.replays_db import iter_replays


def _load_agent():
    spec = importlib.util.spec_from_file_location(
        "agent_imitation", "autoresearch/agent_imitation.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _a_learned_deck():
    # (deck_sig, policy_file_stem) for a >=30g deck; policy files named by m["file"]
    manifest = json.load(open("tools/imitation/data/manifest.json"))
    m = next(m for m in manifest if m["games"] >= 30)
    return m["deck_sig"], m["file"]


def _setenv(monkeypatch, tmp_path, sig, fstem):
    deckcsv = tmp_path / "deck.csv"
    deckcsv.write_text("\n".join(sig.split("_")))
    monkeypatch.setenv("IMIT_DECK", str(deckcsv))
    monkeypatch.setenv("IMIT_POLICY", f"tools/imitation/policies/{fstem}.json")


def test_deck_phase_returns_60(tmp_path, monkeypatch):
    sig, fstem = _a_learned_deck()
    _setenv(monkeypatch, tmp_path, sig, fstem)
    mod = _load_agent()
    out = mod.agent({"select": None})
    assert isinstance(out, list) and len(out) == 60


def test_in_play_returns_legal_indices(tmp_path, monkeypatch):
    sig, fstem = _a_learned_deck()
    _setenv(monkeypatch, tmp_path, sig, fstem)
    mod = _load_agent()
    for _ep, g in iter_replays():
        for s in g["steps"]:
            for seat in (0, 1):
                obs = s[seat].get("observation", {})
                sel = obs.get("select"); cur = obs.get("current")
                if sel and cur and sel.get("option") and len(sel["option"]) >= 2:
                    out = mod.agent({"select": sel, "current": cur})
                    n = len(sel["option"])
                    assert all(0 <= i < n for i in out)
                    assert len(out) >= sel.get("minCount", 1)
                    return
    raise RuntimeError("no decision found")


def test_malformed_obs_does_not_raise(monkeypatch):
    # wrong deck file (json, not ints) -> loader must cope, not crash at import
    monkeypatch.setenv("IMIT_DECK", "tools/imitation/data/manifest.json")
    mod = _load_agent()
    assert mod.agent("garbage") is not None
    assert mod.agent({"select": {"option": [], "minCount": 0}}) == []
