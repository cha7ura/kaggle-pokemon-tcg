"""Validate the panel metric against KNOWN ladder scores before trusting it as a gate.

Runs panel_eval on v1/v2/v4/v5 (all on the lucario deck — policy comparison) and checks the
panel_score reproduces the real ladder ranking, especially v5 highest and v2 >= v4 (the case the
mirror metric got WRONG). If it can't, we don't trust the panel and fall back to pure-ladder A/B.
"""
import json
from panel_eval import panel_eval

LADDER = {"v1": 441.3, "v2": 565.8, "v4": 554.6, "v5": 664.4}
AGENTS = [("v1", "_versions/agent_v1.py"), ("v2", "_versions/agent_v2.py"),
          ("v4", "_versions/agent_v4.py"), ("v5", "_versions/agent_v5.py")]
GAMES = 200


def main():
    scores = {}
    for name, path in AGENTS:
        r = panel_eval(path, "decks/lucario_meta.csv", GAMES)
        scores[name] = r["panel_score"]
        print(f"{name}: panel_score={r['panel_score']:.4f}  ladder={LADDER[name]}  "
              f"(greedy={r['opponents']['greedy']['wr']} dragapult={r['opponents']['dragapult']['wr']} "
              f"v4mirror={r['opponents']['v4_mirror']['wr']} | random={r['random_floor']} crustle={r['crustle']})",
              flush=True)
    panel_order = sorted(scores, key=lambda k: scores[k])
    ladder_order = sorted(LADDER, key=lambda k: LADDER[k])
    v5_top = max(scores, key=scores.get) == "v5"
    v2_ge_v4 = scores["v2"] >= scores["v4"]
    print(json.dumps({"panel_scores": scores, "panel_order_low_to_high": panel_order,
                      "ladder_order_low_to_high": ladder_order,
                      "v5_is_top": v5_top, "v2_ge_v4": v2_ge_v4,
                      "VERDICT": "TRUST panel" if (v5_top and v2_ge_v4) else "panel UNRELIABLE -> pure-ladder A/B"}, indent=2))


if __name__ == "__main__":
    main()
