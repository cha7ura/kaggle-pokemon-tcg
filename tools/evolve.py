"""AlphaEvolve-style auto-loop for the Lucario heuristic (wiki/14-alphaevolve.md).

Automates the keep/revert half of the autoresearch loop: score a candidate agent against the
frozen champion via eval.py (in Docker), apply the Wilson-LB gate, and on PROMOTE replace the
champion + append to the experiment ledger. The *propose* half (mutating agent_lucario.py) stays
with the developer/Claude — no runtime LLM. So this is the evaluator + selection, not the generator.

CLI:  python -m tools.evolve <candidate.py> [--champion champion_lucario.py] [--games 600] [--lb 0.48]
"""
import argparse
import json
import os
import shutil
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTO = os.path.join(REPO, "autoresearch")          # candidate/champion are named relative to here
LOG = os.path.join(REPO, "autoresearch", "log", "experiments.md")


def _host(p):
    """Resolve a name (relative to autoresearch/) to a host path; absolute paths pass through."""
    return os.path.join(AUTO, p)


def run_eval(candidate, champion, games, deck="decks/lucario_meta.csv") -> dict:
    """Run eval.py inside the linux/amd64 engine container; return the parsed JSON verdict."""
    out = subprocess.run(
        ["docker", "run", "--rm", "--platform", "linux/amd64",
         "-v", f"{REPO}:/app", "-w", "/app/autoresearch", "-e", "PYTHONPATH=/app/sdk",
         "python:3.11-slim", "python", "eval.py",
         "--challenger", candidate, "--champion", champion, "--games", str(games), "--deck", deck],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def gate(result, lb_min=0.48) -> bool:
    return result.get("wilson_lb", 0.0) > lb_min


def evolve(candidate, champion, log_path, note, games=600, lb_min=0.48, runner=run_eval):
    result = runner(candidate, champion, games)
    kept = gate(result, lb_min)
    verdict = "PROMOTE" if kept else "REJECT"
    line = (f"{note} | {os.path.basename(candidate)} | games={result.get('games')} "
            f"lb={result.get('wilson_lb')} | {verdict}\n")
    with open(log_path, "a") as f:
        f.write(line)
    if kept:
        shutil.copyfile(_host(candidate), _host(champion))
    return kept, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--champion", default="champion_lucario.py",
                    help="frozen benchmark, named relative to autoresearch/")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--lb", type=float, default=0.48)
    ap.add_argument("--note", default="evolve")
    a = ap.parse_args()
    kept, result = evolve(a.candidate, a.champion, LOG, a.note, a.games, a.lb)
    print(json.dumps({"kept": kept, **result}, indent=2))


if __name__ == "__main__":
    main()
