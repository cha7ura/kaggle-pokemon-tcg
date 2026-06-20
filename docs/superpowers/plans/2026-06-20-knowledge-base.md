# Knowledge Base (Plan 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Karpathy-style project wiki, an arxiv paper-fetch pipeline, and a Stop hook that keeps the wiki current — the legible knowledge base for this Pokémon TCG project.

**Architecture:** A static markdown `wiki/` tree authored from the mapped codebase and the game-AI research; a small Python tool (`tools/fetch_papers.py`) that turns arxiv ids into cited markdown under `wiki/papers/`; and a no-LLM `.claude` Stop hook that nudges Claude to update the wiki when code/research changed but the wiki did not.

**Tech Stack:** Markdown; Python 3.11 stdlib (`urllib`) + optional `pypdf` for tooling only (NOT shipped); Bash for the hook; `pytest` for tool tests.

## Global Constraints

- Nothing in this plan ships to Kaggle. `tools/` and `.claude/` are dev-only; the wiki is docs.
- Tooling MAY use third-party deps (`pypdf`, `pytest`); the **shipped agent** stays stdlib-only — untouched by this plan.
- The Stop hook MUST NOT call an LLM, MUST NOT edit files, and MUST be idempotent (silent if `wiki/` was already touched this turn) — it only emits a context nudge.
- Wiki voice: Karpathy-style — dumb-version-first, then why-it's-wrong, then the fix. First principles, honest about limitations.
- Every research claim in a wiki page cites an entry in `wiki/papers/sources.md`.
- Reference real repo paths verbatim: champion policy `autoresearch/agent_lucario.py`, eval `autoresearch/eval.py`, method `autoresearch/program.md`, ledger `autoresearch/log/experiments.md`, engine `sdk/cg/`, forward-model prior art `autoresearch/agent_search.py`.
- The deep-research workflow report (run this session) is the primary source for pages 10–17; fold its verified claims in, citing sources.

---

### Task 1: Wiki skeleton + index + citation ledger

**Files:**
- Create: `wiki/00-index.md`
- Create: `wiki/papers/sources.md`
- Create: `wiki/papers/.gitkeep`

**Interfaces:**
- Produces: the `wiki/` directory layout and reading order that every later page links into; `wiki/papers/sources.md` as the append-only citation ledger keyed by short id (e.g. `pog2021`, `deepnash2022`).

- [ ] **Step 1: Create the index page**

`wiki/00-index.md` must contain:
- One-paragraph "what this project is" (Kaggle Pokémon TCG AI Battle; ship `agent(obs)->list[int]`; offline stdlib agent; compiled `cg` engine via ctypes/Docker).
- A reading-order list linking every planned page (00–17 + `perfect-info-ancestors.md`), each with a one-line hook. Pages not yet written are listed (they arrive in later tasks) — the index is the map.
- A "two challengers, one gate" sentence pointing at `04-the-challenger-framework.md`.
- An explicit "nothing here ships an LLM; the only LLM is the developer" note.

- [ ] **Step 2: Create the citation ledger**

`wiki/papers/sources.md` starts with a header and a markdown table with columns: `id | title | authors | year | url | retrieved`. Seed it with the six known sources (leave `retrieved` blank — Task 3 fills it on fetch):

```markdown
# Sources

Citation ledger for the wiki. Cite as `[pog2021]` etc.

| id | title | authors | year | url | retrieved |
|----|-------|---------|------|-----|-----------|
| pog2021 | Player of Games (Student of Games) | Schmid et al. | 2021 | https://arxiv.org/abs/2112.03178 | |
| deepnash2022 | Mastering Stratego with Model-Free Multiagent RL (DeepNash) | Perolat et al. | 2022 | https://arxiv.org/abs/2206.15378 | |
| alphastar2019 | Grandmaster level in StarCraft II (AlphaStar) | Vinyals et al. | 2019 | https://www.nature.com/articles/s41586-019-1724-z | |
| alphaevolve2025 | AlphaEvolve: a Gemini-powered coding agent | DeepMind | 2025 | https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/ | |
| pokellmon2024 | PokeLLMon: human-parity agent for Pokémon battles | Hu, Huang, Liu | 2024 | https://arxiv.org/abs/2402.01118 | |
| ptcgbench2026 | PTCG-Bench: Can LLM Agents Master the Pokémon TCG? | Hua et al. | 2026 | https://arxiv.org/abs/2605.29653 | |
```

- [ ] **Step 3: Verify links resolve to planned paths**

Run: `grep -oE '\]\(([0-9a-z-]+\.md)\)' wiki/00-index.md | sort -u`
Expected: lists the page filenames the index references; confirm each matches a page planned in Task 2/4 (no typos). No broken-to-nowhere names beyond the planned set.

- [ ] **Step 4: Commit**

```bash
git add wiki/00-index.md wiki/papers/sources.md wiki/papers/.gitkeep
git commit -m "docs(wiki): skeleton index + citation ledger"
```

---

### Task 2: Core pages from the codebase (01–04)

**Files:**
- Create: `wiki/01-game-and-engine.md`
- Create: `wiki/02-policy-lucario.md`
- Create: `wiki/03-autoresearch-loop.md`
- Create: `wiki/04-the-challenger-framework.md`

**Interfaces:**
- Consumes: the index/reading-order from Task 1.
- Produces: the four "how this project actually works" pages later research pages link back to (esp. `04` and `03`, which the challenger/evolve discussion references).

These are editorial; "tests" are definition-of-done checklists, not pytest. Read the cited source files before writing each page.

- [ ] **Step 1: Write `01-game-and-engine.md`**

Read `sdk/cg/api.py` (the `Observation`/`Select` dataclasses, `to_observation_class`, `search_begin/step/end`) and `submission_lucario/main.py` (the `agent(obs_dict)->list[int]` contract).
DoD — page covers: the agent contract (return indices into `obs.select.option`, length in `[minCount,maxCount]`; return the 60-card deck when `obs.select is None`); the obs/select shapes; weakness/resistance damage; that the engine is a compiled `libcg.so` run under `docker --platform linux/amd64`; and the forward model (`search_begin/step/end`) with a one-line note that hidden state must be `_determinize`d (point at `autoresearch/agent_search.py`). Karpathy voice. ≥1 `file:line` reference.

- [ ] **Step 2: Write `02-policy-lucario.md`**

Read `autoresearch/agent_lucario.py` (`_score_main`, the `W` weight table, `_eff_damage`, ATTACH/PLAY branches, Crustle routing).
DoD — annotated walk of the champion policy: develop-then-attack scoring; what `W` weights mean; `_eff_damage` (weakness ×2, resistance, Crustle ex-wall zeroing); the kept "energy-spread discipline" rule; Boss's Orders drag; the crash-safe try/except → `_legal_fallback`. Explain *why* each meta rule exists. ≥3 `file:line` references.

- [ ] **Step 3: Write `03-autoresearch-loop.md`**

Read `autoresearch/program.md`, `autoresearch/eval.py` (`gauntlet`, `wilson_lb`), `autoresearch/log/experiments.md`.
DoD — explains: the keep/revert loop; seat-swapped gauntlet; why a **Wilson lower bound** (not raw win-rate) is the gate; the current gate (mirror LB ≥ 0.48 AND Dragapult holds); how to read `experiments.md`; why rejected experiments are kept ("rejections are evidence too"). Frame it explicitly as "a manual version of AlphaEvolve" forward-linking to `14-alphaevolve.md`.

- [ ] **Step 4: Write `04-the-challenger-framework.md`**

DoD — explains the discipline that lets neural ambition stay safe: the frozen champion (`autoresearch/champion_lucario.py`) is the benchmark; a challenger ships only if it beats the champion through the *same* `eval.py` Wilson-LB gate; the two challenger producers (`tools/evolve.py` evolved heuristic; `minizero/rnad` trained net via `agent_net.py`+`weights.npz`); same gate, same ledger; the live submission changes only on a win. (These producers are Plan 2; this page documents the contract they'll honor.)

- [ ] **Step 5: Verify pages are non-empty and cross-link**

Run: `wc -l wiki/01-game-and-engine.md wiki/02-policy-lucario.md wiki/03-autoresearch-loop.md wiki/04-the-challenger-framework.md && grep -l "agent_lucario.py" wiki/02-policy-lucario.md`
Expected: each file has substantial content (≥ ~40 lines); `02` references the policy file.

- [ ] **Step 6: Commit**

```bash
git add wiki/01-game-and-engine.md wiki/02-policy-lucario.md wiki/03-autoresearch-loop.md wiki/04-the-challenger-framework.md
git commit -m "docs(wiki): core pages — engine, policy, loop, challenger framework"
```

---

### Task 3: `tools/fetch_papers.py` (TDD) + fetch the six papers

**Files:**
- Create: `tools/fetch_papers.py`
- Create: `tools/__init__.py` (empty)
- Test: `tools/test_fetch_papers.py`
- Create (output, by running): `wiki/papers/<id>.md` ×6; updates `wiki/papers/sources.md`

**Interfaces:**
- Consumes: `wiki/papers/sources.md` (the id→url ledger) from Task 1.
- Produces:
  - `clean_markdown(raw_text: str) -> str` — collapses whitespace, strips control chars, normalizes headings; pure, deterministic, no network.
  - `arxiv_pdf_url(arxiv_id: str) -> str` — `"2402.01118"` → `"https://arxiv.org/pdf/2402.01118"`.
  - `fetch_paper(arxiv_id: str, out_dir: str = "wiki/papers", retrieved: str | None = None) -> str` — downloads, extracts text (pypdf if available, else writes a marked summary stub), writes `<out_dir>/<id>.md`, returns the path. Network — exercised in the manual fetch step, not the unit test.

- [ ] **Step 1: Write the failing test for the pure helpers**

`tools/test_fetch_papers.py`:

```python
from tools.fetch_papers import clean_markdown, arxiv_pdf_url


def test_arxiv_pdf_url():
    assert arxiv_pdf_url("2402.01118") == "https://arxiv.org/pdf/2402.01118"
    assert arxiv_pdf_url("2112.03178") == "https://arxiv.org/pdf/2112.03178"


def test_clean_markdown_collapses_whitespace_and_strips_control():
    raw = "Title\n\n\n\nbody\ttext\x0c with   spaces\n\n\n\nend"
    out = clean_markdown(raw)
    assert "\x0c" not in out            # form feed (pdf page break) removed
    assert "\n\n\n" not in out          # no 3+ consecutive newlines
    assert "with spaces" in out         # runs of spaces collapsed
    assert out.startswith("Title")
    assert out.endswith("end")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chaturaattidiya/Documents/Github/kaggle/pokemon && python -m pytest tools/test_fetch_papers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.fetch_papers'` (or ImportError).

- [ ] **Step 3: Implement `fetch_papers.py`**

```python
"""Fetch arxiv papers -> clean markdown under wiki/papers/. Dev tooling only; not shipped."""
import os
import re
import sys
import urllib.request


def arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}"


def clean_markdown(raw_text: str) -> str:
    text = raw_text.replace("\x0c", "\n")          # pdf page breaks -> newline
    text = re.sub(r"[ \t]+", " ", text)            # collapse spaces/tabs
    text = re.sub(r" *\n", "\n", text)             # trim trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)         # max one blank line
    return text.strip()


def _extract_pdf_text(pdf_bytes: bytes) -> str | None:
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return None


def fetch_paper(arxiv_id: str, out_dir: str = "wiki/papers", retrieved: str | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{arxiv_id}.md")
    url = arxiv_pdf_url(arxiv_id)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (paper-fetch)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        pdf_bytes = resp.read()
    text = _extract_pdf_text(pdf_bytes)
    if text and len(text.strip()) > 500:
        body = clean_markdown(text)
        header = f"# {arxiv_id}\n\nSource: {url}\nRetrieved: {retrieved or ''}\n\n---\n\n"
    else:
        body = ("(PDF text extraction unavailable — install `pypdf` or add a WebFetch summary here. "
                "This stub marks the paper as fetched-but-not-extracted.)")
        header = f"# {arxiv_id}\n\nSource: {url}\nRetrieved: {retrieved or ''}\nStatus: STUB\n\n---\n\n"
    with open(out_path, "w") as f:
        f.write(header + body + "\n")
    return out_path


if __name__ == "__main__":
    for aid in sys.argv[1:]:
        print(fetch_paper(aid))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/chaturaattidiya/Documents/Github/kaggle/pokemon && python -m pytest tools/test_fetch_papers.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Fetch the six papers (manual integration step)**

Run: `cd /Users/chaturaattidiya/Documents/Github/kaggle/pokemon && pip install pypdf >/dev/null 2>&1; python -m tools.fetch_papers 2112.03178 2206.15378 2402.01118 2605.29653`
Expected: prints four `wiki/papers/<id>.md` paths. (AlphaStar and AlphaEvolve are not arxiv PDFs — capture them as summaries directly in their wiki pages in Task 4, citing `alphastar2019`/`alphaevolve2025`.)
Then fill the `retrieved` column in `wiki/papers/sources.md` with `2026-06-20` for the fetched ids.
Note: if a download fails or yields a STUB, that's acceptable — Task 4 supplements from the deep-research report; record which are stubs.

- [ ] **Step 6: Commit**

```bash
git add tools/__init__.py tools/fetch_papers.py tools/test_fetch_papers.py wiki/papers/
git commit -m "feat(tools): arxiv paper fetch pipeline + fetched papers"
```

---

### Task 4: Research synthesis pages (10–17 + ancestors)

**Files:**
- Create: `wiki/10-cfr-and-nash.md`
- Create: `wiki/11-player-of-games.md`
- Create: `wiki/12-deepnash-rnad.md`
- Create: `wiki/13-alphastar-league.md`
- Create: `wiki/14-alphaevolve.md`
- Create: `wiki/15-pokellmon.md`
- Create: `wiki/16-ptcg-bench.md`
- Create: `wiki/17-imperfect-info-on-tcg.md`
- Create: `wiki/perfect-info-ancestors.md`

**Interfaces:**
- Consumes: the deep-research report (this session), `wiki/papers/*.md` (Task 3), `sources.md`, and pages `02/03/04` for back-links.
- Produces: the research spine; `17` is the synthesis page the whole wiki builds toward.

Editorial tasks; DoD checklists below. Every factual claim cites a `sources.md` id. Prefer the deep-research report's *verified* claims; mark anything uncertain as such.

- [ ] **Step 1: Write `10-cfr-and-nash.md`**

DoD: information sets; regret matching; CFR → CFR+ → external-sampling MCCFR; why the *average* strategy converges to Nash; exploitability as the scoreboard. This is the foundation pages 11–13 lean on. Cite as needed.

- [ ] **Step 2: Write `11-player-of-games.md`**

DoD: GT-CFR (growing-tree CFR) + the counterfactual value-and-policy network; sound search in imperfect-info games; "PoG = AlphaZero generalized to hidden info"; the search-at-inference cost (why it's heavy to ship). Cite `[pog2021]`.

- [ ] **Step 3: Write `12-deepnash-rnad.md`**

DoD: R-NaD (regularized Nash dynamics); model-free, **search-free at inference**; reaching (not cycling around) an approximate Nash; Stratego's 10^535 scale; the key property for us — search-free inference = shippable as a NumPy forward pass. Cite `[deepnash2022]`.

- [ ] **Step 4: Write `13-alphastar-league.md`**

DoD: league training; prioritized fictitious self-play; main/exploiter agents; non-transitivity (strategy cycles) and why a single best-response gets exploited. Tie to "why an evolved heuristic should be robust to multiple opponents, not just the mirror." Cite `[alphastar2019]`.

- [ ] **Step 5: Write `14-alphaevolve.md`**

DoD: LLM-proposes-mutation → automated-evaluator-scores → select → feed winners back; Gemini Flash breadth + Pro depth; results (Strassen 4×4, FlashAttention speedup, Borg). Explicit mapping: this *is* `autoresearch/program.md` automated, realized by Plan 2's `tools/evolve.py`. Cite `[alphaevolve2025]`.

- [ ] **Step 6: Write `15-pokellmon.md` and `16-ptcg-bench.md`**

DoD `15`: in-context RL from battle feedback; knowledge-augmented generation; consistent-action / anti-panic-switch; ~49% ladder win-rate. Cite `[pokellmon2024]`.
DoD `16`: benchmark for LLM agents on the *TCG*; the self-evolution **instability** finding and its lesson for our net (uncertain payoff → gate it). Cite `[ptcgbench2026]`.

- [ ] **Step 7: Write `17-imperfect-info-on-tcg.md` (the bridge) and `perfect-info-ancestors.md`**

DoD `17`: why PoG/DeepNash/AlphaStar don't port wholesale to an offline stdlib Kaggle agent (no GPU/net at inference, astronomical infoset count); what *does* transfer — determinization, opponent uncertainty, league-style robustness, regret intuition; and the one shippable neural path: **R-NaD net trained on GPU → NumPy forward pass** (search-free). Back-links to `04` and `12`.
DoD ancestors: short — AlphaGo/AlphaZero/MuZero; MCTS assumes perfect info + determinism; why that assumption breaks here.

- [ ] **Step 8: Verify coverage and citations**

Run: `ls wiki/*.md | wc -l && grep -RoE '\[(pog2021|deepnash2022|alphastar2019|alphaevolve2025|pokellmon2024|ptcgbench2026)\]' wiki/1*.md | sort | uniq -c`
Expected: all planned pages exist; each research page shows ≥1 citation.

- [ ] **Step 9: Commit**

```bash
git add wiki/10-*.md wiki/11-*.md wiki/12-*.md wiki/13-*.md wiki/14-*.md wiki/15-*.md wiki/16-*.md wiki/17-*.md wiki/perfect-info-ancestors.md
git commit -m "docs(wiki): game-AI research synthesis (CFR, PoG, DeepNash, AlphaStar, AlphaEvolve, LLM agents) + tcg bridge"
```

---

### Task 5: Stop hook — auto-update nudge (TDD)

**Files:**
- Create: `.claude/hooks/check-wiki.sh`
- Create: `.claude/settings.json`
- Test: `.claude/hooks/test_check_wiki.sh`

**Interfaces:**
- Consumes: nothing from earlier tasks (operates on git working-tree state).
- Produces: a Stop hook that, given the working tree, prints a nudge to stdout when non-wiki files changed but `wiki/` did not, and prints nothing otherwise. Exit 0 always.

Decision rule (kept simple and idempotent): inspect `git status --porcelain`. Let `CODE` = any changed/untracked path NOT under `wiki/` and NOT under `docs/` and NOT the hook's own dir. Let `WIKI` = any changed path under `wiki/`. Nudge iff `CODE` is non-empty AND `WIKI` is empty. This makes "already touched wiki this turn" → silent, and "only docs/spec changes" → silent (avoids nagging during planning).

- [ ] **Step 1: Write the failing test**

`.claude/hooks/test_check_wiki.sh`:

```bash
#!/usr/bin/env bash
# Test harness for check-wiki.sh — builds throwaway git repos and asserts output.
set -u
HOOK="$(cd "$(dirname "$0")" && pwd)/check-wiki.sh"
fail=0

run_case() {  # $1=desc  $2=expect_nudge(yes/no)  ; caller has prepared $TMP repo + staged state
  out="$(cd "$TMP" && bash "$HOOK")"
  if [ "$2" = yes ]; then
    echo "$out" | grep -qi "wiki" || { echo "FAIL: $1 (expected nudge, got none)"; fail=1; return; }
  else
    [ -z "$out" ] || { echo "FAIL: $1 (expected silence, got: $out)"; fail=1; return; }
  fi
  echo "PASS: $1"
}

# Case A: code changed, wiki untouched -> nudge
TMP="$(mktemp -d)"; (cd "$TMP" && git init -q && mkdir -p wiki && echo x > wiki/00-index.md && git add -A && git commit -qm init && echo "change" >> autoresearch_file.py)
run_case "code changed, no wiki -> nudge" yes
rm -rf "$TMP"

# Case B: wiki also changed -> silent
TMP="$(mktemp -d)"; (cd "$TMP" && git init -q && mkdir -p wiki && echo x > wiki/00-index.md && git add -A && git commit -qm init && echo "c" >> code.py && echo "w" >> wiki/00-index.md)
run_case "code + wiki changed -> silent" no
rm -rf "$TMP"

# Case C: only docs changed -> silent
TMP="$(mktemp -d)"; (cd "$TMP" && git init -q && mkdir -p docs && echo x > docs/a.md && git add -A && git commit -qm init && echo "d" >> docs/a.md)
run_case "only docs changed -> silent" no
rm -rf "$TMP"

# Case D: clean tree -> silent
TMP="$(mktemp -d)"; (cd "$TMP" && git init -q && echo x > a.py && git add -A && git commit -qm init)
run_case "clean tree -> silent" no
rm -rf "$TMP"

exit $fail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash .claude/hooks/test_check_wiki.sh`
Expected: FAIL — `check-wiki.sh` does not exist yet (bash: No such file).

- [ ] **Step 3: Implement the hook**

`.claude/hooks/check-wiki.sh`:

```bash
#!/usr/bin/env bash
# Stop hook: nudge to update the wiki when code/research changed but wiki/ did not.
# No LLM, no file edits — prints a context nudge to stdout, exits 0 always.
set -u
cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || true
command -v git >/dev/null 2>&1 || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

status="$(git status --porcelain 2>/dev/null)"
[ -z "$status" ] && exit 0

code_changed=no
wiki_changed=no
while IFS= read -r line; do
  path="${line:3}"
  case "$path" in
    wiki/*) wiki_changed=yes ;;
    docs/*|.claude/*) : ;;            # planning/config noise — ignore
    *) code_changed=yes ;;
  esac
done <<< "$status"

if [ "$code_changed" = yes ] && [ "$wiki_changed" = no ]; then
  echo "This turn changed code/research but no wiki/ page was updated. If anything affects how the project works, the policy, the method, or the research, update the relevant wiki/ page now — or note explicitly why no update is needed."
fi
exit 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `chmod +x .claude/hooks/check-wiki.sh && bash .claude/hooks/test_check_wiki.sh`
Expected: four `PASS:` lines, exit 0.

- [ ] **Step 5: Register the hook in settings**

`.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          { "type": "command", "command": "bash \"$CLAUDE_PROJECT_DIR/.claude/hooks/check-wiki.sh\"" }
        ]
      }
    ]
  }
}
```

- [ ] **Step 6: Verify settings is valid JSON**

Run: `python -c "import json;json.load(open('.claude/settings.json'));print('ok')"`
Expected: `ok`.

- [ ] **Step 7: Commit**

```bash
git add .claude/hooks/check-wiki.sh .claude/hooks/test_check_wiki.sh .claude/settings.json
git commit -m "feat(hook): Stop hook nudges wiki updates when code changed but wiki did not"
```

---

## Self-Review

**Spec coverage (Plan 1 portion = spec build steps 1–4):**
- Wiki skeleton + 00/01/02/03 → Tasks 1, 2 ✓ (also 04, the challenger page).
- `tools/fetch_papers.py` + fetch papers → Task 3 ✓.
- Research synthesis 04/10–17 + ancestors → Task 4 ✓.
- Stop hook + `.claude/settings.json` → Task 5 ✓.
- minizero / evolve / training → **deferred to Plan 2** (out of scope here) ✓.

**Placeholder scan:** code steps contain full code; wiki tasks use DoD checklists + source files to read (editorial, not placeholders); no "TBD"/"handle edge cases". ✓

**Type consistency:** `clean_markdown`, `arxiv_pdf_url`, `fetch_paper` names/signatures match between the test (Task 3 Step 1) and impl (Step 3). Hook decision rule consistent between description and implementation. `sources.md` ids match those cited in Task 4. ✓

**Open dependency:** Task 4 depends on the deep-research report (running this session); if it has not completed at execution time, write pages 10–17 from `wiki/papers/*.md` + abstracts and mark depth as provisional, then enrich when it lands.
