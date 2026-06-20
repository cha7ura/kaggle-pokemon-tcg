#!/usr/bin/env bash
# Test harness for check-wiki.sh — builds throwaway git repos and asserts output.
set -u
HOOK="$(cd "$(dirname "$0")" && pwd)/check-wiki.sh"
fail=0

run_case() {  # $1=desc  $2=expect_nudge(yes/no) ; caller prepared $TMP repo + working state
  out="$(cd "$TMP" && bash "$HOOK")"
  if [ "$2" = yes ]; then
    echo "$out" | grep -qi "wiki" || { echo "FAIL: $1 (expected nudge, got none)"; fail=1; return; }
  else
    [ -z "$out" ] || { echo "FAIL: $1 (expected silence, got: $out)"; fail=1; return; }
  fi
  echo "PASS: $1"
}

# A: code changed, wiki untouched -> nudge
TMP="$(mktemp -d)"; (cd "$TMP" && git init -q && mkdir -p wiki && echo x > wiki/00-index.md && git add -A && git commit -qm init && echo "change" >> autoresearch_file.py)
run_case "code changed, no wiki -> nudge" yes; rm -rf "$TMP"

# B: code + wiki changed -> silent
TMP="$(mktemp -d)"; (cd "$TMP" && git init -q && mkdir -p wiki && echo x > wiki/00-index.md && git add -A && git commit -qm init && echo "c" >> code.py && echo "w" >> wiki/00-index.md)
run_case "code + wiki changed -> silent" no; rm -rf "$TMP"

# C: only docs changed -> silent
TMP="$(mktemp -d)"; (cd "$TMP" && git init -q && mkdir -p docs && echo x > docs/a.md && git add -A && git commit -qm init && echo "d" >> docs/a.md)
run_case "only docs changed -> silent" no; rm -rf "$TMP"

# D: clean tree -> silent
TMP="$(mktemp -d)"; (cd "$TMP" && git init -q && echo x > a.py && git add -A && git commit -qm init)
run_case "clean tree -> silent" no; rm -rf "$TMP"

exit $fail
