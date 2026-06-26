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
