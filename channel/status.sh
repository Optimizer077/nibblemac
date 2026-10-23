#!/bin/bash
# List channel tasks: id, state, exit code, commits, branch.
CH=$(cd "$(dirname "$0")" && pwd)
printf '%-24s %-9s %-5s %-7s %s\n' TASK STATE EXIT COMMITS BRANCH
for t in "$CH"/inbox/*.md; do [ -e "$t" ] || continue; id=$(basename "$t" .md); m="$CH/logs/$id.meta"
  if [ -f "$m" ]; then st=$(grep -E '^state=' "$m" | tail -1 | cut -d= -f2); ex=$(grep '^exit=' "$m" | cut -d= -f2)
    co=$(grep '^commits=' "$m" | cut -d= -f2); br=$(grep '^branch=' "$m" | cut -d= -f2)
  else st=queued; ex=-; co=-; br=-; fi
  printf '%-24s %-9s %-5s %-7s %s\n' "$id" "$st" "${ex:--}" "${co:--}" "$br"; done
