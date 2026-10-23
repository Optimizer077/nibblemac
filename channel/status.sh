#!/bin/bash
# List the messages in channel.md: id, whether Codex replied, exit code, commits, branch.
CH=$(cd "$(dirname "$0")" && pwd)
printf '%-24s %-9s %-5s %-7s %s\n' ID STATE EXIT COMMITS BRANCH
awk '/^## Claude → Codex — / {print $NF}' "$CH/channel.md" | while read -r id; do m="$CH/logs/$id.meta"
  if [ -f "$m" ]; then st=$(grep '^state=' "$m" | cut -d= -f2); ex=$(grep '^exit=' "$m" | cut -d= -f2)
    co=$(grep '^commits=' "$m" | cut -d= -f2); br=$(grep '^branch=' "$m" | cut -d= -f2)
  else st=unsent; ex=-; co=-; br=-; fi
  printf '%-24s %-9s %-5s %-7s %s\n' "$id" "$st" "${ex:--}" "${co:--}" "$br"; done
