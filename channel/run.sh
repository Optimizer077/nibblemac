#!/bin/bash
# Channel: send Claude's message <id> from channel/channel.md to Codex and append Codex's reply to the same file.
#   bash channel/run.sh <id> [--sandbox read-only|workspace-write|danger-full-access] [--in-place]
# The message is the section "## Claude → Codex — <id>" (up to the next "## " heading). Codex works in its own
# worktree .claude/worktrees/codex-<id> on branch codex/<id> unless --in-place. Its final message is appended as
# "## Codex → Claude — <id>". Full logs: channel/logs/<id>.{jsonl,stderr,meta}. The model is always the one set in
# the owner's Codex config. Pi tasks need danger-full-access (no network otherwise) and "Pi access allowed" in the text.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); CH=$ROOT/channel; CHAN=$CH/channel.md
ID=${1:?usage: run.sh <id> [--sandbox MODE] [--in-place]}; shift
SANDBOX=workspace-write; INPLACE=0
while [ $# -gt 0 ]; do case "$1" in
  --sandbox) SANDBOX=$2; shift 2;; --in-place) INPLACE=1; shift;;
  *) echo "unknown option $1" >&2; exit 2;; esac; done
MSG=$(awk -v h="## Claude → Codex — $ID" '$0==h {on=1; next} on && /^## / {exit} on' "$CHAN")
[ -n "$(echo "$MSG" | tr -d '[:space:]')" ] || { echo "no message '## Claude → Codex — $ID' in $CHAN" >&2; exit 2; }
mkdir -p "$CH/logs"; LOG=$CH/logs/$ID.jsonl; META=$CH/logs/$ID.meta; LAST=$CH/logs/$ID.last.md
if [ $INPLACE = 1 ]; then WT=$ROOT; BR=$(git -C "$ROOT" branch --show-current)
else
  WT=$ROOT/.claude/worktrees/codex-$ID; BR=codex/$ID
  [ -d "$WT" ] || git -C "$ROOT" worktree add -q -b "$BR" "$WT" main || exit 3
fi
BASE=$(git -C "$WT" rev-parse --short HEAD)
{ echo "id=$ID"; echo "start=$(date '+%F %T')"; echo "sandbox=$SANDBOX"; echo "worktree=$WT"; echo "branch=$BR";
  echo "base=$BASE"; echo "state=running"; } > "$META"
ARGS=(exec - -C "$(cygpath -w "$WT")" -s "$SANDBOX" --skip-git-repo-check --json -o "$(cygpath -w "$LAST")")
{ sed "s|codex/<task-id>|$BR|" "$CH/preamble.md"; echo; echo "Task id: $ID"; echo; echo "$MSG"; } \
  | OMP_NUM_THREADS=8 codex "${ARGS[@]}" > "$LOG" 2> "$CH/logs/$ID.stderr"
RC=$?
COMMITS=$(git -C "$WT" log --oneline "$BASE"..HEAD 2>/dev/null | wc -l)
{ echo "end=$(date '+%F %T')"; echo "exit=$RC"; echo "commits=$COMMITS"; echo "head=$(git -C "$WT" rev-parse --short HEAD)"; } >> "$META"
sed -i 's/^state=running$/state=finished/' "$META"
REPLY=$(cat "$LAST" 2>/dev/null)
[ -n "$REPLY" ] || REPLY="(no reply: Codex exited with code $RC; see channel/logs/$ID.jsonl and .stderr)"
{ echo; echo "## Codex → Claude — $ID"; echo "_$(date '+%F %H:%M') · exit $RC · $COMMITS commit(s) on \`$BR\`_"; echo; echo "$REPLY"; } >> "$CHAN"
echo "task $ID finished: exit $RC, $COMMITS commit(s) on $BR; reply appended to channel/channel.md"
exit $RC
