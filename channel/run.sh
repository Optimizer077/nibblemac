#!/bin/bash
# Channel: run one task from channel/inbox/<id>.md in an isolated worktree, record everything.
#   bash channel/run.sh <id> [--sandbox read-only|workspace-write|danger-full-access] [--in-place]
# Output: channel/outbox/<id>.md (Codex's final message), channel/logs/<id>.jsonl (event stream),
#         channel/logs/<id>.meta (start/end, exit code, sandbox, branch, commits). Pi tasks need danger-full-access
#         (the workspace sandbox blocks the network) and must say "Pi access allowed" in the task text.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); CH=$ROOT/channel
ID=${1:?usage: run.sh <task-id> [--sandbox MODE] [--in-place]}; shift
SANDBOX=workspace-write; INPLACE=0  # model: always the one set in the owner's Codex config
while [ $# -gt 0 ]; do case "$1" in
  --sandbox) SANDBOX=$2; shift 2;; --in-place) INPLACE=1; shift;;
  *) echo "unknown option $1" >&2; exit 2;; esac; done
TASK=$CH/inbox/$ID.md; [ -f "$TASK" ] || { echo "no task file $TASK" >&2; exit 2; }
OUT=$CH/outbox/$ID.md; LOG=$CH/logs/$ID.jsonl; META=$CH/logs/$ID.meta
if [ $INPLACE = 1 ]; then WT=$ROOT; BR=$(git -C "$ROOT" branch --show-current)
else
  WT=$ROOT/.claude/worktrees/codex-$ID; BR=codex/$ID
  [ -d "$WT" ] || git -C "$ROOT" worktree add -q -b "$BR" "$WT" main || exit 3
fi
BASE=$(git -C "$WT" rev-parse --short HEAD)
{ echo "id=$ID"; echo "start=$(date '+%F %T')"; echo "sandbox=$SANDBOX"; echo "model=from the Codex config";
  echo "worktree=$WT"; echo "branch=$BR"; echo "base=$BASE"; echo "state=running"; } > "$META"
WIN_WT=$(cygpath -w "$WT"); WIN_OUT=$(cygpath -w "$OUT")
ARGS=(exec - -C "$WIN_WT" -s "$SANDBOX" --skip-git-repo-check --json -o "$WIN_OUT")
{ sed "s|codex/<task-id>|$BR|" "$CH/preamble.md"; echo; echo "Task id: $ID"; echo; cat "$TASK"; } \
  | OMP_NUM_THREADS=8 codex "${ARGS[@]}" > "$LOG" 2> "$CH/logs/$ID.stderr"
RC=$?
COMMITS=$(git -C "$WT" log --oneline "$BASE"..HEAD 2>/dev/null | wc -l)
{ echo "end=$(date '+%F %T')"; echo "exit=$RC"; echo "commits=$COMMITS"; echo "head=$(git -C "$WT" rev-parse --short HEAD)";
  echo "dirty=$(git -C "$WT" status --porcelain | grep -vc '^??')"; echo "state=done"; } >> "$META"
sed -i 's/^state=running$/state=finished/' "$META"
echo "task $ID finished: exit $RC, $COMMITS commit(s) on $BR; answer in channel/outbox/$ID.md"
exit $RC
