#!/usr/bin/env bash
# Driver of the experiment: renders the rules, runs the Codex agent in workspace/ as a loop of fresh sessions with
# notebook handoffs, and records everything in out/<mode>/.
#
# Usage: scripts/run_driver.sh pilot|full --no-look [--effort high] [--model gpt-6.1-sol] [--stop-at N]
#                      [--val-budget N] [--session-wall D] [--total-wall D] [--wall D] [--max-sessions N]
#                      [--batch N] [--fixed-loop] [--no-commit]
#        scripts/run_driver.sh continue full [--stop-at N] [--session-wall D] [--total-wall D] [--max-sessions N]
#
# The published run was started with
#        scripts/run_driver.sh full --no-look --val-budget 25 --stop-at 97
# and, after the budget monitor stopped session 1, re-entered with
#        scripts/run_driver.sh continue full --stop-at 97
#
#   --no-look        REQUIRED here: no image viewer. look.py stays in the workspace but refuses to run
#                    (.batch_config.json "viewer": false) and the rules are rendered without the viewer blocks. The
#                    viewer service of the original harness is not part of this repository.
#   Defaults         model gpt-6.1-sol, reasoning effort high. Val is measure-only and limited to --val-budget passes.
#   --batch N        batch size of the optional step.py helper (default: pilot 10, full 64); written to
#                    workspace/.batch_config.json ({"batch_size": N, "seed": 0, "viewer": false, "loop": "free",
#                    "val_budget": B, "enforce_comments": true, "model": M}) before the first session, together with
#                    a reset of .batch_state.json, .steps.jsonl, .teacher_used and .val_passes.json (fresh runs only).
#   --fixed-loop     write "loop": "fixed" (batch-gated step.py/val rules) instead of the default "loop": "free"
#                    (val may be measured at any time, step.py --batch-size). The published run used "free".
#   pilot            fresh pilot runs temporarily replace workspace/data with a 30 train + 30 val subset (all 10
#                    classes, the first three files by sorted name in each of train and val; default batch size 10).
#                    The full data is stashed in .pilot_stash/data and restored when the run ends (and by an EXIT
#                    trap). Full mode never touches the data.
#   fresh run        after the pre-run commit, workspace/ root is cleaned of leftovers of earlier runs (classify.py,
#                    NOTES.md, *.py except check.py/look.py/step.py/agent.py, *.json/*.csv/*.txt/*.md except
#                    AGENTS.md/BRIEF.md, __pycache__/, .subagents/ and other scratch directories). The directories
#                    data/, teacher/, .agents, .aws, .codex and .git are never deleted.
#   Session loop     a fresh run is a loop of codex sessions that ends when all val passes are used (not when the
#                    agent stops): after a session exits (any exit code) the loop reads "used" from
#                    workspace/.val_passes.json and starts the next session as a FRESH `codex exec` (brief + handoff
#                    text + the last 8,000 characters of NOTES.md) unless: used >= --val-budget; total elapsed >=
#                    --total-wall; out/<mode>/budget_stop exists (budget monitor); --max-sessions sessions ran; or a
#                    session exited non-zero within 30 s with no events twice in a row (one retry with the same
#                    prompt). Each session runs under `timeout --signal=INT --kill-after=60`, with its wall capped by
#                    what is left of --total-wall.
#   continue         re-enter the session loop of an existing out/full run (typically after a budget-monitor stop).
#                    Settings come from out/full/run_meta.json; elapsed time and session count carry over; the
#                    workspace is left exactly as the last session left it (no render, no cleanup, no config rewrite).
#   --val-budget N   val passes in the whole run (default: pilot 5, full 25); written to .batch_config.json and
#                    substituted for {{VAL_BUDGET}} in the rules.
#   --session-wall D wall of one session (default: pilot 20m, full 40m)
#   --total-wall D   wall of the whole loop (default: pilot 25m, full 4h); --wall D is a synonym
#   --max-sessions N at most N sessions (default 12)
#   --stop-at N      stop threshold for scripts/budget_monitor.py, in percent of the Codex 5-hour rate-limit window
#                    (default: $STOP_AT or 90). The monitor is started for every run with --kill (SIGINT to
#                    `codex exec` when crossed) and stopped at the end; output in out/<mode>/budget.stdout and
#                    out/<mode>/budget.jsonl.
#   --no-commit      skip the local pre/post-run git commits (the commits record the workspace state; nothing is
#                    pushed).
#   Per session files in out/<mode>/: brief_session_N.md, events_session_N.jsonl, last_message_session_N.md,
#                    codex_stderr_session_N.log; events.jsonl (and codex_stderr.log) are the concatenation of all
#                    sessions, last_message.md is the last session's; val_passes.json is the copy of
#                    workspace/.val_passes.json; run_meta.json holds the settings and timing.
#
# Notes:
#  - Codex's workspace-write sandbox has network access OFF by default, so no extra flag is needed to keep the
#    agent offline. The sandbox can still READ the whole file system: keep results of earlier runs (including
#    results/ of this repository) out of reach before starting a run (see README.md).
#  - workspace/teacher/ must exist before a run; this script never creates, changes or deletes it.
#  - The sub-agent daemon (scripts/subagent_daemon.py, serves workspace/agent.py) must already be running:
#    `mkdir -p out && nohup python3 scripts/subagent_daemon.py >> out/subagent.stdout 2>&1 &` from the repo root.
#    This script only warns if it cannot find it. After the run workspace/.subagents/log.jsonl is copied to
#    out/<mode>/subagents_log.jsonl and the number of sub-agents is printed.
set -u

ACTION=run
if [ "${1:-}" = continue ]; then ACTION=continue; shift; fi
MODE="${1:-}"
[ $# -gt 0 ] && shift
EFFORT="high"
MODEL="gpt-6.1-sol"
WALL_OVERRIDE=""
STOP_AT="${STOP_AT:-90}"
BATCH_OVERRIDE=""
LOOP=free
LOOK=1
VAL_BUDGET_OVERRIDE=""
SESSION_WALL_OVERRIDE=""
TOTAL_WALL_OVERRIDE=""
MAX_SESSIONS=12
NO_COMMIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --effort) EFFORT="${2:?--effort needs a value}"; shift 2 ;;
    --model)  MODEL="${2:?--model needs a value}"; shift 2 ;;
    --wall)   WALL_OVERRIDE="${2:?--wall needs a value}"; shift 2 ;;
    --val-budget) VAL_BUDGET_OVERRIDE="${2:?--val-budget needs a value}"; shift 2 ;;
    --session-wall) SESSION_WALL_OVERRIDE="${2:?--session-wall needs a value}"; shift 2 ;;
    --total-wall) TOTAL_WALL_OVERRIDE="${2:?--total-wall needs a value}"; shift 2 ;;
    --max-sessions) MAX_SESSIONS="${2:?--max-sessions needs a value}"; shift 2 ;;
    --no-commit) NO_COMMIT=1; shift ;;
    --stop-at) STOP_AT="${2:?--stop-at needs a value}"; shift 2 ;;
    --batch)  BATCH_OVERRIDE="${2:?--batch needs a value}"; shift 2 ;;
    --no-look) LOOK=0; shift ;;
    --fixed-loop) LOOP=fixed; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

USAGE="usage: $0 pilot|full --no-look [--effort high] [--model gpt-6.1-sol] [--stop-at N] [--val-budget N] [--session-wall D] [--total-wall D] [--wall D] [--max-sessions N] [--batch N] [--fixed-loop] [--no-commit]
       $0 continue full [--stop-at N] [--session-wall D] [--total-wall D] [--max-sessions N]"
case "$MODE" in
  pilot) NTRAIN=30;  NVAL=30;  BATCH=10; VAL_BUDGET=5;  SESSION_WALL=20m; TOTAL_WALL=25m
         EXTRA="This is a short pilot with 30 train + 30 val images; you decide the strategy; use the teacher first; produce classify.py and NOTES.md." ;;
  full)  NTRAIN=505;  NVAL=495; BATCH=64; EXTRA=""; VAL_BUDGET=25; SESSION_WALL=40m; TOTAL_WALL=4h ;;
  *) echo "$USAGE" >&2; exit 2 ;;
esac
if [ "$ACTION" = run ] && [ "$LOOK" = 1 ]; then
  echo "the image viewer is not part of this repository; run with --no-look (as the published run did)" >&2; exit 2
fi
[ -n "$WALL_OVERRIDE" ] && TOTAL_WALL="$WALL_OVERRIDE"
[ -n "$TOTAL_WALL_OVERRIDE" ] && TOTAL_WALL="$TOTAL_WALL_OVERRIDE"
[ -n "$SESSION_WALL_OVERRIDE" ] && SESSION_WALL="$SESSION_WALL_OVERRIDE"
[ -n "$VAL_BUDGET_OVERRIDE" ] && VAL_BUDGET="$VAL_BUDGET_OVERRIDE"
case "$VAL_BUDGET" in ''|*[!0-9]*|0) echo "--val-budget must be a positive integer" >&2; exit 2 ;; esac
case "$MAX_SESSIONS" in ''|*[!0-9]*|0) echo "--max-sessions must be a positive integer" >&2; exit 2 ;; esac
[ -n "$BATCH_OVERRIDE" ] && BATCH="$BATCH_OVERRIDE"
case "$BATCH" in ''|*[!0-9]*|0) echo "--batch must be a positive integer" >&2; exit 2 ;; esac
case "$STOP_AT" in ''|*[!0-9.]*) echo "--stop-at must be a number (percent)" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
command -v codex >/dev/null || { echo "codex (the Codex CLI) is not on PATH" >&2; exit 1; }
[ -f workspace/teacher/cnn_505.npz ] || { echo "workspace/teacher/cnn_505.npz is missing" >&2; exit 1; }
[ -d workspace/data/train ] && [ -d workspace/data/val ] || { echo "workspace/data/{train,val} missing; run data/build_splits.py first" >&2; exit 1; }

# Commit the whole project (local record of the workspace before/after the run); nothing is pushed.
commit_push() {
  [ "$NO_COMMIT" = 1 ] && { echo "(--no-commit: skipped commit: $1)"; return 0; }
  git add -A && git commit -qm "$1" || true
}

# Budget monitor: started for every run, killed when the driver exits.
MONITOR_PID=""
start_monitor() {  # $1 = out dir
  rm -f "$1/budget_stop"   # stale marker of an earlier run; the monitor re-creates it when it kills this run
  STOP_LINES_BEFORE=$(grep -c 'budget stop' "$1/budget.stdout" 2>/dev/null || true); STOP_LINES_BEFORE=${STOP_LINES_BEFORE:-0}
  nohup python3 scripts/budget_monitor.py --kill --stop-at "$STOP_AT" --log "$1/budget.jsonl" \
    --marker "$1/budget_stop" >> "$1/budget.stdout" 2>&1 &
  MONITOR_PID=$!
  echo "budget monitor started (pid $MONITOR_PID, stop at ${STOP_AT}% of the 5h window, log $1/budget.jsonl)"
}
# True if the budget monitor stopped the current run (marker file, or a new "budget stop" line in budget.stdout).
monitor_stopped() {  # $1 = out dir
  [ -e "$1/budget_stop" ] && return 0
  local now; now=$(grep -c 'budget stop' "$1/budget.stdout" 2>/dev/null || true)
  [ "${now:-0}" -gt "${STOP_LINES_BEFORE:-0}" ]
}
stop_monitor() { [ -n "$MONITOR_PID" ] && kill "$MONITOR_PID" 2>/dev/null; MONITOR_PID=""; }

# Pilot data subset: stash the full workspace/data and build a 30 train + 30 val copy (all 10 classes, the first
# three files by sorted name in each of train and val). Full mode never calls these.
PILOT_STASHED=0
build_pilot_subset() {  # $1 = workspace dir, $2 = stash dir (must not exist)
  local ws="$1" stash="$2"
  if [ -e "$stash" ]; then echo "ERROR: $stash already exists (leftover of an interrupted pilot?); refusing to continue" >&2; return 1; fi
  [ -d "$ws/data" ] || { echo "ERROR: $ws/data not found" >&2; return 1; }
  mkdir -p "$stash" && mv "$ws/data" "$stash/data" || return 1
  PILOT_STASHED=1
  python3 - "$stash/data" "$ws/data" <<'PY' || return 1
import os, shutil, sys
src, dst = sys.argv[1:]
classes = sorted(os.listdir(os.path.join(src, "train")))
for split in ("train", "val"):
    for c in classes:
        files = sorted(os.listdir(os.path.join(src, split, c)))
        os.makedirs(os.path.join(dst, split, c), exist_ok=True)
        for f in files[:3]:
            shutil.copy2(os.path.join(src, split, c, f), os.path.join(dst, split, c, f))
print("pilot subset built in %s: classes %s, 3 train + 3 val images each (30 + 30)" % (dst, ", ".join(classes)))
PY
}
restore_pilot_subset() {  # $1 = workspace dir, $2 = stash dir; no-op unless this run stashed the data
  local ws="$1" stash="$2"
  [ "$PILOT_STASHED" = 1 ] && [ -d "$stash/data" ] || return 0
  rm -rf "$ws/data" && mv "$stash/data" "$ws/data" && rmdir "$stash" && PILOT_STASHED=0 \
    && echo "pilot data restored: $ws/data ($(find "$ws/data" -name '*.JPEG' | wc -l) images)"
}
trap 'stop_monitor; restore_pilot_subset "$ROOT/workspace" "$ROOT/.pilot_stash"' EXIT
budget_once() { python3 scripts/budget_monitor.py --once 2>&1 | tail -n 1; }
json_str() { python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"; }

# Duration (as for timeout(1): 90, 30s, 30m, 2h) -> whole minutes, rounded up.
to_minutes() {
  python3 - "$1" <<'PY'
import sys, math
v = sys.argv[1]
mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}
n = float(v[:-1]) * mult[v[-1]] if v[-1] in mult else float(v)
print(max(1, math.ceil(n / 60)))
PY
}

# Duration (as for timeout(1)) -> whole seconds, rounded up.
to_seconds() {
  python3 - "$1" <<'PY'
import sys, math
v = sys.argv[1]
mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}
n = float(v[:-1]) * mult[v[-1]] if v[-1] in mult else float(v)
print(max(1, math.ceil(n)))
PY
}

# Val passes used so far (workspace/.val_passes.json "used"; 0 if missing or unreadable).
val_used() {
  python3 -c 'import json; print(int(json.load(open("workspace/.val_passes.json")).get("used", 0)))' 2>/dev/null || echo 0
}

# Write the prompt of session $1 (>= 2) to $2: the rendered brief, a blank line, the handoff text, the NOTES.md tail.
#   $3 = sessions so far, $4 = val passes used, $5 = elapsed seconds, $6 = exit code of the previous session
write_handoff_prompt() {
  python3 - "$OUT/brief_rendered.md" "$2" "$3" "$4" "$VAL_BUDGET" "$5" "$(to_minutes "$TOTAL_WALL")" "$6" workspace/NOTES.md <<'PY'
import sys
brief, dst, sessions, used, budget, elapsed, total_min, prev_exit, notes = sys.argv[1:]
out = open(brief, encoding="utf-8", errors="replace").read()
if not out.endswith("\n"):
    out += "\n"
out += "\n"
out += ("THIS IS A RESTARTED CONVERSATION (to keep it short). Sessions before this one: %s. Val passes used: %s of %s. "
        "Elapsed: %dm of %sm. The workspace is as you left it: classify.py and NOTES.md on disk are your current state; "
        "read NOTES.md first, then classify.py, then continue. The session continues until all %s val passes are used.\n"
        % (sessions, used, budget, int(elapsed) // 60, total_min, budget))
if prev_exit == "0" and int(used) < int(budget):
    out += ("You finished with val passes left; that does not end the session. Use the remaining passes deliberately "
            "(see 'How to work' in AGENTS.md).\n")
try:
    tail = open(notes, encoding="utf-8", errors="replace").read()[-8000:]
except OSError:
    tail = None
out += "\nYOUR NOTEBOOK (NOTES.md, last 8,000 characters):\n````\n"
out += (tail if tail is not None else "(no NOTES.md yet)")
if not out.endswith("\n"):
    out += "\n"
out += "````\n"
open(dst, "w", encoding="utf-8").write(out)
PY
}

# Copy the sub-agent log of this run to the out dir and print how many sub-agents ran.
copy_subagent_log() {  # $1 = out dir
  if [ -s workspace/.subagents/log.jsonl ]; then
    cp workspace/.subagents/log.jsonl "$1/subagents_log.jsonl"
    echo "sub-agents run: $(python3 -c 'import json,sys; print(len({json.loads(l)["id"] for l in open(sys.argv[1]) if l.strip()}))' "$1/subagents_log.jsonl") (log: $1/subagents_log.jsonl)"
  else
    echo "sub-agents run: 0"
  fi
}

# Print the Codex thread id (thread.started event) of an events file.
extract_session_id() {
  python3 - "$1" <<'PY'
import sys, json
try:
    lines = open(sys.argv[1])
except OSError:
    sys.exit(0)
for i, line in enumerate(lines):
    if i > 50: break
    try: e = json.loads(line)
    except ValueError: continue
    if e.get("type") == "thread.started" and e.get("thread_id"):
        print(e["thread_id"]); break
PY
}

OUT="out/$MODE"
TS=$(date +%Y%m%d_%H%M%S)

# Preserve the artifacts of an earlier run of the same mode (not on `continue`: that run dir is the one continued).
if [ -e "$OUT" ] && [ "$ACTION" != continue ]; then
  mv "$OUT" "out/${MODE}_${TS}" && echo "moved previous $OUT -> out/${MODE}_${TS}"
fi
mkdir -p "$OUT"

pgrep -f 'subagent_daemon\.py' >/dev/null || echo "WARNING: no scripts/subagent_daemon.py process found (sub-agent daemon; agent.py will not work)." >&2

# Pilot only: swap workspace/data for a 30 train + 30 val subset (restored by restore_pilot_subset / the EXIT trap).
if [ "$MODE" = pilot ]; then
  build_pilot_subset "$ROOT/workspace" "$ROOT/.pilot_stash" || exit 1
fi

if [ "$ACTION" = continue ]; then
  OUT="out/$MODE"
  [ -f "$OUT/run_meta.json" ] && [ -f "$OUT/brief_rendered.md" ] || { echo "no $OUT/run_meta.json or brief_rendered.md; cannot continue" >&2; exit 1; }
  eval "$(python3 - "$OUT/run_meta.json" <<'PY'
import json, sys, shlex
m = json.load(open(sys.argv[1]))
for k, v in [("MODEL", m["model"]), ("EFFORT", m["effort"]), ("VAL_BUDGET", m["val_budget"]), ("SESSION_WALL", m["session_wall"]),
             ("TOTAL_WALL", m["total_wall"]), ("LOOP", m.get("loop", "free")), ("LOOK", 1 if m.get("viewer", True) else 0),
             ("NTRAIN", m["n_train"]), ("NVAL", m["n_val"]), ("BATCH", m["batch"]), ("PREV_SESSIONS", m["sessions"]),
             ("PREV_ELAPSED", m["elapsed_seconds"]), ("PREV_STOP", m.get("stop_reason", ""))]:
    print("%s=%s" % (k, shlex.quote(str(v))))
PY
)"
  [ -n "$TOTAL_WALL_OVERRIDE" ] && TOTAL_WALL="$TOTAL_WALL_OVERRIDE"
  [ -n "$SESSION_WALL_OVERRIDE" ] && SESSION_WALL="$SESSION_WALL_OVERRIDE"
  [ "$MODE" = pilot ] && { echo "continue is for full runs (the pilot subset is not kept between runs)" >&2; exit 1; }
  echo "continuing $OUT: $PREV_SESSIONS session(s) and ${PREV_ELAPSED}s so far (stopped: $PREV_STOP); model $MODEL, val budget $VAL_BUDGET, total wall $TOTAL_WALL"
  if [ "$(val_used)" -ge "$VAL_BUDGET" ]; then echo "all $VAL_BUDGET val passes are already used; nothing to continue"; exit 0; fi
  commit_push "pre-continue $MODE"
  start_monitor "$OUT"
  BUDGET_BEFORE=$(budget_once); echo "budget before: $BUDGET_BEFORE"
  START_EPOCH=$(( $(date +%s) - PREV_ELAPSED ))     # elapsed carries over: the total wall is a limit on the whole run
  START_ISO=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["start"])' "$OUT/run_meta.json")
  S=$PREV_SESSIONS; CONTINUED=1
else
  CONTINUED=0
# Render the rules from the templates in rules/ (--no-look keeps the no-viewer blocks). workspace/AGENTS.md and
# workspace/BRIEF.md are the rendered copies; edit rules/*.md, not them. The brief then gets the data/batch numbers.
python3 scripts/render_rules.py rules/AGENTS.md workspace/AGENTS.md --look "$LOOK" --set VAL_BUDGET="$VAL_BUDGET" || exit 1
python3 scripts/render_rules.py rules/BRIEF.md workspace/BRIEF.md --look "$LOOK" --set VAL_BUDGET="$VAL_BUDGET" || exit 1
echo "rules rendered: viewer $([ "$LOOK" = 1 ] && echo enabled || echo "DISABLED (--no-look)")"
sed -e "s/{{N_TRAIN}}/$NTRAIN/g" -e "s/{{N_VAL}}/$NVAL/g" -e "s/{{BATCH}}/$BATCH/g" -e "s/{{VAL_BUDGET}}/$VAL_BUDGET/g" \
    workspace/BRIEF.md > "$OUT/brief_rendered.md"
if [ -n "$EXTRA" ]; then
  printf '\n%s\n' "$EXTRA" >> "$OUT/brief_rendered.md"
fi

commit_push "pre-run $MODE"

# Clean leftovers of earlier runs from the workspace root.
echo "cleaning workspace/ root:"
( cd workspace && find . -maxdepth 1 \( -type f \( -name classify.py -o -name NOTES.md \
    -o \( \( -name '*.py' -o -name '*.json' -o -name '*.csv' -o -name '*.txt' -o -name '*.md' \) \
         ! -name check.py ! -name look.py ! -name step.py ! -name agent.py ! -name AGENTS.md ! -name BRIEF.md \) -o -name '.teacher_used' -o -name '.batch_state.json' -o -name '.steps.jsonl' \) \
    -o \( -type d ! -name . ! -name data ! -name teacher ! -name '.agents' ! -name '.aws' ! -name '.codex' ! -name '.git' \) \) -print -exec rm -rf {} + | sed 's|^\./|  removed |' )

# Run config (read by check.py, step.py, look.py and agent.py), fresh state.
printf '{"batch_size": %s, "seed": 0, "viewer": %s, "loop": "%s", "val_budget": %s, "enforce_comments": true, "model": "%s"}\n' \
  "$BATCH" "$([ "$LOOK" = 1 ] && echo true || echo false)" "$LOOP" "$VAL_BUDGET" "$MODEL" > workspace/.batch_config.json
rm -f workspace/.batch_state.json workspace/.batch_state.lock workspace/.steps.jsonl workspace/.teacher_used \
      workspace/.val_passes.json workspace/.val_passes.lock
echo "batch config: workspace/.batch_config.json = batch_size $BATCH, seed 0, viewer $LOOK, loop $LOOP, val_budget $VAL_BUDGET, enforce_comments true"

start_monitor "$OUT"
BUDGET_BEFORE=$(budget_once); echo "budget before: $BUDGET_BEFORE"

START_EPOCH=$(date +%s)
START_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
S=0
: > "$OUT/events.jsonl"; : > "$OUT/codex_stderr.log"
fi   # end of the fresh-run setup

# Session loop: runs until all val passes are used (see the header). Every session is a fresh `codex exec`.
TOTAL_WALL_S=$(to_seconds "$TOTAL_WALL"); SESSION_WALL_S=$(to_seconds "$SESSION_WALL")
echo "session loop: val budget $VAL_BUDGET, session wall $SESSION_WALL, total wall $TOTAL_WALL, max sessions $MAX_SESSIONS$([ "$CONTINUED" = 1 ] && echo " (continuing after session $S)")"
RETRY=0; NOEV_STREAK=0; EXIT=0; STOP_REASON=""; PREV_EXIT=""
[ "$CONTINUED" = 1 ] && PREV_EXIT=124   # the earlier run was interrupted, so no "you finished with passes left" line
while :; do
  S=$((S + 1))
  NOW=$(date +%s); LOOP_ELAPSED=$((NOW - START_EPOCH)); USED=$(val_used)
  PF="$OUT/brief_session_$S.md"
  if [ "$RETRY" = 1 ]; then
    cp "$OUT/brief_session_$((S - 1)).md" "$PF"    # the previous session did not start: same prompt again
  elif [ "$S" = 1 ]; then
    cp "$OUT/brief_rendered.md" "$PF"
  else
    write_handoff_prompt "$S" "$PF" "$((S - 1))" "$USED" "$LOOP_ELAPSED" "$PREV_EXIT"
  fi
  RETRY=0
  EVS="$OUT/events_session_$S.jsonl"; ERRS="$OUT/codex_stderr_session_$S.log"; LASTS="$OUT/last_message_session_$S.md"
  SW=$SESSION_WALL_S; REMAIN=$((TOTAL_WALL_S - LOOP_ELAPSED)); [ "$REMAIN" -lt "$SW" ] && SW=$REMAIN
  [ "$SW" -lt 1 ] && SW=1
  echo "session $S start: wall ${SW}s, elapsed ${LOOP_ELAPSED}s of ${TOTAL_WALL_S}s, val passes used $USED of $VAL_BUDGET"
  S_START=$(date +%s)
  timeout --signal=INT --kill-after=60 "${SW}s" codex exec -C workspace -m "$MODEL" \
    -c model_reasoning_effort="$EFFORT" -s workspace-write --skip-git-repo-check --json \
    -o "$LASTS" "$(cat "$PF")" \
    > "$EVS" 2> "$ERRS"
  EXIT=$?
  S_END=$(date +%s); S_ELAPSED=$((S_END - S_START)); LOOP_ELAPSED=$((S_END - START_EPOCH)); USED=$(val_used)
  cat "$EVS" >> "$OUT/events.jsonl"; cat "$ERRS" >> "$OUT/codex_stderr.log"
  [ -f "$LASTS" ] && cp "$LASTS" "$OUT/last_message.md"
  PREV_EXIT=$EXIT
  echo "session $S end: exit $EXIT (124 = hit session wall), ${S_ELAPSED}s, elapsed ${LOOP_ELAPSED}s of ${TOTAL_WALL_S}s, val passes used $USED of $VAL_BUDGET"

  if monitor_stopped "$OUT"; then STOP_REASON="budget monitor stop"; break; fi
  if [ "$USED" -ge "$VAL_BUDGET" ]; then STOP_REASON="all val passes used"; break; fi
  if [ "$LOOP_ELAPSED" -ge "$TOTAL_WALL_S" ]; then STOP_REASON="total wall reached"; break; fi
  if [ "$S" -ge "$MAX_SESSIONS" ] && [ "$CONTINUED" = 0 ]; then STOP_REASON="max sessions reached"; break; fi
  if [ "$CONTINUED" = 1 ] && [ "$((S - PREV_SESSIONS))" -ge "$MAX_SESSIONS" ]; then STOP_REASON="max sessions reached (this continuation)"; break; fi
  if [ "$EXIT" -ne 0 ] && [ "$S_ELAPSED" -lt 30 ] && [ ! -s "$EVS" ]; then   # codex failed to start
    NOEV_STREAK=$((NOEV_STREAK + 1))
    if [ "$NOEV_STREAK" -ge 2 ]; then STOP_REASON="codex failed to start twice in a row"; break; fi
    echo "session $S exited $EXIT within 30 s with no events; retrying once with the same prompt"
    RETRY=1
  else
    NOEV_STREAK=0
  fi
done
echo "session loop stopped: $STOP_REASON ($S session(s))"
SESSIONS=$S
[ -f workspace/.val_passes.json ] && cp workspace/.val_passes.json "$OUT/val_passes.json"
VAL_USED=$(val_used)

END_EPOCH=$(date +%s)
END_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ELAPSED=$((END_EPOCH - START_EPOCH))
BUDGET_AFTER=$(budget_once); echo "budget after: $BUDGET_AFTER"
stop_monitor
copy_subagent_log "$OUT"
restore_pilot_subset "$ROOT/workspace" "$ROOT/.pilot_stash"   # before the post-run commit (the trap is a safety net)

# thread id of the last session that has one
SESSION_ID=""
for ((i = SESSIONS; i >= 1; i--)); do
  SESSION_ID=$(extract_session_id "$OUT/events_session_$i.jsonl"); [ -n "$SESSION_ID" ] && break
done
[ -n "$SESSION_ID" ] && printf '%s\n' "$SESSION_ID" > "$OUT/session_id" || echo "WARNING: no thread.started event found in $OUT/events*.jsonl" >&2

cat > "$OUT/run_meta.json" <<JSON
{
  "mode": "$MODE",
  "model": "$MODEL",
  "effort": "$EFFORT",
  "wall_limit": "$TOTAL_WALL",
  "val_budget": $VAL_BUDGET,
  "val_passes_used": $VAL_USED,
  "sessions": $SESSIONS,
  "session_wall": "$SESSION_WALL",
  "total_wall": "$TOTAL_WALL",
  "stop_reason": $(json_str "$STOP_REASON"),
  "views": "$([ "$LOOK" = 1 ] && echo 'viewer enabled' || echo 'no viewer (--no-look): look.py disabled')",
  "viewer": $([ "$LOOK" = 1 ] && echo true || echo false),
  "n_train": $NTRAIN,
  "n_val": $NVAL,
  "batch": $BATCH,
  "loop": "$LOOP",
  "start": "$START_ISO",
  "end": "$END_ISO",
  "elapsed_seconds": $ELAPSED,
  "exit_code": $EXIT,
  "session_id": "$SESSION_ID",
  "budget_before": $(json_str "$BUDGET_BEFORE"),
  "budget_after": $(json_str "$BUDGET_AFTER")
}
JSON

commit_push "post-$([ "$CONTINUED" = 1 ] && echo continue || echo run) $MODE"

echo "== run_driver summary =="
echo "mode: $MODE  model: $MODEL  effort: $EFFORT  batch: $BATCH  loop: $LOOP  viewer: $LOOK"
echo "exit code: $EXIT (124 = hit wall-clock limit; of the last session)  elapsed: ${ELAPSED}s"
echo "val budget: $VAL_BUDGET  val passes used: $VAL_USED  sessions: $SESSIONS (max $MAX_SESSIONS)  session wall: $SESSION_WALL  total wall: $TOTAL_WALL  stopped: $STOP_REASON"
echo "events: $(wc -l < "$OUT/events.jsonl") lines  classify.py: $( [ -f workspace/classify.py ] && stat -c %s workspace/classify.py || echo missing ) bytes"
echo "outputs in $OUT/"
exit 0
