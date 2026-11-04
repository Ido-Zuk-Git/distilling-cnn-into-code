#!/usr/bin/env python3
"""Start a sub-agent (a fresh Codex session in this workspace) and print its final report.

Usage: python3 agent.py "task text" [--wall 10m] [--effort high] [--model NAME]  (default: the run's model)
       python3 agent.py --list        ids and status of past/running sub-agents

The task text is one argument (may be long and multi-line). The sub-agent has the same workspace, AGENTS.md and
sandbox as you, but none of your conversation: put everything it needs into the task. --wall is the sub-agent's
time limit (e.g. 90s, 10m; default 10m, maximum 30m). The call blocks until the report arrives (exit 0) or fails
(prints ERROR: ..., exit 1). Several sub-agents can run in parallel (run agent.py calls in the background).
"""
import argparse, json, os, sys, time, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, ".subagents")
REQ, RES, LOG = os.path.join(D, "req"), os.path.join(D, "res"), os.path.join(D, "log.jsonl")
MAX_WALL = 1800


def parse_duration(v):
    v = str(v).strip().lower()
    mult = {"s": 1, "m": 60, "h": 3600}
    try:
        return int(float(v[:-1]) * mult[v[-1]]) if v and v[-1] in mult else int(float(v))
    except (ValueError, KeyError):
        raise argparse.ArgumentTypeError("bad duration %r (use e.g. 90s, 10m)" % v)


def list_agents():
    last = {}
    try:
        for line in open(LOG):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            last[r.get("id")] = r
    except OSError:
        pass
    if not last:
        print("no sub-agents yet")
        return 0
    for rid, r in last.items():
        print("%s  %-8s %s" % (rid, r.get("status"), time.strftime("%H:%M:%S", time.localtime(r.get("ts", 0)))))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?")
    ap.add_argument("--wall", type=parse_duration, default=600)
    ap.add_argument("--effort", default="high")
    # Default model = the run's model (written by the driver into .batch_config.json), so sub-agents match the orchestrator.
    try:
        _default_model = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".batch_config.json"))).get("model") or "gpt-6.1-sol"
    except (OSError, ValueError):
        _default_model = "gpt-6.1-sol"
    ap.add_argument("--model", default=_default_model)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        return list_agents()
    if not a.task or not a.task.strip():
        ap.print_usage(sys.stderr)
        print("ERROR: a task text is required", file=sys.stderr)
        return 2
    wall = max(30, min(a.wall, MAX_WALL))
    os.makedirs(REQ, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    rid = time.strftime("sa_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    tmp = os.path.join(REQ, rid + ".tmp")
    with open(tmp, "w") as f:
        json.dump({"id": rid, "task": a.task, "wall": "%ds" % wall, "effort": a.effort, "model": a.model,
                   "cwd": HERE, "ts": time.time()}, f)
    os.rename(tmp, os.path.join(REQ, rid + ".json"))  # atomic publish
    print("sub-agent %s started (wall %ds)" % (rid, wall), file=sys.stderr, flush=True)
    res = os.path.join(RES, rid + ".json")
    end = time.time() + wall + 60
    while time.time() < end:
        if os.path.exists(res):
            try:
                with open(res) as f:
                    r = json.load(f)
            except ValueError:
                time.sleep(0.2)
                continue
            try:
                os.remove(res)
            except OSError:
                pass
            text, err = (r.get("text") or "").strip(), r.get("error")
            if text:
                print(text)
                if err:
                    print("WARNING: sub-agent ended abnormally: %s" % err, file=sys.stderr)
                return 0
            print("ERROR: %s" % (err or "sub-agent returned no report"))
            return 1
        time.sleep(0.5)
    try:
        os.remove(os.path.join(REQ, rid + ".json"))  # still unclaimed: withdraw it
    except OSError:
        pass
    print("ERROR: no answer within %ds; the sub-agent daemon may be down (sub-agent %s)" % (wall + 60, rid))
    return 1


if __name__ == "__main__":
    sys.exit(main())
