#!/usr/bin/env python3
"""Sub-agent daemon: serves workspace/.subagents/req/*.json by running a fresh `codex exec` session in the workspace
(workspace-write sandbox, no network, same flags as the driver's main agent) and writing the final report to
workspace/.subagents/res/<id>.json. Run from the repo root, outside the agents' sandbox:
    mkdir -p out && nohup python3 scripts/subagent_daemon.py >> out/subagent.stdout 2>&1 &
Logs (events, stderr, last message) go to out/subagents/<id>/ (or out/subagents/<run>/<id>/ with --run TAG), outside
the workspace so that the agents cannot tamper with them. Requests are claimed by renaming req/<id>.json to
req/<id>.claimed, and at most --workers run at once (others stay unclaimed in req/)."""
import argparse, json, os, re, subprocess, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = ("You are a sub-agent started by the orchestrating agent in this workspace. AGENTS.md is binding for you too. "
          "Do only the task below, then end with a report of what you did, what you found, which files you changed, "
          "and anything the orchestrator should know. Do not run agent.py yourself: only the orchestrator delegates. Task:")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MODEL_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
EFFORTS = {"low", "medium", "high", "xhigh"}
LOCK = threading.Lock()


def parse_duration(v, default):
    try:
        v = str(v).strip().lower()
        mult = {"s": 1, "m": 60, "h": 3600}
        return int(float(v[:-1]) * mult[v[-1]]) if v and v[-1] in mult else int(float(v))
    except (ValueError, KeyError):
        return default


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def append(path, obj):
    with LOCK:
        with open(path, "a") as f:
            f.write(json.dumps(obj) + "\n")


def run_one(a, claimed, rid, req):
    t0 = time.time()
    wall = max(30, min(parse_duration(req.get("wall", "10m"), 600), a.max_wall))
    model = str(req.get("model") or a.model)
    effort = str(req.get("effort") or "high")
    task = str(req.get("task") or "")
    err, text, code = None, "", None
    outdir = os.path.join(a.outbase, rid)
    events = os.path.join(outdir, "events.jsonl")
    print("[subagent] start %s wall=%ds model=%s effort=%s" % (rid, wall, model, effort), flush=True)
    append(a.wlog, {"id": rid, "status": "running", "ts": time.time()})
    try:
        if not task.strip():
            raise ValueError("empty task")
        if not MODEL_RE.match(model):
            raise ValueError("bad model name")
        if effort not in EFFORTS:
            raise ValueError("bad effort (use low/medium/high/xhigh)")
        os.makedirs(outdir, exist_ok=True)
        last = os.path.join(outdir, "last_message.md")
        try:
            os.remove(last)
        except OSError:
            pass
        cmd = ["timeout", "--signal=INT", "--kill-after=60", str(wall), "codex", "exec", "-C", a.workspace,
               "-m", model, "-c", 'model_reasoning_effort="%s"' % effort, "--sandbox", "workspace-write",
               "--skip-git-repo-check", "--json", "-o", last, HEADER + "\n\n" + task]
        with open(events, "w") as ev, open(os.path.join(outdir, "stderr.log"), "w") as er:
            code = subprocess.run(cmd, cwd=a.workspace, stdin=subprocess.DEVNULL, stdout=ev, stderr=er).returncode
        try:
            text = open(last, encoding="utf-8", errors="replace").read().strip()
        except OSError:
            text = ""
        if code == 124:
            err = "sub-agent hit its wall-clock limit (%ds)" % wall
        elif code != 0:
            err = "codex exited with code %s" % code
        if not text and not err:
            err = "sub-agent produced no final message"
    except Exception as e:
        traceback.print_exc()
        err = "daemon error: %r" % e
    lat = round(time.time() - t0, 2)
    try:
        write_json(os.path.join(a.resd, rid + ".json"),
                   {"id": rid, "text": text, "error": err, "exit": code, "latency_s": lat, "events": events})
        os.remove(claimed)
    except Exception:
        traceback.print_exc()
    try:
        append(a.log, {"id": rid, "task": task[:200], "wall": wall, "model": model, "effort": effort, "exit": code,
                       "latency_s": lat, "events": events, "text_len": len(text), "error": err, "ts": time.time()})
        append(a.wlog, {"id": rid, "status": "error" if err else "done", "ts": time.time()})
    except Exception:
        traceback.print_exc()
    print("[subagent] done %s exit=%s %ss text=%d chars%s" % (rid, code, lat, len(text),
          "  [%s]" % err[:80] if err else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=os.path.join(ROOT, "workspace"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--run", default="", help="optional tag: logs go to out/subagents/<run>/<id>/ (default out/subagents/<id>/)")
    ap.add_argument("--model", default="gpt-6.1-sol", help="default model for requests that name none")
    ap.add_argument("--max-wall", default="30m", help="requests above this wall are clamped (default 30m)")
    ap.add_argument("--poll", type=float, default=0.5)
    a = ap.parse_args()
    a.workspace = os.path.abspath(a.workspace)
    a.max_wall = parse_duration(a.max_wall, 1800)
    a.outbase = os.path.join(ROOT, "out", "subagents", a.run) if a.run else os.path.join(ROOT, "out", "subagents")
    a.log = os.path.join(ROOT, "out", "subagents.jsonl")
    sd = os.path.join(a.workspace, ".subagents")
    reqd, a.resd, a.wlog = os.path.join(sd, "req"), os.path.join(sd, "res"), os.path.join(sd, "log.jsonl")
    for d in (reqd, a.resd, a.outbase):
        os.makedirs(d, exist_ok=True)
    # leftover claims: requeue if younger than 300 s, else drop
    for f in os.listdir(reqd):
        if f.endswith(".claimed"):
            p = os.path.join(reqd, f)
            try:
                if time.time() - os.path.getmtime(p) < 300:
                    os.rename(p, p[:-len(".claimed")] + ".json")
                else:
                    os.remove(p)
            except OSError:
                pass
    print("[subagent] started pid=%d workspace=%s workers=%d max_wall=%ds out=%s"
          % (os.getpid(), a.workspace, a.workers, a.max_wall, a.outbase), flush=True)
    slots = threading.Semaphore(a.workers)
    ex = ThreadPoolExecutor(max_workers=a.workers)

    def job(claimed, rid, req):
        try:
            run_one(a, claimed, rid, req)
        finally:
            slots.release()

    while True:
        if not slots.acquire(blocking=False):
            time.sleep(a.poll)
            continue
        started = False
        try:
            os.makedirs(reqd, exist_ok=True); os.makedirs(a.resd, exist_ok=True)   # the driver's cleanup may remove the queue dir between runs
            files = sorted((os.path.join(reqd, f) for f in os.listdir(reqd) if f.endswith(".json")),
                           key=os.path.getmtime)
            for fp in files:
                rid = os.path.basename(fp)[:-5]
                claimed = fp[:-5] + ".claimed"
                try:
                    os.rename(fp, claimed)  # atomic claim
                except OSError:
                    continue
                try:
                    req = json.load(open(claimed))
                    assert isinstance(req, dict)
                except Exception:
                    req = {}
                if not ID_RE.match(rid):
                    print("[subagent] ignored request with a bad id", flush=True)
                    os.remove(claimed)
                    continue
                ex.submit(job, claimed, rid, req)
                started = True
                break
        except Exception:
            traceback.print_exc()
        if not started:
            slots.release()
            time.sleep(a.poll)


if __name__ == "__main__":
    main()
