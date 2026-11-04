#!/usr/bin/env python3
"""Run your classifier on the NEXT training batch (SGD-like loop).

Usage: python3 step.py classify.py      run the program on the next batch of the current epoch
       python3 step.py --status         print the state summary (runs nothing)
       python3 step.py --reset          delete .batch_state.json (start over with a fresh epoch plan)
       python3 step.py classify.py --batch-size N
                                        use batches of N images (overrides the config's batch_size). On a fresh
                                        plan (first call, after --reset, or after a finished epoch) N applies at
                                        once; in the middle of an epoch it applies from the next epoch. The size
                                        sticks until you change it again.

Every call consumes one batch. The plan of an epoch is a stratified shuffle of all data/train images, dealt into
batches of the configured size (.batch_config.json, written by the driver). After the last batch of an epoch
(epoch_done) the next call starts a new epoch with a new shuffle. look.py shows only the images of the current
batch, each once; check.py --images data/val is allowed only after the last batch of an epoch (these two rules apply
only in loop mode "fixed" of .batch_config.json; in the default mode "free" look.py shows any train image and
val may be measured at any time).
The same rule checks and the same isolated, timed runner as check.py are used.
"""
import fcntl
import json
import os
import random
import sys
import tempfile
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check  # noqa: E402  (same directory; the scorer)

CONFIG = os.path.join(HERE, ".batch_config.json")
STATE = os.path.join(HERE, ".batch_state.json")
LOCK = os.path.join(HERE, ".batch_state.lock")
STEPS = os.path.join(HERE, ".steps.jsonl")
TRAIN = os.path.join(HERE, "data", "train")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_state(st):
    """Atomic write (tmp + os.replace), under the lock shared with look.py."""
    with open(LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        tmp = STATE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, STATE)


def rel(p):
    return os.path.relpath(p, HERE)


def make_plan(batch_size, seed, epoch):
    """Stratified batches: shuffle with Random(seed+epoch), then round-robin over the classes (class order
    reshuffled each round) so every batch has roughly equal class counts. Returns list of lists of relative paths."""
    rng = random.Random(seed + epoch)
    per_class = {}
    for c in sorted(os.listdir(TRAIN)):
        d = os.path.join(TRAIN, c)
        if os.path.isdir(d):
            fs = [os.path.join("data", "train", c, f) for f in sorted(os.listdir(d))
                  if os.path.splitext(f)[1].lower() in check.EXTS]
            rng.shuffle(fs)
            per_class[c] = fs
    seq, i = [], 0
    while any(i < len(v) for v in per_class.values()):
        cs = [c for c in per_class if i < len(per_class[c])]
        rng.shuffle(cs)
        seq += [per_class[c][i] for c in cs]
        i += 1
    return [seq[j:j + batch_size] for j in range(0, len(seq), batch_size)]


def true_class(relpath):
    return relpath.replace("\\", "/").split("/")[-2]


def summary(st):
    res = st["results"]
    A, N = sum(r["correct"] for r in res), sum(r["n"] for r in res)
    fixed = check.loop_mode() == "fixed"
    return ("epoch %d  batch %d/%d done  (%s)  epoch so far %d/%d%s"
            % (st["epoch"], st["batch"], st["n_batches"],
               ("epoch finished" + ("; val check allowed" if fixed else "")) if st["epoch_done"] else "next: step.py", A, N, "" if not N else " = %.3f" % (A / N)))


def status():
    st = load_json(STATE)
    if not st:
        print("no state yet: run step.py classify.py")
        return 0
    print(summary(st))
    print("current batch: %d images, %d viewed with look.py" % (len(st["current"]), len(st["viewed"])))
    for r in st["results"]:
        print("  batch %d: %d/%d  %.1f ms/img" % (r["batch"], r["correct"], r["n"], r["ms"]))
    for h in st["history"]:
        print("  epoch %d: train %d/%d" % (h["epoch"], h["train_correct"], h["train_n"]))
    return 0


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    if a[0] == "--status":
        return status()
    if a[0] == "--reset":
        try:
            os.remove(STATE)
            print("state deleted")
        except OSError:
            print("no state to delete")
        return 0
    new_size = None
    if "--batch-size" in a:
        i = a.index("--batch-size")
        try:
            new_size = int(a[i + 1])
            assert new_size > 0
        except (IndexError, ValueError, AssertionError):
            print("ERROR: --batch-size needs a positive integer")
            return 2
        a = a[:i] + a[i + 2:]
    if not a:
        print(__doc__)
        return 2
    prog = a[0]
    cfg = load_json(CONFIG)
    if not cfg:
        print("ERROR: .batch_config.json missing or unreadable")
        return 1
    batch_size, seed = int(cfg["batch_size"]), int(cfg.get("seed", 0))

    try:
        size = check.check_rules(prog)
    except check.Rejected as e:
        print(f"REJECTED: {e}")
        return 2
    except OSError as e:
        print(f"REJECTED: cannot read {prog}: {e}")
        return 2
    rej = check.comment_rejection(prog)   # same comment rule as check.py; no batch is consumed
    if rej:
        print("\n".join(rej))
        return 2

    st = load_json(STATE)
    if st is None or not st.get("batches"):
        batch_size = new_size or batch_size
        st = {"epoch": 1, "batch": 0, "n_batches": 0, "batches": make_plan(batch_size, seed, 1), "current": [],
              "viewed": [], "results": [], "epoch_done": False, "history": [], "batch_size": batch_size}
        st["n_batches"] = len(st["batches"])
    elif st["epoch_done"]:
        batch_size = new_size or st.get("next_batch_size") or st.get("batch_size") or batch_size
        st["batch_size"] = batch_size
        st.pop("next_batch_size", None)
        st["history"].append({"epoch": st["epoch"], "train_correct": sum(r["correct"] for r in st["results"]),
                              "train_n": sum(r["n"] for r in st["results"])})
        st["epoch"] += 1
        st["batches"] = make_plan(batch_size, seed, st["epoch"])
        st.update(batch=0, n_batches=len(st["batches"]), results=[], epoch_done=False, current=[], viewed=[])
    if not st["batches"]:
        print("ERROR: no training images found under data/train")
        return 1
    if new_size and new_size != st.get("batch_size", batch_size):
        st["next_batch_size"] = new_size
        print("NOTE: an epoch is in progress (batch size %d); batch size %d applies from the next epoch "
              "(or run step.py --reset first to restart with it now)" % (st.get("batch_size", batch_size), new_size))

    k = st["batch"] + 1
    cur = st["batches"][k - 1]
    images = [(os.path.join(HERE, p), true_class(p)) for p in cur]

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    tmpdir = tempfile.mkdtemp(prefix="step_iso_")
    try:
        shutil.copyfile(prog, os.path.join(tmpdir, "classify.py"))
        try:
            res = check.run_all(tmpdir, images, check.DEFAULT_TIMEOUT, min(8, os.cpu_count() or 1))
        except check.Rejected as e:
            print(f"REJECTED: {e}")
            return 2
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    n = len(cur)
    print("epoch %d  batch %d/%d  (%d images)" % (st["epoch"], k, st["n_batches"], n))
    correct, n_err, n_to, ms_ok, per_image = 0, 0, 0, [], []
    for p, (path, true), (pred, ms, err) in zip(cur, images, res):
        if err is None and pred not in check.CLASSES:
            err = f"invalid label: {pred!r}"[:200]
        ok = err is None and pred == true
        correct += ok
        if err is not None:
            n_err += 1
            n_to += err.startswith("timeout")
        else:
            ms_ok.append(ms)
        print("%s  %s  true=%s  pred=%s  %.0fms%s" % ("OK" if ok else "XX", p, true, pred if err is None else "-",
                                                      ms, "" if err is None else "  [%s]" % err))
        per_image.append({"path": p, "true": true, "pred": pred if err is None else None, "ms": round(ms, 1),
                          "error": err})
    mean_ms = sum(ms_ok) / len(ms_ok) if ms_ok else 0.0
    st["batch"] = k
    st["current"] = list(cur)
    st["viewed"] = []
    st["results"].append({"batch": k, "correct": correct, "n": n, "ms": round(mean_ms, 1)})
    st["epoch_done"] = k == st["n_batches"]
    A, N = sum(r["correct"] for r in st["results"]), sum(r["n"] for r in st["results"])
    print("batch accuracy %d/%d" % (correct, n))
    print("epoch so far %d/%d" % (A, N))
    print("mean ms/img %.1f (excl. timeouts)" % mean_ms)
    if n_err:
        print("WARNING: %d image(s) errored or timed out (%d timeouts, limit %.1fs per image); they count as wrong"
              % (n_err, n_to, check.DEFAULT_TIMEOUT))
    save_state(st)
    with open(STEPS, "a") as f:
        f.write(json.dumps({"ts": time.time(), "epoch": st["epoch"], "batch": k, "n_batches": st["n_batches"],
                            "program": os.path.abspath(prog), "program_bytes": size, "correct": correct, "n": n,
                            "mean_ms": round(mean_ms, 1), "images": per_image}) + "\n")
    if check.loop_mode() == "fixed":
        print("look.py accepts these %d images once each until the next step; val check is allowed after batch %d/%d"
              % (n, st["n_batches"], st["n_batches"]))
    elif st["epoch_done"]:
        print("epoch %d finished; the next call starts a new epoch" % st["epoch"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
