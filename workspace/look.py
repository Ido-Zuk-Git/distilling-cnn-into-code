#!/usr/bin/env python3
"""Ask the viewer service to describe an image.
Usage: python3 look.py <image> ["question"]   |   python3 look.py --batch <image1> <image2> ... ["question"]
In loop mode "fixed" (.batch_config.json) it shows only images of the current step.py batch, each once per batch
(run step.py first); in loop mode "free" (the default) any data/train image can be viewed. Val images never can."""
import fcntl, json, os, sys, time, uuid

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".looker")
REQ, RES = os.path.join(D, "req"), os.path.join(D, "res")
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, ".batch_state.json")
LOCK = os.path.join(HERE, ".batch_state.lock")


def load_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def mark_viewed(path):
    """Append path to state['viewed'] (re-read under a lock, atomic rewrite)."""
    with open(LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = load_state()
        if st is None:
            return
        if path not in st["viewed"]:
            st["viewed"].append(path)
        tmp = STATE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, STATE)
TIMEOUT = 300


def loop_mode():
    """"fixed" or "free" from .batch_config.json (missing key or file: "free")."""
    try:
        with open(os.path.join(HERE, ".batch_config.json")) as f:
            return "fixed" if json.load(f).get("loop", "free") == "fixed" else "free"
    except (OSError, ValueError, AttributeError):
        return "free"


def look(path, question=None):
    os.makedirs(REQ, exist_ok=True); os.makedirs(RES, exist_ok=True)
    rid = uuid.uuid4().hex
    body = {"image": os.path.abspath(path), "question": question, "ts": time.time()}
    tmp = os.path.join(REQ, rid + ".tmp")
    with open(tmp, "w") as f:
        json.dump(body, f)
    os.rename(tmp, os.path.join(REQ, rid + ".json"))  # atomic publish
    res = os.path.join(RES, rid + ".json")
    end = time.time() + TIMEOUT
    while time.time() < end:
        if os.path.exists(res):
            try:
                with open(res) as f:
                    r = json.load(f)
            except ValueError:
                time.sleep(0.2); continue
            try: os.remove(res)
            except OSError: pass
            return r.get("text") or ("ERROR: " + str(r.get("error")))
        time.sleep(0.5)
    try: os.remove(os.path.join(REQ, rid + ".json"))
    except OSError: pass
    return None


def main():
    a = sys.argv[1:]
    # No-viewer runs (.batch_config.json has "viewer": false): nothing may describe an image for the agent.
    try:
        with open(os.path.join(HERE, ".batch_config.json")) as f:
            if json.load(f).get("viewer", True) is False:
                print("ERROR: there is no image viewer in this run; inspect the pixels with your own code and use the teacher.")
                sys.exit(1)
    except (OSError, ValueError):
        pass
    # The teacher comes first: refuse to show any image until the teacher CNN has been inspected (it leaves this marker).
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".teacher_used")):
        print("ERROR: inspect the teacher first (e.g. python3 teacher/teacher.py layers / weights / activations); look.py works only after the teacher has been used.")
        sys.exit(1)
    if not a or a[0] in ("-h", "--help"):
        print(__doc__); sys.exit(2)
    if a[0] == "--batch":
        rest = a[1:]
        q = None
        if rest and not os.path.isfile(rest[-1]):
            q, rest = rest[-1], rest[:-1]
        paths = rest
        if not paths:
            print("ERROR: --batch needs at least one existing image path"); sys.exit(2)
    else:
        paths, q = [a[0]], (a[1] if len(a) > 1 else None)
    # Val images can never be viewed (checked before anything is sent, on the given and the resolved path).
    vsep = os.path.join("data", "val")
    for p in paths:
        for q_ in (p.replace("\\", "/"), os.path.abspath(p), os.path.realpath(p)):
            q_ = q_.replace("\\", "/")
            if "data/val" in q_ or vsep in q_:
                print("ERROR: val images cannot be viewed"); sys.exit(1)
    free = loop_mode() == "free"
    # Batch gate (fixed mode only): all checks first, so a call is all-or-nothing on validation.
    if not free:
        st = load_state()
        if st is None:
            print("ERROR: run step.py first; look.py only shows images of the current batch"); sys.exit(1)
        cur = {os.path.realpath(os.path.join(HERE, c)) for c in st.get("current", [])}
        viewed = {os.path.realpath(os.path.join(HERE, c)) for c in st.get("viewed", [])}
        seen = set()
        for p in paths:
            rp = os.path.realpath(os.path.abspath(p))
            if rp not in cur:
                print("ERROR: %s is not in the current batch (epoch %s batch %s); look.py shows only the current batch's images"
                      % (p, st.get("epoch"), st.get("batch"))); sys.exit(1)
            if rp in viewed or rp in seen:
                print("ERROR: %s was already viewed in this batch; each image can be viewed once per batch" % p); sys.exit(1)
            seen.add(rp)
    # Submit all requests at once (the viewer answers in parallel), then print in the order given.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(16, len(paths))) as ex:
        texts = list(ex.map(lambda p: look(p, q), paths))
    for p, t in zip(paths, texts):
        if t is None:
            print("ERROR: timed out after %ds waiting for the viewer; the viewer service may be down." % TIMEOUT)
            sys.exit(1)
        if a[0] == "--batch":
            print("### " + p)
        print(t)
        if not free and not t.startswith("ERROR:"):
            mark_viewed(os.path.relpath(os.path.realpath(os.path.abspath(p)), os.path.realpath(HERE)))


if __name__ == "__main__":
    main()
