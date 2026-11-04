# NOTE: scripts/score.py is mirrored byte-for-byte to workspace/check.py; edit score.py, then `cp` it so both stay in sync.
"""Score a single-file image classifier (classify.py) on a directory of images.

Usage: score.py <classify.py> --images <dir> [--out results.json] [--preds preds.csv]
                [--timeout 1] [--workers N] [--viewed-list PATH] [--no-exclude]

On a directory named `val`, images that the agent has viewed with look.py (listed in
<project root>/out/looker_viewed_val.json) are excluded from the score.

Images: <dir>/<class_name>/*.{JPEG,jpg,png}. Exit 0 = scored, 2 = REJECTED (rule violation).

Val gate (workspace copy only): when --images points into data/val and a .batch_config.json sits next to this
script (i.e. this is workspace/check.py, not scripts/score.py), val is scored only after the current epoch of
step.py is finished (.batch_state.json next to the script has "epoch_done": true). In "free" loop mode
(.batch_config.json "loop": "free", the default when the key is missing) there is no epoch gate.
In the workspace copy a val run prints only the accuracy line, the errors line and any rule/comment report (no
per-class accuracy, no confusion matrix, no per-image data); --out and --preds are refused for val there.

Val pass budget (workspace copy only): when .batch_config.json has "val_budget": B, every successfully scored run on
data/val is counted in .val_passes.json (under an flock on .val_passes.lock, held for the whole val run so parallel
sub-agent runs cannot overshoot); once B passes are used, val runs are refused. A REJECTED run (rules or comments),
a refusal or a crash costs nothing.

Comment enforcement (workspace copy only): with "enforce_comments": true in .batch_config.json a program that has
any numeric line without a nearby comment is REJECTED (exit 2) before anything is scored; step.py applies the same
check through comment_rejection().
"""
import argparse
import ast
import csv
import datetime
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import tokenize
import traceback
from collections import deque
from pathlib import Path
from multiprocessing import connection, get_context

CLASSES = ["tench", "english_springer", "cassette_player", "chain_saw", "church",
           "french_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute"]
ALLOWED_THIRD_PARTY = {"numpy", "PIL", "scipy", "cv2"}
BANNED_MODULES = {"subprocess", "socket", "urllib", "requests", "importlib",
                  # loaders / serialisers of stored parameters or arrays
                  "pickle", "shelve", "marshal", "torch", "torchvision", "h5py", "joblib"}
# the teacher CNN workspace module (run 3): classify.py may not import it, directly or by any path
TEACHER_MODULES = {"teacher"}
# attributes banned on a numpy/scipy alias chain (np.load(...), numpy.lib.npyio.load(...), scipy.io.loadmat(...)):
# they read parameters or arrays from files
BANNED_NP_ATTRS = {"load", "fromfile", "loadtxt", "genfromtxt", "memmap", "loadmat"}
# names / attributes of pathlib objects that read files
PATHLIB_NAMES = {"Path", "PurePath", "PosixPath", "WindowsPath", "PurePosixPath", "PureWindowsPath"}
BANNED_PATHLIB_ATTRS = {"read_bytes", "read_text", "open"}
BANNED_NAMES = {"__import__", "importlib", "eval", "exec"}
BANNED_OS_ATTRS = {"system", "popen", "execv", "execve", "execvp", "execl", "execlp",
                   "spawnl", "spawnv", "spawnlp", "spawnvp", "posix_spawn", "startfile"}
# OpenCV names that load parameters/models/files or read data from outside the image passed to predict().
BANNED_CV2 = {"dnn", "dnn_superres", "ml", "face", "CascadeClassifier",
              "HOGDescriptor_getDefaultPeopleDetector", "HOGDescriptor_getDaimlerPeopleDetector",
              "getDefaultPeopleDetector", "getDaimlerPeopleDetector", "setSVMDetector",
              "FileStorage", "readNet", "readNetFromONNX", "readNetFromTensorflow",
              "readNetFromCaffe", "readNetFromTorch", "readNetFromDarknet",
              "imread", "imreadmulti", "VideoCapture"}
BANNED_CV2_CALLS = {"load", "read", "restore"}   # banned only when called on a cv2.* chain
EXTS = {".jpeg", ".jpg", ".png"}
IMPORT_TIMEOUT = 60.0
DEFAULT_TIMEOUT = 1.0   # seconds per image: the agent's programs must be fast


class Rejected(Exception):
    pass


# ----------------------------------------------------------------------------- rules
def _cv2_banned(name):
    return name in BANNED_CV2 or name.startswith("readNet") or re.search(r"_load$", name) is not None


def _chain(node):
    """Attribute chain of `node` as (root_name_or_None, [attrs...]); looks through calls and subscripts,
    so cv2.CascadeClassifier(x).load(y) yields ('cv2', ['CascadeClassifier', 'load'])."""
    attrs = []
    while True:
        if isinstance(node, ast.Attribute):
            attrs.append(node.attr)
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Subscript):
            node = node.value
        else:
            break
    return (node.id if isinstance(node, ast.Name) else None), attrs[::-1]


def _chain_path(node):
    """Like _chain, but also looks through binary operators (Path(x) / "a") and keeps the root as a Name."""
    while True:
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.BinOp):
            node = node.left
        else:
            break
    return node.id if isinstance(node, ast.Name) else None


def check_rules(path):
    size = os.path.getsize(path)
    try:
        src = open(path, "rb").read()
        tree = ast.parse(src, filename=path)
    except (SyntaxError, ValueError) as e:
        raise Rejected(f"cannot parse: {e}")
    stdlib = set(sys.stdlib_module_names)

    def check_mod(name, lineno):
        top = name.split(".")[0]
        if top in TEACHER_MODULES:
            raise Rejected(f"line {lineno}: import of '{top}' is not allowed: classify.py must not import or load "
                           f"the teacher CNN (or anything derived from it)")
        if top in BANNED_MODULES:
            raise Rejected(f"line {lineno}: banned module '{top}'")
        if top not in stdlib and top not in ALLOWED_THIRD_PARTY:
            raise Rejected(f"line {lineno}: import of '{top}' not allowed (only numpy, PIL, scipy, cv2, stdlib)")

    # names bound to the cv2 module (import cv2 / import cv2 as X)
    cv2_names = {"cv2"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "cv2":
                    cv2_names.add(a.asname or "cv2")

    # names bound to numpy/scipy (import numpy as np / from numpy import lib / import scipy.io as sio) and to
    # pathlib objects (import pathlib / from pathlib import Path / p = Path(...) / q = p / "x")
    np_names, pl_names = set(), set(PATHLIB_NAMES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top in ("numpy", "scipy"):
                    np_names.add(a.asname or top)
                if top == "pathlib":
                    pl_names.add(a.asname or top)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            for a in node.names:
                if top in ("numpy", "scipy"):
                    np_names.add(a.asname or a.name)
                if top == "pathlib":
                    pl_names.add(a.asname or a.name)
    changed = True
    while changed:  # variables assigned from a pathlib expression are pathlib objects too
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                if _chain_path(node.value) in pl_names:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for t in targets:
                        if isinstance(t, ast.Name) and t.id not in pl_names:
                            pl_names.add(t.id)
                            changed = True

    for node in ast.walk(tree):
        ln = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for a in node.names:
                check_mod(a.name, ln)
                if a.name.split(".")[0] == "cv2":
                    for part in a.name.split(".")[1:]:
                        if _cv2_banned(part):
                            raise Rejected(f"line {ln}: cv2.{part} is banned (no pretrained/loaded models)")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise Rejected(f"line {ln}: relative import not allowed")
            check_mod(node.module or "", ln)
            if (node.module or "").split(".")[0] == "cv2":
                for part in (node.module or "").split(".")[1:] + [a.name for a in node.names]:
                    if _cv2_banned(part):
                        raise Rejected(f"line {ln}: cv2.{part} is banned (no pretrained/loaded models)")
            if node.module == "os":
                for a in node.names:
                    if a.name in BANNED_OS_ATTRS:
                        raise Rejected(f"line {ln}: os.{a.name} not allowed")
            if (node.module or "").split(".")[0] in ("numpy", "scipy"):
                for a in node.names:
                    if a.name in BANNED_NP_ATTRS:
                        raise Rejected(f"line {ln}: {node.module}.{a.name} is banned (no loading of parameters/files)")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise Rejected(f"line {ln}: use of '{node.id}' not allowed")
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "os" and node.attr in BANNED_OS_ATTRS:
                raise Rejected(f"line {ln}: os.{node.attr} not allowed")
            if node.attr in ("__import__", "import_module"):
                raise Rejected(f"line {ln}: dynamic import '{node.attr}' not allowed")
            root, attrs = _chain(node)
            if root in np_names:
                for a in attrs:
                    if a in BANNED_NP_ATTRS:
                        raise Rejected(f"line {ln}: .{a} on numpy/scipy is banned (no loading of parameters/files)")
            if node.attr in BANNED_PATHLIB_ATTRS and _chain_path(node) in pl_names:
                raise Rejected(f"line {ln}: pathlib .{node.attr} is banned (predict may read only the image passed in)")
            if root in cv2_names:
                for a in attrs:
                    if _cv2_banned(a):
                        raise Rejected(f"line {ln}: cv2.{a} is banned (no pretrained/loaded models, no file reads)")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in BANNED_CV2_CALLS and _chain(f)[0] in cv2_names:
                raise Rejected(f"line {ln}: .{f.attr}() on a cv2 object is banned (no loading of parameters/files)")
            is_open = (isinstance(f, ast.Name) and f.id == "open") or \
                      (isinstance(f, ast.Attribute) and f.attr == "open")
            if is_open:
                first = node.args[0] if node.args else next(
                    (k.value for k in node.keywords if k.arg in ("file", "fp")), None)
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    raise Rejected(f"line {ln}: open() with a string literal path ('{first.value}')")
    return size


# ----------------------------------------------------------------------------- comment report
EXEMPT_NUMBERS = {0, 1, 2, -1, 255}
MAX_REPORT_LINES = 30


def comment_report(path):
    """Informational only: comment coverage of the source. Never rejects."""
    try:
        src = open(path, "rb").read().decode("utf-8")
        lines = src.splitlines()
        tree = ast.parse(src)
        doc_rows = set()   # rows of docstring-only statements
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                doc_rows.update(range(node.lineno, node.end_lineno + 1))
        skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                tokenize.ENDMARKER}
        code_rows, comment_rows, num_rows = set(), set(), set()
        logical = []        # (first_row, last_row) of every logical line
        start = None
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                comment_rows.add(tok.start[0])
                continue
            if tok.type in skip:
                if tok.type == tokenize.NEWLINE and start is not None:
                    logical.append((start, tok.start[0]))
                    start = None
                continue
            if start is None:
                start = tok.start[0]
            rows = range(tok.start[0], tok.end[0] + 1)
            if tok.start[0] in doc_rows:
                continue
            code_rows.update(rows)
            if tok.type == tokenize.NUMBER:
                try:
                    v = ast.literal_eval(tok.string)
                    exempt = v in EXEMPT_NUMBERS
                except Exception:
                    exempt = False
                if not exempt:
                    num_rows.add(tok.start[0])
        # a numeric line is covered by a comment on its own line, on the line just before it, or anywhere
        # in the same multi-line statement / the line before that statement starts
        flagged = []
        for r in sorted(num_rows):
            lo, hi = r, r
            for a, b in logical:
                if a <= r <= b:
                    lo, hi = a, b
                    break
            if not any(x in comment_rows for x in range(lo - 1, hi + 1)) and r - 1 not in comment_rows:
                flagged.append(r)
        return {"code_lines": len(code_rows), "commented_lines": len(comment_rows),
                "numeric_lines_without_comment": len(flagged),
                "uncommented_numeric": [{"line": r, "src": lines[r - 1].strip()} for r in flagged]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def format_comment_report(rep):
    if "error" in rep:
        return [f"comment coverage: unavailable ({rep['error']})"]
    out = [f"comment coverage: {rep['commented_lines']}/{rep['code_lines']} lines; "
           f"{rep['numeric_lines_without_comment']} numeric lines without a nearby comment"]
    for e in rep["uncommented_numeric"][:MAX_REPORT_LINES]:
        out.append(f"  L{e['line']}: {e['src']}")
    return out


# ----------------------------------------------------------------------------- worker
def _worker(conn, tmpdir):
    os.chdir(tmpdir)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [tmpdir] + [p for p in sys.path if p not in ("", here)]
    dn = open(os.devnull, "w")
    sys.stdout = dn
    sys.stderr = dn
    try:
        import classify  # noqa: the isolated copy
        predict = classify.predict
    except BaseException:
        conn.send(("importerr", traceback.format_exc()[-500:]))
        return
    conn.send(("ready",))
    while True:
        try:
            path = conn.recv()
        except EOFError:
            return
        if path is None:
            return
        t0 = time.perf_counter()
        out, err = None, None
        try:
            out = predict(path)
            if not isinstance(out, str):
                err = f"non-str return: {type(out).__name__}"
                out = None
        except BaseException as e:
            err = f"{type(e).__name__}: {e}"[:200]
        conn.send(("res", out, (time.perf_counter() - t0) * 1000.0, err))


class _W:
    def __init__(self, ctx, tmpdir):
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child, tmpdir), daemon=True)
        self.proc.start()
        child.close()
        self.state = "starting"
        self.task = None
        self.deadline = time.monotonic() + IMPORT_TIMEOUT

    def kill(self):
        try:
            self.proc.kill()
            self.proc.join(2)
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass


def run_all(tmpdir, images, timeout, nworkers):
    """Returns list of (pred, ms, err) aligned with images."""
    ctx = get_context("fork")
    n = len(images)
    results = [None] * n
    pending = deque(range(n))
    done = 0
    ws = [_W(ctx, tmpdir) for _ in range(max(1, min(nworkers, n)))]
    try:
        while done < n:
            for w in ws:
                if w.state == "idle" and pending:
                    w.task = pending.popleft()
                    w.conn.send(images[w.task][0])
                    w.state = "busy"
                    w.deadline = time.monotonic() + timeout
            dls = [w.deadline for w in ws if w.state in ("busy", "starting")]
            wait_for = max(0.0, min(dls) - time.monotonic()) if dls else 0.1
            ready = connection.wait([w.conn for w in ws], timeout=min(wait_for, 1.0))
            for w in list(ws):
                dead = None
                if w.conn in ready:
                    try:
                        msg = w.conn.recv()
                    except (EOFError, OSError):
                        msg = None
                    if msg is None:
                        dead = "worker crashed"
                    elif msg[0] == "importerr":
                        raise Rejected("module failed to import in isolation: " + msg[1].strip().splitlines()[-1])
                    elif msg[0] == "ready":
                        w.state = "idle"
                    else:
                        results[w.task] = (msg[1], msg[2], msg[3])
                        done += 1
                        w.state, w.task = "idle", None
                elif w.state in ("busy", "starting") and time.monotonic() > w.deadline:
                    if w.state == "starting":
                        raise Rejected(f"module import took > {IMPORT_TIMEOUT:.0f}s")
                    dead = "timeout"
                if dead:
                    if w.state == "starting":
                        raise Rejected("worker died during import")
                    results[w.task] = (None, timeout * 1000.0,
                                       f"timeout (>{timeout}s)" if dead == "timeout" else "worker crashed")
                    done += 1
                    w.kill()
                    if pending or done < n:
                        ws[ws.index(w)] = _W(ctx, tmpdir)
        return results
    finally:
        for w in ws:
            try:
                if w.state != "busy":
                    w.conn.send(None)
            except Exception:
                pass
        for w in ws:
            w.proc.join(1)
            w.kill()


# ----------------------------------------------------------------------------- main
def find_images(root):
    out = []
    for cls_dir in sorted(os.listdir(root)):
        d = os.path.join(root, cls_dir)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if os.path.splitext(fn)[1].lower() in EXTS:
                out.append((os.path.abspath(os.path.join(d, fn)), cls_dir))
    return out


def load_viewed(path):
    """Set of '<class>/<filename>' keys for viewed val images (entries 'data/val/<class>/<file>')."""
    try:
        with open(path) as f:
            lst = json.load(f)
    except FileNotFoundError:
        return set()
    except (OSError, ValueError) as e:
        print(f"warning: cannot read viewed list {path}: {e}", file=sys.stderr)
        return set()
    keys = set()
    for ent in lst if isinstance(lst, list) else []:
        parts = str(ent).replace("\\", "/").split("/")
        if len(parts) >= 3 and "/".join(parts[:2]) == "data/val":
            keys.add("/".join(parts[-2:]))
    return keys


def loop_mode():
    """"fixed" or "free" from the .batch_config.json next to this script (missing key/unreadable: "free")."""
    try:
        with open(Path(__file__).resolve().parent / ".batch_config.json") as f:
            return "fixed" if json.load(f).get("loop", "free") == "fixed" else "free"
    except (OSError, ValueError, AttributeError):
        return "free"


def restricted_val(images_dir):
    """True for the workspace copy (.batch_config.json beside this script) on a --images dir inside data/val:
    only the accuracy and errors lines are shown."""
    if not (Path(__file__).resolve().parent / ".batch_config.json").exists():
        return False
    return os.sep + os.path.join("data", "val") + os.sep in os.path.realpath(images_dir) + os.sep


# ----------------------------------------------------------------------------- workspace config, comment rule, val passes
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / ".batch_config.json"
VAL_PASSES_PATH = HERE / ".val_passes.json"
VAL_PASSES_LOCK = HERE / ".val_passes.lock"


def workspace_config():
    """The .batch_config.json beside this script as a dict, or None when there is none (scripts/score.py run by the
    operator) or it is unreadable."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else None
    except (OSError, ValueError):
        return None


def comment_rejection(path):
    """Workspace copy with "enforce_comments": true: if the program has numeric lines without a nearby comment,
    return the lines to print (the report, then the REJECTED line); else None."""
    cfg = workspace_config()
    if not cfg or cfg.get("enforce_comments") is not True:
        return None
    rep = comment_report(path)
    n = rep.get("numeric_lines_without_comment", 0) if "error" not in rep else 0
    if n <= 0:
        return None
    first = ", ".join(str(e["line"]) for e in rep["uncommented_numeric"][:10])
    return format_comment_report(rep) + [
        f"REJECTED: {n} numeric line(s) without a nearby comment (every non-trivial line and every constant "
        f"must be explained); lines: {first}"]


class ValPasses:
    """Val pass budget of the workspace copy. Use as a context manager around a whole val run: it takes an exclusive
    flock, refuses (error message in .refusal) when no pass is left, and count() records a successful scoring."""

    def __init__(self, budget):
        self.budget = budget
        self.used = 0
        self.refusal = None
        self.lock = None

    def _load(self):
        try:
            with open(VAL_PASSES_PATH) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def __enter__(self):
        self.lock = open(VAL_PASSES_LOCK, "w")
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        self.used = int(self._load().get("used", 0))
        if self.used >= self.budget:
            self.refusal = (f"ERROR: all {self.budget} val passes are used; "
                            "the session is over as far as measurement goes")
        return self

    def __exit__(self, *exc):
        try:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
        except Exception:
            pass
        return False

    def left_line(self):
        return f"val passes: {self.used} of {self.budget} used; {self.budget - self.used} left (this run costs one if it is scored)"

    def count(self, program, accuracy):
        d = self._load()
        log = d.get("log") if isinstance(d.get("log"), list) else []
        try:
            sha = hashlib.sha1(open(program, "rb").read()).hexdigest()
        except OSError:
            sha = None
        log.append({"ts": datetime.datetime.now().isoformat(timespec="seconds"), "accuracy": accuracy,
                    "program_sha1": sha, "by": os.getcwd() or os.path.basename(sys.argv[0])})
        self.used += 1
        tmp = str(VAL_PASSES_PATH) + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"used": self.used, "budget": self.budget, "log": log}, f, indent=1)
        os.replace(tmp, VAL_PASSES_PATH)
        return self.used


def val_budget_for(images_dir):
    """The val pass budget (int) when this is the workspace copy, the config has "val_budget" and --images is inside
    data/val; else None."""
    cfg = workspace_config()
    if not cfg or not restricted_val(images_dir):
        return None
    vb = cfg.get("val_budget")
    return int(vb) if isinstance(vb, int) and not isinstance(vb, bool) and vb >= 0 else None


def val_gate(images_dir):
    """Return an error message if val may not be measured yet, else None. Applies only to the workspace copy
    (a .batch_config.json next to this script) and only to a --images dir inside data/val."""
    here = Path(__file__).resolve().parent
    real = os.path.realpath(images_dir) + os.sep
    if os.sep + os.path.join("data", "val") + os.sep not in real:
        return None
    if not (here / ".batch_config.json").exists():
        return None
    if loop_mode() == "free":
        return None
    st = {}
    try:
        with open(here / ".batch_state.json") as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = {}
    if st.get("epoch_done") is True:
        return None
    return ("ERROR: val is measured once per epoch: finish all batches of the current epoch with step.py first "
            f"(epoch {st.get('epoch', 0)}, batch {st.get('batch', 0)}/{st.get('n_batches', '?')} done)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("classify")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out")
    ap.add_argument("--preds")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--viewed-list",
                    default=str(Path(__file__).resolve().parent.parent / "out" / "looker_viewed_val.json"))
    ap.add_argument("--no-exclude", action="store_true")
    a = ap.parse_args()

    msg = val_gate(a.images)
    if msg:
        print(msg)
        return 1
    brief = restricted_val(a.images)
    if brief and (a.out or a.preds):
        print("ERROR: --out and --preds are not available for val here")
        return 1

    budget = val_budget_for(a.images)
    if budget is None:
        return run(a, brief, None)
    with ValPasses(budget) as vp:   # the lock is held for the whole val run
        if vp.refusal:
            print(vp.refusal)
            return 1
        print(vp.left_line())
        return run(a, brief, vp)


def run(a, brief, vp):

    # keep numpy/BLAS from oversubscribing cores across parallel workers
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")

    try:
        size = check_rules(a.classify)
    except Rejected as e:
        print(f"REJECTED: {e}")
        return 2
    except OSError as e:
        print(f"REJECTED: cannot read {a.classify}: {e}")
        return 2
    rej = comment_rejection(a.classify)
    if rej:
        print("\n".join(rej))
        return 2

    images = find_images(a.images)
    if not images:
        print(f"error: no images found under {a.images}", file=sys.stderr)
        return 1
    excluded = 0
    if not a.no_exclude and os.path.basename(os.path.abspath(a.images)) == "val":
        viewed = load_viewed(a.viewed_list)
        kept = [(p, c) for p, c in images if f"{c}/{os.path.basename(p)}" not in viewed]
        excluded = len(images) - len(kept)
        images = kept
        if not images:
            print(f"error: all {excluded} val images have been viewed and are excluded", file=sys.stderr)
            return 1
    bad = sorted({c for _, c in images if c not in CLASSES})
    if bad:
        print(f"error: unknown class directories: {bad}", file=sys.stderr)
        return 1

    tmpdir = tempfile.mkdtemp(prefix="score_iso_")
    try:
        shutil.copyfile(a.classify, os.path.join(tmpdir, "classify.py"))
        t_start = time.time()
        try:
            res = run_all(tmpdir, images, a.timeout, a.workers)
        except Rejected as e:
            print(f"REJECTED: {e}")
            return 2
        wall = time.time() - t_start
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    idx = {c: i for i, c in enumerate(CLASSES)}
    conf = [[0] * 10 for _ in range(10)]
    unscored = [0] * 10          # per true class: errors/timeouts/invalid labels
    tot = [0] * 10
    hit = [0] * 10
    rows = []
    n_err = n_to = 0
    ms_ok = []
    for (path, true), (pred, ms, err) in zip(images, res):
        t = idx[true]
        tot[t] += 1
        if err is None and pred not in idx:
            err = f"invalid label: {pred!r}"[:200]
        if err is not None:
            n_err += 1
            n_to += err.startswith("timeout")
            unscored[t] += 1
        else:
            conf[t][idx[pred]] += 1
            hit[t] += pred == true
        if not (err or "").startswith("timeout"):
            ms_ok.append(ms)
        rows.append((path, true, pred if err is None else "", ms, err or ""))
    n = len(images)
    acc = sum(hit) / n
    per_class = {c: (hit[i] / tot[i] if tot[i] else None) for i, c in enumerate(CLASSES)}
    mean_ms = sum(ms_ok) / len(ms_ok) if ms_ok else None
    vp_line = None
    if vp is not None:   # a scored val run (also a broken program that errored on every image) costs one pass
        k = vp.count(a.classify, acc)
        vp_line = f"val pass {k} of {vp.budget} used; {vp.budget - k} left"
        if n_err == n:
            vp_line += " (every image errored or timed out: the pass still counts)"

    print(f"program: {a.classify}  size: {size} bytes")
    crep = comment_report(a.classify)
    for line in format_comment_report(crep):
        print(line)
    if excluded:
        print(f"excluded {excluded} viewed val image(s) (they no longer count toward val)")
    print(f"images: {n}  accuracy: {acc:.4f} ({sum(hit)}/{n})")
    print(f"errors: {n_err} (timeouts: {n_to})  mean ms/img: "
          + (f"{mean_ms:.1f}" if mean_ms is not None else "n/a") + " (excl. timeouts)"
          + f"  wall: {wall:.1f}s")
    if brief:
        if vp_line:
            print(vp_line)
        return 0
    print("per-class accuracy:")
    for i, c in enumerate(CLASSES):
        print(f"  {c:<17s} {per_class[c] if per_class[c] is None else format(per_class[c], '.3f')}  ({hit[i]}/{tot[i]})")
    print("confusion (rows=true, cols=pred, in class order above; 'err' = error/timeout/invalid):")
    print("       " + " ".join(f"{i:>3d}" for i in range(10)) + "  err")
    for i, c in enumerate(CLASSES):
        print(f"  {i:>2d}   " + " ".join(f"{v:>3d}" for v in conf[i]) + f" {unscored[i]:>4d}   {c}")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "program": os.path.abspath(a.classify),
                "program_bytes": size,
                "comment_report": crep,
                "n_images": n,
                "n_scored": n,
                "excluded": excluded,
                "accuracy": acc,
                "per_class_accuracy": per_class,
                "classes": CLASSES,
                "confusion": conf,
                "unscored_per_true_class": unscored,
                "n_errors": n_err,
                "n_timeouts": n_to,
                "mean_ms": mean_ms,
                "timeout_s": a.timeout,
                "wall_s": wall,
            }, f, indent=2)
    if a.preds:
        with open(a.preds, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "true", "pred", "ms", "error"])
            for p, t, pr, ms, er in rows:
                w.writerow([p, t, pr, f"{ms:.1f}", er])
    return 0


if __name__ == "__main__":
    sys.exit(main())
