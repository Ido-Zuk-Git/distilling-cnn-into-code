#!/usr/bin/env python3
"""Numpy/scipy-only forward pass of a small CNN trained on workspace/data/train (505 images).

Python API:
    import teacher
    m = teacher.load()
    m.layers()                       -> [(name, kind, output_shape), ...]
    m.forward(path_or_array)         -> logits, shape (10,)
    m.forward(x, return_activations=True) -> (logits, OrderedDict name -> array)
    m.predict(path)                  -> (class_name, probs (10,))
    m.weights(layer)                 -> OrderedDict of parameter arrays for that layer

CLI: layers | predict | activations | weights | top

The model may only be run on images from data/train; paths containing "data/val" are refused.
"""
import argparse, hashlib, json, os, sys, time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE.parent / "data" / "train"
CACHE_DIR = HERE / ".cache"
REFUSE = "ERROR: the teacher may only be run on data/train images"


def check_path(p):
    """Exit(1) unless p is allowed (anything under data/val is refused)."""
    for q in (str(p), os.path.realpath(str(p))):
        if "data/val" in q.replace("\\", "/"):
            print(REFUSE)
            sys.exit(1)


class Teacher:
    def __init__(self, directory=HERE):
        d = Path(directory)
        self.arch = json.load(open(d / "arch.json"))
        z = np.load(d / "cnn_505.npz")
        self.p = {k: z[k] for k in z.files}
        self.classes = self.arch["classes"]
        # Record that the teacher has been inspected: look.py refuses to show images until this marker exists.
        try:
            (HERE.parent / ".teacher_used").touch()
        except OSError:
            pass
        pre = self.arch["preprocess"]
        self.resize = pre["resize_shortest_side"]; self.crop = pre["center_crop"]
        self.mean = np.array(pre["mean"], np.float32).reshape(3, 1, 1)
        self.std = np.array(pre["std"], np.float32).reshape(3, 1, 1)
        self.blocks = self.arch["blocks"]
        self._shapes = None

    # ---- preprocessing -------------------------------------------------
    def load_image(self, x, already_cropped=False):
        """path or RGB uint8 HxWx3 array -> float32 (3,crop,crop) normalised tensor."""
        if isinstance(x, (str, os.PathLike)):
            check_path(x)
            im = Image.open(x).convert("RGB")
        else:
            a = np.asarray(x)
            if a.dtype != np.uint8:
                a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
            im = Image.fromarray(a)
        if not already_cropped:
            w, h = im.size; s = self.resize / min(w, h)
            im = im.resize((max(self.resize, round(w * s)), max(self.resize, round(h * s))), Image.BILINEAR)
            a = np.asarray(im); h, w = a.shape[:2]
            i, j = (h - self.crop) // 2, (w - self.crop) // 2
            a = a[i:i + self.crop, j:j + self.crop]
        else:
            a = np.asarray(im)
        t = np.ascontiguousarray(a.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return (t - self.mean) / self.std

    # ---- ops -----------------------------------------------------------
    @staticmethod
    def _conv(x, w, k, pad):
        c, h, wd = x.shape
        xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
        win = sliding_window_view(xp, (k, k), axis=(1, 2))           # (C,H,W,k,k)
        col = win.transpose(1, 2, 0, 3, 4).reshape(h * wd, c * k * k)  # (HW, C*k*k)
        out = col @ w.reshape(w.shape[0], -1).T                      # (HW, O)
        return np.ascontiguousarray(out.T.reshape(w.shape[0], h, wd), dtype=np.float32)

    @staticmethod
    def _pool(x, k):
        c, h, w = x.shape
        return x[:, :h // k * k, :w // k * k].reshape(c, h // k, k, w // k, k).max(axis=(2, 4))

    def _bn(self, x, key, eps):
        p = self.p
        scale = p[key + ".weight"] / np.sqrt(p[key + ".running_var"] + eps)
        shift = p[key + ".bias"] - p[key + ".running_mean"] * scale
        return x * scale[:, None, None] + shift[:, None, None]

    def _run_block(self, b, x):
        for l in b["ops"]:
            t = l["type"]
            if t == "conv":
                x = self._conv(x, self.p[l["key"] + ".weight"], l["kernel"], l["padding"])
                if l.get("bias"): x = x + self.p[l["key"] + ".bias"][:, None, None]
            elif t == "bn": x = self._bn(x, l["key"], l["eps"])
            elif t == "relu": x = np.maximum(x, 0)
            elif t == "maxpool": x = self._pool(x, l["kernel"])
            elif t == "gap": x = x.mean(axis=(1, 2))
            elif t == "linear": x = self.p[l["key"] + ".weight"] @ x + self.p[l["key"] + ".bias"]
            elif t == "softmax":
                e = np.exp(x - x.max()); x = e / e.sum()
            else: raise ValueError(t)
        return x.astype(np.float32) if x.dtype != np.float32 else x

    # ---- API -----------------------------------------------------------
    def forward(self, x, return_activations=False, already_cropped=False):
        """x: image path or RGB uint8 HxWx3 array (full pipeline: resize shortest side to 144, centre crop 128).
        -> logits (10,), and with return_activations an OrderedDict name -> array
        (convN = post BN+ReLU, poolN, gap, fc = logits, probs = softmax)."""
        t = self.load_image(x, already_cropped)
        acts = OrderedDict()
        for b in self.blocks:
            t = self._run_block(b, t); acts[b["name"]] = t
        logits = acts["fc"]
        return (logits, acts) if return_activations else logits

    def predict(self, path):
        logits = self.forward(path)
        e = np.exp(logits - logits.max()); pr = e / e.sum()
        return self.classes[int(pr.argmax())], pr

    def layers(self):
        if self._shapes is None:
            _, acts = self.forward(np.zeros((self.crop, self.crop, 3), np.uint8), True, already_cropped=True)
            self._shapes = [(b["name"], b["kind"], tuple(acts[b["name"]].shape)) for b in self.blocks]
        return list(self._shapes)

    def weights(self, layer):
        b = self._block(layer)
        keys = b["params"]
        if not keys: raise ValueError(f"layer {layer} has no weights (pool/gap/probs layers are parameter-free)")
        return OrderedDict((k, self.p[k]) for k in keys)

    def _block(self, name):
        for b in self.blocks:
            if b["name"] == name: return b
        raise SystemExit(f"unknown layer {name!r}; choose from: " + ", ".join(b["name"] for b in self.blocks))


def load(directory=HERE):
    return Teacher(directory)


# ---- CLI ---------------------------------------------------------------
def _fmt_probs(m, pr):
    return "  ".join(f"{c}={v:.3f}" for c, v in zip(m.classes, pr))


def cmd_layers(m, a):
    print(f"{'name':8s} {'kind':28s} output_shape")
    for n, k, s in m.layers(): print(f"{n:8s} {k:28s} {s}")


def cmd_predict(m, a):
    for p in a.images: check_path(p)
    for p in a.images:
        c, pr = m.predict(p)
        print(f"{p}\n  predicted: {c}\n  probs: {_fmt_probs(m, pr)}")


def cmd_activations(m, a):
    for p in a.images: check_path(p)
    for p in a.images:
        _, acts = m.forward(p, True)
        if a.layer not in acts: raise SystemExit(f"unknown layer {a.layer!r}; choose from: " + ", ".join(acts))
        A = acts[a.layer]
        print(f"{p}  layer {a.layer}  shape {A.shape}")
        if A.ndim == 3:
            flat = A.reshape(A.shape[0], -1); mx = flat.max(1); mean = flat.mean(1); am = flat.argmax(1)
            order = np.argsort(-mx)[:a.top_channels]
            print(f"{'ch':>4s} {'max':>10s} {'mean':>10s}  argmax(row,col)")
            for c in order: print(f"{c:4d} {mx[c]:10.4f} {mean[c]:10.4f}  ({am[c] // A.shape[2]},{am[c] % A.shape[2]})")
        else:
            print("  (1-D layer) values: " + np.array2string(A, precision=3, max_line_width=120))
        if a.out:
            out = a.out if len(a.images) == 1 else f"{Path(a.out).with_suffix('')}_{Path(p).stem}.npz"
            np.savez(out, activation=A); print(f"saved full map to {out}")


def cmd_weights(m, a):
    W = m.weights(a.layer)
    print(f"layer {a.layer}")
    for k, v in W.items():
        print(f"  {k}: shape {v.shape}  min {v.min():.4f}  max {v.max():.4f}")
        if v.ndim >= 2:
            nrm = np.sqrt((v.reshape(v.shape[0], -1) ** 2).sum(1))
            print(f"    per-output-channel L2 norm: min {nrm.min():.4f} mean {nrm.mean():.4f} max {nrm.max():.4f}")
            print("    " + np.array2string(nrm, precision=3, max_line_width=110, threshold=64, edgeitems=8).replace("\n", "\n    "))
    if a.out:
        np.savez(a.out, **{k.replace(".", "_"): v for k, v in W.items()}); print(f"saved to {a.out}")


def _train_paths(m):
    """(paths, labels) of every train image, class-major, sorted."""
    paths, labels = [], []
    for ci, c in enumerate(m.classes):
        for p in sorted((TRAIN_DIR / c).glob("*.JPEG")): paths.append(str(p)); labels.append(ci)
    return paths, labels


def _cache_key(paths):
    """Short hash of the sorted list of train image paths (relative to workspace/)."""
    rel = sorted(os.path.relpath(p, HERE.parent) for p in paths)
    return hashlib.sha1("\n".join(rel).encode()).hexdigest()[:12]


def _build_cache(m):
    """One pass over data/train: per-image per-channel max and argmax for every conv/pool layer.
    Writes .cache/key.txt (hash of the train listing) last, so a half-built cache is never trusted."""
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / "key.txt").unlink(missing_ok=True)
    paths, labels = _train_paths(m)
    names = [b["name"] for b in m.blocks if b["name"].startswith(("conv", "pool"))]
    shapes = {n: s for n, _, s in m.layers()}
    mx = {n: [] for n in names}; am = {n: [] for n in names}
    for p in paths:
        _, acts = m.forward(p, True)
        for n in names:
            f = acts[n].reshape(acts[n].shape[0], -1); mx[n].append(f.max(1)); am[n].append(f.argmax(1))
    for n in names:
        np.savez(CACHE_DIR / f"{n}.npz", max=np.array(mx[n], np.float32), argmax=np.array(am[n], np.int32),
                 paths=np.array([os.path.relpath(p, HERE.parent) for p in paths]), labels=np.array(labels),
                 width=int(shapes[n][2]))
    (CACHE_DIR / "key.txt").write_text(_cache_key(paths))


def _ensure_cache(m, layer):
    """Return the cache file for layer, (re)building it when the train listing differs from the cached one."""
    f = CACHE_DIR / f"{layer}.npz"
    key = _cache_key(_train_paths(m)[0])
    kf = CACHE_DIR / "key.txt"
    if not (f.exists() and kf.exists() and kf.read_text().strip() == key):
        t0 = time.time(); print("building cache over data/train (one-off, train listing changed or no cache) ...", file=sys.stderr)
        _build_cache(m); print(f"cache built in {time.time() - t0:.1f}s", file=sys.stderr)
    return f


def cmd_top(m, a):
    if a.layer not in [b["name"] for b in m.blocks if b["name"].startswith(("conv", "pool"))]:
        raise SystemExit("top supports conv1..conv5 and pool1..pool5")
    z = np.load(_ensure_cache(m, a.layer))
    C = z["max"].shape[1]
    if not 0 <= a.channel < C: raise SystemExit(f"channel must be in 0..{C - 1}")
    mx = z["max"][:, a.channel]; w = int(z["width"])
    exists = np.array([(HERE.parent / str(p)).exists() for p in z["paths"]])  # skip entries whose file is gone
    order = [i for i in np.argsort(-mx) if exists[i]][:a.k]
    print(f"top {a.k} train images for {a.layer} channel {a.channel}")
    print(f"{'max':>9s}  {'argmax(row,col)':16s} {'class':18s} path")
    for i in order:
        pos = int(z["argmax"][i, a.channel])
        print(f"{mx[i]:9.4f}  ({pos // w:2d},{pos % w:2d})        {m.classes[int(z['labels'][i])]:18s} {z['paths'][i]}")


def _teacher_preds(m, paths, workers=8):
    """{relpath: [pred_idx, confidence]} for every train image, computed once and cached in
    .cache/teacher_preds.json (keyed by the same train-listing hash as the activation cache)."""
    key = _cache_key(paths); f = CACHE_DIR / "teacher_preds.json"
    try:
        d = json.load(open(f))
        if d.get("key") == key and all(os.path.relpath(p, HERE.parent) in d["preds"] for p in paths): return d["preds"]
    except (OSError, ValueError, KeyError): pass
    print(f"computing teacher predictions for {len(paths)} train images (one-off) ...", file=sys.stderr)
    import multiprocessing as mp
    t0 = time.time(); out = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap(_teacher_worker, paths, chunksize=4), 1):
            out.append(r)
            if i % 50 == 0 or i == len(paths): print(f"  teacher {i}/{len(paths)}  {time.time() - t0:.0f}s", file=sys.stderr)
    preds = {os.path.relpath(p, HERE.parent): r for p, r in zip(paths, out)}
    CACHE_DIR.mkdir(exist_ok=True)
    json.dump({"key": key, "preds": preds}, open(f, "w"))
    return preds


_W = {}  # per-worker-process state


def _teacher_worker(path):
    """Pool worker: teacher prediction [class_index, confidence] for one train image."""
    if "m" not in _W: _W["m"] = load()
    c, pr = _W["m"].predict(path)
    return [_W["m"].classes.index(c), float(pr.max())]


def _program_init(classify_path):
    """Pool initialiser: import the agent's classify.py by path in this fresh process, workspace as cwd.
    The teacher's own directory is removed from sys.path so the program never sees the teacher module."""
    import importlib.util
    sys.path[:] = [q for q in sys.path if os.path.realpath(q or ".") != str(HERE)]
    os.chdir(HERE.parent)
    spec = importlib.util.spec_from_file_location("_agent_classify", classify_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    _W["predict"] = mod.predict


def _program_worker(path):
    """Pool worker: the program's label for one image ('ERROR: ...' if predict raised)."""
    try: return str(_W["predict"](path))
    except Exception as e: return f"ERROR: {type(e).__name__}: {e}"


def cmd_disagree(m, a):
    import multiprocessing as mp
    cp = Path(a.classify).resolve()
    if not cp.is_file(): raise SystemExit(f"no such file: {a.classify}")
    check_path(cp)
    if a.cls is not None and a.cls not in m.classes: raise SystemExit(f"unknown class {a.cls!r}; choose from: " + ", ".join(m.classes))
    paths, labels = _train_paths(m)
    for p in paths: check_path(p)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"  # 8 procs x 1 thread
    tp = _teacher_preds(m, paths, a.workers)
    print(f"running {cp.name} on {len(paths)} train images with {a.workers} workers ...", file=sys.stderr)
    t0 = time.time(); prog = []
    with mp.get_context("spawn").Pool(a.workers, initializer=_program_init, initargs=(str(cp),)) as pool:
        for i, r in enumerate(pool.imap(_program_worker, paths, chunksize=2), 1):
            prog.append(r)
            if i % 50 == 0 or i == len(paths): print(f"  program {i}/{len(paths)}  {time.time() - t0:.0f}s", file=sys.stderr)
    rows = []
    for p, y, pl in zip(paths, labels, prog):
        rel = os.path.relpath(p, HERE.parent); ti, conf = tp[rel]; true = m.classes[y]
        rows.append(dict(path=rel, true=true, prog=pl, teach=m.classes[ti], conf=conf,
                         prog_ok=pl == true, teach_ok=ti == y))
    n = len(rows)
    both = sum(r["prog_ok"] and r["teach_ok"] for r in rows)
    t_only = sum(r["teach_ok"] and not r["prog_ok"] for r in rows)
    p_only = sum(r["prog_ok"] and not r["teach_ok"] for r in rows)
    neither = n - both - t_only - p_only
    # "disagree" = the two labels differ. Order: teacher right/program wrong, both wrong (different labels), program right/teacher wrong.
    def grp(r): return 0 if r["teach_ok"] and not r["prog_ok"] else (1 if not r["prog_ok"] else 2)
    dis = sorted((r for r in rows if r["prog"] != r["teach"]), key=lambda r: (grp(r), -r["conf"], r["path"]))
    if a.cls: dis = [r for r in dis if r["true"] == a.cls]
    names = {0: "TEACHER_RIGHT", 1: "BOTH_WRONG", 2: "PROGRAM_RIGHT"}
    print(f"disagreements{' for true class ' + a.cls if a.cls else ''}: {len(dis)} (showing {min(a.limit, len(dis))}); sorted teacher-right/program-wrong first, then by teacher confidence")
    print(f"{'case':14s} {'true':17s} {'program':17s} {'teacher':17s} {'t.conf':>6s}  path")
    for r in dis[:a.limit]:
        print(f"{names[grp(r)]:14s} {r['true']:17s} {r['prog']:17s} {r['teach']:17s} {r['conf']:6.2f}  {r['path']}")
    print(f"\nsummary over all {n} train images (program acc {100 * (both + p_only) / n:.1f}%, teacher acc {100 * (both + t_only) / n:.1f}%)")
    print(f"  both right {both} | teacher right/program wrong {t_only} | program right/teacher wrong {p_only} | both wrong {neither}")
    print("  program errors per true class (wrong/total; of which teacher right):")
    for c in m.classes:
        rc = [r for r in rows if r["true"] == c]; w = [r for r in rc if not r["prog_ok"]]
        print(f"    {c:17s} {len(w):3d}/{len(rc):3d}   teacher right on {sum(r['teach_ok'] for r in w):3d} of these")


def _where(r, c, H, W, size):
    """(y_px, x_px, y_frac, x_frac, words) of the centre of cell (r,c) of an HxW map, in 'size'-pixel input coordinates."""
    fy, fx = (r + 0.5) / H, (c + 0.5) / W
    v = "top" if fy < 1 / 3 else ("middle" if fy < 2 / 3 else "bottom"); h = "left" if fx < 1 / 3 else ("centre" if fx < 2 / 3 else "right")
    return fy * size, fx * size, fy, fx, f"{v}-{h}"


def cmd_explain(m, a):
    check_path(a.image)
    layers = [x.strip() for x in a.layers.split(",") if x.strip()]
    for l in layers:
        if l not in [b["name"] for b in m.blocks if b["name"].startswith("conv")]: raise SystemExit("explain supports conv1..conv5")
    logits, acts = m.forward(a.image, True)
    e = np.exp(logits - logits.max()); pr = e / e.sum()
    print(f"{a.image}   (coordinates are in the {m.crop}px centre crop the teacher sees, after resizing the short side to {m.resize})")
    print(f"teacher: {m.classes[int(pr.argmax())]}  ({pr.max():.2f})   probs: " + " ".join(f"{c}={v:.2f}" for c, v in zip(m.classes, pr)))
    me = os.path.relpath(os.path.realpath(a.image), HERE.parent)
    for l in layers:
        A = acts[l]; C, H, W = A.shape; flat = A.reshape(C, -1)
        z = np.load(_ensure_cache(m, l)); cw = int(z["width"]); cmx = z["max"]
        exists = np.array([(HERE.parent / str(p)).exists() and str(p) != me for p in z["paths"]])
        print(f"\n== {l}: {C} channels, {H}x{W} map; top {a.top_channels} channels on this image by max activation ==")
        for c in np.argsort(-flat.max(1))[:a.top_channels]:
            pos = int(flat[c].argmax()); r, cc = pos // W, pos % W
            y, x, fy, fx, words = _where(r, cc, H, W, m.crop)
            print(f"ch {c:3d}  max {flat[c].max():7.3f}  at cell ({r},{cc}) = input (y={y:.0f},x={x:.0f}px) = ({100 * fy:.0f}% down, {100 * fx:.0f}% across) {words}")
            for i in [i for i in np.argsort(-cmx[:, c]) if exists[i]][:a.k]:
                q = int(z["argmax"][i, c]); _, _, qy, qx, qw = _where(q // cw, q % cw, cw, cw, m.crop)
                print(f"     also fires on: {cmx[i, c]:7.3f} {m.classes[int(z['labels'][i])]:17s} {z['paths'][i]}  ({100 * qy:.0f}% down, {100 * qx:.0f}% across) {qw}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("layers", help="list layers with output shapes")
    s = sp.add_parser("predict", help="class + probabilities"); s.add_argument("images", nargs="+")
    s = sp.add_parser("activations", help="per-channel stats of a layer's activation map")
    s.add_argument("images", nargs="+"); s.add_argument("--layer", required=True)
    s.add_argument("--out"); s.add_argument("--top-channels", type=int, default=10)
    s = sp.add_parser("weights", help="weight statistics of a layer"); s.add_argument("--layer", required=True); s.add_argument("--out")
    s = sp.add_parser("top", help="train images that most activate a channel")
    s.add_argument("--layer", required=True); s.add_argument("--channel", type=int, required=True); s.add_argument("--k", type=int, default=10)
    a = ap.parse_args(argv)
    for p in list(getattr(a, "images", [])) + [x for x in [getattr(a, "image", None), getattr(a, "classify", None)] if x]: check_path(p)
    m = load()
    {"layers": cmd_layers, "predict": cmd_predict, "activations": cmd_activations, "weights": cmd_weights, "top": cmd_top,
     "disagree": cmd_disagree, "explain": cmd_explain}[a.cmd](m, a)


if __name__ == "__main__":
    main()
