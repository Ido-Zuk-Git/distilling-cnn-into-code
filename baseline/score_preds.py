"""Shared helpers: image listing and result JSON in the same format as scripts/score.py --out.

CLI: score_preds.py <preds.csv> <out.json>   (rescore a path,true,pred csv)
"""
import csv, datetime, json, os, sys
from pathlib import Path



def _pop(name, takes_value):
    """Pop a --flag (and its value) out of sys.argv so the baseline scripts keep their positional CLI."""
    if name in sys.argv:
        i = sys.argv.index(name)
        if takes_value:
            v = sys.argv[i + 1]; del sys.argv[i:i + 2]; return v
        del sys.argv[i]; return True
    return None


# Optional switches (default behaviour unchanged = fit on train+val = 1,000 images, scored on HIDDEN):
#   --train-only      fit on workspace/data/train only; workspace/data/val is monitoring only (val_set())
#   --hidden DIR      hidden test dir (contains <class>/*.JPEG), overrides HIDDEN
#   --out-dir DIR     write <name>.json / <name>_preds.csv here instead of out/baselines
#   --seeds 0,1,2     (cnn_small.py) seeds to run
TRAIN_ONLY = bool(_pop("--train-only", False))
_h = _pop("--hidden", True)
_o = _pop("--out-dir", True)
SEEDS_ARG = _pop("--seeds", True)

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["tench", "english_springer", "cassette_player", "chain_saw", "church",
           "french_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute"]
OUT = Path(_o) if _o else ROOT / "out" / "baselines"
HIDDEN = _h or str(ROOT / "hidden" / "test")   # built by data/build_splits.py (git-ignored)


def list_images(root):
    """-> (paths, labels) sorted; root/<class>/*.JPEG"""
    paths, labels = [], []
    for ci, c in enumerate(CLASSES):
        for p in sorted((Path(root) / c).glob("*.JPEG")):
            paths.append(str(p)); labels.append(ci)
    return paths, labels


def val_set():
    """workspace/data/val (495 images in the current layout); monitoring only, never fit on it in --train-only mode."""
    return list_images(ROOT / "workspace/data/val")


def train_set():
    p1, l1 = list_images(ROOT / "workspace/data/train")
    if TRAIN_ONLY:
        assert len(p1) == 505, len(p1)
        return p1, l1
    p2, l2 = val_set()
    assert len(p1) + len(p2) == 1000, (len(p1), len(p2))
    return p1 + p2, l1 + l2


def test_set(root=None):
    p, l = list_images(root or HIDDEN)
    assert len(p) == 2000, len(p)
    return p, l


def result(paths, true, pred, program, ms_per_image, extra=None):
    n = len(paths); k = len(CLASSES)
    conf = [[0] * k for _ in range(k)]
    for t, p in zip(true, pred):
        conf[t][p] += 1
    pca = {c: conf[i][i] / max(1, sum(conf[i])) for i, c in enumerate(CLASSES)}
    d = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
         "program": program, "n_images": n, "n_scored": n,
         "accuracy": sum(conf[i][i] for i in range(k)) / n,
         "per_class_accuracy": pca, "classes": CLASSES, "confusion": conf,
         "mean_ms": ms_per_image}
    d.update(extra or {})
    return d


def write(name, paths, true, pred, program, ms_per_image, extra=None):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}_preds.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "true", "pred"])
        for p, t, q in zip(paths, true, pred):
            w.writerow([os.path.relpath(p, HIDDEN), CLASSES[t], CLASSES[q]])   # <class>/<file>
    d = result(paths, true, pred, program, ms_per_image, extra)
    (OUT / f"{name}.json").write_text(json.dumps(d, indent=2))
    print(name, "acc=%.4f" % d["accuracy"])
    return d


if __name__ == "__main__":
    rows = list(csv.DictReader(open(sys.argv[1])))
    t = [CLASSES.index(r["true"]) for r in rows]; p = [CLASSES.index(r["pred"]) for r in rows]
    d = result([r["path"] for r in rows], t, p, sys.argv[1], 0.0)
    Path(sys.argv[2]).write_text(json.dumps(d, indent=2)); print(d["accuracy"])
