#!/usr/bin/env python3
"""Download Imagenette (160 px) and rebuild the train / val / hidden-test splits of the experiment.

Usage (from the repo root):
    python3 data/build_splits.py                 # download imagenette2-160.tgz into data/cache/ and build all splits
    python3 data/build_splits.py --src DIR       # use an already extracted imagenette2-160/ directory instead
    python3 data/build_splits.py --no-hidden     # build only workspace/data/{train,val} (do this before a run)
    python3 data/build_splits.py --hidden-only   # build only hidden/test (do this after the run, to score once)

Writes
    workspace/data/train/<class>/*.JPEG   505 images   (the agent tunes on these and runs the teacher on them)
    workspace/data/val/<class>/*.JPEG     495 images   (measure-only, overall accuracy, limited passes)
    hidden/test/<class>/*.JPEG            2,000 images (scored once at the end; never put it inside workspace/)
Membership comes from data/splits.json; train and val files are checked against the sha256 recorded there.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
ROOT = Path(__file__).resolve().parent.parent


def download(cache: Path) -> Path:
    """Download and extract imagenette2-160 into cache/; return the extracted directory."""
    cache.mkdir(parents=True, exist_ok=True)
    tgz, src = cache / "imagenette2-160.tgz", cache / "imagenette2-160"
    if src.is_dir():
        return src
    if not tgz.exists():
        print(f"downloading {URL} -> {tgz}", flush=True)
        tmp = tgz.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.rename(tgz)
    print(f"extracting {tgz}", flush=True)
    with tarfile.open(tgz) as tf:
        try:
            tf.extractall(cache, filter="data")
        except TypeError:  # Python < 3.12 without extraction filters
            tf.extractall(cache)
    return src


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="extracted imagenette2-160 directory (contains train/ and val/)")
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"), help="download directory (default data/cache)")
    ap.add_argument("--no-hidden", action="store_true", help="do not build hidden/test")
    ap.add_argument("--hidden-only", action="store_true", help="build only hidden/test")
    ap.add_argument("--force", action="store_true", help="replace existing workspace/data and hidden/test")
    a = ap.parse_args()

    splits = json.load(open(ROOT / "data" / "splits.json"))
    src = Path(a.src) if a.src else download(Path(a.cache))
    if not (src / "train").is_dir() or not (src / "val").is_dir():
        sys.exit(f"{src} does not look like imagenette2-160 (needs train/ and val/)")

    if a.no_hidden and a.hidden_only:
        sys.exit("--no-hidden and --hidden-only exclude each other")
    dests = {} if a.hidden_only else {"train": ROOT / "workspace" / "data" / "train", "val": ROOT / "workspace" / "data" / "val"}
    if not a.no_hidden:
        dests["hidden_test"] = ROOT / "hidden" / "test"
    for d in dests.values():
        if d.exists() and any(d.rglob("*.JPEG")):
            if not a.force:
                sys.exit(f"{d} already contains images; use --force to rebuild it")
            shutil.rmtree(d)

    bad, counts = [], {k: 0 for k in dests}
    for cls, info in splits["classes"].items():
        wnid = info["wnid"]
        for split, dst in dests.items():
            out = dst / cls
            out.mkdir(parents=True, exist_ok=True)
            folder = "val" if split == "hidden_test" else "train"   # train and val both come from Imagenette's train/
            files = info[split]
            for name in files:
                s = src / folder / wnid / name
                if not s.exists():
                    bad.append(f"missing: {s}")
                    continue
                if split != "hidden_test" and sha256(s) != files[name]:
                    bad.append(f"sha256 mismatch: {s}")
                    continue
                shutil.copy2(s, out / name)
                counts[split] += 1
    for split, n in counts.items():
        print(f"{split}: {n} images in {dests[split].relative_to(ROOT)}")
    if bad:
        print(f"{len(bad)} problem(s):", *bad[:20], sep="\n  ", file=sys.stderr)
        sys.exit(1)
    expected = {k: v for k, v in splits["counts"].items() if k in dests}
    if counts != expected:
        sys.exit(f"unexpected counts {counts}, expected {expected}")
    print("splits rebuilt and verified")


if __name__ == "__main__":
    main()
