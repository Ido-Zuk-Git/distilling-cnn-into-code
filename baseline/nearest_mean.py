"""Baseline 3: 32x32 RGB pixels, standardise with train stats; nearest class mean (cosine), 1-NN, 5-NN (cosine)."""
import time, numpy as np
from PIL import Image
from score_preds import train_set, test_set, val_set, write, TRAIN_ONLY

def feat(p):
    return np.asarray(Image.open(p).convert("RGB").resize((32, 32), Image.BILINEAR), np.float32).ravel()

def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

tp, tl = train_set(); ep, el = test_set()
t0 = time.time()
X = np.stack([feat(p) for p in tp]); y = np.array(tl)
mu, sd = X.mean(0), X.std(0) + 1e-6
Xs = unit((X - mu) / sd)
means = unit(np.stack([((X[y == c] - mu) / sd).mean(0) for c in range(10)]))
train_s = time.time() - t0

def predict(x, mode):
    q = unit((x - mu) / sd)
    if mode == "mean":
        return int(np.argmax(means @ q))
    s = Xs @ q; k = 1 if mode == "1nn" else 5
    idx = np.argsort(-s)[:k]
    return int(np.bincount(y[idx], weights=s[idx], minlength=10).argmax())  # similarity-weighted vote

if TRAIN_ONLY:  # monitoring only: val images are never used to fit
    vp, vl = val_set(); Xv = np.stack([feat(p) for p in vp])
for mode, name in [("mean", "nearest_mean"), ("1nn", "nearest_mean_1nn"), ("5nn", "nearest_mean_5nn")]:
    t0 = time.time(); pred = []
    M0 = time.time(); extra_acc = {}
    if TRAIN_ONLY:  # train acc of 1-NN/5-NN includes the self-match (trivially 100%/high)
        extra_acc = {"train_acc": float(np.mean([predict(x, mode) == t for x, t in zip(X, tl)])),
                     "val_acc": float(np.mean([predict(x, mode) == t for x, t in zip(Xv, vl)])), "n_val_monitor": len(vl)}
    t0 = time.time()
    for p in ep:
        pred.append(predict(feat(p), mode))
    ms = (time.time() - t0) / len(ep) * 1000
    write(name, ep, el, pred, f"baseline/nearest_mean.py ({mode})", ms,
          {**extra_acc, "wall_time_s": time.time() - M0 + train_s, "n_train": len(tp), "feature_dim": 3072, "train_time_s": train_s, "params": 10 * 3072 if mode == "mean" else len(tp) * 3072,
           "method": {"mean": "nearest class mean, cosine", "1nn": "1-NN cosine", "5nn": "5-NN cosine, similarity-weighted vote"}[mode]})
