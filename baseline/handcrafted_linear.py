"""Baseline 2: HOG + HSV colour hist (2x2 grid) + LAB mean/std (3x3 grid) -> standardise -> LogReg (C=1) and LinearSVC (C=0.5)."""
import time, numpy as np
from PIL import Image
from joblib import Parallel, delayed
from skimage.feature import hog
from skimage.color import rgb2hsv, rgb2lab, rgb2gray
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from score_preds import train_set, test_set, val_set, write, TRAIN_ONLY

def feat(p):
    im = Image.open(p).convert("RGB").resize((128, 128), Image.BILINEAR)
    a = np.asarray(im, np.float64) / 255
    h = hog(rgb2gray(a), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
    hsv = rgb2hsv(a); lab = rgb2lab(a); f = [h]
    for i in range(2):
        for j in range(2):
            s = hsv[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64].reshape(-1, 3)
            for c in range(3):
                f.append(np.histogram(s[:, c], bins=16, range=(0, 1))[0] / len(s))
    edges = [0, 43, 86, 128]
    for i in range(3):
        for j in range(3):
            s = lab[edges[i]:edges[i + 1], edges[j]:edges[j + 1]].reshape(-1, 3)
            f += [s.mean(0), s.std(0)]
    return np.concatenate(f).astype(np.float32)

tp, tl = train_set(); ep, el = test_set()
t0 = time.time()
Xtr = np.stack(Parallel(n_jobs=20)(delayed(feat)(p) for p in tp)); fe_train = time.time() - t0
Xte = np.stack(Parallel(n_jobs=20)(delayed(feat)(p) for p in ep))
if TRAIN_ONLY:  # monitoring only: val images are never used to fit (scaler included)
    vp, vl = val_set(); Xva = np.stack(Parallel(n_jobs=20)(delayed(feat)(p) for p in vp))
dim = Xtr.shape[1]; print("feature dim", dim)
t0 = time.time(); sc = StandardScaler().fit(Xtr); Xs = sc.transform(Xtr); sc_t = time.time() - t0
# per-image inference cost (single-process, one image at a time: features + scaler + classifier)
for name, clf in [("handcrafted_logreg", LogisticRegression(C=1.0, max_iter=2000)),
                  ("handcrafted_linsvm", LinearSVC(C=0.5, max_iter=5000))]:
    t0 = time.time(); clf.fit(Xs, tl); ft = time.time() - t0
    pred = clf.predict(sc.transform(Xte)).tolist()
    val_extra = {"val_acc": float((clf.predict(sc.transform(Xva)) == np.array(vl)).mean()), "n_val_monitor": len(vl)} if TRAIN_ONLY else {}
    t0 = time.time()
    for p in ep[:200]:
        clf.predict(sc.transform(feat(p)[None]))
    ms = (time.time() - t0) / 200 * 1000
    write(name, ep, el, pred, f"baseline/handcrafted_linear.py ({type(clf).__name__})", ms,
          {**val_extra, "wall_time_s": fe_train + sc_t + ft, "n_train": len(tp), "feature_dim": dim, "params": int(clf.coef_.size + clf.intercept_.size),
           "train_time_s": fe_train + sc_t + ft, "train_time_features_s": fe_train,
           "train_acc": float((clf.predict(Xs) == np.array(tl)).mean()),
           "ms_note": "single-image features+predict, mean of first 200 test images"})
