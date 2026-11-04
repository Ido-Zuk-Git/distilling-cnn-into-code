"""Baseline 1: small CNN from scratch, 128px, 40 epochs, AdamW + one-cycle, label smoothing 0.1, batch 64. Seeds 0,1,2."""
import time, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
SAVE = None  # optional --save PATH: write the seed-0 model's state_dict (torch) after training
if "--save" in sys.argv:
    _i = sys.argv.index("--save"); SAVE = sys.argv[_i + 1]; del sys.argv[_i:_i + 2]
from PIL import Image
from score_preds import train_set, test_set, val_set, write, CLASSES, TRAIN_ONLY, SEEDS_ARG
torch.set_num_threads(20)
EPOCHS, BS, LR, WD = 40, 64, 2e-3, 0.05
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def load(p):  # resize shortest side to 144 -> uint8 HWC
    im = Image.open(p).convert("RGB"); w, h = im.size; s = 144 / min(w, h)
    return np.asarray(im.resize((max(144, round(w * s)), max(144, round(h * s))), Image.BILINEAR))

def center(a):
    h, w = a.shape[:2]; i, j = (h - 128) // 2, (w - 128) // 2
    return a[i:i + 128, j:j + 128]

def to_t(batch):  # list of uint8 HWC 128 -> float NCHW in [0,1]
    return torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 255

def augment(arrs, rng):
    out = []
    for a in arrs:
        h, w = a.shape[:2]; i, j = rng.integers(0, h - 127), rng.integers(0, w - 127)
        c = a[i:i + 128, j:j + 128]
        out.append(c[:, ::-1] if rng.random() < 0.5 else c)
    x = to_t([np.ascontiguousarray(o) for o in out]); n = x.shape[0]
    u = lambda: (1 + (torch.rand(n, 1, 1, 1) * 0.4 - 0.2))
    x = x * u()                                              # brightness
    m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * u() + m   # contrast
    g = x.mean(1, keepdim=True); x = (x - g) * u() + g           # saturation
    return x.clamp(0, 1)

def net():
    L, c = [], 3
    for o in [32, 64, 128, 256, 256]:
        L += [nn.Conv2d(c, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2)]; c = o
    return nn.Sequential(*L, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, 10))

def run(seed, tr, ytr, te, yte_n, va=None, yva=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = net(); nparams = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * int(np.ceil(len(tr) / BS))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    y = torch.tensor(ytr); t0 = time.time(); val_trace = []; val_s = 0.0
    for ep in range(EPOCHS):
        m.train(); perm = rng.permutation(len(tr)); tot = 0
        for k in range(0, len(tr), BS):
            idx = perm[k:k + BS]
            if len(idx) < 2: continue
            x = (augment([tr[i] for i in idx], rng) - MEAN) / STD
            loss = F.cross_entropy(m(x), y[idx], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(idx)
        if va is not None:  # monitoring only (no early stopping / model selection): per-epoch val acc, final-epoch weights are used
            tv = time.time(); m.eval()
            with torch.no_grad():
                vacc = float(np.mean([int(m((to_t([center(a)]) - MEAN) / STD).argmax(1)) == t for a, t in zip(va, yva)]))
            val_trace.append(vacc); m.train(); val_s += time.time() - tv
            print(f"seed {seed} ep {ep+1} val_acc {vacc:.4f}", flush=True)
        if ep % 5 == 4 or ep == EPOCHS - 1: print(f"seed {seed} ep {ep+1} loss {tot/len(tr):.3f} t={time.time()-t0:.0f}s", flush=True)
    train_s = time.time() - t0 - val_s   # excludes per-epoch val monitoring time
    m.eval(); pred = []; t1 = time.time()
    if SAVE and seed == 0: torch.save(m.state_dict(), SAVE)
    with torch.no_grad():
        for p in te:  # batch-1, includes JPEG decode + resize + crop
            x = (to_t([center(load(p))]) - MEAN) / STD
            pred.append(int(m(x).argmax(1)))
    ms = (time.time() - t1) / len(te) * 1000
    with torch.no_grad():
        tracc = np.mean([int(m((to_t([center(a)]) - MEAN) / STD).argmax(1)) == t for a, t in zip(tr[:200], ytr[:200])])
    info = {"val_trace": val_trace, "val_eval_s": val_s}
    if va is not None:  # full train acc (all train images, centre crop) + final val acc
        with torch.no_grad():
            info["train_acc_full"] = float(np.mean([int(m((to_t([center(a)]) - MEAN) / STD).argmax(1)) == t for a, t in zip(tr, ytr)]))
        info["val_acc_final"] = val_trace[-1]
    return pred, train_s, ms, nparams, float(tracc), info

if __name__ == "__main__":
    tp, tl = train_set(); ep_, el = test_set()
    tr = [load(p) for p in tp]
    va = yva = None
    if TRAIN_ONLY:  # val images: monitoring only
        vp, yva = val_set(); va = [load(p) for p in vp]
    res = {}
    for seed in ([int(x) for x in SEEDS_ARG.split(",")] if SEEDS_ARG else [0, 1, 2]):
        res[seed] = run(seed, tr, tl, ep_, el, va, yva)
        acc = np.mean(np.array(res[seed][0]) == np.array(el))
        print(f"seed {seed}: hidden acc {acc:.4f} train {res[seed][1]:.0f}s ms/img {res[seed][2]:.1f}", flush=True)
        if SEEDS_ARG is None and seed == 0 and res[0][1] >= 600: print("seed0 >= 10 min, stopping"); break
    accs = [float(np.mean(np.array(res[s][0]) == np.array(el))) for s in res]
    pred, train_s, ms, npar, _, info = res[0]
    write("cnn_small", ep_, el, pred, "baseline/cnn_small.py (seed 0)", ms,
          {**info, "wall_time_s": train_s + info["val_eval_s"], "n_val_monitor": len(yva) if yva else None,
           "n_train": len(tp), "params": npar, "train_time_s": train_s,
           "seeds": list(res), "seed_accuracies": accs, "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
           "train_time_s_per_seed": [res[s][1] for s in res],
           "per_class_accuracy_mean": {c: float(np.mean([np.mean((np.array(res[s][0]) == np.array(el))[np.array(el) == i]) for s in res])) for i, c in enumerate(CLASSES)},
           "ms_note": "batch-1 inference incl. JPEG decode/resize/crop, 20 torch threads",
           "recipe": f"5x[conv3x3-BN-ReLU-maxpool] 32-64-128-256-256, GAP, linear; AdamW lr {LR} wd {WD} one-cycle, ls 0.1, bs {BS}, {EPOCHS} ep, 128px"})
