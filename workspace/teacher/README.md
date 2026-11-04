# teacher: a small CNN trained on the 505 training images

A 5-block convolutional network (981,802 parameters) trained from scratch on the 505 images in
`workspace/data/train` and nothing else. It is delivered as numpy weights (`cnn_505.npz`, ~3.9 MB)
plus a numpy-only forward pass (`teacher.py`); no torch is needed.

## How it was trained and how it scores
- Recipe (baseline/cnn_small.py, seed 0): AdamW lr 2e-3 wd 0.05, one-cycle schedule, label smoothing 0.1,
  batch 64, 40 epochs, 128 px, random crop from a 144 px resize + flip + brightness/contrast/saturation jitter.
  The final-epoch weights are used (no early stopping or model selection).
- Accuracy: 99.0% on its 505 training images, 68.1% on the 495 validation images, 68.9% on the hidden test set.
  It memorises the training set, so activations on train images look cleaner than they will on unseen images.

## Classes (logit index order)
0 tench, 1 english_springer, 2 cassette_player, 3 chain_saw, 4 church, 5 french_horn, 6 garbage_truck, 7 gas_pump, 8 golf_ball, 9 parachute

## Preprocessing (identical to the training-time eval transform)
RGB; resize shortest side to 144 (PIL bilinear, other side `round(w*s)`); centre crop 128x128;
divide by 255; subtract mean (0.485, 0.456, 0.406) and divide by std (0.229, 0.224, 0.225) per channel; CHW.

## Layers (output shapes for a 128 px input)
| name | operation | output |
|---|---|---|
| conv1 | conv3x3 3->32 (pad 1, no bias) + BN (eps 1e-5) + ReLU | 32x128x128 |
| pool1 | maxpool 2x2 | 32x64x64 |
| conv2 | conv3x3 32->64 + BN + ReLU | 64x64x64 |
| pool2 | maxpool 2x2 | 64x32x32 |
| conv3 | conv3x3 64->128 + BN + ReLU | 128x32x32 |
| pool3 | maxpool 2x2 | 128x16x16 |
| conv4 | conv3x3 128->256 + BN + ReLU | 256x16x16 |
| pool4 | maxpool 2x2 | 256x8x8 |
| conv5 | conv3x3 256->256 + BN + ReLU | 256x8x8 |
| pool5 | maxpool 2x2 | 256x4x4 |
| gap | global average pool | 256 |
| fc | linear 256->10 (logits) | 10 |
| probs | softmax | 10 |

`conv1..conv5` activations are post BN+ReLU. Weights: `features.N.*` (N = index in the 20-layer feature stack:
conv at 0,4,8,12,16; BN at 1,5,9,13,17) and `fc.weight`/`fc.bias`. Full details are in `arch.json`.

## Using it
From `workspace/`:

    import sys; sys.path.insert(0, "teacher"); import teacher
    m = teacher.load()
    m.layers()                                    # [(name, kind, output_shape), ...]
    logits = m.forward("data/train/church/x.JPEG")                 # (10,)
    logits, acts = m.forward(path, return_activations=True)        # acts: ordered dict name -> array
    cls, probs = m.predict(path)
    m.weights("conv2")                            # dict of that layer's parameter arrays

`forward` also accepts an RGB uint8 HxWx3 array (same resize/crop pipeline; pass `already_cropped=True` to skip it).
About 0.15 s per image on CPU.

Command line (from `workspace/`):

    python3 teacher/teacher.py layers
    python3 teacher/teacher.py predict IMG [IMG ...]
    python3 teacher/teacher.py activations IMG --layer conv3 [--top-channels 10] [--out maps.npz]
    python3 teacher/teacher.py weights --layer conv1 [--out w.npz]
    python3 teacher/teacher.py top --layer conv3 --channel 5 [--k 10]

`top` ranks the training images by a channel's maximum activation (per-layer cache in `teacher/.cache/`,
built on first use, about a minute). The teacher refuses image paths under `data/val`.
