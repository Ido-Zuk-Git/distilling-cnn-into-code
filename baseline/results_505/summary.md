# Baselines refit on the 505 images in workspace/data/train (train-only)

Fit set: 505 train images (500 original + 5 moved church images). `workspace/data/val` (495) used for monitoring only. Hidden test: 2,000 images. Same hyperparameters/seed as the 1,000-image runs.

| method | hidden acc (505 fit) | hidden acc (1,000 fit, previous) | delta (pts) | train acc | val acc (495) | params | wall time (s) | best / worst class (505 fit) |
|---|---|---|---|---|---|---|---|---|
| nearest class mean (cosine) | 25.55% | 28.15% | -2.60 | 30.1% | 26.5% | 30,720 | 1 | parachute 69.0 / gas_pump 2.5 |
| 1-NN (cosine) | 22.10% | 26.85% | -4.75 | 100.0% | 27.3% | 1,551,360 | 14 | parachute 46.0 / gas_pump 9.5 |
| 5-NN (cosine) | 27.30% | 29.70% | -2.40 | 75.6% | 29.5% | 1,551,360 | 10 | parachute 65.5 / gas_pump 6.5 |
| HOG+colour hist + linear SVM | 40.45% | 40.85% | -0.40 | 100.0% | 38.6% | 83,470 | 10 | parachute 58.5 / french_horn 18.0 |
| HOG+colour hist + logistic regression | 45.05% | 48.10% | -3.05 | 100.0% | 42.4% | 83,470 | 7 | cassette_player 63.5 / golf_ball 24.0 |
| small CNN (seed 0, 40 ep) | 68.90% | 75.45% | -6.55 | 99.0% | 68.1% | 981,802 | 1045 | tench 80.5 / chain_saw 49.0 |

Notes: 1-NN/5-NN train acc includes self-match. Previous 1,000-image numbers are read from out/baselines/*.json (match README/RESULTS: 28.2/26.9/29.7/40.9/48.1/75.45). Wall time = feature extraction + fit (+ monitoring/test-time work for NN modes; CNN wall time 1045 s = 446 s pure training + 599 s of per-epoch val monitoring (batch-1, 20 threads); compare 446 s with the earlier 782 s on 1,000 images. Previous 1,000-fit values shown with 2 decimals: 28.15/26.85/29.70/40.85/48.10/75.45).

## Per-class hidden accuracy (%), 505 fit

| class | nearest class mean (cosine) | 1-NN (cosine) | 5-NN (cosine) | HOG+colour hist + linear SVM | HOG+colour hist + logistic regression | small CNN (seed 0, 40 ep) |
|---|---|---|---|---|---|---|
| tench | 31.5 | 25.0 | 30.5 | 52.5 | 54.0 | 80.5 |
| english_springer | 26.0 | 16.0 | 18.0 | 24.0 | 30.0 | 73.0 |
| cassette_player | 35.5 | 31.5 | 42.0 | 56.5 | 63.5 | 74.5 |
| chain_saw | 15.0 | 12.5 | 11.5 | 24.0 | 34.5 | 49.0 |
| church | 17.5 | 11.0 | 14.5 | 44.0 | 48.0 | 71.5 |
| french_horn | 4.5 | 18.0 | 19.5 | 18.0 | 27.0 | 62.0 |
| garbage_truck | 33.5 | 18.5 | 31.0 | 51.0 | 55.0 | 76.5 |
| gas_pump | 2.5 | 9.5 | 6.5 | 50.5 | 60.0 | 64.0 |
| golf_ball | 20.5 | 33.0 | 34.0 | 25.5 | 24.0 | 61.5 |
| parachute | 69.0 | 46.0 | 65.5 | 58.5 | 54.5 | 76.5 |

## CNN per-epoch val accuracy (495 images, monitoring only; final-epoch weights used)

`15.4 16.0 17.2 22.6 36.6 33.1 45.1 30.1 41.2 47.9 48.7 26.1 51.7 37.4 52.5 53.1 55.6 52.1 48.1 47.9 38.8 52.9 53.9 48.1 60.4 58.0 64.8 61.8 62.0 65.9 66.7 67.9 67.5 68.3 70.3 69.5 68.9 68.3 67.7 68.1`
