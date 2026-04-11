#!/usr/bin/env python3
"""
save_cover_type.py
------------------
Saves a binary subset of the UCI Cover Type dataset as a CSV file for use
with the Chapel GBM comparison driver.

Binary task: class 1 (Spruce/Fir) vs class 2 (Lodgepole Pine), the two
largest classes (~495 k rows, 43 % / 57 % split).  All 54 features are
kept (10 quantitative + 4 wilderness-area + 40 soil-type indicators).

Rows are shuffled before saving so that a sequential 80/20 split sees the
same train/test samples in both Chapel and Python.
Labels: 0 = class 2 (Lodgepole Pine), 1 = class 1 (Spruce/Fir).

Usage:
    python save_cover_type.py [output.csv] [--seed=10331]

Output columns (in order):
    54 numeric features, cover_type (label, last)
"""

import sys
import csv
import numpy as np
from sklearn.datasets import fetch_covtype

outfile = "data/cover_type.csv"
seed    = 10331
for arg in sys.argv[1:]:
    if arg.startswith("--seed="):
        seed = int(arg.split("=")[1])
    else:
        outfile = arg

data = fetch_covtype()
X, y = data.data, data.target

# Keep only class 1 and class 2; recode as 1 and 0 respectively.
mask = (y == 1) | (y == 2)
X, y = X[mask], y[mask]
y = (y == 1).astype(float)   # 1 = Spruce/Fir, 0 = Lodgepole Pine

rng = np.random.default_rng(seed)
idx = rng.permutation(len(y))
X, y = X[idx], y[idx]

feature_names = data.feature_names

with open(outfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(list(feature_names) + ["cover_type"])
    for i in range(len(y)):
        writer.writerow(list(X[i]) + [y[i]])

n_pos = int(y.sum())
n_neg = len(y) - n_pos
print(f"Saved {len(y)} rows x {len(feature_names)} features to {outfile} "
      f"(shuffled, seed={seed})")
print(f"Label distribution: {n_pos} Spruce/Fir (1), {n_neg} Lodgepole Pine (0)")
