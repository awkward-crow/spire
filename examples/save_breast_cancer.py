#!/usr/bin/env python3
"""
save_breast_cancer.py
---------------------
Saves the Wisconsin Breast Cancer dataset as a CSV file for use with the
Chapel GBM comparison driver.

Rows are shuffled before saving so that a simple sequential 80/20 split
in both the Chapel and Python scripts sees the same train/test samples.
Labels: 0 = malignant, 1 = benign.

Usage:
    python save_breast_cancer.py [output.csv] [--seed=10331]

Output columns (in order):
    30 numeric features (mean, se, worst radius/texture/…), diagnosis (label, last)
"""

import sys
import csv
import numpy as np
from sklearn.datasets import load_breast_cancer

outfile = "data/breast_cancer.csv"
seed    = 10331
for arg in sys.argv[1:]:
    if arg.startswith("--seed="):
        seed = int(arg.split("=")[1])
    else:
        outfile = arg

data = load_breast_cancer()

rng = np.random.default_rng(seed)
idx = rng.permutation(len(data.target))

with open(outfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(list(data.feature_names) + ["diagnosis"])
    for i in idx:
        writer.writerow(list(data.data[i]) + [float(data.target[i])])

print(f"Saved {len(data.target)} rows x {len(data.feature_names)} features "
      f"to {outfile} (shuffled, seed={seed})")
print(f"Label distribution: {data.target.sum()} benign, "
      f"{(data.target == 0).sum()} malignant")
