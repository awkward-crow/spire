#!/usr/bin/env python3
"""
save_california_housing.py
--------------------------
Saves the California Housing dataset as a CSV file for use with the
Chapel GBM comparison driver.

Rows are shuffled before saving so that a simple sequential 80/20 split
in both the Chapel and Python scripts sees the same train/test samples.

Usage:
    python save_california_housing.py [output.csv] [--seed=10331]

Output columns (in order):
    MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,
    Latitude, Longitude, MedHouseVal (label, last column)
"""

import sys
import csv
import numpy as np
from sklearn.datasets import fetch_california_housing

outfile = "data/california_housing.csv"
seed    = 10331
for arg in sys.argv[1:]:
    if arg.startswith("--seed="):
        seed = int(arg.split("=")[1])
    else:
        outfile = arg

data = fetch_california_housing()
feature_names = list(data.feature_names)
label_name    = "MedHouseVal"

rng = np.random.default_rng(seed)
idx = rng.permutation(len(data.target))

with open(outfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(feature_names + [label_name])
    for i in idx:
        writer.writerow(list(data.data[i]) + [data.target[i]])

print(f"Saved {len(data.target)} rows x {len(feature_names)} features "
      f"to {outfile} (shuffled, seed={seed})")
