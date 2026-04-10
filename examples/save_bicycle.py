#!/usr/bin/env python3
"""
save_bicycle.py
---------------
Downloads the UCI Bike Sharing Dataset (hourly counts) via scikit-learn's
OpenML loader and saves it as a CSV for use with the Chapel quantile
regression driver.

Rows are shuffled before saving so a sequential train/test split in both
the Chapel and Python scripts sees the same samples.

Usage:
    python save_bicycle.py [output.csv] [--seed=42]

Output columns (in order):
    season, year, month, hour, holiday, weekday, workingday, weather,
    temp, feel_temp, humidity, windspeed, count (label, last column)

Dataset: https://openml.org/d/45101  (Bike_Sharing_Demand, hourly)
"""

import sys
import csv
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import fetch_openml

outfile = "data/bicycle.csv"
seed    = 42
for arg in sys.argv[1:]:
    if arg.startswith("--seed="):
        seed = int(arg.split("=")[1])
    else:
        outfile = arg

print("Fetching Bike Sharing Demand dataset from OpenML …")
ds = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True,
                  parser="auto")

# Keep the 12 numeric/ordinal features.
# Category columns are encoded as their integer codes.
keep_features = ["season", "year", "month", "hour", "holiday", "weekday",
                 "workingday", "weather", "temp", "feel_temp", "humidity",
                 "windspeed"]
label = "count"

df = ds.frame.copy()
for col in keep_features:
    if hasattr(df[col], "cat"):
        df[col] = df[col].cat.codes
X = df[keep_features].astype(float)
y = df[label].astype(float)

rng = np.random.default_rng(seed)
idx = rng.permutation(len(y))

with open(outfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(keep_features + [label])
    for i in idx:
        writer.writerow(list(X.iloc[i]) + [float(y.iloc[i])])

print(f"Saved {len(y)} rows x {len(keep_features)} features "
      f"to {outfile} (shuffled, seed={seed})")
