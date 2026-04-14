#!/usr/bin/env python3
"""
lightgbm_bicycle.py
-------------------
Trains LightGBM quantile regression models (tau = 0.1, 0.5, 0.9) on the
UCI Bike Sharing Dataset (hourly counts) using the same hyperparameters
and train/test split as Bicycle.chpl.  Reports pinball loss and 80%
interval metrics for direct comparison.

Usage:
    python lightgbm_bicycle.py [bicycle.csv] [--nTrees=100] [--numLeaves=31] [--colsample=1.0]
"""

import sys
import csv
import time
import logging
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)

# ---- Args ------------------------------------------------------------
csvfile    = "data/bicycle.csv"
colsample  = 1.0
n_trees    = 100
num_leaves = 31
for arg in sys.argv[1:]:
    if arg.startswith("--colsample="):
        colsample = float(arg.split("=")[1])
    elif arg.startswith("--nTrees="):
        n_trees = int(arg.split("=")[1])
    elif arg.startswith("--numLeaves="):
        num_leaves = int(arg.split("=")[1])
    else:
        csvfile = arg

# ---- Load data -------------------------------------------------------
with open(csvfile) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows   = [list(map(float, r)) for r in reader]

feature_names = header[:-1]
rows    = np.array(rows)
X = pd.DataFrame(rows[:, :-1], columns=feature_names)
y = rows[:, -1]

# ---- Train/test split ------------------------------------------------
n       = len(y)
n_train = int(n * 0.8)
X_train, y_train = X.iloc[:n_train], y[:n_train]
X_test,  y_test  = X.iloc[n_train:], y[n_train:]

# ---- Hyperparameters — match Bicycle.chpl ----------------------------
# max_depth=-1: unconstrained — only num_leaves limits tree shape
# (leaf-wise growth, matching Chapel's new builder).
base_params = {
    "n_estimators":      n_trees,
    "num_leaves":        num_leaves,
    "max_depth":         -1,
    "min_child_samples": 1,
    "min_split_gain":    0.0,
    "learning_rate":     0.1,
    "reg_lambda":        1.0,
    "min_child_weight":  1.0,
    "colsample_bytree":  colsample,
    "verbose":           -1,
}

# ---- Pinball loss helper ---------------------------------------------
def pinball(pred, true, tau):
    r = true - pred
    return np.mean(np.where(r > 0, tau * r, (tau - 1) * r))

# ---- Train and evaluate each quantile --------------------------------
print("=== Bicycle Quantile Regression — LightGBM ===")
print(f"Samples: {n}  Features: {X.shape[1]}  colsample_bytree: {colsample}")
print(f"Train: {n_train}  Test: {n - n_train}")
print(f"nTrees: {n_trees}  numLeaves: {num_leaves}")
print()

preds = {}
print("Pinball loss:")
for tau in (0.1, 0.9):
    t0 = time.time()
    model = lgb.LGBMRegressor(objective="quantile", alpha=tau, **base_params)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)
    train_loss = pinball(train_pred, y_train, tau)
    test_loss  = pinball(test_pred,  y_test,  tau)

    print(f"  tau={tau:.1f}  pinball (train={train_loss:.4f}  test={test_loss:.4f})  ({elapsed:.2f}s)")
    preds[tau] = test_pred

# ---- 80% interval metrics --------------------------------------------
q10, q90 = preds[0.1], preds[0.9]
covered   = np.mean((y_test >= q10) & (y_test <= q90)) * 100.0
mean_width = np.mean(q90 - q10)

print()
print("80% prediction interval (q10–q90):")
print(f"  coverage:     {covered:.1f}%  (target 80%)")
print(f"  mean width:   {mean_width:.1f} counts/hour")
