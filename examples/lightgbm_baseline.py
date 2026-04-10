#!/usr/bin/env python3
"""
lightgbm_baseline.py
--------------------
Trains a LightGBM regressor on California Housing using the same
hyperparameters and train/test split as CaliforniaHousing.chpl.

Usage:
    python lightgbm_baseline.py [california_housing.csv]

Prints RMSE on train and test sets.
"""

import sys
import math
import csv
import warnings
import numpy as np
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---- Load data -------------------------------------------------------
csvfile = sys.argv[1] if len(sys.argv) > 1 else "data/california_housing.csv"

with open(csvfile) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows   = [list(map(float, r)) for r in reader]

rows = np.array(rows)
X = rows[:, :-1]
y = rows[:, -1]

# ---- Train/test split (first 80% train, last 20% test) ---------------
n       = len(y)
n_train = int(n * 0.8)
X_train, y_train = X[:n_train], y[:n_train]
X_test,  y_test  = X[n_train:], y[n_train:]

# ---- Hyperparameters — match CaliforniaHousing.chpl ------------------
# num_leaves=16 forces complete binary trees at max_depth=4 (2^4 = 16),
# matching Chapel's level-wise growth.  min_child_samples=1 and
# min_split_gain=0.0 remove LightGBM's additional pruning guards.
params = {
    "n_estimators":      50,
    "max_depth":         4,
    "num_leaves":        16,
    "learning_rate":     0.1,
    "reg_lambda":        1.0,
    "min_child_weight":  1.0,
    "min_child_samples": 1,
    "min_split_gain":    0.0,
    "verbose":           -1,
}

# ---- Train -----------------------------------------------------------
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)

# ---- Evaluate --------------------------------------------------------
def rmse(pred, true):
    return math.sqrt(np.mean((pred - true) ** 2))

train_rmse = rmse(model.predict(X_train), y_train)
test_rmse  = rmse(model.predict(X_test),  y_test)

print("=== California Housing Regression — LightGBM ===")
print(f"Samples: {n}  Features: {X.shape[1]}")
print(f"Train: {n_train}  Test: {n - n_train}")
print()
print(f"  RMSE (train): {train_rmse:.4f}")
print(f"  RMSE (test):  {test_rmse:.4f}")
