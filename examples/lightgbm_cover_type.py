#!/usr/bin/env python3
"""
lightgbm_cover_type.py
----------------------
Trains a LightGBM binary classifier on the Cover Type dataset (class 1
Spruce/Fir vs class 2 Lodgepole Pine) using the same hyperparameters and
train/test split as CoverType.chpl.
Reports log-loss and accuracy for direct comparison.

Usage:
    python lightgbm_cover_type.py [cover_type.csv] [--colsample=0.8]
"""

import sys
import csv
import logging
import warnings
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss, accuracy_score

warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)

# ---- Args ------------------------------------------------------------
csvfile   = "data/cover_type.csv"
colsample = 1.0
for arg in sys.argv[1:]:
    if arg.startswith("--colsample="):
        colsample = float(arg.split("=")[1])
    else:
        csvfile = arg

# ---- Load data -------------------------------------------------------

with open(csvfile) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows   = [list(map(float, r)) for r in reader]

rows = np.array(rows)
X    = rows[:, :-1].astype(np.float64)
y    = rows[:, -1]

# ---- Train/test split ------------------------------------------------
n       = len(y)
n_train = int(n * 0.8)
X_train, y_train = X[:n_train], y[:n_train]
X_test,  y_test  = X[n_train:], y[n_train:]

# ---- Hyperparameters — match CoverType.chpl --------------------------
# num_leaves=16 (=2^4) and min_child_samples=1 / min_split_gain=0 force
# LightGBM to grow complete binary trees of depth 4, matching Chapel's
# level-wise builder.
params = {
    "n_estimators":      50,
    "max_depth":         4,
    "num_leaves":        16,
    "min_child_samples": 1,
    "min_split_gain":    0.0,
    "learning_rate":     0.1,
    "reg_lambda":        1.0,
    "min_child_weight":  1.0,
    "colsample_bytree":  colsample,
    "verbose":           -1,
}

# ---- Train -----------------------------------------------------------
model = lgb.LGBMClassifier(objective="binary", **params)
model.fit(X_train, y_train)

# ---- Evaluate --------------------------------------------------------
train_prob = model.predict_proba(X_train)[:, 1]
test_prob  = model.predict_proba(X_test)[:, 1]

train_ll  = log_loss(y_train, train_prob)
test_ll   = log_loss(y_test,  test_prob)
train_acc = accuracy_score(y_train, model.predict(X_train)) * 100.0
test_acc  = accuracy_score(y_test,  model.predict(X_test))  * 100.0

print("=== Cover Type Classification — LightGBM ===")
print(f"Samples: {n}  Features: {X.shape[1]}  colsample_bytree: {colsample}")
print(f"Train: {n_train}  Test: {n - n_train}")
print()
print("Log-loss:")
print(f"  train: {train_ll:.6f}  test: {test_ll:.6f}")
print()
print("Accuracy:")
print(f"  train: {train_acc:.4f}%  test: {test_acc:.4f}%")
