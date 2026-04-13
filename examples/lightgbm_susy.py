#!/usr/bin/env python3
"""
lightgbm_susy.py
----------------
Trains a LightGBM binary classifier on the SUSY dataset using the same
hyperparameters and train/test split as SUSY.chpl.
Reports log-loss and accuracy for direct comparison.

Usage:
    python lightgbm_susy.py [susy.csv]
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

# ---- Load data -------------------------------------------------------
csvfile = sys.argv[1] if len(sys.argv) > 1 else "data/susy.csv"

print(f"Loading {csvfile} …")
with open(csvfile) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows   = [list(map(float, r)) for r in reader]

rows = np.array(rows, dtype=np.float64)
X    = rows[:, :-1]
y    = rows[:, -1]

# ---- Train/test split ------------------------------------------------
n       = len(y)
n_train = int(n * 0.8)
X_train, y_train = X[:n_train], y[:n_train]
X_test,  y_test  = X[n_train:], y[n_train:]

# ---- Hyperparameters — match SUSY.chpl -------------------------------
params = {
    "n_estimators":      50,
    "max_depth":         4,
    "num_leaves":        16,
    "min_child_samples": 1,
    "min_split_gain":    0.0,
    "learning_rate":     0.1,
    "reg_lambda":        1.0,
    "min_child_weight":  1.0,
    "verbose":           -1,
}

# ---- Train -----------------------------------------------------------
print("Training LightGBM …")
model = lgb.LGBMClassifier(objective="binary", **params)
model.fit(X_train, y_train)

# ---- Evaluate --------------------------------------------------------
train_prob = model.predict_proba(X_train)[:, 1]
test_prob  = model.predict_proba(X_test)[:, 1]

train_ll  = log_loss(y_train, train_prob)
test_ll   = log_loss(y_test,  test_prob)
train_acc = accuracy_score(y_train, model.predict(X_train)) * 100.0
test_acc  = accuracy_score(y_test,  model.predict(X_test))  * 100.0

print()
print("=== SUSY Classification — LightGBM ===")
print(f"Samples: {n}  Features: {X.shape[1]}")
print(f"Train: {n_train}  Test: {n - n_train}")
print()
print("Log-loss:")
print(f"  train: {train_ll:.6f}  test: {test_ll:.6f}")
print()
print("Accuracy:")
print(f"  train: {train_acc:.4f}%  test: {test_acc:.4f}%")
