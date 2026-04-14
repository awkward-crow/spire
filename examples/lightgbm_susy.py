#!/usr/bin/env python3
"""
lightgbm_susy.py
----------------
Trains a LightGBM binary classifier on the SUSY dataset using the same
hyperparameters and train/test split as SUSY.chpl.
Reports log-loss and accuracy for direct comparison.

Usage:
    python lightgbm_susy.py [susy.csv] [--nTrees=100] [--numLeaves=31]
"""

import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, accuracy_score

warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)

# ---- Args ------------------------------------------------------------
csvfile    = "data/susy.csv"
n_trees    = 100
num_leaves = 31
for arg in sys.argv[1:]:
    if arg.startswith("--nTrees="):
        n_trees = int(arg.split("=")[1])
    elif arg.startswith("--numLeaves="):
        num_leaves = int(arg.split("=")[1])
    else:
        csvfile = arg

# ---- Load data -------------------------------------------------------
print(f"Loading {csvfile} …", flush=True)
t0  = time.time()
df  = pd.read_csv(csvfile)
X   = df.iloc[:, :-1].values.astype(np.float64)
y   = df.iloc[:, -1].values
print(f"  {len(y)} rows, {X.shape[1]} features  ({time.time()-t0:.1f}s)", flush=True)

# ---- Train/test split ------------------------------------------------
n       = len(y)
n_train = int(n * 0.8)
X_train, y_train = X[:n_train], y[:n_train]
X_test,  y_test  = X[n_train:], y[n_train:]

# ---- Hyperparameters — match SUSY.chpl -------------------------------
# max_depth=-1: unconstrained leaf-wise growth, only num_leaves limits shape.
params = {
    "n_estimators":      n_trees,
    "num_leaves":        num_leaves,
    "max_depth":         -1,
    "min_child_samples": 1,
    "min_split_gain":    0.0,
    "learning_rate":     0.1,
    "reg_lambda":        1.0,
    "min_child_weight":  1.0,
    "verbose":           -1,
}

# ---- Train -----------------------------------------------------------
print(f"Training  nTrees={n_trees}  numLeaves={num_leaves} …", flush=True)
t0 = time.time()
model = lgb.LGBMClassifier(objective="binary", **params)
model.fit(X_train, y_train)
elapsed = time.time() - t0

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
print(f"nTrees: {n_trees}  numLeaves: {num_leaves}  (elapsed: {elapsed:.2f}s)")
print()
print("Log-loss:")
print(f"  train: {train_ll:.6f}  test: {test_ll:.6f}")
print()
print("Accuracy:")
print(f"  train: {train_acc:.4f}%  test: {test_acc:.4f}%")
