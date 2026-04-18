#!/usr/bin/env -S python3 -u
"""
save_susy.py
------------
Downloads the UCI SUSY dataset and saves a (shuffled) CSV for use with
the Chapel GBM comparison driver.

Binary task: 1 = SUSY signal (supersymmetric particles), 0 = background.
5 000 000 samples, 18 features:
  - 8 kinematic properties measured by detectors (lepton pT, eta, phi, ...)
  - 10 high-level features derived from the above

Usage:
    python save_susy.py [output.csv] [--seed=10331] [--nrows=N]

Output columns (in order):
    18 numeric features, label (last)
"""

import sys
import os
import subprocess
import numpy as np
import polars as pl

URL      = "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz"
gz_cache = "data/SUSY.csv.gz"
outfile  = "data/susy.csv"
seed     = 10331
nrows    = None   # None = use all 5 000 000 rows

for arg in sys.argv[1:]:
    if arg.startswith("--seed="):
        seed = int(arg.split("=")[1])
    elif arg.startswith("--nrows="):
        nrows = int(arg.split("=")[1])
    else:
        outfile = arg

feature_names = [
    "lepton_1_pt", "lepton_1_eta", "lepton_1_phi",
    "lepton_2_pt", "lepton_2_eta", "lepton_2_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "MET_rel", "axial_MET", "M_R", "M_TR_2", "R", "MT2",
    "S_R", "M_Delta_R", "dPhi_r_b", "cos_theta_r1",
]

# Download with wget (-c = resume, --tries = retry on failure)
if not os.path.exists(gz_cache):
    print(f"Downloading SUSY dataset from UCI …")
    print(f"  {URL}")
    print(f"  (the file is ~2.4 GB compressed)")
    subprocess.run(
        ["wget", "-c", "--tries=10", "--waitretry=5", "-O", gz_cache, URL],
        check=True,
    )
else:
    print(f"Using cached {gz_cache}", flush=True)

# Original column order: label first, then 18 features.
print(f"Reading {gz_cache} …", flush=True)
df = pl.read_csv(
    gz_cache,
    has_header=False,
    n_rows=nrows,
    new_columns=["label"] + feature_names,
)
print(f"  {len(df):>9,} rows — done", flush=True)

# Reorder: features first, label last (consistent with other datasets).
df = df.select(feature_names + ["label"])

# Shuffle rows.
rng = np.random.default_rng(seed)
df  = df[rng.permutation(len(df))]

n_pos = int((df["label"] == 1.0).sum())
n_neg = len(df) - n_pos

print(f"Writing {outfile} …", flush=True)
df.write_csv(outfile)

print(f"Saved {len(df)} rows x {len(feature_names)} features to {outfile} "
      f"(shuffled, seed={seed})")
print(f"Label distribution: {n_pos} signal (1), {n_neg} background (0)")
