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
import csv
import gzip
import os
import subprocess
import numpy as np

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
    print(f"Using cached {gz_cache}")

print(f"Reading {gz_cache} …")
with gzip.open(gz_cache) as gz:
    reader = csv.reader(line.decode() for line in gz)
    rows = []
    for i, row in enumerate(reader):
        if nrows is not None and i >= nrows:
            break
        label = row[0]
        feats = row[1:]
        rows.append(feats + [label])
        if (i + 1) % 100_000 == 0:
            print(f"  {i + 1:>9,} rows …")

print(f"  {len(rows):>9,} rows — done")

rng = np.random.default_rng(seed)
idx = rng.permutation(len(rows))
rows = [rows[i] for i in idx]

labels = [float(r[-1]) for r in rows]
n_pos  = sum(1 for l in labels if l == 1.0)
n_neg  = len(labels) - n_pos

with open(outfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(feature_names + ["label"])
    writer.writerows(rows)

print(f"Saved {len(rows)} rows x {len(feature_names)} features to {outfile} "
      f"(shuffled, seed={seed})")
print(f"Label distribution: {n_pos} signal (1), {n_neg} background (0)")
