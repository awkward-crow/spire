#!/usr/bin/env -S python3 -u
"""
save_higgs.py
-------------
Downloads the UCI HIGGS dataset and saves it as HDF5 for use with the
Chapel GBM driver.

Binary task: 1 = Higgs signal, 0 = background.
11 000 000 samples, 28 features.  The dataset is already randomly ordered
(Monte Carlo simulation output), so no shuffle is needed.

HDF5 layout:
    /X   float32[nSamples, 28]   feature matrix, row-major
    /y   float32[nSamples]       labels (0.0 or 1.0)

Usage:
    python save_higgs.py [output.h5] [--nrows=N]
"""

import sys
import os
import subprocess
import numpy as np
import polars as pl
import h5py

URL      = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
gz_cache = "data/HIGGS.csv.gz"
outfile  = "data/higgs.h5"
nrows    = None

for arg in sys.argv[1:]:
    if arg.startswith("--nrows="):
        nrows = int(arg.split("=")[1])
    else:
        outfile = arg

feature_names = [
    "lepton_pt", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_btag",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_btag",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_btag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_btag",
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb",
]

if not os.path.exists(gz_cache):
    print(f"Downloading HIGGS dataset from UCI …")
    print(f"  {URL}")
    print(f"  (the file is ~2.6 GB compressed)")
    subprocess.run(
        ["wget", "-c", "--tries=10", "--waitretry=5", "-O", gz_cache, URL],
        check=True,
    )
else:
    print(f"Using cached {gz_cache}", flush=True)

# Original column order: label first, then 28 features.
print(f"Reading {gz_cache} …", flush=True)
df = pl.read_csv(
    gz_cache,
    has_header=False,
    n_rows=nrows,
    new_columns=["label"] + feature_names,
)
print(f"  {len(df):>10,} rows — done", flush=True)

X = df[feature_names].to_numpy().astype(np.float32)
y = df["label"].to_numpy().astype(np.float32)

n_pos = int((y == 1.0).sum())
n_neg = len(y) - n_pos

print(f"Writing {outfile} …", flush=True)
with h5py.File(outfile, "w") as f:
    f.create_dataset("X", data=X, compression="gzip", compression_opts=1)
    f.create_dataset("y", data=y, compression="gzip", compression_opts=1)

mb = os.path.getsize(outfile) / 1e6
print(f"Saved {len(y)} rows x {len(feature_names)} features to {outfile}")
print(f"File size: {mb:.0f} MB")
print(f"Label distribution: {n_pos} signal (1), {n_neg} background (0)")
