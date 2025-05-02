#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Dataset Loader and Visualization Script (Standard vs Balanced)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Edited      : 2025-05-02
Description : This script compares standard and balanced IMU datasets by:
              1. Loading left and right hand raw IMU sequences and labels
              2. Creating standard and class-balanced datasets for training
              3. Initializing DataLoaders and printing subject distributions
              4. Plotting randomly sampled time-domain signals from both datasets

              Useful for verifying data balancing and augmentation impacts.
===============================================================================
"""

# ── Imports ───────────────────────────────────────────────────────
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter
from components.datasets import IMUDatasetBalanced, IMUDataset
import random
import matplotlib.pyplot as plt

# ── Dataset Settings ─────────────────────────────────────────────
DATASET = "DXI"
SAMPLING_FREQ_ORIGINAL = 64
DOWNSAMPLE_FACTOR = 4
SAMPLING_FREQ = SAMPLING_FREQ_ORIGINAL // DOWNSAMPLE_FACTOR
BATCH_SIZE = 64
NUM_WORKERS = 4
SEQUENCE_LENGTH = SAMPLING_FREQ * 60  # 1 minute of data

if DATASET.startswith("DX"):
    NUM_CLASSES = 2
    sub_version = DATASET.replace("DX", "").upper() or "I"
    DATA_DIR = f"./dataset/DX/DX-{sub_version}"
    TASK = "binary"
elif DATASET.startswith("FD"):
    NUM_CLASSES = 3
    sub_version = DATASET.replace("FD", "").upper() or "I"
    DATA_DIR = f"./dataset/FD/FD-{sub_version}"
    TASK = "multiclass"
else:
    raise ValueError(f"Invalid dataset: {DATASET}")


# ── Load Raw Pickles ─────────────────────────────────────────────
def load_list(path):
    with open(path, "rb") as f:
        return np.array(pickle.load(f), dtype=object)


X_L = load_list(os.path.join(DATA_DIR, "X_L.pkl"))
Y_L = load_list(os.path.join(DATA_DIR, "Y_L.pkl"))
X_R = load_list(os.path.join(DATA_DIR, "X_R.pkl"))
Y_R = load_list(os.path.join(DATA_DIR, "Y_R.pkl"))

# concatenate left/right streams per subject
X = np.array([np.concatenate([xl, xr], axis=0) for xl, xr in zip(X_L, X_R)], dtype=object)
Y = np.array([np.concatenate([yl, yr], axis=0) for yl, yr in zip(Y_L, Y_R)], dtype=object)

# ── Dataset Instantiation ──────────────────────────────────────────
dataset_std = IMUDataset(
    X=X,
    Y=Y,
    sequence_length=SEQUENCE_LENGTH,
    downsample_factor=DOWNSAMPLE_FACTOR,
    apply_antialias=True,
)


dataset_bal = IMUDatasetBalanced(
    X=X,
    Y=Y,
    sequence_length=SEQUENCE_LENGTH,
    downsample_factor=DOWNSAMPLE_FACTOR,
    apply_antialias=True,
)

# ── DataLoaders ─────────────────────────────────────────────────
loader_std = DataLoader(
    dataset_std,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

loader_bal = DataLoader(
    dataset_bal,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

# Optional: subject distribution counts
print("Standard subject distribution:", Counter(dataset_std.subject_indices))
print("Balanced subject distribution:", Counter(dataset_bal.subject_indices))

# ── Time Domain Plot ────────────────────────────────────────────
# Randomly select a sequence from standard dataset
x_std, _ = next(iter(loader_std))
random_idx_std = random.randint(0, x_std.shape[0] - 1)
seq_std = x_std[random_idx_std].numpy()
time = np.arange(seq_std.shape[0]) / SAMPLING_FREQ
plt.figure()
for ch in range(seq_std.shape[1]):
    plt.plot(time, seq_std[:, ch])
plt.title("Time Domain Plot - Standard Dataset (random sequence)")
plt.xlabel("Time (s)")
plt.ylabel("Normalized IMU signal")
plt.show()

# Randomly select a sequence from balanced dataset
x_bal, _ = next(iter(loader_bal))
random_idx_bal = random.randint(0, x_bal.shape[0] - 1)
seq_bal = x_bal[random_idx_bal].numpy()
time = np.arange(seq_bal.shape[0]) / SAMPLING_FREQ
plt.figure()
for ch in range(seq_bal.shape[1]):
    plt.plot(time, seq_bal[:, ch])
plt.title("Time Domain Plot - Balanced Dataset (random sequence)")
plt.xlabel("Time (s)")
plt.ylabel("Normalized IMU signal")
plt.show()
