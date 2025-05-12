#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Dataset Loader and Visualization Script (Standard vs Balanced)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Edited      : 2025-05-12
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
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from torch.utils.data import DataLoader, Dataset

from components.datasets import IMUDataset


class IMUDatasetBalanced(Dataset):
    def __init__(
        self,
        X,
        Y,
        sequence_length=128,
        downsample_factor=1,
        apply_antialias=True,
    ):
        """
        Dataset that balances long zero-label segments across entire recordings before
        slicing into fixed-length windows, ensuring each window is exactly `sequence_length`.

        Steps:
          1. For each subject, optional anti-alias + downsample + normalize.
          2. Identify runs of constant label; crop any zero-only runs to half the length
             of the neighboring runs (ignoring missing neighbors).
          3. Reconstruct a new sequence by concatenating the balanced runs.
          4. Window the balanced sequence into fixed-length segments of `sequence_length`, zero-
             padding the final remainder if necessary.

        Parameters:
            X (list of np.ndarray): IMU data per subject, shape (N, 6)
            Y (list of np.ndarray): Label arrays per subject, shape (N,)
            sequence_length (int): Window length after balancing.
            downsample_factor (int): Factor for decimation.
            apply_antialias (bool): Whether to low-pass filter before decimation.
        """
        if downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1.")

        self.sequence_length = sequence_length
        self.downsample_factor = downsample_factor
        self.apply_antialias = apply_antialias

        self.data = []
        self.labels = []
        self.subject_indices = []

        for subject_idx, (imu_data, label_seq) in enumerate(zip(X, Y)):
            # Step 1: downsample + normalize
            if downsample_factor > 1:
                imu_data = self._downsample(imu_data)
                label_seq = label_seq[::downsample_factor]
            imu_data = self._normalize(imu_data)

            # Step 2: segment by class then balance zero runs
            runs = self._segment_by_class(imu_data, label_seq)
            balanced_runs = self._balance_runs(runs)

            # Step 3: reconstruct balanced full sequence
            imu_balanced = np.concatenate([seg for seg, _ in balanced_runs], axis=0)
            lbl_balanced = np.concatenate([lbl for _, lbl in balanced_runs], axis=0)

            # Step 4: slice into fixed windows
            num_samples = len(lbl_balanced)
            end_full = num_samples - (num_samples % sequence_length)
            for start in range(0, end_full, sequence_length):
                x = imu_balanced[start : start + sequence_length]
                y = lbl_balanced[start : start + sequence_length]
                self.data.append(x)
                self.labels.append(y)
                self.subject_indices.append(subject_idx)

            # Step 5: zero-pad the last segment if needed
            rem = num_samples - end_full
            if rem > 0:
                x = imu_balanced[end_full:]
                y = lbl_balanced[end_full:]
                pad_len = sequence_length - rem
                x = np.pad(x, ((0, pad_len), (0, 0)), mode="constant", constant_values=0)
                y = np.pad(y, (0, pad_len), mode="constant", constant_values=0)
                self.data.append(x)
                self.labels.append(y)
                self.subject_indices.append(subject_idx)

    def _segment_by_class(self, data, labels):
        """
        Splits continuous arrays into runs of constant label.
        Returns list of (imu_run, label_run).
        """
        runs = []
        current_label = labels[0]
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                runs.append((data[start:i], labels[start:i]))
                start = i
                current_label = labels[i]
        runs.append((data[start:], labels[start:]))
        return runs

    def _balance_runs(self, runs):
        """
        Crop zero-only runs based on neighbors.
        """
        balanced = []
        n = len(runs)
        for idx, (seg, lbl) in enumerate(runs):
            if np.all(lbl == 0):
                left_len = len(runs[idx - 1][0]) // 2 if idx > 0 else 0
                right_len = len(runs[idx + 1][0]) // 2 if idx < n - 1 else 0
                target = left_len + right_len
                L = len(lbl)
                start = max((L - target) // 2, 0)
                seg = seg[start : start + target]
                lbl = lbl[start : start + target]
            balanced.append((seg, lbl))
        return balanced

    def _downsample(self, data):
        if self.apply_antialias:
            nyq = 0.5 * data.shape[0]
            cutoff = (0.5 / self.downsample_factor) * nyq
            b, a = signal.butter(4, cutoff / nyq, btype="low", analog=False)
            data = signal.filtfilt(b, a, data, axis=0)
        return data[:: self.downsample_factor]

    def _normalize(self, data):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        return (data - mean) / (std + 1e-9)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


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
