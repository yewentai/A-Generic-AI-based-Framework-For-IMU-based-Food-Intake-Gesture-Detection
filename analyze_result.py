#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Training Result Analysis Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Last Edited : 2025-04-03
Description : This script loads training and validation statistics from saved
              result directories, generates training loss plots for each fold,
              and evaluates final validation performance using various metrics.
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json

from dl_validate import THRESHOLD  # Segment-wise F1 threshold value

# ==============================================================================
#                            CONFIGURATION SECTION
# ==============================================================================

# Define the root directory where result folders are saved
RESULT_ROOT = "result"

# Automatically select the latest result version by modification timestamp
all_versions = glob.glob(os.path.join(RESULT_ROOT, "*"))
result_version = max(all_versions, key=os.path.getmtime).split(os.sep)[-1]

# Alternatively, uncomment to set manually:
# result_version = "202503281533"

RESULT_PATH = os.path.join(RESULT_ROOT, result_version)
FLAG_DATASET_MIRROR = False  # Toggle for mirrored dataset variant

# File paths for training and validation statistics
train_stats_file = os.path.join(RESULT_PATH, "train_stats.npy")
valid_stats_file = os.path.join(
    RESULT_ROOT,
    "validate_stats_mirrored.npy" if FLAG_DATASET_MIRROR else "validate_stats.npy",
)

# ==============================================================================
#                    TRAINING LOSS CURVE: ONE PLOT PER FOLD
# ==============================================================================

train_stats = np.load(train_stats_file, allow_pickle=True).tolist()
folds = sorted(set(entry["fold"] for entry in train_stats))  # Unique folds

for fold in folds:
    # Filter entries belonging to the current fold
    stats_fold = [entry for entry in train_stats if entry["fold"] == fold]

    # Get all unique epochs for this fold
    epochs = sorted(set(entry["epoch"] for entry in stats_fold))

    # Initialize containers for loss components
    loss_per_epoch = {epoch: [] for epoch in epochs}
    loss_ce_per_epoch = {epoch: [] for epoch in epochs}
    loss_mse_per_epoch = {epoch: [] for epoch in epochs}

    # Group values by epoch
    for entry in stats_fold:
        epoch = entry["epoch"]
        loss_per_epoch[epoch].append(entry["train_loss"])
        loss_ce_per_epoch[epoch].append(entry["train_loss_ce"])
        loss_mse_per_epoch[epoch].append(entry["train_loss_mse"])

    # Compute mean loss for each epoch
    mean_loss = [np.mean(loss_per_epoch[e]) for e in epochs]
    mean_ce = [np.mean(loss_ce_per_epoch[e]) for e in epochs]
    mean_mse = [np.mean(loss_mse_per_epoch[e]) for e in epochs]

    # Plot training loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_loss, label="Total Loss", color="blue")
    plt.plot(epochs, mean_ce, label="Cross Entropy Loss", linestyle="--", color="red")
    plt.plot(epochs, mean_mse, label="MSE Loss", linestyle=":", color="green")
    plt.yscale("log")

    plt.title(f"Training Loss Over Epochs (Fold {fold})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, f"train_loss_fold{fold}.png"), dpi=300)
    plt.close()

# ==============================================================================
#                  VALIDATION METRICS ANALYSIS & VISUALIZATION
# ==============================================================================

validate_stats = np.load(valid_stats_file, allow_pickle=True).tolist()

# Metric containers for all folds
label_distribution = []
f1_scores_sample = []
f1_scores_segment = []
cohen_kappa_scores = []
matthews_corrcoef_scores = []

# Collect metrics fold by fold
for entry in validate_stats:
    label_dist = entry["label_distribution"]
    label_distribution.append(label_dist)

    # Weighted sample-wise F1
    total_sample = 0.0
    weighted_f1_sample = 0.0
    for label, stats in entry["metrics_sample"].items():
        label_int = int(label)
        weight = label_dist.get(label_int, 0)
        weighted_f1_sample += stats["f1"] * weight
        total_sample += weight
    f1_scores_sample.append(weighted_f1_sample / total_sample if total_sample > 0 else 0.0)

    # Weighted segment-wise F1
    total_segment = 0.0
    weighted_f1_segment = 0.0
    for label, stats in entry["metrics_segment"].items():
        label_int = int(label)
        weight = label_dist.get(label_int, 0)
        weighted_f1_segment += stats["f1"] * weight
        total_segment += weight
    f1_scores_segment.append(weighted_f1_segment / total_segment if total_segment > 0 else 0.0)

    # Append other metrics
    cohen_kappa_scores.append(entry["cohen_kappa"])
    matthews_corrcoef_scores.append(entry["matthews_corrcoef"])

# Plot bar chart for all validation metrics across folds
plt.figure(figsize=(12, 6))
width = 0.2
fold_indices = np.arange(1, len(cohen_kappa_scores) + 1)

plt.bar(fold_indices - width * 1.5, cohen_kappa_scores, width=width, label="Cohen Kappa", color="orange")
plt.bar(fold_indices - width / 2, matthews_corrcoef_scores, width=width, label="Matthews Corrcoef", color="purple")
plt.bar(fold_indices + width / 2, f1_scores_sample, width=width, label="Weighted Sample-wise F1", color="blue")
plt.bar(
    fold_indices + width * 1.5,
    f1_scores_segment,
    width=width,
    label=f"Weighted Segment-wise F1 (Threshold={THRESHOLD})",
    color="green",
)

plt.xticks(fold_indices)
plt.xlabel("Fold")
plt.ylabel("Score")
title_suffix = " (Mirrored)" if FLAG_DATASET_MIRROR else ""
filename_suffix = "_mirrored" if FLAG_DATASET_MIRROR else ""
plt.title(f"Validation Metrics Across Folds{title_suffix}")
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, f"validate_metrics{filename_suffix}.png"), dpi=300)
plt.close()
