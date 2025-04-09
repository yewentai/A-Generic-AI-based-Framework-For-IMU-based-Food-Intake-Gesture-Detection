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
import json

from dl_validate import THRESHOLD_LIST

# ==============================================================================
#                            CONFIGURATION SECTION
# ==============================================================================

FLAG_DATASET_MIRROR = False  # Toggle for mirrored dataset variant

# Define the root directory where result folders are saved
result_dir = "result"

# result_version = max(glob.glob(os.path.join(result_dir, "*")), key=os.path.getmtime).split(os.sep)[-1]
result_version = "202503281533"  # <- Manually set version

result_dir = os.path.join(result_dir, result_version)

# File paths for training and validation statistics
train_stats_file = os.path.join(result_dir, "train_stats.npy")
valid_stats_file = os.path.join(
    result_dir,
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
    plt.savefig(os.path.join(result_dir, f"train_loss_fold{fold}.png"), dpi=300)
    plt.close()

# ==============================================================================
#                  VALIDATION METRICS ANALYSIS & VISUALIZATION
# ==============================================================================

validate_stats = np.load(valid_stats_file, allow_pickle=True).tolist()

# Initialize storage for aggregated metrics
segment_metrics = {str(t): {} for t in THRESHOLD_LIST}
sample_metrics = {}
cohen_kappas = []
matthews_ccs = []

# Aggregate metrics across all folds
for fold_stat in validate_stats:
    # Collect sample-wise metrics
    cohen_kappas.append(fold_stat["cohen_kappa"])
    matthews_ccs.append(fold_stat["matthews_corrcoef"])

    # Process segment metrics for each threshold
    for threshold in THRESHOLD_LIST:
        t = str(threshold)
        for class_label, metrics in fold_stat["metrics_segment"][t].items():
            if class_label not in segment_metrics[t]:
                segment_metrics[t][class_label] = {"f1": [], "tp": [], "fp": [], "fn": []}
            segment_metrics[t][class_label]["f1"].append(metrics["f1"])
            segment_metrics[t][class_label]["tp"].append(metrics["tp"])
            segment_metrics[t][class_label]["fp"].append(metrics["fp"])
            segment_metrics[t][class_label]["fn"].append(metrics["fn"])

    # Process sample-wise class metrics
    for class_label, metrics in fold_stat["metrics_sample"].items():
        if class_label not in sample_metrics:
            sample_metrics[class_label] = {"f1": [], "tp": [], "fp": [], "fn": []}
        sample_metrics[class_label]["f1"].append(metrics["f1"])
        sample_metrics[class_label]["tp"].append(metrics["tp"])
        sample_metrics[class_label]["fp"].append(metrics["fp"])
        sample_metrics[class_label]["fn"].append(metrics["fn"])

# ==============================================================================
#                      METRIC CALCULATIONS & VISUALIZATION
# ==============================================================================

# Create analysis directory
analysis_dir = os.path.join(result_dir, "analysis")
os.makedirs(analysis_dir, exist_ok=True)

# 1. Segment-wise F1 Scores by Threshold
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(THRESHOLD_LIST)))

for t, color in zip(THRESHOLD_LIST, colors):
    t_str = str(t)
    class_labels = sorted(segment_metrics[t_str].keys())
    mean_f1s = [np.nanmean(segment_metrics[t_str][cl]["f1"]) for cl in class_labels]

    plt.bar([f"Class {cl}\n(t={t})" for cl in class_labels], mean_f1s, color=color, alpha=0.7, label=f"Threshold {t}")

plt.title("Segment-wise F1 Scores by Class and Threshold")
plt.ylabel("Mean F1 Score")
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "segment_f1_by_threshold.png"), dpi=300)
plt.close()

# 2. Sample-wise Class Metrics
class_labels = sorted(sample_metrics.keys())
mean_f1s = [np.nanmean(sample_metrics[cl]["f1"]) for cl in class_labels]
std_f1s = [np.nanstd(sample_metrics[cl]["f1"]) for cl in class_labels]

plt.figure(figsize=(10, 6))
plt.bar([f"Class {cl}" for cl in class_labels], mean_f1s, yerr=std_f1s, capsize=5, color="skyblue", alpha=0.7)
plt.title("Sample-wise F1 Scores by Class")
plt.ylabel("Mean F1 Score ± SD")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "sample_f1_by_class.png"), dpi=300)
plt.close()

# 3. Agreement Metrics Visualization
metrics = {"Cohen's Kappa": cohen_kappas, "Matthews CC": matthews_ccs}

plt.figure(figsize=(8, 6))
plt.boxplot(metrics.values(), labels=metrics.keys())
plt.title("Agreement Metric Distributions Across Folds")
plt.ylabel("Score Value")
plt.ylim(-0.1, 1.1)
plt.grid(True, axis="y", linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "agreement_metrics.png"), dpi=300)
plt.close()

# ==============================================================================
#                             SUMMARY REPORT
# ==============================================================================

# Create comprehensive summary report
summary = {
    "segment_metrics": {},
    "sample_metrics": {},
    "agreement_metrics": {
        "cohens_kappa": {"mean": np.mean(cohen_kappas), "std": np.std(cohen_kappas)},
        "matthews_cc": {"mean": np.mean(matthews_ccs), "std": np.std(matthews_ccs)},
    },
    "label_distribution": validate_stats[0]["label_distribution"],
}

# Add segment metrics
for threshold in THRESHOLD_LIST:
    t = str(threshold)
    summary["segment_metrics"][t] = {}
    for class_label in segment_metrics[t].keys():
        summary["segment_metrics"][t][class_label] = {
            "f1": {
                "mean": float(np.nanmean(segment_metrics[t][class_label]["f1"])),
                "std": float(np.nanstd(segment_metrics[t][class_label]["f1"])),
            },
            "tp": int(np.sum(segment_metrics[t][class_label]["tp"])),
            "fp": int(np.sum(segment_metrics[t][class_label]["fp"])),
            "fn": int(np.sum(segment_metrics[t][class_label]["fn"])),
        }

# Add sample metrics
for class_label in sample_metrics.keys():
    summary["sample_metrics"][class_label] = {
        "f1": {
            "mean": float(np.nanmean(sample_metrics[class_label]["f1"])),
            "std": float(np.nanstd(sample_metrics[class_label]["f1"])),
        },
        "tp": int(np.sum(sample_metrics[class_label]["tp"])),
        "fp": int(np.sum(sample_metrics[class_label]["fp"])),
        "fn": int(np.sum(sample_metrics[class_label]["fn"])),
    }

# Save summary report
summary_file = os.path.join(analysis_dir, "validation_summary.json")
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Analysis complete. Results saved to: {analysis_dir}")
