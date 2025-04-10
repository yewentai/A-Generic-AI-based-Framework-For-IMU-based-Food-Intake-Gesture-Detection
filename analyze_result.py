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

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dl_validate import THRESHOLD_LIST

if len(sys.argv) >= 2:
    result_version = sys.argv[1]
else:
    raise ValueError("Usage: python dl_validate.py <result_version> [mirror|no_mirror]")

if len(sys.argv) >= 3 and sys.argv[2].lower() == "mirror":
    FLAG_DATASET_MIRROR = True
else:
    FLAG_DATASET_MIRROR = False

PLOT_LOSS_CURVE = False  # Toggle for plotting training loss curves


def main():
    # ==============================================================================
    #                            CONFIGURATION SECTION
    # ==============================================================================

    # result_version = max(glob.glob(os.path.join("result", "*")), key=os.path.getmtime).split(os.sep)[-1]
    # result_version = "202503281533"  # <- Manually set version

    result_dir = os.path.join("result", result_version)

    # File paths for training and validation statistics
    train_stats_file = os.path.join(result_dir, "train_stats.npy")
    valid_stats_file = os.path.join(
        result_dir,
        "validate_stats_mirrored.npy" if FLAG_DATASET_MIRROR else "validate_stats.npy",
    )

    # ==============================================================================
    #                    TRAINING LOSS CURVE: ONE PLOT PER FOLD
    # ==============================================================================

    if PLOT_LOSS_CURVE:
        loss_curve_plot_dir = os.path.join(result_dir, "loss_curve")
        os.makedirs(loss_curve_plot_dir, exist_ok=True)
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
            plt.savefig(os.path.join(loss_curve_plot_dir, f"train_loss_fold{fold}.png"), dpi=300)
            plt.close()

    # ==============================================================================
    #                  VALIDATION METRICS ANALYSIS & VISUALIZATION
    # ==============================================================================

    # Load validation statistics and create analysis directory
    validate_stats = np.load(valid_stats_file, allow_pickle=True).tolist()
    analysis_dir = os.path.join(result_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    weighted_f1_sample = []  # To store weighted F1 scores (sample-wise) per fold
    weighted_f1_segment = []  # To store weighted F1 scores (segment-wise) per fold
    threshold_list = THRESHOLD_LIST  # Assumed threshold list is available in the environment

    for fold_stat in validate_stats:
        fold = fold_stat["fold"]

        # Extract weighted sample F1 score
        weighted_f1_sample.append(fold_stat["metrics_sample"]["weighted_f1"])

        # Extract weighted segment F1 score across thresholds
        segment_f1_at_thresholds = {}
        for threshold in threshold_list:
            segment_f1_at_thresholds[threshold] = fold_stat["metrics_segment"][str(threshold)]["weighted_f1"]
        weighted_f1_segment.append(segment_f1_at_thresholds)

    # 1. Combined bar plot for sample-wise and segment-wise weighted F1 scores per fold
    x = np.arange(1, len(weighted_f1_sample) + 1)  # Fold indices
    bar_width = 0.35  # Width of the bars

    # Sample-wise weighted F1 scores
    plt.figure(figsize=(12, 8))
    plt.bar(x - bar_width / 2, weighted_f1_sample, bar_width, label="Sample-wise Weighted F1", color="blue")

    # Segment-wise weighted F1 scores at different thresholds
    for i, threshold in enumerate(threshold_list):
        segment_f1_at_threshold = [
            fold_stat["metrics_segment"][str(threshold)]["weighted_f1"] for fold_stat in validate_stats
        ]
        plt.bar(
            x + (i - len(threshold_list) / 2) * bar_width / len(threshold_list),
            segment_f1_at_threshold,
            bar_width / len(threshold_list),
            label=f"Segment-wise Weighted F1 (T{threshold})",
        )

    plt.title("Weighted F1 Scores per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Weighted F1 Score")
    plt.xticks(x)
    plt.legend(title="Metrics", loc="best")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "weighted_f1_combined_per_fold.png"), dpi=300)
    plt.close()

    # Combined boxplot for sample-wise and segment-wise weighted F1 scores
    plt.figure(figsize=(12, 8))

    # Prepare data for boxplot
    data = [weighted_f1_sample]  # Start with sample-wise weighted F1 scores
    labels = ["Sample-wise"]  # Label for sample-wise F1 scores

    # Add segment-wise weighted F1 scores for each threshold
    for threshold in threshold_list:
        segment_f1_scores = [
            fold_stat["metrics_segment"][str(threshold)]["weighted_f1"] for fold_stat in validate_stats
        ]
        data.append(segment_f1_scores)
        labels.append(f"Segment-wise (T{threshold})")

    # Create boxplot
    sns.boxplot(data=data)
    plt.title("Boxplot of Weighted F1 Scores Across Folds")
    plt.ylabel("Weighted F1 Score")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "boxplot_combined_weighted_f1.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
