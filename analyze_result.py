#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Training Result Analysis Script (Enhanced Mode Comparison)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Last Edited : 2025-04-14
Description : This script analyzes training results across validation modes
              with enhanced comparative visualizations and per-class metrics.
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dl_validate import THRESHOLD_LIST

# --- Configurations ---
PLOT_LOSS_CURVE = False  # Toggle for plotting training loss curves
COLOR_PALETTE = sns.color_palette("husl", len(THRESHOLD_LIST) + 1)  # Color palette for plots


if __name__ == "__main__":
    result_root = "result"
    # versions = [] # Uncomment this line to manually specify versions
    versions = [d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d))]
    versions.sort()

    for version in versions:

        result_dir = os.path.join("result", version)
        analysis_dir = os.path.join(result_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        # Load validation statistics
        valid_stats_file = os.path.join(result_dir, "validate_stats.npy")
        all_stats = np.load(valid_stats_file, allow_pickle=True).item()
        # ==============================================================================
        #                    TRAINING LOSS CURVE: ONE PLOT PER FOLD
        # ==============================================================================

        if PLOT_LOSS_CURVE:
            loss_curve_plot_dir = os.path.join(result_dir, "loss_curve")
            os.makedirs(loss_curve_plot_dir, exist_ok=True)
            train_stats_file = os.path.join(result_dir, "train_stats.npy")
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

        # ==========================================================================
        #                      CROSS-MODE COMPARISON ANALYSIS
        # ==========================================================================
        mode_metrics = {}
        class_metrics = {}

        # Process each validation mode
        for mode_name, mode_stats in all_stats.items():
            # Initialize storage
            mode_metrics[mode_name] = {"sample_wise": [], "segment_wise": {t: [] for t in THRESHOLD_LIST}}

            class_metrics[mode_name] = {
                str(t): {str(c): [] for c in range(1, len(mode_stats[0]["metrics_sample"]) - 1)}
                for t in ["sample"] + THRESHOLD_LIST
            }

            # Aggregate metrics across folds
            for fold_stat in mode_stats:
                # Sample-wise metrics
                sample_f1 = fold_stat["metrics_sample"]["weighted_f1"]
                mode_metrics[mode_name]["sample_wise"].append(sample_f1)

                # Segment-wise metrics
                for threshold in THRESHOLD_LIST:
                    seg_f1 = fold_stat["metrics_segment"][str(threshold)]["weighted_f1"]
                    mode_metrics[mode_name]["segment_wise"][threshold].append(seg_f1)

                # Class-level metrics
                for c in range(1, len(fold_stat["metrics_sample"]) - 1):
                    # Sample-wise class F1
                    class_metrics[mode_name]["sample"][str(c)].append(fold_stat["metrics_sample"][str(c)]["f1"])

                    # Segment-wise class F1
                    for threshold in THRESHOLD_LIST:
                        class_metrics[mode_name][str(threshold)][str(c)].append(
                            fold_stat["metrics_segment"][str(threshold)][str(c)]["f1"]
                        )

        # ==========================================================================
        #                          COMPARATIVE VISUALIZATIONS
        # ==========================================================================

        plt.figure(figsize=(14, 8))

        mode_names = list(mode_metrics.keys())
        x = np.arange(len(THRESHOLD_LIST))
        width = 0.8 / len(mode_names)

        for i, mode in enumerate(mode_names):
            # Calculate mean F1 for each threshold
            means = [np.mean(mode_metrics[mode]["segment_wise"][t]) for t in THRESHOLD_LIST]
            # Calculate standard deviation
            stds = [np.std(mode_metrics[mode]["segment_wise"][t]) for t in THRESHOLD_LIST]

            positions = x + (i - len(mode_names) / 2 + 0.5) * width
            plt.bar(positions, means, width, label=f"{mode}", color=COLOR_PALETTE[i], alpha=0.7)

            # Add error bars
            plt.errorbar(positions, means, yerr=stds, fmt="none", ecolor="black", capsize=5, alpha=0.5)

        # Add sample-wise data as a bar
        for i, mode in enumerate(mode_names):
            sample_mean = np.mean(mode_metrics[mode]["sample_wise"])
            sample_std = np.std(mode_metrics[mode]["sample_wise"])

            # Position for sample-wise bar
            sample_position = len(THRESHOLD_LIST) + (i - len(mode_names) / 2 + 0.5) * width
            plt.bar(
                sample_position,
                sample_mean,
                width,
                color=COLOR_PALETTE[i],
                alpha=0.7,
            )

            # Add error bars
            plt.errorbar(
                sample_position,
                sample_mean,
                yerr=sample_std,
                fmt="none",
                ecolor="black",
                capsize=5,
                alpha=0.5,
            )

        plt.xlabel("Segmentation Threshold")
        plt.ylabel("Mean Weighted F1 Score")
        plt.title(f"Segment-wise and Sample-wise F1 Scores (Version: {version})")
        plt.xticks(list(x) + [len(THRESHOLD_LIST)], [str(t) for t in THRESHOLD_LIST] + ["Sample"])
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(analysis_dir, f"threshold_impact_{version}.png"), dpi=300)
        plt.close()
