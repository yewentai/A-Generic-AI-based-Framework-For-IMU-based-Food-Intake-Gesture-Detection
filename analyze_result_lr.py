#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Training Result Analysis Script (Dual-Hand Analysis)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Last Edited : 2025-04-11
Description : This script loads training and validation statistics from saved
              result directories, generates training loss plots for each fold,
              and evaluates final validation performance for left and right
              hands using various metrics.
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
    result_dir = os.path.join("result", result_version)

    train_stats_file = os.path.join(result_dir, "train_stats.npy")
    valid_stats_file = os.path.join(
        result_dir,
        "validate_stats_separate.npy",
    )

    if PLOT_LOSS_CURVE:
        loss_curve_plot_dir = os.path.join(result_dir, "loss_curve")
        os.makedirs(loss_curve_plot_dir, exist_ok=True)
        train_stats = np.load(train_stats_file, allow_pickle=True).tolist()
        folds = sorted(set(entry["fold"] for entry in train_stats))

        for fold in folds:
            stats_fold = [entry for entry in train_stats if entry["fold"] == fold]
            epochs = sorted(set(entry["epoch"] for entry in stats_fold))
            loss_per_epoch = {epoch: [] for epoch in epochs}
            loss_ce_per_epoch = {epoch: [] for epoch in epochs}
            loss_mse_per_epoch = {epoch: [] for epoch in epochs}

            for entry in stats_fold:
                epoch = entry["epoch"]
                loss_per_epoch[epoch].append(entry["train_loss"])
                loss_ce_per_epoch[epoch].append(entry["train_loss_ce"])
                loss_mse_per_epoch[epoch].append(entry["train_loss_mse"])

            mean_loss = [np.mean(loss_per_epoch[e]) for e in epochs]
            mean_ce = [np.mean(loss_ce_per_epoch[e]) for e in epochs]
            mean_mse = [np.mean(loss_mse_per_epoch[e]) for e in epochs]

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

    validate_stats = np.load(valid_stats_file, allow_pickle=True).tolist()
    analysis_dir = os.path.join(result_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    threshold_list = THRESHOLD_LIST

    for hand in ["left", "right"]:
        weighted_f1_sample = []
        weighted_f1_segment = []

        for fold_stat in validate_stats:
            sample_key = f"metrics_sample_{hand}"
            segment_key = f"metrics_segment_{hand}"

            # Sample-wise F1
            weighted_f1_sample.append(fold_stat[sample_key]["weighted_f1"])

            # Segment-wise F1
            segment_f1_at_thresholds = {}
            for threshold in threshold_list:
                segment_f1_at_thresholds[threshold] = fold_stat[segment_key][str(threshold)]["weighted_f1"]
            weighted_f1_segment.append(segment_f1_at_thresholds)

        # Bar Plot per fold
        x = np.arange(1, len(weighted_f1_sample) + 1)
        bar_width = 0.35

        plt.figure(figsize=(12, 8))
        plt.bar(x - bar_width / 2, weighted_f1_sample, bar_width, label="Sample-wise Weighted F1", color="blue")

        for i, threshold in enumerate(threshold_list):
            segment_f1_at_threshold = [fold_seg[str(threshold)] for fold_seg in weighted_f1_segment]
            plt.bar(
                x + (i - len(threshold_list) / 2) * bar_width / len(threshold_list),
                segment_f1_at_threshold,
                bar_width / len(threshold_list),
                label=f"Segment-wise Weighted F1 (T{threshold})",
            )

        plt.title(f"Weighted F1 Scores per Fold - {hand.capitalize()} Hand")
        plt.xlabel("Fold")
        plt.ylabel("Weighted F1 Score")
        plt.xticks(x)
        plt.legend(title="Metrics", loc="best")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f"{hand}_hand_weighted_f1_combined_per_fold.png"), dpi=300)
        plt.close()

        # Boxplot
        plt.figure(figsize=(12, 8))
        data = [weighted_f1_sample]
        labels = ["Sample-wise"]

        for threshold in threshold_list:
            segment_f1_scores = [fold_seg[str(threshold)] for fold_seg in weighted_f1_segment]
            data.append(segment_f1_scores)
            labels.append(f"Segment-wise (T{threshold})")

        sns.boxplot(data=data)
        plt.title(f"Boxplot of Weighted F1 Scores - {hand.capitalize()} Hand")
        plt.ylabel("Weighted F1 Score")
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f"{hand}_hand_boxplot_combined_weighted_f1.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
