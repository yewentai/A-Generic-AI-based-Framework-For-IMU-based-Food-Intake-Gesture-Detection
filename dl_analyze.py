#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Training Result Analysis Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Last Edited : 2025-04-14
Description : This script analyzes training results across validation modes
              with enhanced comparative visualizations and per-class metrics.
===============================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dl_validate_mirror import THRESHOLD_LIST

# --- Configurations ---
PLOT_LOSS_CURVE = False  # Toggle for plotting training loss curves
COLOR_PALETTE = sns.color_palette("husl", len(THRESHOLD_LIST) + 1)  # Color palette for plots

# Define validation file priority order
VALIDATION_FILES = [
    "validation_mirroring.npy",  # Mirror augmentation validation
    "validation_rotation.npy",  # Rotation augmentation validation
]

if __name__ == "__main__":
    result_root = "result"
    versions = [d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d))]
    versions.sort()

    for version in versions:
        result_dir = os.path.join("result", version)
        analysis_dir = os.path.join(result_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        # Load configuration file
        config_file = os.path.join(result_dir, "config.json")
        with open(config_file, "r") as f:
            config_info = json.load(f)
        mirror_enabled = config_info.get("augmentation_hand_mirroring", False) or config_info.get(
            "dataset_mirroring", False
        )
        rotation_enabled = config_info.get("augmentation_planar_rotation", False)

        # Determine which validation files exist and should be processed
        existing_validation_files = []
        for vfile in VALIDATION_FILES:
            if os.path.exists(os.path.join(result_dir, vfile)):
                existing_validation_files.append(vfile)

        # Skip if no validation files found
        if not existing_validation_files:
            continue

        # Process each validation file
        for validation_file in existing_validation_files:
            validation_stats_file = os.path.join(result_dir, validation_file)
            all_stats = np.load(validation_stats_file, allow_pickle=True).item()

            # Determine analysis type based on filename
            if "mirror" in validation_file:
                analysis_type = "Mirroring"
            elif "rotation" in validation_file:
                analysis_type = "Rotation"
            else:
                analysis_type = "Standard"

            # ======================================================================
            #                    TRAINING LOSS CURVE: ONE PLOT PER FOLD
            # ======================================================================
            if PLOT_LOSS_CURVE:
                loss_curve_plot_dir = os.path.join(result_dir, "loss_curve")
                os.makedirs(loss_curve_plot_dir, exist_ok=True)

                # Find corresponding training stats file
                train_file = validation_file.replace("validate", "train")
                train_stats_file = os.path.join(result_dir, train_file)

                if os.path.exists(train_stats_file):
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

                        plt.title(f"Training Loss Over Epochs (Fold {fold}) - {analysis_type}")
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss (Log Scale)")
                        plt.grid(True, which="both", linestyle="--", alpha=0.6)
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(loss_curve_plot_dir, f"train_loss_fold{fold}_{analysis_type.lower()}.png"),
                            dpi=300,
                        )
                        plt.close()

            # ==================================================================
            #                      CROSS-MODE COMPARISON ANALYSIS
            # ==================================================================
            mode_metrics = {}
            class_metrics = {}

            for mode_name, mode_stats in all_stats.items():
                mode_metrics[mode_name] = {"sample_wise": [], "segment_wise": {t: [] for t in THRESHOLD_LIST}}
                class_metrics[mode_name] = {
                    str(t): {str(c): [] for c in range(1, len(mode_stats[0]["metrics_sample"]) - 1)}
                    for t in ["sample"] + THRESHOLD_LIST
                }

                for fold_stat in mode_stats:
                    sample_f1 = fold_stat["metrics_sample"]["weighted_f1"]
                    mode_metrics[mode_name]["sample_wise"].append(sample_f1)

                    for threshold in THRESHOLD_LIST:
                        seg_f1 = fold_stat["metrics_segment"][str(threshold)]["weighted_f1"]
                        mode_metrics[mode_name]["segment_wise"][threshold].append(seg_f1)

                    for c in range(1, len(fold_stat["metrics_sample"]) - 1):
                        class_metrics[mode_name]["sample"][str(c)].append(fold_stat["metrics_sample"][str(c)]["f1"])
                        for threshold in THRESHOLD_LIST:
                            class_metrics[mode_name][str(threshold)][str(c)].append(
                                fold_stat["metrics_segment"][str(threshold)][str(c)]["f1"]
                            )

            # ==================================================================
            #                          COMPARATIVE VISUALIZATIONS
            # ==================================================================
            mode_names = list(mode_metrics.keys())

            # 1. Segment-wise and Sample-wise F1 Scores
            plt.figure(figsize=(14, 8))
            x = np.arange(len(THRESHOLD_LIST))
            width = 0.8 / len(mode_names)

            for i, mode in enumerate(mode_names):
                means = [np.mean(mode_metrics[mode]["segment_wise"][t]) for t in THRESHOLD_LIST]
                stds = [np.std(mode_metrics[mode]["segment_wise"][t]) for t in THRESHOLD_LIST]

                positions = x + (i - len(mode_names) / 2 + 0.5) * width
                plt.bar(positions, means, width, label=f"{mode}", color=COLOR_PALETTE[i], alpha=0.7)
                plt.errorbar(positions, means, yerr=stds, fmt="none", ecolor="black", capsize=5, alpha=0.5)

            for i, mode in enumerate(mode_names):
                sample_mean = np.mean(mode_metrics[mode]["sample_wise"])
                sample_std = np.std(mode_metrics[mode]["sample_wise"])

                sample_position = len(THRESHOLD_LIST) + (i - len(mode_names) / 2 + 0.5) * width
                plt.bar(sample_position, sample_mean, width, color=COLOR_PALETTE[i], alpha=0.7)
                plt.errorbar(
                    sample_position, sample_mean, yerr=sample_std, fmt="none", ecolor="black", capsize=5, alpha=0.5
                )

            plt.xlabel("Segmentation Threshold")
            plt.ylabel("Mean Weighted F1 Score")
            plt.title(f"Segment-wise and Sample-wise F1 Scores ({analysis_type})")
            plt.xticks(list(x) + [len(THRESHOLD_LIST)], [str(t) for t in THRESHOLD_LIST] + ["Sample"])
            plt.grid(True, axis="y", linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()

            plt.savefig(os.path.join(analysis_dir, f"segment_sample_f1_{analysis_type.lower()}.png"), dpi=300)
            plt.close()

            # 2. Sample-wise F1 scores per fold
            plt.figure(figsize=(14, 8))
            bar_width = 0.8 / len(mode_names)
            folds = range(1, len(mode_metrics[mode_names[0]]["sample_wise"]) + 1)

            for i, mode in enumerate(mode_names):
                fold_f1_scores = mode_metrics[mode]["sample_wise"]
                positions = [fold + (i - len(mode_names) / 2 + 0.5) * bar_width for fold in folds]
                plt.bar(positions, fold_f1_scores, width=bar_width, label=f"{mode}", color=COLOR_PALETTE[i], alpha=0.7)

            plt.xticks(folds, [f"Fold {fold}" for fold in folds])
            plt.xlabel("Fold")
            plt.ylabel("Weighted F1 Score")
            plt.title(f"Sample-wise F1 Scores per Fold ({analysis_type})")
            plt.grid(True, axis="y", linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()

            plt.savefig(os.path.join(analysis_dir, f"sample_f1_per_fold_{analysis_type.lower()}.png"), dpi=300)
            plt.close()
