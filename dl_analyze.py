#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Training Result Analysis Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-08
Description : This script analyzes training results:
              1. Comparative analysis of segment-wise and sample-wise metrics
              2. Support for multiple validation modes (original, mirrored, rotated)
              3. Per-class performance analysis
              4. Cross-validation fold analysis
===============================================================================
"""

import os
import json
import logging

import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette

if __name__ == "__main__":
    result_root = "result"
    versions = [
        "DXI_BOTH_MSTCN_HUBER",
        "DXI_BOTH_MSTCN_L1",
        "DXI_BOTH_MSTCN_MSE",
    ]  # Uncomment to manually specify versions
    # versions = [d for d in sorted(os.listdir(result_root)) if os.path.isdir(os.path.join(result_root, d))]

    for version in versions:
        # Set up logging
        logger = logging.getLogger(f"validation_{version}")
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.info(f"\n=== Validating Version: {version} ===")

        # Load threshold_list and validation flag`
        result_dir = os.path.join(result_root, version)
        config_file = os.path.join(result_dir, "validation_config.json")
        if not os.path.exists(config_file):
            logger.warning(f"Config file does not exist: {config_file}.")
            continue
        with open(config_file, "r") as f:
            config = json.load(f)
        threshold_list = config.get("threshold_list", []) or []
        flag_segment_validation = config.get("flag_segment_validation", False)
        if not threshold_list:
            logger.warning("No thresholds provided in config; segment-wise analysis will be skipped.")

        # Gather all stats files
        stats_files = sorted(
            os.path.join(result_dir, f)
            for f in os.listdir(result_dir)
            if f.startswith("validation_stats") and f.endswith(".npy")
        )
        if not stats_files:
            logger.warning(f"No validation_stats.npy files found for version {version}, skipping.")
            continue

        # Extract dataset names in the same order
        dataset_names = [os.path.splitext(os.path.basename(f))[0].split("_", 2)[-1] for f in stats_files]

        for stats_file, dataset_name in zip(stats_files, dataset_names):
            if not os.path.exists(stats_file):
                logger.warning(f"Stats file missing: {stats_file}, skipping.")
                continue

            analysis_dir = os.path.join(result_dir, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            all_stats = np.load(stats_file, allow_pickle=True).item()
            mode_names = list(all_stats.keys())

            # Initialize containers
            mode_metrics = {
                name: {"sample_wise": [], "segment_wise": {t: [] for t in threshold_list}} for name in mode_names
            }

            num_classes = None
            class_metrics = {}

            # Populate metrics
            for mode_name, stats_list in all_stats.items():
                # Determine number of classes (exclude 'weighted_f1')
                if num_classes is None and stats_list:
                    num_classes = len(stats_list[0]["metrics_sample"]) - 1

                # Prepare per-class containers
                class_metrics[mode_name] = {
                    "sample": {str(c): [] for c in range(1, num_classes + 1)},
                    **{str(t): {str(c): [] for c in range(1, num_classes + 1)} for t in threshold_list},
                }

                for fold_stat in stats_list:
                    # Sample-wise
                    sample_f1 = fold_stat["metrics_sample"]["weighted_f1"]
                    mode_metrics[mode_name]["sample_wise"].append(sample_f1)

                    # Segment-wise
                    for t in threshold_list:
                        seg_f1 = fold_stat["metrics_segment"][str(t)]["weighted_f1"]
                        mode_metrics[mode_name]["segment_wise"][t].append(seg_f1)

                    # Per-class
                    for c in range(1, num_classes + 1):
                        class_metrics[mode_name]["sample"][str(c)].append(fold_stat["metrics_sample"][str(c)]["f1"])
                        for t in threshold_list:
                            class_metrics[mode_name][str(t)][str(c)].append(
                                fold_stat["metrics_segment"][str(t)][str(c)]["f1"]
                            )

            colors = color_palette("husl", len(mode_names))

            # 1. Segment-wise vs Sample-wise F1
            if flag_segment_validation and threshold_list:
                plt.figure(figsize=(12, 6))
                x = np.arange(len(threshold_list))
                width = 0.8 / (len(mode_names) + 1)

                # Segment-wise bars
                for i, mode_name in enumerate(mode_names):
                    means = [np.mean(mode_metrics[mode_name]["segment_wise"][t]) for t in threshold_list]
                    stds = [np.std(mode_metrics[mode_name]["segment_wise"][t]) for t in threshold_list]
                    positions = x + (i - len(mode_names) / 2 + 0.5) * width
                    bars = plt.bar(
                        positions,
                        means,
                        width,
                        label=f"{mode_name} (Segment)",
                        edgecolor="black",
                        linewidth=0.5,
                        color=colors[i],
                        alpha=0.8,
                    )
                    plt.errorbar(positions, means, yerr=stds, fmt="none", ecolor="black", capsize=4, alpha=0.6)
                    for bar, m in zip(bars, means):
                        plt.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            f"{m:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

                # Sample-wise bars
                for i, mode_name in enumerate(mode_names):
                    mean_s = np.mean(mode_metrics[mode_name]["sample_wise"])
                    std_s = np.std(mode_metrics[mode_name]["sample_wise"])
                    pos = len(threshold_list) + (i - len(mode_names) / 2 + 0.5) * width
                    bar = plt.bar(
                        pos,
                        mean_s,
                        width,
                        label=f"{mode_name} (Sample)",
                        edgecolor="black",
                        linewidth=0.5,
                        color=colors[i],
                        alpha=0.8,
                    )
                    plt.errorbar(pos, mean_s, yerr=std_s, fmt="none", ecolor="black", capsize=4, alpha=0.6)
                    plt.text(pos, mean_s, f"{mean_s:.2f}", ha="center", va="bottom", fontsize=8)

                plt.xlabel("Segmentation Threshold")
                plt.ylabel("Weighted F1 Score")
                plt.xticks(list(x) + [len(threshold_list)], [str(t) for t in threshold_list] + ["Sample"])
                plt.title(f"Version {version} ({dataset_name}): Segment-wise vs Sample-wise F1 Scores")
                plt.grid(axis="y", linestyle="--", alpha=0.5)
                plt.legend(loc="lower right", fontsize=9)
                plt.tight_layout()
                plt.savefig(os.path.join(analysis_dir, f"f1_comparison_{dataset_name}.png"), dpi=300)
                plt.close()

            # 2. Sample-wise F1 Scores per Fold
            plt.figure(figsize=(12, 6))
            bar_width = 0.8 / len(mode_names)
            num_folds = len(next(iter(mode_metrics.values()))["sample_wise"])
            folds = np.arange(1, num_folds + 1)

            highest_mean_score = 0  # Initialize variable to track the highest mean score

            for i, mode_name in enumerate(mode_names):
                scores = mode_metrics[mode_name]["sample_wise"]
                mean_score = np.mean(scores)
                highest_mean_score = max(highest_mean_score, mean_score)  # Update highest mean score
                positions = folds + (i - len(mode_names) / 2 + 0.5) * bar_width
                bars = plt.bar(
                    positions,
                    scores,
                    bar_width,
                    label=f"{mode_name} (Mean: {mean_score:.2f})",
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.8,
                )
                for bar, s in zip(bars, scores):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{s:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            plt.xlabel("Fold")
            plt.ylabel("Weighted F1 Score")
            plt.xticks(folds, [f"Fold {f}" for f in folds])
            plt.title(f"Version {version} ({dataset_name}): Sample-wise F1 Scores per Fold")
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.legend(loc="lower right", fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"f1_per_fold_{dataset_name}.png"), dpi=300)
            plt.close()

            logger.info(f"Mean Sample-wise F1 Score for {mode_name}: {mean_score:.2f}")
            logger.info(f"Analysis for version {version} ({dataset_name}) completed. Plots saved to {analysis_dir}")

            # Print the highest mean value with 3 digits after the decimal
            logger.info(f"Highest Mean Sample-wise F1 Score: {highest_mean_score:.3f}")
