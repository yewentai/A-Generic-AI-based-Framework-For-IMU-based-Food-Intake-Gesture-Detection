#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Training Result Analysis Script (DX/FD Datasets)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-29
Description : This script analyzes training results for DX/FD datasets with:
              1. Comparative analysis of segment-wise and sample-wise metrics
              2. Support for multiple validation modes (original, mirrored, rotated)
              3. Per-class performance analysis
              4. Cross-validation fold analysis
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette
from dl_validate import THRESHOLD_LIST  # [0.1, 0.25, 0.5, 0.75]

if __name__ == "__main__":
    result_root = "result"
    # Discover all version directories
    # versions = ["FDI_BOTH_MSTCN_1"]  # Uncomment to manually specify versions
    versions = [d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d))]
    versions.sort()

    for version in versions:
        result_dir = os.path.join(result_root, version)
        stats_files = [
            os.path.join(result_dir, f)
            for f in os.listdir(result_dir)
            if f.startswith("validation_stats") and f.endswith(".npy")
        ]
        if not stats_files:
            print(f"No validation_stats.npy files found for version {version}, skipping.")
            continue

        # Extract dataset names from the file names
        dataset_names = [
            os.path.splitext(f)[0].split("_", 2)[-1]
            for f in os.listdir(result_dir)
            if f.startswith("validation_stats") and f.endswith(".npy")
        ]

        for stats_file, dataset_name in zip(stats_files, dataset_names):
            if not os.path.exists(stats_file):
                print(f"No validation_stats.npy found for version {version}, skipping.")
                continue

            # Create analysis output dir
            analysis_dir = os.path.join(result_dir, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            # Load all validation statistics
            all_stats = np.load(stats_file, allow_pickle=True).item()

            # Prepare containers for metrics
            mode_names = list(all_stats.keys())
            mode_metrics = {
                name: {"sample_wise": [], "segment_wise": {t: [] for t in THRESHOLD_LIST}} for name in mode_names
            }

            # Extract per-class metrics if needed
            # class_metrics[name]["sample" or threshold][class_label] = list of f1s
            num_classes = None
            class_metrics = {}

            for mode_name, stats_list in all_stats.items():
                # Determine number of classes from first fold sample metrics
                if num_classes is None and stats_list:
                    num_classes = len(stats_list[0]["metrics_sample"]) - 1  # exclude weighted_f1 key

                # initialize class_metrics structure
                class_metrics[mode_name] = {"sample": {str(c): [] for c in range(1, num_classes + 1)}}
                for t in THRESHOLD_LIST:
                    class_metrics[mode_name][str(t)] = {str(c): [] for c in range(1, num_classes + 1)}

                # Populate metrics
                for fold_stat in stats_list:
                    # sample-wise weighted F1
                    sample_f1 = fold_stat["metrics_sample"]["weighted_f1"]
                    mode_metrics[mode_name]["sample_wise"].append(sample_f1)

                    # segment-wise weighted F1 per threshold
                    for t in THRESHOLD_LIST:
                        seg_f1 = fold_stat["metrics_segment"][str(t)]["weighted_f1"]
                        mode_metrics[mode_name]["segment_wise"][t].append(seg_f1)

                    # per-class sample and segment f1s
                    for c in range(1, num_classes + 1):
                        class_metrics[mode_name]["sample"][str(c)].append(fold_stat["metrics_sample"][str(c)]["f1"])
                        for t in THRESHOLD_LIST:
                            class_metrics[mode_name][str(t)][str(c)].append(
                                fold_stat["metrics_segment"][str(t)][str(c)]["f1"]
                            )

            colors = color_palette("husl", len(mode_names))

            # ======================================================================
            # 1. Segment-wise and Sample-wise F1 Score Comparison
            # ======================================================================
            plt.figure(figsize=(12, 6))
            x = np.arange(len(THRESHOLD_LIST))
            width = 0.8 / len(mode_names)

            # Plot segment-wise bars
            for i, mode_name in enumerate(mode_names):
                means = [np.mean(mode_metrics[mode_name]["segment_wise"][t]) for t in THRESHOLD_LIST]
                stds = [np.std(mode_metrics[mode_name]["segment_wise"][t]) for t in THRESHOLD_LIST]
                positions = x + (i - len(mode_names) / 2 + 0.5) * width

                bars = plt.bar(
                    positions,
                    means,
                    width,
                    label=f"{mode_name} (Segment)",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5,
                    color=colors[i],
                )
                plt.errorbar(positions, means, yerr=stds, fmt="none", ecolor="black", capsize=4, alpha=0.6)

                # Annotate bar values
                for bar, m in zip(bars, means):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{m:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            # Plot sample-wise bars at end
            for i, mode_name in enumerate(mode_names):
                sample_mean = np.mean(mode_metrics[mode_name]["sample_wise"])
                sample_std = np.std(mode_metrics[mode_name]["sample_wise"])
                sample_pos = len(THRESHOLD_LIST) + (i - len(mode_names) / 2 + 0.5) * width

                bar = plt.bar(
                    sample_pos, sample_mean, width, alpha=0.8, edgecolor="black", linewidth=0.5, color=colors[i]
                )
                plt.errorbar(sample_pos, sample_mean, yerr=sample_std, fmt="none", ecolor="black", capsize=4, alpha=0.6)
                plt.text(sample_pos, sample_mean, f"{sample_mean:.2f}", ha="center", va="bottom", fontsize=8)

            plt.xlabel("Segmentation Threshold")
            plt.ylabel("Weighted F1 Score")
            plt.xticks(list(x) + [len(THRESHOLD_LIST)], [str(t) for t in THRESHOLD_LIST] + ["Sample"])
            plt.title(f"Version {version} ({dataset_name}): Segment-wise vs Sample-wise F1 Scores")
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.legend(loc="lower right", fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"f1_comparison_{dataset_name}.png"), dpi=300)
            plt.close()

            # ======================================================================
            # 2. Sample-wise F1 Scores per Fold
            # ======================================================================
            plt.figure(figsize=(12, 6))
            bar_width = 0.8 / len(mode_names)
            num_folds = len(next(iter(mode_metrics.values()))["sample_wise"])
            folds = np.arange(1, num_folds + 1)

            for i, mode_name in enumerate(mode_names):
                scores = mode_metrics[mode_name]["sample_wise"]
                positions = folds + (i - len(mode_names) / 2 + 0.5) * bar_width

                bars = plt.bar(
                    positions, scores, bar_width, alpha=0.8, edgecolor="black", linewidth=0.5, label=f"{mode_name}"
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

            print(f"Analysis for version {version} ({dataset_name}) completed. Plots saved to {analysis_dir}")
