#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Training Result Analysis Script (Extended)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Last Edited : 2025-04-22
Description : This script analyzes training results for Oreba/Clemson datasets
              produced by the extended validation script, with comparative
              visualizations of segment-wise and sample-wise metrics.
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dl_validate_ext import THRESHOLD_LIST  # [0.1, 0.25, 0.5, 0.75]

if __name__ == "__main__":
    result_root = "result"
    # Discover all version directories
    versions = [d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d))]
    versions.sort()

    for version in versions:
        result_dir = os.path.join(result_root, version)

        # Check for both Oreba and Clemson stats files
        stats_files = {
            "Oreba": os.path.join(result_dir, "validation_stats_Oreba.npy"),
            "Clemson": os.path.join(result_dir, "validation_stats_Clemson.npy"),
        }

        for dataset_name, stats_file in stats_files.items():
            if not os.path.exists(stats_file):
                print(f"No {stats_file} found for version {version}, skipping.")
                continue

            # Create analysis output dir
            analysis_dir = os.path.join(result_dir, "analysis", dataset_name)
            os.makedirs(analysis_dir, exist_ok=True)

            # Load validation statistics
            validate_stats = np.load(stats_file, allow_pickle=True)

            # Prepare containers for metrics
            metrics = {"sample_wise": [], "segment_wise": {t: [] for t in THRESHOLD_LIST}}

            # Extract per-class metrics
            num_classes = None
            class_metrics = {"sample": {}, "segment": {t: {} for t in THRESHOLD_LIST}}

            # Determine number of classes from first fold sample metrics
            if len(validate_stats) > 0:
                num_classes = len(validate_stats[0]["metrics_sample"]) - 1  # exclude weighted_f1 key

                # Initialize class metrics structure
                for c in range(1, num_classes + 1):
                    class_metrics["sample"][str(c)] = []
                    for t in THRESHOLD_LIST:
                        class_metrics["segment"][t][str(c)] = []

            # Populate metrics
            for fold_stat in validate_stats:
                # sample-wise weighted F1
                sample_f1 = fold_stat["metrics_sample"]["weighted_f1"]
                metrics["sample_wise"].append(sample_f1)

                # segment-wise weighted F1 per threshold
                for t in THRESHOLD_LIST:
                    seg_f1 = fold_stat["metrics_segment"][str(t)]["weighted_f1"]
                    metrics["segment_wise"][t].append(seg_f1)

                # per-class sample and segment f1s
                for c in range(1, num_classes + 1):
                    class_metrics["sample"][str(c)].append(fold_stat["metrics_sample"][str(c)]["f1"])
                    for t in THRESHOLD_LIST:
                        class_metrics["segment"][t][str(c)].append(fold_stat["metrics_segment"][str(t)][str(c)]["f1"])

            # ======================================================================
            # 1. Segment-wise and Sample-wise F1 Score Comparison
            # ======================================================================
            plt.figure(figsize=(12, 6))
            x = np.arange(len(THRESHOLD_LIST))
            width = 0.8

            # Plot segment-wise bars
            means = [np.mean(metrics["segment_wise"][t]) for t in THRESHOLD_LIST]
            stds = [np.std(metrics["segment_wise"][t]) for t in THRESHOLD_LIST]

            bars = plt.bar(
                x, means, width, label="Segment-wise", alpha=0.8, edgecolor="black", linewidth=0.5, color="skyblue"
            )
            plt.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", capsize=4, alpha=0.6)

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

            # Plot sample-wise bar at end
            sample_mean = np.mean(metrics["sample_wise"])
            sample_std = np.std(metrics["sample_wise"])
            sample_pos = len(THRESHOLD_LIST)

            bar = plt.bar(
                sample_pos, sample_mean, width, alpha=0.8, edgecolor="black", linewidth=0.5, color="lightgreen"
            )
            plt.errorbar(sample_pos, sample_mean, yerr=sample_std, fmt="none", ecolor="black", capsize=4, alpha=0.6)
            plt.text(sample_pos, sample_mean, f"{sample_mean:.2f}", ha="center", va="bottom", fontsize=8)

            plt.xlabel("Segmentation Threshold")
            plt.ylabel("Weighted F1 Score")
            plt.xticks(list(x) + [len(THRESHOLD_LIST)], [str(t) for t in THRESHOLD_LIST] + ["Sample"])
            plt.title(f"Version {version} - {dataset_name}: Segment-wise vs Sample-wise F1 Scores")
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.legend(loc="lower right", fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"f1_comparison_{version}.png"), dpi=300)
            plt.close()

            # ======================================================================
            # 2. Sample-wise F1 Scores per Fold
            # ======================================================================
            plt.figure(figsize=(12, 6))
            bar_width = 0.8
            num_folds = len(metrics["sample_wise"])
            folds = np.arange(1, num_folds + 1)

            scores = metrics["sample_wise"]
            bars = plt.bar(folds, scores, bar_width, alpha=0.8, edgecolor="black", linewidth=0.5, color="lightgreen")

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
            plt.title(f"Version {version} - {dataset_name}: Sample-wise F1 Scores per Fold")
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"f1_per_fold_{version}.png"), dpi=300)
            plt.close()

            # ======================================================================
            # 3. Sample-wise Per-Class F1 Scores
            # ======================================================================
            plt.figure(figsize=(15, 8))

            # Plot sample-wise per-class F1s
            x = np.arange(num_classes)
            width = 0.8

            sample_means = [np.mean(class_metrics["sample"][str(c)]) for c in range(1, num_classes + 1)]
            sample_stds = [np.std(class_metrics["sample"][str(c)]) for c in range(1, num_classes + 1)]

            bars = plt.bar(
                x,
                sample_means,
                width,
                label="Sample-wise",
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
                color="lightgreen",
            )
            plt.errorbar(x, sample_means, yerr=sample_stds, fmt="none", ecolor="black", capsize=4, alpha=0.6)

            plt.xlabel("Class")
            plt.ylabel("F1 Score")
            plt.title(f"Version {version} - {dataset_name}: Sample-wise Per-Class F1 Scores")
            plt.xticks(x, [f"Class {c}" for c in range(1, num_classes + 1)])
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"f1_sample_per_class_{version}.png"), dpi=300)
            plt.close()

            print(
                f"Sample-wise analysis for version {version} - {dataset_name} completed. Plots saved to {analysis_dir}"
            )
