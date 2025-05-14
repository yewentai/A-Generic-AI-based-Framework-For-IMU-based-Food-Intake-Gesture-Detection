#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Training Result Analysis Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-14
Description : This script analyzes training results:
              1. Comparative analysis of segment-wise and sample-wise metrics
              2. Support for multiple validation modes (original, mirrored, rotated)
              3. Per-class performance analysis
              4. Cross-validation fold analysis
===============================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette

if __name__ == "__main__":
    result_root = "results/new"
    # versions = ["DXI_MSTCN_DM"]
    versions = [d for d in sorted(os.listdir(result_root)) if os.path.isdir(os.path.join(result_root, d))]

    for version in versions:
        result_dir = os.path.join(result_root, version)
        config_file = os.path.join(result_dir, "validation_config.json")
        stats_file = os.path.join(result_dir, "validation_stats.npy")

        # Skip if required files are missing
        if not os.path.exists(config_file) or not os.path.exists(stats_file):
            print(f"Skipping {version}: missing config or stats file.")
            continue

        # Load configuration (if needed later)
        with open(config_file, "r") as f:
            config = json.load(f)

        # Load validation statistics
        all_stats = np.load(stats_file, allow_pickle=True).item()
        mode_names = list(all_stats.keys())

        # Safely extract sample-wise and segment-wise aggregated scores
        mode_metrics = {}
        for mode_name, stats_list in all_stats.items():
            sample_scores = []
            segment_scores = []
            for s in stats_list:
                # Sample-wise
                ms = s.get("metrics_sample")
                if isinstance(ms, dict):
                    # dict: preserve insertion order, last value is aggregated score
                    try:
                        sample_scores.append(list(ms.values())[-1])
                    except (IndexError, TypeError):
                        sample_scores.append(np.nan)
                else:
                    # list/array
                    try:
                        sample_scores.append(ms[-1])
                    except Exception:
                        sample_scores.append(np.nan)
                # Segment-wise
                seg = s.get("metrics_segment")
                if isinstance(seg, dict):
                    try:
                        segment_scores.append(list(seg.values())[-1])
                    except (IndexError, TypeError):
                        segment_scores.append(np.nan)
                else:
                    try:
                        segment_scores.append(seg[-1])
                    except Exception:
                        segment_scores.append(np.nan)
            mode_metrics[mode_name] = {
                "sample_wise": sample_scores,
                "segment_wise": segment_scores,
            }

        # Derive dataset name from stats file
        dataset_name = os.path.splitext(os.path.basename(stats_file))[0].split("_", 2)[-1]

        # Prepare output directory
        analysis_dir = os.path.join(result_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        # Plot sample-wise F1 scores per fold for each validation mode
        colors = color_palette("husl", len(mode_names))
        plt.figure(figsize=(12, 6))
        bar_width = 0.8 / len(mode_names)
        num_folds = len(next(iter(mode_metrics.values()))["sample_wise"])
        folds = np.arange(1, num_folds + 1)

        for i, mode_name in enumerate(mode_names):
            scores = mode_metrics[mode_name]["sample_wise"]
            mean_score = np.nanmean(scores)
            positions = folds + (i - len(mode_names) / 2 + 0.5) * bar_width
            bars = plt.bar(
                positions,
                scores,
                bar_width,
                color=colors[i],
                label=f"{mode_name} (Mean: {mean_score:.3f})",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.8,
            )
            for bar, s in zip(bars, scores):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{s:.3f}" if not np.isnan(s) else "nan",
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

        print(f"Saved sample-wise F1 plot for {version} ({dataset_name})")
