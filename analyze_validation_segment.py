#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Training Result Analysis Script (Sorted Legend)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-14
Description : Plot segment-wise F1 scores at different thresholds for all versions
              with legend sorted by average F1 score
===============================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette

if __name__ == "__main__":
    # List all versions to compare
    result_root = "results/DXI"
    versions = [d for d in sorted(os.listdir(result_root)) if os.path.isdir(os.path.join(result_root, d))]

    # Dictionary to collect segment-wise metrics per version
    all_version_metrics = {}

    for version in versions:
        result_dir = os.path.join(result_root, version)
        config_file = os.path.join(result_dir, "validation_config.json")
        stats_file = os.path.join(result_dir, "validation_stats.npy")

        # Skip if required files are missing
        if not os.path.exists(config_file) or not os.path.exists(stats_file):
            print(f"Skipping {version}: missing config or stats file.")
            continue

        # Load thresholds
        with open(config_file, "r") as f:
            config = json.load(f)
        threshold_list = config.get("threshold_list", []) or []
        if not threshold_list:
            print(f"Skipping {version}: no thresholds specified.")
            continue

        # Load validation statistics
        all_stats = np.load(stats_file, allow_pickle=True).item()

        # Collect segment-wise weighted F1 per threshold
        seg_metrics = {str(t): [] for t in threshold_list}
        for mode_name, stats_list in all_stats.items():
            for fold_stat in stats_list:
                for t in threshold_list:
                    seg_data = fold_stat.get("metrics_segment", {}).get(str(t), {})
                    seg_metrics[str(t)].append(seg_data.get("weighted_f1", np.nan))

        all_version_metrics[version] = seg_metrics

    # Check if data exists
    if not all_version_metrics:
        print("No data to plot.")
        exit(0)

    # Determine common threshold axis (assumes all versions share same thresholds)
    sample_version = next(iter(all_version_metrics))
    thresholds = [float(t) for t in all_version_metrics[sample_version].keys()]

    # Compute average mean F1 per version for sorting
    avg_metrics = {}
    for version, seg_dict in all_version_metrics.items():
        means = [np.nanmean(seg_dict[str(t)]) for t in thresholds]
        avg_metrics[version] = np.nanmean(means)

    # Sort versions by descending average F1 score
    sorted_versions = sorted(all_version_metrics.keys(), key=lambda v: avg_metrics[v], reverse=True)

    # Prepare plot
    colors = color_palette("tab10", len(sorted_versions))
    markers = [
        "o",
        "s",
        "D",
        "^",
        "v",
        "P",
        "X",
        "*",
        "h",
        "H",
        "p",
        "d",
        "8",
        ">",
        "<",
    ]  # Distinct markers

    plt.figure(figsize=(10, 6))

    # Plot in sorted order
    for idx, version in enumerate(sorted_versions):
        seg_dict = all_version_metrics[version]
        means = [np.nanmean(seg_dict[str(t)]) for t in thresholds]
        stds = [np.nanstd(seg_dict[str(t)]) for t in thresholds]
        marker = markers[idx % len(markers)]
        plt.errorbar(
            thresholds,
            means,
            yerr=stds,
            label=f"{version} (avg: {avg_metrics[version]:.3f})",
            marker=marker,
            markersize=4,
            linewidth=1.0,
            alpha=0.8,
            color=colors[idx],
        )

    plt.xlabel("Segmentation Threshold")
    plt.ylabel("Weighted F1 Score")
    plt.title("Segment-wise F1 Scores Across Thresholds (Sorted by Avg F1)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize=8, title="Version (Avg F1)")
    plt.tight_layout()

    # Save plot
    out_file = os.path.join(result_root, "all_versions_f1_segment_thresholds_sorted.png")
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved sorted segment-wise F1 plot: {out_file}")
