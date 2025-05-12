#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Training Result Analysis Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-12
Description : Plot segment-wise F1 scores at different thresholds for all versions
===============================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette

if __name__ == "__main__":
    # List all versions to compare
    result_root = "results/smooth/DXI"
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

    # Plot all versions on the same figure
    if not all_version_metrics:
        print("No data to plot.")
        exit(0)

    # Use consistent colors for each version
    colors = color_palette("husl", len(all_version_metrics))
    plt.figure(figsize=(10, 6))

    # Determine common threshold axis
    # Assume all versions share the same thresholds as the first entry
    sample_version = next(iter(all_version_metrics))
    thresholds = [float(t) for t in all_version_metrics[sample_version].keys()]

    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "+", "x", "X", "d"]  # List of markers
    for idx, (version, seg_dict) in enumerate(all_version_metrics.items()):
        means = [np.nanmean(seg_dict[str(t)]) for t in thresholds]
        stds = [np.nanstd(seg_dict[str(t)]) for t in thresholds]
        marker = markers[idx % len(markers)]  # Cycle through markers if more versions than markers
        plt.errorbar(
            thresholds, means, yerr=stds, label=version, marker=marker, linewidth=1, alpha=0.8, color=colors[idx]
        )

    plt.xlabel("Segmentation Threshold")
    plt.ylabel("Weighted F1 Score")
    plt.title("Segment-wise F1 Scores Across Thresholds for All Versions")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    # Save plot
    out_file = os.path.join(result_root, "all_versions_f1_segment_thresholds.png")
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Saved combined segment-wise F1 plot: {out_file}")
