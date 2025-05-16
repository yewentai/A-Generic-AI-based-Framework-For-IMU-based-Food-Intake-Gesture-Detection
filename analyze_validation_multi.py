#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Training Result Analysis Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-15
Description : Plot segment-wise F1 scores at different thresholds for all versions
              with legend sorted by average F1 score.
===============================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette

if __name__ == "__main__":
    # Base directory for results
    result_root = "results/smooth/DXI"

    # Find all version directories
    raw_versions = [d for d in sorted(os.listdir(result_root)) if os.path.isdir(os.path.join(result_root, d))]

    # Collect metrics per version
    all_version_metrics = {}
    all_version_sample_metrics = {}

    for raw_version in raw_versions:
        # Determine display name
        display_version = raw_version
        if "S-" in raw_version:
            display_version = raw_version.split("S-")[-1]

        # Paths to config and stats
        result_dir = os.path.join(result_root, raw_version)
        config_file = os.path.join(result_dir, "validation_config.json")
        stats_file = os.path.join(result_dir, "validation_stats.npy")

        # Skip missing
        if not os.path.exists(config_file) or not os.path.exists(stats_file):
            print(f"Skipping {display_version}: missing config or stats file.")
            continue

        # Load thresholds from config
        with open(config_file, "r") as f:
            config = json.load(f)
        threshold_list = config.get("threshold_list", [])
        if not threshold_list:
            print(f"Skipping {display_version}: no thresholds specified.")
            continue

        # Load statistics
        all_stats = np.load(stats_file, allow_pickle=True).item()

        # ---- Segment-wise metrics ----
        seg_metrics = {str(t): [] for t in threshold_list}
        # ---- Sample-wise metrics ----
        sample_metrics = []

        for mode_name, stats_list in all_stats.items():
            for fold_stat in stats_list:
                # Collect segment-wise weighted F1
                for t in threshold_list:
                    seg_data = fold_stat.get("metrics_segment", {}).get(str(t), {})
                    seg_metrics[str(t)].append(seg_data.get("weighted_f1", np.nan))
                # Collect sample-wise weighted F1
                sample_metrics.append(fold_stat.get("metrics_sample", {}).get("weighted_f1", np.nan))

        all_version_metrics[display_version] = seg_metrics
        all_version_sample_metrics[display_version] = sample_metrics

    # Exit if no data
    if not all_version_metrics:
        print("No data to plot.")
        exit(0)

    # --- Segment-wise Plot ---
    # Determine thresholds (assumes same for all)
    sample_version = next(iter(all_version_metrics))
    thresholds = [float(t) for t in all_version_metrics[sample_version].keys()]

    # Compute average mean F1 per version for sorting
    avg_segment = {
        v: np.nanmean([np.nanmean(all_version_metrics[v][str(t)]) for t in thresholds]) for v in all_version_metrics
    }

    # Sort versions by descending segment-wise average F1
    sorted_versions = sorted(all_version_metrics.keys(), key=lambda v: avg_segment[v], reverse=True)

    # Plot segment-wise performance
    colors = color_palette("tab10", len(sorted_versions))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "H", "p", "d", "8", ">", "<"]
    plt.figure(figsize=(10, 6))
    for idx, version in enumerate(sorted_versions):
        seg_vals = all_version_metrics[version]
        means = [np.nanmean(seg_vals[str(t)]) for t in thresholds]
        stds = [np.nanstd(seg_vals[str(t)]) for t in thresholds]
        plt.errorbar(
            thresholds,
            means,
            yerr=stds,
            label=version,
            marker=markers[idx % len(markers)],
            markersize=4,
            linewidth=1.0,
            alpha=0.8,
            color=colors[idx],
        )
    plt.xlabel("Segmentation Threshold")
    plt.ylabel("Weighted F1 Score")
    plt.title("Segment-wise F1 Scores Across Thresholds")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize=8, title="Smooth Loss")
    plt.tight_layout()
    out_file_seg = os.path.join(result_root, "DXI_smooth_segmentwise.png")
    plt.savefig(out_file_seg, dpi=300)
    plt.close()
    print(f"Saved sorted segment-wise F1 plot: {out_file_seg}")
