#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Training Result Comparison Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-21
Description : Plot segment-wise F1 scores at different thresholds for specified
              versions and evaluation modes. Used for direct comparison.
===============================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette
import matplotlib as mpl

mpl.rcParams.update({"font.size": 16})  # Update default font size


def load_version_mode_metrics(result_root, version, mode_filter):
    result_dir = os.path.join(result_root, version)
    config_file = os.path.join(result_dir, "validation_config.json")
    stats_file = os.path.join(result_dir, "validation_stats.npy")

    if not os.path.exists(config_file) or not os.path.exists(stats_file):
        print(f"Skipping {version}: missing config or stats file.")
        return None, None

    with open(config_file, "r") as f:
        config = json.load(f)
    threshold_list = config.get("threshold_list", [])
    if not threshold_list:
        print(f"Skipping {version}: no thresholds specified.")
        return None, None

    all_stats = np.load(stats_file, allow_pickle=True).item()
    seg_metrics = {str(t): [] for t in threshold_list}

    for mode_name, stats_list in all_stats.items():
        if mode_name != mode_filter:
            continue
        for fold_stat in stats_list:
            for t in threshold_list:
                seg_data = fold_stat.get("metrics_segment", {}).get(str(t), {})
                seg_metrics[str(t)].append(seg_data.get("weighted_f1", np.nan))

    return threshold_list, seg_metrics


if __name__ == "__main__":
    result_root = "results/aug/DXI"
    # List of (version, mode) combinations to compare
    combs = [
        # ("DXI_MSTCN", "planar_rotated"),
        # ("DXI_MSTCN_AR", "planar_rotated"),
        ("DXI_MSTCN", "axis_permuted"),
        ("DXI_MSTCN_AP", "axis_permuted"),
        # Add more (version, mode) tuples as needed
    ]

    all_curves = {}
    thresholds = None

    for version, mode in combs:
        threshold_list, seg_metrics = load_version_mode_metrics(result_root, version, mode)
        if threshold_list is None or seg_metrics is None:
            continue
        thresholds = [float(t) for t in threshold_list]
        means = [np.nanmean(seg_metrics[str(t)]) for t in threshold_list]
        stds = [np.nanstd(seg_metrics[str(t)]) for t in threshold_list]
        all_curves[version] = {"mean": means, "std": stds}

    if not all_curves:
        print("No data to plot.")
        exit(0)

    # Plot
    colors = color_palette("tab10", len(all_curves))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "H", "p", "d", "8", ">", "<"]

    # Map version names to descriptive labels for the legend
    legend_labels = {
        "DXI_MSTCN": "No augmentation",
        "DXI_MSTCN_AR": "Augmentation Planar Rotation",
        "DXI_MSTCN_AP": "Augmentation Axis Permutation",
    }

    # Sort all_curves by average value (descending)
    sorted_items = sorted(all_curves.items(), key=lambda x: np.nanmean(x[1]["mean"]), reverse=True)

    plt.figure(figsize=(10, 6))
    for idx, (version, vals) in enumerate(sorted_items):
        label = legend_labels.get(version, version)
        plt.errorbar(
            thresholds,
            vals["mean"],
            yerr=vals["std"],
            label=label,
            marker=markers[idx % len(markers)],
            markersize=4,
            linewidth=1.0,
            alpha=0.8,
            color=colors[idx],
        )
    plt.xlabel("Segmentation Threshold")
    plt.ylabel("Weighted F1 Score")
    # Format mode name for title and filename
    mode_title = mode.replace("_", " ").title()
    plt.title(f"Segment-wise F1 Scores Comparison (Dataset: DXI, Model: MSTCN)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    out_file = os.path.join(result_root, f"compare_{mode}_DXI.pdf")
    plt.savefig(out_file, format="pdf", dpi=300)
    plt.close()
    print(f"Saved comparison plot: {out_file}")

    # --- Save data for LaTeX table ---
    table_data = {}
    for version, vals in sorted_items:
        label = legend_labels.get(version, version)
        table_data[label] = {
            str(t): {"mean": round(float(m), 4), "std": round(float(s), 4)}
            for t, m, s in zip(thresholds, vals["mean"], vals["std"])
        }

    json_out_path = os.path.join(result_root, f"compare_{mode}_DXI_table.json")
    with open(json_out_path, "w") as f:
        json.dump(table_data, f, indent=4)
    print(f"Saved F1 statistics for LaTeX table: {json_out_path}")
