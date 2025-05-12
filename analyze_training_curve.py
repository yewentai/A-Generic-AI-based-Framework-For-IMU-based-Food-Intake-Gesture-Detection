#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Training Loss Curve Plotter
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-12
Description : This script plots fold-wise training loss curves (total, cross-entropy,
              and smooth losses) for a specified experiment version. It loads training
              statistics from a .npy file and generates log-scale plots for visual
              inspection of training behavior across epochs.
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate  # make sure to install with: pip install tabulate


def plot_loss_curves(result_root):
    versions = sorted(d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d)))

    for version in versions:
        result_dir = os.path.join(result_root, version)
        stats_path = os.path.join(result_dir, "training_stats.npy")
        loss_curve_plot_dir = os.path.join(result_dir, "loss_curve")
        os.makedirs(loss_curve_plot_dir, exist_ok=True)

        if not os.path.exists(stats_path):
            print(f"  [!] No training_stats.npy found for version {version}, skipping.")
            continue

        training_stats = np.load(stats_path, allow_pickle=True).tolist()
        folds = sorted(set(entry["fold"] for entry in training_stats))

        for fold in folds:
            # collect stats for this fold
            stats_fold = [e for e in training_stats if e["fold"] == fold]
            epochs = sorted(set(e["epoch"] for e in stats_fold))

            loss_per_epoch = {epoch: [] for epoch in epochs}
            loss_ce_per_epoch = {epoch: [] for epoch in epochs}
            loss_smooth_per_epoch = {epoch: [] for epoch in epochs}

            for entry in stats_fold:
                ep = entry["epoch"]
                loss_per_epoch[ep].append(entry["train_loss"])
                loss_ce_per_epoch[ep].append(entry["train_loss_ce"])
                loss_smooth_per_epoch[ep].append(entry["train_loss_smooth"])

            mean_loss = [np.mean(loss_per_epoch[e]) for e in epochs]
            mean_ce = [np.mean(loss_ce_per_epoch[e]) for e in epochs]
            mean_sm = [np.mean(loss_smooth_per_epoch[e]) for e in epochs]

            plt.figure(figsize=(10, 6))
            plt.plot(epochs, mean_loss, label="Total Loss")
            plt.plot(epochs, mean_ce, label="Cross Entropy Loss", linestyle="--")
            plt.plot(epochs, mean_sm, label="Smooth Loss", linestyle=":")
            plt.yscale("log")

            plt.title(f"Training Loss Over Epochs (Version {version}, Fold {fold})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (log scale)")
            plt.grid(True, which="both", linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()

            out_path = os.path.join(loss_curve_plot_dir, f"train_loss_fold{fold}.png")
            plt.savefig(out_path, dpi=300)
            plt.close()
        print(f"[+] Plotted loss curves for version {version}")


def summarize_losses(result_root):
    versions = sorted(d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d)))

    summary = []
    for version in versions:
        stats_path = os.path.join(result_root, version, "training_stats.npy")
        if not os.path.exists(stats_path):
            continue

        stats = np.load(stats_path, allow_pickle=True).tolist()
        all_ce = np.array([e["train_loss_ce"] for e in stats])
        all_smooth = np.array([e["train_loss_smooth"] for e in stats])

        mean_ce = all_ce.mean()
        mean_smooth = all_smooth.mean()
        ratio = mean_ce / mean_smooth if mean_smooth != 0 else np.nan

        summary.append(
            {
                "Version": version,
                "Mean CE Loss": mean_ce,
                "Mean Smooth Loss": mean_smooth,
                "CE Ratio/Smooth": ratio,
            }
        )

    print("\nQuantitative comparison of CE vs. Smooth loss by version:\n")
    print(tabulate(summary, headers="keys", tablefmt="github", floatfmt=".4f"))


if __name__ == "__main__":
    # root directory containing per-version subdirectories
    result_root = "results/smooth/DXI"

    print("== Plotting loss curves ==")
    plot_loss_curves(result_root)

    print("\n== Summarizing CE vs. Smooth losses ==")
    summarize_losses(result_root)
