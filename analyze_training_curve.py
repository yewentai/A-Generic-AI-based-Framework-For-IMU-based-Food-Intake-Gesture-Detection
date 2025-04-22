#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Training Loss Curve Plotter
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-14
Description : This script plots training loss curves per fold for the specified
              experiment version and validation type.
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    result_root = "result"
    versions = ["202504151704_LEFT"]  # Specify the version to analyze
    # versions = [d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d))]
    versions.sort()

    for version in versions:
        result_dir = os.path.join(result_root, version)
        train_file = "train_stats.npy"
        loss_curve_plot_dir = os.path.join(result_dir, "loss_curve")
        os.makedirs(loss_curve_plot_dir, exist_ok=True)

        loss_curve_plot_dir = os.path.join(result_dir, "loss_curve")
        os.makedirs(loss_curve_plot_dir, exist_ok=True)

        # Find corresponding training stats file
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

                plt.title(f"Training Loss Over Epochs (Fold {fold})")
                plt.xlabel("Epoch")
                plt.ylabel("Loss (Log Scale)")
                plt.grid(True, which="both", linestyle="--", alpha=0.6)
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(loss_curve_plot_dir, f"train_loss_fold{fold}.png"),
                    dpi=300,
                )
                plt.close()
