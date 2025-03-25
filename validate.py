#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Validation Script
-------------------------------------------------------------------------------
Author      : Joseph Yep (improved by ChatGPT)
Email       : yewentai126@gmail.com
Version     : 2.1
Created     : 2025-03-24
Description : Validates a trained MSTCN model on the full IMU dataset using
              saved checkpoints for each fold. The script performs inference,
              computes evaluation metrics, and generates performance plots.
Usage       : $ python validate.py
===============================================================================
"""

import os
import json
import glob
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import CohenKappa, MatthewsCorrCoef
from torch.utils.data import DataLoader, Subset

# Import custom modules from the components package
from components.model_mstcn import MSTCN
from components.post_processing import post_process_predictions
from components.evaluation import segment_evaluation
from components.datasets import IMUDataset
from components.pre_processing import hand_mirroring


def main():
    # -------------------------------------------------------------------------
    # Configuration and Setup
    # -------------------------------------------------------------------------
    RESULT_DIR = "result"
    all_versions = glob.glob(os.path.join(RESULT_DIR, "*"))
    if not all_versions:
        raise ValueError("No result directories found in 'result'.")
    # Select the latest result directory by modification time
    RESULT_VERSION = max(all_versions, key=os.path.getmtime).split(os.sep)[-1]
    result_dir = os.path.join(RESULT_DIR, RESULT_VERSION)
    config_file = os.path.join(result_dir, "config.json")

    # Load configuration parameters from JSON file
    with open(config_file, "r") as f:
        config_info = json.load(f)

    # Set key parameters from configuration
    DATASET = config_info["dataset"]
    NUM_CLASSES = config_info["num_classes"]
    NUM_STAGES = config_info["num_stages"]
    NUM_LAYERS = config_info["num_layers"]
    NUM_HEADS = config_info["num_heads"]
    INPUT_DIM = config_info["input_dim"]
    NUM_FILTERS = config_info["num_filters"]
    KERNEL_SIZE = config_info["kernel_size"]
    DROPOUT = config_info["dropout"]
    SAMPLING_FREQ = config_info["sampling_freq"]
    WINDOW_SIZE = config_info["window_size"]
    BATCH_SIZE = config_info["batch_size"]
    FLAG_MIRROR = config_info.get("mirroring", False)
    NUM_FOLDS = config_info.get("num_folds", 7)
    DEBUG_PLOT = False
    THRESHOLD = 0.5

    # Determine dataset directory and task based on dataset type
    if DATASET.startswith("DX"):
        sub_version = DATASET.replace("DX", "").upper() or "I"
        DATA_DIR = os.path.join("dataset", "DX", f"DX-{sub_version}")
        TASK = "binary"
    elif DATASET.startswith("FD"):
        sub_version = DATASET.replace("FD", "").upper() or "I"
        DATA_DIR = os.path.join("dataset", "FD", f"FD-{sub_version}")
        TASK = "multiclass"
    else:
        raise ValueError(f"Invalid dataset: {DATASET}")

    # Use a smaller number of workers for validation
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Data Loading and Pre-processing
    # -------------------------------------------------------------------------
    # Define file paths for left and right hand data
    x_left_path = os.path.join(DATA_DIR, "X_L.pkl")
    y_left_path = os.path.join(DATA_DIR, "Y_L.pkl")
    x_right_path = os.path.join(DATA_DIR, "X_R.pkl")
    y_right_path = os.path.join(DATA_DIR, "Y_R.pkl")

    # Load data from pickle files
    with open(x_left_path, "rb") as f:
        X_L = np.array(pickle.load(f), dtype=object)
    with open(y_left_path, "rb") as f:
        Y_L = np.array(pickle.load(f), dtype=object)
    with open(x_right_path, "rb") as f:
        X_R = np.array(pickle.load(f), dtype=object)
    with open(y_right_path, "rb") as f:
        Y_R = np.array(pickle.load(f), dtype=object)

    # Optionally apply hand mirroring to left-hand data
    if FLAG_MIRROR:
        X_L = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)

    # Merge left and right data into a single dataset
    X = np.concatenate([X_L, X_R], axis=0)
    Y = np.concatenate([Y_L, Y_R], axis=0)
    full_dataset = IMUDataset(X, Y, sequence_length=WINDOW_SIZE)

    # Optionally load FDIII data for augmentation (for FDII/FDI datasets)
    fdiii_dataset = None
    if DATASET in ["FDII", "FDI"]:
        fdiii_dir = os.path.join("dataset", "FD", "FD-III")
        with open(os.path.join(fdiii_dir, "X_L.pkl"), "rb") as f:
            X_L_fdiii = np.array(pickle.load(f), dtype=object)
        with open(os.path.join(fdiii_dir, "Y_L.pkl"), "rb") as f:
            Y_L_fdiii = np.array(pickle.load(f), dtype=object)
        with open(os.path.join(fdiii_dir, "X_R.pkl"), "rb") as f:
            X_R_fdiii = np.array(pickle.load(f), dtype=object)
        with open(os.path.join(fdiii_dir, "Y_R.pkl"), "rb") as f:
            Y_R_fdiii = np.array(pickle.load(f), dtype=object)
        X_fdiii = np.concatenate([X_L_fdiii, X_R_fdiii], axis=0)
        Y_fdiii = np.concatenate([Y_L_fdiii, Y_R_fdiii], axis=0)
        fdiii_dataset = IMUDataset(X_fdiii, Y_fdiii, sequence_length=WINDOW_SIZE)

    # Retrieve cross-validation folds from configuration
    validate_folds = config_info.get("validate_folds")
    if validate_folds is None:
        raise ValueError("No 'validate_folds' found in the configuration file.")

    # -------------------------------------------------------------------------
    # Cross-Validation Loop for Model Validation
    # -------------------------------------------------------------------------
    validating_statistics = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for fold, validate_subjects in enumerate(
        tqdm(validate_folds, desc="K-Fold", leave=True)
    ):
        # Define the checkpoint path for the current fold
        checkpoint_path = os.path.join(result_dir, f"best_model_fold{fold+1}.pth")
        if not os.path.exists(checkpoint_path):
            continue

        # Initialize the MSTCN model and load its checkpoint
        model = MSTCN(
            num_stages=NUM_STAGES,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            input_dim=INPUT_DIM,
            num_filters=NUM_FILTERS,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT,
        ).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Remove 'module.' prefix if present (from DistributedDataParallel)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.eval()

        # Determine validation indices based on subject IDs
        validate_indices = [
            i
            for i, subject in enumerate(full_dataset.subject_indices)
            if subject in validate_subjects
        ]
        validate_loader = DataLoader(
            Subset(full_dataset, validate_indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        all_predictions = []
        all_labels = []

        # Inference loop: process each batch in validation loader
        with torch.no_grad():
            for batch_x, batch_y in tqdm(
                validate_loader, desc=f"Validating Fold {fold+1}", leave=False
            ):
                batch_x = batch_x.permute(0, 2, 1).to(device)
                outputs = model(batch_x)
                # Use output from the last stage for prediction
                last_stage_output = outputs[:, -1, :, :]
                probabilities = F.softmax(last_stage_output, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                all_predictions.extend(predictions.view(-1).cpu().numpy())
                all_labels.extend(batch_y.view(-1).cpu().numpy())

        # Compute label distribution from the ground truth
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        label_distribution = {
            float(label): int(count) for label, count in zip(unique_labels, counts)
        }

        preds_tensor = torch.tensor(all_predictions)
        labels_tensor = torch.tensor(all_labels)

        # Calculate sample-wise metrics (for each class excluding class 0)
        metrics_sample = {}
        for label in range(1, NUM_CLASSES):
            tp = torch.sum((preds_tensor == label) & (labels_tensor == label)).item()
            fp = torch.sum((preds_tensor == label) & (labels_tensor != label)).item()
            fn = torch.sum((preds_tensor != label) & (labels_tensor == label)).item()
            denominator = 2 * tp + fp + fn
            f1 = 2 * tp / denominator if denominator != 0 else 0.0
            metrics_sample[str(label)] = {"tp": tp, "fp": fp, "fn": fn, "f1": f1}

        # Compute additional evaluation metrics: Cohen Kappa and Matthews CorrCoef
        cohen_kappa_val = CohenKappa(num_classes=NUM_CLASSES, task=TASK)(
            preds_tensor, labels_tensor
        ).item()
        matthews_corrcoef_val = MatthewsCorrCoef(num_classes=NUM_CLASSES, task=TASK)(
            preds_tensor, labels_tensor
        ).item()

        # Post-process predictions and evaluate segment-wise metrics
        processed_predictions = post_process_predictions(
            np.array(all_predictions), SAMPLING_FREQ
        )
        all_labels_np = np.array(all_labels)
        metrics_segment = {}
        for label in range(1, NUM_CLASSES):
            fn_seg, fp_seg, tp_seg = segment_evaluation(
                processed_predictions,
                all_labels_np,
                class_label=label,
                threshold=THRESHOLD,
                debug_plot=DEBUG_PLOT,
            )
            f1_seg = (
                2 * tp_seg / (2 * tp_seg + fp_seg + fn_seg)
                if (fp_seg + fn_seg) != 0
                else 0.0
            )
            metrics_segment[str(label)] = {
                "tp": int(tp_seg),
                "fp": int(fp_seg),
                "fn": int(fn_seg),
                "f1": float(f1_seg),
            }

        # Store validation statistics for this fold
        fold_statistics = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "fold": fold + 1,
            "metrics_sample": metrics_sample,
            "metrics_segment": metrics_segment,
            "cohen_kappa": cohen_kappa_val,
            "matthews_corrcoef": matthews_corrcoef_val,
            "label_distribution": label_distribution,
        }
        validating_statistics.append(fold_statistics)

    # -------------------------------------------------------------------------
    # Save Evaluation Results
    # -------------------------------------------------------------------------
    validate_stats_file = os.path.join(result_dir, "validate_stats.npy")
    np.save(validate_stats_file, validating_statistics)
    print(f"\nValidation statistics saved to {validate_stats_file}")

    # -------------------------------------------------------------------------
    # Plotting Section: Training Curves and Performance Metrics
    # -------------------------------------------------------------------------
    # Load saved training and validation statistics
    train_stats_file = os.path.join(result_dir, "train_stats.npy")
    validate_stats_file = os.path.join(result_dir, "validate_stats.npy")
    train_stats = list(np.load(train_stats_file, allow_pickle=True))
    validate_stats = list(np.load(validate_stats_file, allow_pickle=True))

    # Plot training loss curves per fold
    folds = sorted(set(entry["fold"] for entry in train_stats))
    for fold in folds:
        fold_stats = [entry for entry in train_stats if entry["fold"] == fold]
        epochs = sorted(set(entry["epoch"] for entry in fold_stats))
        loss_total = {ep: [] for ep in epochs}
        loss_ce = {ep: [] for ep in epochs}
        loss_mse = {ep: [] for ep in epochs}

        for entry in fold_stats:
            loss_total[entry["epoch"]].append(entry["train_loss"])
            loss_ce[entry["epoch"]].append(entry["train_loss_ce"])
            loss_mse[entry["epoch"]].append(entry["train_loss_mse"])

        mean_loss_total = [np.mean(loss_total[ep]) for ep in epochs]
        mean_loss_ce = [np.mean(loss_ce[ep]) for ep in epochs]
        mean_loss_mse = [np.mean(loss_mse[ep]) for ep in epochs]

        plt.figure(figsize=(10, 6))
        plt.plot(
            epochs,
            mean_loss_total,
            marker="o",
            linestyle="-",
            color="blue",
            label="Total Loss",
        )
        plt.plot(
            epochs,
            mean_loss_ce,
            marker="s",
            linestyle="--",
            color="red",
            label="CE Loss",
        )
        plt.plot(
            epochs,
            mean_loss_mse,
            marker="^",
            linestyle=":",
            color="green",
            label="MSE Loss",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title(f"Training Losses (Fold {fold})")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"train_loss_fold{fold}.png"), dpi=300)
        plt.close()

    # Plot fold-wise performance metrics
    cohen_kappa_scores = []
    matthews_scores = []
    weighted_f1_sample = []
    weighted_f1_segment = []

    for entry in validate_stats:
        label_dist = entry["label_distribution"]
        total_weight_sample = sum(label_dist.get(lbl, 0) for lbl in label_dist)
        weighted_f1_sample_val = sum(
            entry["metrics_sample"].get(str(lbl), {}).get("f1", 0.0)
            * label_dist.get(lbl, 0)
            for lbl in label_dist
        )
        f1_sample = (
            weighted_f1_sample_val / total_weight_sample
            if total_weight_sample > 0
            else 0.0
        )

        total_weight_segment = sum(label_dist.get(lbl, 0) for lbl in label_dist)
        weighted_f1_segment_val = sum(
            entry["metrics_segment"].get(str(lbl), {}).get("f1", 0.0)
            * label_dist.get(lbl, 0)
            for lbl in label_dist
        )
        f1_segment = (
            weighted_f1_segment_val / total_weight_segment
            if total_weight_segment > 0
            else 0.0
        )

        cohen_kappa_scores.append(entry["cohen_kappa"])
        matthews_scores.append(entry["matthews_corrcoef"])
        weighted_f1_sample.append(f1_sample)
        weighted_f1_segment.append(f1_segment)

    fold_indices = np.arange(1, len(cohen_kappa_scores) + 1)
    bar_width = 0.2
    plt.figure(figsize=(12, 6))
    plt.bar(
        fold_indices - bar_width * 1.5,
        cohen_kappa_scores,
        width=bar_width,
        label="Cohen Kappa",
        color="orange",
    )
    plt.bar(
        fold_indices - bar_width / 2,
        matthews_scores,
        width=bar_width,
        label="Matthews CorrCoef",
        color="purple",
    )
    plt.bar(
        fold_indices + bar_width / 2,
        weighted_f1_sample,
        width=bar_width,
        label="Weighted Sample-wise F1",
        color="blue",
    )
    plt.bar(
        fold_indices + bar_width * 1.5,
        weighted_f1_segment,
        width=bar_width,
        label=f"Weighted Segment-wise F1 (Threshold={THRESHOLD})",
        color="green",
    )
    plt.xticks(fold_indices)
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.title("Fold-wise Performance Metrics")
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "validate_metrics.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
