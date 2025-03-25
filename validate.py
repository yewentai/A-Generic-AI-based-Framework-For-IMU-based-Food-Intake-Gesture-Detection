#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Validating Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Version     : 2.0
Created     : 2025-03-24
Description : This script validates a trained MSTCN model on the full IMU dataset.
              It loads the dataset and saved checkpoints for each fold,
              runs inference, and computes comprehensive evaluation metrics.
Usage       : Execute the script in your terminal:
              $ python validate.py
===============================================================================
"""

import os
import json
import glob
import pickle
from matplotlib.dates import TH
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from torchmetrics import CohenKappa, MatthewsCorrCoef
from torch.utils.data import DataLoader, Subset

# Import model and utility modules from components package
from components.model_mstcn import MSTCN
from components.post_processing import post_process_predictions
from components.evaluation import segment_evaluation
from components.datasets import IMUDataset
from components.pre_processing import hand_mirroring

# -----------------------------------------------------------------------------
#                   Configuration and Parameters
# -----------------------------------------------------------------------------

# Automatically select the lavalidate version based on timestamp
RESULT_DIR = "result"
all_versions = glob.glob(os.path.join(RESULT_DIR, "*"))
RESULT_VERSION = max(all_versions, key=os.path.getmtime).split(os.sep)[-1]
result_dir = os.path.join(RESULT_DIR, RESULT_VERSION)
CONFIG_FILE = os.path.join(RESULT_DIR, RESULT_VERSION, "config.json")

# Load the configuration parameters from the JSON file
with open(CONFIG_FILE, "r") as f:
    config_info = json.load(f)

# Set configuration parameters from the loaded JSON
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

if DATASET.startswith("DX"):
    sub_version = DATASET.replace("DX", "").upper() or "I"
    DATA_DIR = f"./dataset/DX/DX-{sub_version}"
    TASK = "binary"
elif DATASET.startswith("FD"):
    sub_version = DATASET.replace("FD", "").upper() or "I"
    DATA_DIR = f"./dataset/FD/FD-{sub_version}"
    TASK = "multiclass"
else:
    raise ValueError(f"Invalid dataset: {DATASET}")

# Validating DataLoader parameters
NUM_WORKERS = 4  # Fewer workers are typically sufficient for validating

# -----------------------------------------------------------------------------
#                        Data Loading and Pre-processing
# -----------------------------------------------------------------------------

# Define file paths for the dataset (left and right data)
X_L_PATH = os.path.join(DATA_DIR, "X_L.pkl")
Y_L_PATH = os.path.join(DATA_DIR, "Y_L.pkl")
X_R_PATH = os.path.join(DATA_DIR, "X_R.pkl")
Y_R_PATH = os.path.join(DATA_DIR, "Y_R.pkl")

# Load data from pickle files
with open(X_L_PATH, "rb") as f:
    X_L = np.array(pickle.load(f), dtype=object)
with open(Y_L_PATH, "rb") as f:
    Y_L = np.array(pickle.load(f), dtype=object)
with open(X_R_PATH, "rb") as f:
    X_R = np.array(pickle.load(f), dtype=object)
with open(Y_R_PATH, "rb") as f:
    Y_R = np.array(pickle.load(f), dtype=object)

# Optionally apply hand mirroring if the flag is set
if FLAG_MIRROR:
    X_L = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)

# Merge left and right data
X = np.concatenate([X_L, X_R], axis=0)
Y = np.concatenate([Y_L, Y_R], axis=0)

# Create the full dataset with the defined window size
full_dataset = IMUDataset(X, Y, sequence_length=WINDOW_SIZE)

# Augment Dataset with FDIII (if using FDII/FDI)
fdiii_dataset = None
if DATASET in ["FDII", "FDI"]:
    fdiii_dir = "./dataset/FD/FD-III"
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

# Load cross-validation folds from the JSON configuration
validate_folds = config_info.get("validate_folds")
if validate_folds is None:
    raise ValueError("No 'validate_folds' found in the configuration file.")

# -----------------------------------------------------------------------------
#                    Cross-Validation Loop for Validating
# -----------------------------------------------------------------------------

# Create a results storage for cross-validation evaluations
validating_statistics = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for fold, validate_subjects in enumerate(
    tqdm(validate_folds, desc="K-Fold", leave=True)
):
    # Construct the checkpoint path for the current fold
    CHECKPOINT_PATH = os.path.join(
        RESULT_DIR, RESULT_VERSION, f"best_model_fold{fold+1}.pth"
    )

    # Check if the checkpoint for this fold exists
    if not os.path.exists(CHECKPOINT_PATH):
        continue

    # Initialize the MSTCN model with the specified parameters
    model = MSTCN(
        num_stages=NUM_STAGES,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        input_dim=INPUT_DIM,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)

    # Load the checkpoint for the current fold
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    # Remove 'module.' prefix if it exists.
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    # Split validation indices based on subject IDs
    validate_indices = [
        i
        for i, subject in enumerate(full_dataset.subject_indices)
        if subject in validate_subjects
    ]

    # Create DataLoader for the validation subset of the current fold
    validate_loader = DataLoader(
        Subset(full_dataset, validate_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Explicitly set the model to evaluation mode
    model.eval()

    # Inference and Prediction Collection
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(
            validate_loader, desc=f"Validating Fold {fold+1}", leave=False
        ):
            # Rearrange dimensions and send to device
            batch_x = batch_x.permute(0, 2, 1).to(device)
            outputs = model(batch_x)
            # Use the output from the last stage for predictions
            last_stage_output = outputs[:, -1, :, :]
            probabilities = F.softmax(last_stage_output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            all_predictions.extend(predictions.view(-1).cpu().numpy())
            all_labels.extend(batch_y.view(-1).cpu().numpy())

    # -----------------------------------------------------------------------------
    #                        Evaluation Metrics Calculation
    # -----------------------------------------------------------------------------

    # Compute label distribution
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    label_distribution = {
        float(label): int(count) for label, count in zip(unique_labels, counts)
    }

    preds_tensor = torch.tensor(all_predictions)
    labels_tensor = torch.tensor(all_labels)

    # Sample-wise metrics for each class
    metrics_sample = {}
    for label in range(1, NUM_CLASSES):
        tp = torch.sum((preds_tensor == label) & (labels_tensor == label)).item()
        fp = torch.sum((preds_tensor == label) & (labels_tensor != label)).item()
        fn = torch.sum((preds_tensor != label) & (labels_tensor == label)).item()
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator != 0 else 0.0
        metrics_sample[f"{label}"] = {"fn": fn, "fp": fp, "tp": tp, "f1": f1}

    # Additional sample-wise metrics
    cohen_kappa_val = CohenKappa(num_classes=NUM_CLASSES, task=TASK)(
        preds_tensor, labels_tensor
    ).item()
    matthews_corrcoef_val = MatthewsCorrCoef(num_classes=NUM_CLASSES, task=TASK)(
        preds_tensor, labels_tensor
    ).item()

    # Post-process predictions
    all_predictions = post_process_predictions(np.array(all_predictions), SAMPLING_FREQ)
    all_labels = np.array(all_labels)

    # Segment-wise evaluation metrics
    metrics_segment = {}
    for label in range(1, NUM_CLASSES):
        fn, fp, tp = segment_evaluation(
            all_predictions,
            all_labels,
            class_label=label,
            threshold=THRESHOLD,
            debug_plot=DEBUG_PLOT,
        )
        f1 = 2 * tp / (2 * tp + fp + fn) if (fp + fn) != 0 else 0.0
        metrics_segment[f"{label}"] = {
            "fn": int(fn),
            "fp": int(fp),
            "tp": int(tp),
            "f1": float(f1),
        }

    # Record validating statistics for the current fold
    fold_statistics = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "fold": fold + 1,
        "metrics_segment": metrics_segment,
        "metrics_sample": metrics_sample,
        "cohen_kappa": cohen_kappa_val,
        "matthews_corrcoef": matthews_corrcoef_val,
        "label_distribution": label_distribution,
    }
    validating_statistics.append(fold_statistics)

# -----------------------------------------------------------------------------
#                         Save Evaluation Results
# -----------------------------------------------------------------------------

# Save validating statistics
TESTING_STATS_FILE = os.path.join(result_dir, "validate_stats.npy")
np.save(TESTING_STATS_FILE, validating_statistics)
print(f"\nValidating statistics saved to {TESTING_STATS_FILE}")


# =============================================================================
#                               PLOTTING SECTION
# =============================================================================


# Define the specific directory for this version
RESULT_PATH = os.path.join(RESULT_DIR, RESULT_VERSION)

# Load train and validation statistics from saved .npy files
train_stats_file = os.path.join(RESULT_PATH, f"train_stats.npy")
validate_stats_file = os.path.join(RESULT_PATH, f"validate_stats.npy")
train_stats = np.load(train_stats_file, allow_pickle=True)
validate_stats = np.load(validate_stats_file, allow_pickle=True)

# Convert loaded data to lists for easier processing
train_stats = list(train_stats)
validate_stats = list(validate_stats)

# -------------------------
# Plot train Curves per Fold

# Get list of unique folds present in train_stats
folds = sorted(set(entry["fold"] for entry in train_stats))

for fold in folds:
    # Filter train stats for this fold only
    stats_fold = [entry for entry in train_stats if entry["fold"] == fold]
    # Group by epoch
    epochs = sorted(set(entry["epoch"] for entry in stats_fold))
    loss_per_epoch = {epoch: [] for epoch in epochs}
    loss_ce_per_epoch = {epoch: [] for epoch in epochs}
    loss_mse_per_epoch = {epoch: [] for epoch in epochs}

    for entry in stats_fold:
        loss_per_epoch[entry["epoch"]].append(entry["train_loss"])
        loss_ce_per_epoch[entry["epoch"]].append(entry["train_loss_ce"])
        loss_mse_per_epoch[entry["epoch"]].append(entry["train_loss_mse"])

    # Compute mean loss values for each epoch
    mean_loss_per_epoch = [np.mean(loss_per_epoch[epoch]) for epoch in epochs]
    mean_loss_ce_per_epoch = [np.mean(loss_ce_per_epoch[epoch]) for epoch in epochs]
    mean_loss_mse_per_epoch = [np.mean(loss_mse_per_epoch[epoch]) for epoch in epochs]

    # Plot train losses for this fold
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        mean_loss_per_epoch,
        marker="o",
        linestyle="-",
        color="blue",
        label="Total Loss",
    )
    plt.plot(
        epochs,
        mean_loss_ce_per_epoch,
        marker="s",
        linestyle="--",
        color="red",
        label="CE Loss",
    )
    plt.plot(
        epochs,
        mean_loss_mse_per_epoch,
        marker="^",
        linestyle=":",
        color="green",
        label="MSE Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title(f"Train Losses Over Epochs (Fold {fold})")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, f"train_loss_fold{fold}.png"), dpi=300)
    plt.close()

# -------------------------
# Plot Fold-wise Performance Metrics (Validation)

# Initialize lists to store validation metrics for each fold
label_distribution = []  # Label distribution
f1_scores_sample = []  # Sample-wise F1 scores
f1_scores_segment = []  # Segment-wise F1 scores
cohen_kappa_scores = []  # Cohen's kappa scores
matthews_corrcoef_scores = []  # Matthews correlation coefficient scores

for entry in validate_stats:
    label_dist = entry["label_distribution"]  # dictionary: {label: count}

    # Compute weighted average F1 for sample-wise metrics
    total_weight_sample = 0.0
    weighted_f1_sample = 0.0
    for label_str, stats in entry["metrics_sample"].items():
        label_int = int(label_str)
        weight = label_dist.get(label_int, 0)
        weighted_f1_sample += stats["f1"] * weight
        total_weight_sample += weight
    f1_sample_weighted = (
        weighted_f1_sample / total_weight_sample if total_weight_sample > 0 else 0.0
    )

    # Compute weighted average F1 for segment-wise metrics
    total_weight_segment = 0.0
    weighted_f1_segment = 0.0
    for label_str, stats in entry["metrics_segment"].items():
        label_int = int(label_str)
        weight = label_dist.get(label_int, 0)
        weighted_f1_segment += stats["f1"] * weight
        total_weight_segment += weight
    f1_segment_weighted = (
        weighted_f1_segment / total_weight_segment if total_weight_segment > 0 else 0.0
    )

    label_distribution.append(label_dist)
    f1_scores_sample.append(f1_sample_weighted)
    f1_scores_segment.append(f1_segment_weighted)
    cohen_kappa_scores.append(entry["cohen_kappa"])
    matthews_corrcoef_scores.append(entry["matthews_corrcoef"])

# Create a bar plot for validation metrics across folds
plt.figure(figsize=(12, 6))
width = 0.2  # Width of each bar
fold_indices = np.arange(1, len(cohen_kappa_scores) + 1)

plt.bar(
    fold_indices - width * 1.5,
    cohen_kappa_scores,
    width=width,
    label="Cohen Kappa Coefficient",
    color="orange",
)
plt.bar(
    fold_indices - width / 2,
    matthews_corrcoef_scores,
    width=width,
    label="Matthews Correlation Coefficient",
    color="purple",
)
plt.bar(
    fold_indices + width / 2,
    f1_scores_sample,
    width=width,
    label="Weighted Sample-wise F1 Score",
    color="blue",
)
plt.bar(
    fold_indices + width * 1.5,
    f1_scores_segment,
    width=width,
    label=f"Weighted Segment-wise F1 Score (Threshold={THRESHOLD})",
    color="green",
)

plt.xticks(fold_indices)
plt.xlabel("Fold")
plt.ylabel("Score")
plt.title("Fold-wise Performance Metrics")
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, "validate_metrics.png"), dpi=300)
plt.close()
