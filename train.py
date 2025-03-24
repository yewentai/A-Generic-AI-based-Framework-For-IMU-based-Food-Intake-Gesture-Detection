#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Training Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Version     : 1.0
Created     : 2025-03-22
Description : This script trains an MSTCN model on IMU (Inertial Measurement Unit) data
              using cross-validation. It supports multiple datasets (DXI/DXII or FDI/FDII/FDIII)
              and dynamically generates result and checkpoint directories based on the
              current datetime. The script includes configurable parameters for model
              architecture, training hyperparameters, and data augmentation options.
===============================================================================
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from datetime import datetime
from tqdm import tqdm
from torchmetrics import CohenKappa, MatthewsCorrCoef

# Import custom modules from components package
from components.augmentation import augment_orientation
from components.datasets import (
    IMUDataset,
    create_balanced_subject_folds,
    load_predefined_validate_folds,
)
from components.evaluation import segment_evaluation
from components.pre_processing import hand_mirroring
from components.checkpoint import save_best_model
from components.post_processing import post_process_predictions
from components.model_mstcn_backup import MSTCN, MSTCN_Loss

# =============================================================================
#                         Configuration Parameters
# =============================================================================

DATASET = "FDI"  # Options: DXI/DXII or FDI/FDII/FDIII
NUM_STAGES = 2
NUM_LAYERS = 9
NUM_HEADS = 8
INPUT_DIM = 6
NUM_FILTERS = 128
KERNEL_SIZE = 3
DROPOUT = 0.3
LAMBDA_COEF = 0.15
TAU = 4
LEARNING_RATE = 5e-4
SAMPLING_FREQ_ORIGINAL = 64
DOWNSAMPLE_FACTOR = 4
SAMPLING_FREQ = SAMPLING_FREQ_ORIGINAL // DOWNSAMPLE_FACTOR
WINDOW_LENGTH = 60
WINDOW_SIZE = SAMPLING_FREQ * WINDOW_LENGTH
NUM_FOLDS = 7
NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 16
DEBUG_PLOT = False
DEBUG_LOG = False
FLAG_AUGMENT = True
FLAG_MIRROR = True

# =============================================================================
#                     Dataset Configuration and Loading
# =============================================================================

if DATASET.startswith("DX"):
    NUM_CLASSES = 2
    dataset_type = "DX"
    sub_version = (
        DATASET.replace("DX", "").upper() or "I"
    )  # Handles formats like DX/DXII
    DATA_DIR = f"./dataset/DX/DX-{sub_version}"
    TASK = "binary"
elif DATASET.startswith("FD"):
    NUM_CLASSES = 3
    dataset_type = "FD"
    sub_version = DATASET.replace("FD", "").upper() or "I"
    DATA_DIR = f"./dataset/FD/FD-{sub_version}"
    TASK = "multiclass"
else:
    raise ValueError(f"Invalid dataset: {DATASET}")

# Define file paths for the dataset
X_L_PATH = os.path.join(DATA_DIR, "X_L.pkl")
Y_L_PATH = os.path.join(DATA_DIR, "Y_L.pkl")
X_R_PATH = os.path.join(DATA_DIR, "X_R.pkl")
Y_R_PATH = os.path.join(DATA_DIR, "Y_R.pkl")

# Load left-hand and right-hand data from pickle files
with open(X_L_PATH, "rb") as f:
    X_L = np.array(pickle.load(f), dtype=object)
with open(Y_L_PATH, "rb") as f:
    Y_L = np.array(pickle.load(f), dtype=object)
with open(X_R_PATH, "rb") as f:
    X_R = np.array(pickle.load(f), dtype=object)
with open(Y_R_PATH, "rb") as f:
    Y_R = np.array(pickle.load(f), dtype=object)

# =============================================================================
#                    Result and Checkpoint Setup
# =============================================================================

# Generate version prefix from current datetime (first 12 characters)
version_prefix = datetime.now().strftime("%Y%m%d%H%M")[:12]

# Create result and checkpoint directories using version_prefix
result_dir = os.path.join("result", version_prefix)
checkpoint_dir = os.path.join("checkpoints", version_prefix)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Define file paths for saving statistics and configuration
TRAINING_STATS_FILE = os.path.join(result_dir, f"train_stats.npy")
TESTING_STATS_FILE = os.path.join(result_dir, f"validate_stats.npy")
CONFIG_FILE = os.path.join(result_dir, f"config.txt")

training_statistics = []
validating_statistics = []

# =============================================================================
#                          Data Pre-processing
# =============================================================================

# Apply hand mirroring if flag is set
if FLAG_MIRROR:
    X_L = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)

# Merge left-hand and right-hand data
X = np.concatenate([X_L, X_R], axis=0)
Y = np.concatenate([Y_L, Y_R], axis=0)

# Create the full dataset using the defined window size
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

# =============================================================================
#                     Cross-Validation Setup
# =============================================================================

# Create balanced cross-validation folds
validate_folds = create_balanced_subject_folds(full_dataset, num_folds=NUM_FOLDS)
# To use predefined folds, uncomment the following line:
# validate_folds = load_predefined_validate_folds()

# =============================================================================
#                          Device Configuration
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
#                     Main Cross-Validation Loop
# =============================================================================

for fold, validate_subjects in enumerate(
    tqdm(validate_folds, desc="K-Fold", leave=True)
):
    # Process only the first fold for demonstration; remove the condition to run all folds.
    if fold != 0:
        continue

    # Split training and validation indices based on subject IDs
    train_indices = [
        i
        for i, subject in enumerate(full_dataset.subject_indices)
        if subject not in validate_subjects
    ]
    validate_indices = [
        i
        for i, subject in enumerate(full_dataset.subject_indices)
        if subject in validate_subjects
    ]

    # Augment training data with FDIII if applicable
    if DATASET in ["FDII", "FDI"] and fdiii_dataset is not None:
        train_dataset = ConcatDataset(
            [Subset(full_dataset, train_indices), fdiii_dataset]
        )
    else:
        train_dataset = Subset(full_dataset, train_indices)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    validate_loader = DataLoader(
        Subset(full_dataset, validate_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = MSTCN(
        num_stages=NUM_STAGES,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        input_dim=INPUT_DIM,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize best loss for saving the best model (lower loss is better)
    best_loss = float("inf")

    # =============================================================================
    #                               Training Loop
    # =============================================================================

    for epoch in tqdm(range(NUM_EPOCHS), desc=f"Fold {fold+1}", leave=False):
        model.train()
        training_loss = 0.0
        training_loss_ce = 0.0
        training_loss_mse = 0.0

        for batch_x, batch_y in train_loader:
            # Optionally apply data augmentation
            if FLAG_AUGMENT:
                batch_x = augment_orientation(batch_x)
            # Rearrange dimensions and move data to the configured device
            batch_x = batch_x.permute(0, 2, 1).to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            ce_loss, mse_loss = MSTCN_Loss(outputs, batch_y)
            loss = ce_loss + LAMBDA_COEF * mse_loss
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_loss_ce += ce_loss.item()
            training_loss_mse += mse_loss.item()

        # ------------------ Optional Debug Logging ------------------
        if DEBUG_LOG:
            model.eval()
            # Collect training predictions and labels
            train_predictions = []
            train_labels = []
            with torch.no_grad():
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.permute(0, 2, 1).to(device)
                    outputs = model(batch_x)
                    # For MS-TCN, use the output of the last stage
                    last_stage_output = outputs[:, -1, :, :]
                    probabilities = F.softmax(last_stage_output, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                    train_predictions.extend(predictions.view(-1).cpu().numpy())
                    train_labels.extend(batch_y.view(-1).cpu().numpy())

            # Calculate Matthews CorrCoef for training set
            train_preds_tensor = torch.tensor(train_predictions)
            train_labels_tensor = torch.tensor(train_labels)
            train_mcc = MatthewsCorrCoef(num_classes=NUM_CLASSES, task=TASK)(
                train_preds_tensor, train_labels_tensor
            ).item()

            # Collect validation predictions and labels
            val_predictions = []
            val_labels = []
            with torch.no_grad():
                for batch_x, batch_y in validate_loader:
                    batch_x = batch_x.permute(0, 2, 1).to(device)
                    outputs = model(batch_x)
                    last_stage_output = outputs[:, -1, :, :]
                    probabilities = F.softmax(last_stage_output, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                    val_predictions.extend(predictions.view(-1).cpu().numpy())
                    val_labels.extend(batch_y.view(-1).cpu().numpy())

            # Calculate Matthews CorrCoef for validation set
            val_preds_tensor = torch.tensor(val_predictions)
            val_labels_tensor = torch.tensor(val_labels)
            val_mcc = MatthewsCorrCoef(num_classes=NUM_CLASSES, task=TASK)(
                val_preds_tensor, val_labels_tensor
            ).item()

        # ------------------ Save the Best Model Based on Loss ------------------
        best_loss = save_best_model(
            model,
            fold=fold + 1,
            current_metric=loss.item(),
            best_metric=best_loss,
            checkpoint_dir=checkpoint_dir,
            mode="min",
        )

        # Record training statistics for the current epoch
        stats = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "fold": fold + 1,
            "epoch": epoch + 1,
            "train_loss": training_loss / len(train_loader),
            "train_loss_ce": training_loss_ce / len(train_loader),
            "train_loss_mse": training_loss_mse / len(train_loader),
        }
        if DEBUG_LOG:
            stats["train_matthews_corrcoef"] = train_mcc
            stats["val_matthews_corrcoef"] = val_mcc
        training_statistics.append(stats)

    # =============================================================================
    #                              Evaluation Phase
    # =============================================================================

    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in validate_loader:
            batch_x = batch_x.permute(0, 2, 1).to(device)
            outputs = model(batch_x)
            last_stage_output = outputs[:, -1, :, :]
            probabilities = F.softmax(last_stage_output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            all_predictions.extend(predictions.view(-1).cpu().numpy())
            all_labels.extend(batch_y.view(-1).cpu().numpy())

        # Append any remaining predictions (if applicable)
        all_predictions.extend(predictions.view(-1).cpu().numpy())
        all_labels.extend(batch_y.view(-1).cpu().numpy())

    # ------------------ Compute Evaluation Metrics ------------------

    # Label distribution
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    label_distribution = dict(zip(unique_labels, counts))

    # Sample-wise evaluation metrics
    preds_tensor = torch.tensor(all_predictions)
    labels_tensor = torch.tensor(all_labels)
    metrics_sample = {}
    for label in range(1, NUM_CLASSES):
        tp = torch.sum((preds_tensor == label) & (labels_tensor == label)).item()
        fp = torch.sum((preds_tensor == label) & (labels_tensor != label)).item()
        fn = torch.sum((preds_tensor != label) & (labels_tensor == label)).item()
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator != 0 else 0.0
        metrics_sample[f"{label}"] = {"fn": fn, "fp": fp, "tp": tp, "f1": f1}

    # Additional metrics: Cohen's Kappa and Matthews CorrCoef
    cohen_kappa_val = CohenKappa(num_classes=NUM_CLASSES, task=TASK)(
        preds_tensor, labels_tensor
    ).item()
    matthews_corrcoef_val = MatthewsCorrCoef(num_classes=NUM_CLASSES, task=TASK)(
        preds_tensor, labels_tensor
    ).item()

    # ------------------ Post-processing ------------------
    all_predictions = post_process_predictions(np.array(all_predictions), SAMPLING_FREQ)
    all_labels = np.array(all_labels)

    # ------------------ Segment-wise Evaluation ------------------
    metrics_segment = {}
    for label in range(1, NUM_CLASSES):
        fn, fp, tp = segment_evaluation(
            all_predictions,
            all_labels,
            class_label=label,
            threshold=0.5,
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
    validating_statistics.append(
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "fold": fold + 1,
            "metrics_segment": metrics_segment,
            "metrics_sample": metrics_sample,
            "cohen_kappa": cohen_kappa_val,
            "matthews_corrcoef": matthews_corrcoef_val,
            "label_distribution": label_distribution,
        }
    )

# =============================================================================
#                         Save Results and Configuration
# =============================================================================

np.save(TRAINING_STATS_FILE, training_statistics)
np.save(TESTING_STATS_FILE, validating_statistics)

config_info = {
    "dataset": DATASET,
    "num_classes": NUM_CLASSES,
    "num_stages": NUM_STAGES,
    "num_layers": NUM_LAYERS,
    "num_heads": NUM_HEADS,
    "input_dim": INPUT_DIM,
    "num_filters": NUM_FILTERS,
    "kernel_size": KERNEL_SIZE,
    "dropout": DROPOUT,
    "lambda_coef": LAMBDA_COEF,
    "tau": TAU,
    "learning_rate": LEARNING_RATE,
    "sampling_freq": SAMPLING_FREQ,
    "window_size": WINDOW_SIZE,
    "num_folds": NUM_FOLDS,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "augmentation": FLAG_AUGMENT,
    "mirroring": FLAG_MIRROR,
}

with open(CONFIG_FILE, "w") as f:
    # Write basic configuration information
    for key, value in config_info.items():
        f.write(f"{key}: {value}\n")
    # Write cross-validation fold information
    f.write("validate_folds:\n")
    for i, fold in enumerate(validate_folds, start=1):
        f.write(f"  Fold {i}: {fold}\n")
