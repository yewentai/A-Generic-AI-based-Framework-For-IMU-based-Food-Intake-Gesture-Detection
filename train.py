#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Training Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Version     : 3.0
Created     : 2025-03-26
Description : This script trains an MSTCN model on IMU (Inertial Measurement Unit) data
              using cross-validation. It supports multiple datasets (DXI/DXII or FDI/FDII/FDIII)
              and dynamically generates result and checkpoint directories based on the
              current datetime. The script includes configurable parameters for model
              architecture, training hyperparameters, and data augmentation options.
===============================================================================
"""

import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from datetime import datetime
from tqdm import tqdm

# Import custom modules from components package
from components.augmentation import augment_orientation
from components.datasets import (
    IMUDataset,
    create_balanced_subject_folds,
    load_predefined_validate_folds,
)
from components.pre_processing import hand_mirroring
from components.checkpoint import save_best_model
from components.model_cnnlstm import CNNLSTM, CNNLSTM_Loss
from components.model_tcn import TCN, TCN_Loss
from components.model_mstcn import MSTCN, MSTCN_Loss

# =============================================================================
#                         Configuration Parameters
# =============================================================================

# Dataset
DATASET = "FDI"  # Options: DXI/DXII or FDI/FDII/FDIII
SAMPLING_FREQ_ORIGINAL = 64
DOWNSAMPLE_FACTOR = 4
SAMPLING_FREQ = SAMPLING_FREQ_ORIGINAL // DOWNSAMPLE_FACTOR
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

# Dataloader
WINDOW_LENGTH = 60
WINDOW_SIZE = SAMPLING_FREQ * WINDOW_LENGTH
BATCH_SIZE = 64
NUM_WORKERS = 16

# Model
MODEL = "CNN_LSTM"  # Options: CNN_LSTM, TCN, MSTCN
INPUT_DIM = 6
LAMBDA_COEF = 0.15
if MODEL in ["TCN", "MSTCN"]:
    NUM_LAYERS = 9
    NUM_FILTERS = 128
    KERNEL_SIZE = 3
    DROPOUT = 0.3
    if MODEL == "MSTCN":
        NUM_STAGES = 2
elif MODEL == "CNN_LSTM":
    CONV_FILTERS = (32, 64, 128)
    LSTM_HIDDEN = 128
else:
    raise ValueError(f"Invalid model: {MODEL}")

# Training
LEARNING_RATE = 5e-4
NUM_FOLDS = 7
NUM_EPOCHS = 100
FLAG_AUGMENT = False
FLAG_MIRROR = False
FLAG_SKIP = False

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
#                       Directory and File Management
# =============================================================================

# Generate version prefix from current datetime (first 12 characters)
version_prefix = datetime.now().strftime("%Y%m%d%H%M")[:12]

# Create result directories using version_prefix
result_dir = os.path.join("result", version_prefix)
os.makedirs(result_dir, exist_ok=True)

# Define file paths for saving statistics and configuration
training_stas_file = os.path.join(result_dir, f"train_stats.npy")
config_file = os.path.join(result_dir, "config.json")

# =============================================================================
#                       Data Loading and Pre-processing
# =============================================================================
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
#                     Main Cross-Validation Loop
# =============================================================================

# Create validation folds based on the dataset type
if DATASET == "FDI":
    validate_folds = load_predefined_validate_folds()
else:
    validate_folds = create_balanced_subject_folds(full_dataset, num_folds=NUM_FOLDS)

training_statistics = []

for fold, validate_subjects in enumerate(
    tqdm(validate_folds, desc="K-Fold", leave=True)
):
    # Process only the first fold for demonstration; remove the condition to run all folds.
    if FLAG_SKIP:
        FLAG_SKIP = False
        continue

    # Split training indices based on subject IDs
    train_indices = [
        i
        for i, subject in enumerate(full_dataset.subject_indices)
        if subject not in validate_subjects
    ]

    # Augment training data with FDIII if applicable
    if DATASET in ["FDII", "FDI"] and fdiii_dataset is not None:
        train_dataset = ConcatDataset(
            [Subset(full_dataset, train_indices), fdiii_dataset]
        )
    else:
        train_dataset = Subset(full_dataset, train_indices)

    # Create DataLoaders for training
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Create Model and Optimizer
    if MODEL == "TCN":
        model = TCN(
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            input_dim=INPUT_DIM,
            num_filters=NUM_FILTERS,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT,
        ).to(device)
    elif MODEL == "MSTCN":
        model = MSTCN(
            num_stages=NUM_STAGES,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            input_dim=INPUT_DIM,
            num_filters=NUM_FILTERS,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT,
        ).to(device)
    elif MODEL == "CNN_LSTM":
        model = CNNLSTM(
            input_channels=INPUT_DIM,
            conv_filters=CONV_FILTERS,
            lstm_hidden=LSTM_HIDDEN,
            num_classes=NUM_CLASSES,
        ).to(device)
    else:
        raise ValueError(f"Invalid model: {MODEL}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize best loss for saving the best model (lower loss is better)
    best_loss = float("inf")

    # Training Loop
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
            loss_fn = {
                "MSTCN": MSTCN_Loss,
                "TCN": TCN_Loss,
                "CNN_LSTM": CNNLSTM_Loss,
            }[MODEL]
            ce_loss, mse_loss = loss_fn(outputs, batch_y)
            loss = ce_loss + LAMBDA_COEF * mse_loss
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_loss_ce += ce_loss.item()
            training_loss_mse += mse_loss.item()

        # Save the Best Model Based on Loss
        best_loss = save_best_model(
            model,
            fold=fold + 1,
            current_metric=loss.item(),
            best_metric=best_loss,
            checkpoint_dir=result_dir,
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
        training_statistics.append(stats)

# =============================================================================
#                         Save Results and Configuration
# =============================================================================

np.save(training_stas_file, training_statistics)
print(f"Training statistics saved to {training_stas_file}")

# Base config info
config_info = {
    "dataset": DATASET,
    "num_classes": NUM_CLASSES,
    "model": MODEL,
    "input_dim": INPUT_DIM,
    "learning_rate": LEARNING_RATE,
    "sampling_freq": SAMPLING_FREQ,
    "window_size": WINDOW_SIZE,
    "num_folds": NUM_FOLDS,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "augmentation": FLAG_AUGMENT,
    "mirroring": FLAG_MIRROR,
    "validate_folds": validate_folds,
}

# Model-specific parameters
if MODEL == "CNN_LSTM":
    config_info["conv_filters"] = CONV_FILTERS
    config_info["lstm_hidden"] = LSTM_HIDDEN
elif MODEL in ["TCN", "MSTCN"]:
    config_info["num_layers"] = NUM_LAYERS
    config_info["num_filters"] = NUM_FILTERS
    config_info["kernel_size"] = KERNEL_SIZE
    config_info["dropout"] = DROPOUT
    if MODEL in ["MSTCN"]:
        config_info["num_stages"] = NUM_STAGES
else:
    raise ValueError(f"Invalid model: {MODEL}")

# Save the configuration as JSON
with open(config_file, "w") as f:
    json.dump(config_info, f, indent=4)
print(f"Configuration saved to {config_file}")
