#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Training Script
-------------------------------------------------------------------------------
Author      : Joseph Yep (improved by ChatGPT)
Email       : yewentai126@gmail.com
Version     : 2.1
Created     : 2025-03-25
Description : Trains an MSTCN model on IMU data using cross-validation.
              Supports multiple datasets (DXI/DXII or FDI/FDII/FDIII) and
              dynamically creates result and checkpoint directories.
              Configurable parameters include model architecture, training
              hyperparameters, and data augmentation options.
===============================================================================
"""

import os
import json
import pickle
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm

# Custom modules from the components package
from components.augmentation import augment_orientation
from components.datasets import (
    IMUDataset,
    create_balanced_subject_folds,
    load_predefined_validate_folds,
)
from components.pre_processing import hand_mirroring
from components.checkpoint import save_best_model
from components.model_mstcn import MSTCN, MSTCN_Loss


def main():
    # ============================ Configuration =============================
    # Dataset and model training configuration
    DATASET = "FDI"  # Options: "DXI"/"DXII" or "FDI"/"FDII"/"FDIII"
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
    WINDOW_LENGTH = 60  # in seconds
    WINDOW_SIZE = SAMPLING_FREQ * WINDOW_LENGTH
    NUM_FOLDS = 7
    NUM_EPOCHS = 200
    BATCH_SIZE = 64
    NUM_WORKERS = 16
    FLAG_AUGMENT = False
    FLAG_MIRROR = True

    # ====================== Dataset Path Configuration =======================
    if DATASET.startswith("DX"):
        NUM_CLASSES = 2
        version_label = DATASET.replace("DX", "").upper() or "I"
        data_dir = os.path.join("dataset", "DX", f"DX-{version_label}")
        TASK = "binary"
    elif DATASET.startswith("FD"):
        NUM_CLASSES = 3
        version_label = DATASET.replace("FD", "").upper() or "I"
        data_dir = os.path.join("dataset", "FD", f"FD-{version_label}")
        TASK = "multiclass"
    else:
        raise ValueError(f"Invalid dataset: {DATASET}")

    # File paths for left-hand and right-hand data
    x_l_path = os.path.join(data_dir, "X_L.pkl")
    y_l_path = os.path.join(data_dir, "Y_L.pkl")
    x_r_path = os.path.join(data_dir, "X_R.pkl")
    y_r_path = os.path.join(data_dir, "Y_R.pkl")

    # =========================== Data Loading ================================
    with open(x_l_path, "rb") as f:
        x_l = np.array(pickle.load(f), dtype=object)
    with open(y_l_path, "rb") as f:
        y_l = np.array(pickle.load(f), dtype=object)
    with open(x_r_path, "rb") as f:
        x_r = np.array(pickle.load(f), dtype=object)
    with open(y_r_path, "rb") as f:
        y_r = np.array(pickle.load(f), dtype=object)

    # ===================== Result and Checkpoint Setup =======================
    # Generate version prefix using current datetime (first 12 characters)
    version_prefix = datetime.now().strftime("%Y%m%d%H%M")[:12]
    result_dir = os.path.join("result", version_prefix)
    os.makedirs(result_dir, exist_ok=True)

    # File paths for saving training statistics and configuration
    training_stats_file = os.path.join(result_dir, "train_stats.npy")
    config_file = os.path.join(result_dir, "config.json")
    training_statistics = []

    # ========================= Data Pre-processing ============================
    # Optionally apply hand mirroring to the left-hand data
    if FLAG_MIRROR:
        x_l = np.array([hand_mirroring(sample) for sample in x_l], dtype=object)

    # Merge left-hand and right-hand data
    x_data = np.concatenate([x_l, x_r], axis=0)
    y_data = np.concatenate([y_l, y_r], axis=0)

    # Create the full IMU dataset with the defined window size
    full_dataset = IMUDataset(x_data, y_data, sequence_length=WINDOW_SIZE)

    # =================== Optional FDIII Data Augmentation =====================
    fdiii_dataset = None
    if DATASET in ["FDII", "FDI"]:
        fdiii_dir = os.path.join("dataset", "FD", "FD-III")
        with open(os.path.join(fdiii_dir, "X_L.pkl"), "rb") as f:
            x_l_fdiii = np.array(pickle.load(f), dtype=object)
        with open(os.path.join(fdiii_dir, "Y_L.pkl"), "rb") as f:
            y_l_fdiii = np.array(pickle.load(f), dtype=object)
        with open(os.path.join(fdiii_dir, "X_R.pkl"), "rb") as f:
            x_r_fdiii = np.array(pickle.load(f), dtype=object)
        with open(os.path.join(fdiii_dir, "Y_R.pkl"), "rb") as f:
            y_r_fdiii = np.array(pickle.load(f), dtype=object)
        x_fdiii = np.concatenate([x_l_fdiii, x_r_fdiii], axis=0)
        y_fdiii = np.concatenate([y_l_fdiii, y_r_fdiii], axis=0)
        fdiii_dataset = IMUDataset(x_fdiii, y_fdiii, sequence_length=WINDOW_SIZE)

    # ===================== Cross-Validation Setup ============================
    # Create balanced cross-validation folds based on subject IDs
    validate_folds = create_balanced_subject_folds(full_dataset, num_folds=NUM_FOLDS)
    # To use predefined folds, uncomment the following:
    # validate_folds = load_predefined_validate_folds()

    # ==================== Device Configuration =================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =================== Main Cross-Validation Training Loop =====================
    for fold, validate_subjects in enumerate(
        tqdm(validate_folds, desc="K-Fold", leave=True)
    ):
        # (Optional) Process only the first fold for demonstration:
        # if fold != 0:
        #     continue

        # Get training indices (subjects not in the current validation set)
        train_indices = [
            idx
            for idx, subject in enumerate(full_dataset.subject_indices)
            if subject not in validate_subjects
        ]

        # Augment training data with FDIII dataset if applicable
        if DATASET in ["FDII", "FDI"] and fdiii_dataset is not None:
            train_dataset = ConcatDataset(
                [Subset(full_dataset, train_indices), fdiii_dataset]
            )
        else:
            train_dataset = Subset(full_dataset, train_indices)

        # Create DataLoader for the training set
        train_loader = DataLoader(
            Subset(full_dataset, train_indices),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize the MSTCN model and optimizer
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

        # Initialize best loss for checkpointing (lower is better)
        best_loss = float("inf")

        # ========================== Training Loop ================================
        for epoch in tqdm(range(NUM_EPOCHS), desc=f"Fold {fold+1}", leave=False):
            model.train()
            epoch_loss = 0.0
            epoch_loss_ce = 0.0
            epoch_loss_mse = 0.0

            for batch_x, batch_y in train_loader:
                # Optionally apply data augmentation
                if FLAG_AUGMENT:
                    batch_x = augment_orientation(batch_x)
                # Permute dimensions and move tensors to the configured device
                batch_x = batch_x.permute(0, 2, 1).to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss_ce, loss_mse = MSTCN_Loss(outputs, batch_y)
                loss = loss_ce + LAMBDA_COEF * loss_mse
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_loss_ce += loss_ce.item()
                epoch_loss_mse += loss_mse.item()

            # Save the best model based on the current loss
            best_loss = save_best_model(
                model,
                fold=fold + 1,
                current_metric=loss.item(),
                best_metric=best_loss,
                checkpoint_dir=result_dir,
                mode="min",
            )

            # Log training statistics for the current epoch
            stats = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": epoch_loss / len(train_loader),
                "train_loss_ce": epoch_loss_ce / len(train_loader),
                "train_loss_mse": epoch_loss_mse / len(train_loader),
            }
            training_statistics.append(stats)

    # ====================== Save Training Results and Config ===================
    np.save(training_stats_file, training_statistics)

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
        "validate_folds": validate_folds,
    }

    with open(config_file, "w") as f:
        json.dump(config_info, f, indent=4)

    # Run the validation script after training
    os.system("python validate.py")


if __name__ == "__main__":
    main()
