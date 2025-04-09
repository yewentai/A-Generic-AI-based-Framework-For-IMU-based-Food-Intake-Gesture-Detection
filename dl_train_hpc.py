#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Training Script (Distributed Version)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-03-29
Description : This script trains an MSTCN model on IMU data using cross-validation.
              It has been adapted to run on an HPC with multiple GPUs using PyTorch’s
              DistributedDataParallel. The code initializes a distributed process group,
              wraps the model with DDP, and uses DistributedSampler to partition data
              among GPUs. Only the process with rank 0 handles checkpoint saving and logging.
===============================================================================
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, ConcatDataset, DistributedSampler
from datetime import datetime

# Import your custom modules
from components.augmentation import (
    augment_hand_mirroring,
    augment_axis_permutation,
    augment_planar_rotation,
    augment_spatial_orientation,
)
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
    sub_version = DATASET.replace("DX", "").upper() or "I"  # Handles formats like DX/DXII
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
MODEL = "MSTCN"  # Options: CNN_LSTM, TCN, MSTCN
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
FLAG_AUGMENT_HAND_MIRRORING = False
FLAG_AUGMENT_AXIS_PERMUTATION = False
FLAG_AUGMENT_PLANAR_ROTATION = False
FLAG_AUGMENT_SPATIAL_ORIENTATION = False
FLAG_DATASET_MIRROR = True

# =============================================================================
#                             Main Training Function
# =============================================================================


def main(local_rank=None, world_size=None):
    if local_rank is None or world_size is None:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        print(f"[Rank {local_rank}] Using device: {device}")
        overall_start = datetime.now()
        print("Training started at:", overall_start)

        # Generate version prefix from current datetime (first 12 characters)
        version_prefix = datetime.now().strftime("%Y%m%d%H%M")[:12]

        # Create result directories using version_prefix
        result_dir = os.path.join("result", version_prefix)
        os.makedirs(result_dir, exist_ok=True)

        # Define file paths for saving statistics and configuration
        training_stas_file = os.path.join(result_dir, f"train_stats.npy")
        config_file = os.path.join(result_dir, "config.json")

    # -------------------- Dataset Loading --------------------
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
    if FLAG_DATASET_MIRROR:
        X_L = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)

    # Merge left-hand and right-hand data
    X = np.concatenate([X_L, X_R], axis=0)
    Y = np.concatenate([Y_L, Y_R], axis=0)

    # Create the full dataset using the defined window size
    full_dataset = IMUDataset(X, Y, sequence_length=WINDOW_SIZE, downsample_factor=DOWNSAMPLE_FACTOR)

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
        if FLAG_DATASET_MIRROR:
            X_L_fdiii = np.array([hand_mirroring(sample) for sample in X_L_fdiii], dtype=object)
        X_fdiii = np.concatenate([X_L_fdiii, X_R_fdiii], axis=0)
        Y_fdiii = np.concatenate([Y_L_fdiii, Y_R_fdiii], axis=0)
        fdiii_dataset = IMUDataset(
            X_fdiii,
            Y_fdiii,
            sequence_length=WINDOW_SIZE,
            downsample_factor=DOWNSAMPLE_FACTOR,
        )

    # Create validation folds based on the dataset type
    if DATASET == "FDI":
        validate_folds = load_predefined_validate_folds()
    else:
        validate_folds = create_balanced_subject_folds(full_dataset, num_folds=NUM_FOLDS)

    training_statistics = []

    # -------------------- Cross-Validation Loop --------------------
    for fold, validate_subjects in enumerate(validate_folds):
        # Split training and validation indices based on subject IDs
        train_indices = [
            i for i, subject in enumerate(full_dataset.subject_indices) if subject not in validate_subjects
        ]

        # Augment training data with FDIII if applicable
        if DATASET in ["FDII", "FDI"] and fdiii_dataset is not None:
            base_train_dataset = Subset(full_dataset, train_indices)
            train_dataset = ConcatDataset([base_train_dataset, fdiii_dataset])
        else:
            train_dataset = Subset(full_dataset, train_indices)

        # Create DistributedSampler for training data to partition it across GPUs
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # -------------------- Model Setup --------------------
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
        # Wrap the model with DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        best_loss = float("inf")

        # -------------------- Training Loop --------------------
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_sampler.set_epoch(epoch)  # Update sampler for shuffling
            training_loss = 0.0
            training_loss_ce = 0.0
            training_loss_mse = 0.0

            for batch_x, batch_y in train_loader:
                # Shape of batch_x: [batch_size, seq_len, channels]
                # Shape of batch_y: [batch_size, seq_len]
                # Optionally apply data augmentation
                if FLAG_AUGMENT_HAND_MIRRORING:
                    batch_x, batch_y = augment_hand_mirroring(batch_x, batch_y, probability=1, is_additive=True)
                if FLAG_AUGMENT_AXIS_PERMUTATION:
                    batch_x, batch_y = augment_axis_permutation(batch_x, batch_y, probability=0.5, is_additive=True)
                if FLAG_AUGMENT_PLANAR_ROTATION:
                    batch_x, batch_y = augment_planar_rotation(batch_x, batch_y, probability=0.5, is_additive=True)
                if FLAG_AUGMENT_SPATIAL_ORIENTATION:
                    batch_x, batch_y = augment_spatial_orientation(batch_x, batch_y, probability=0.5, is_additive=True)
                # Rearrange dimensions because CNN in PyTorch expect the channel dimension to be the second dimension (index 1)
                # Shape of batch_x: [batch_size, channels, seq_len]
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

            # -------------------- Logging and Checkpointing (Rank 0 Only) --------------------
            if local_rank == 0:
                avg_loss = training_loss / len(train_loader)
                best_loss = save_best_model(
                    model,
                    fold=fold + 1,
                    current_metric=loss.item(),
                    best_metric=best_loss,
                    checkpoint_dir=result_dir,
                    mode="min",
                )
                stats = {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "fold": fold + 1,
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "train_loss_ce": training_loss_ce / len(train_loader),
                    "train_loss_mse": training_loss_mse / len(train_loader),
                }
                training_statistics.append(stats)

    # -------------------- Save Results and Configuration (Rank 0) --------------------
    if local_rank == 0:
        np.save(training_stas_file, training_statistics)
        print(f"Training statistics saved to {training_stas_file}")

        # Build the initial part of the config with keys up to "model"
        config_info = {
            "dataset": DATASET,
            "num_classes": NUM_CLASSES,
            "sampling_freq": SAMPLING_FREQ,
            "window_size": WINDOW_SIZE,
            "model": MODEL,
            "input_dim": INPUT_DIM,
        }

        # Insert model-specific parameters right after the "model" key
        if MODEL == "CNN_LSTM":
            config_info["conv_filters"] = CONV_FILTERS
            config_info["lstm_hidden"] = LSTM_HIDDEN
        elif MODEL in ["TCN", "MSTCN"]:
            config_info["num_layers"] = NUM_LAYERS
            config_info["num_filters"] = NUM_FILTERS
            config_info["kernel_size"] = KERNEL_SIZE
            config_info["dropout"] = DROPOUT
            if MODEL == "MSTCN":
                config_info["num_stages"] = NUM_STAGES
        else:
            raise ValueError(f"Invalid model: {MODEL}")

        # Add the remaining configuration parameters after the model-specific ones
        config_info.update(
            {
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "num_folds": NUM_FOLDS,
                "num_epochs": NUM_EPOCHS,
                "augmentation_hand_mirroring": FLAG_AUGMENT_HAND_MIRRORING,
                "augmentation_axis_permutation": FLAG_AUGMENT_AXIS_PERMUTATION,
                "augmentation_planar_rotation": FLAG_AUGMENT_PLANAR_ROTATION,
                "augmentation_spatial_orientation": FLAG_AUGMENT_SPATIAL_ORIENTATION,
                "dataset_mirror": FLAG_DATASET_MIRROR,
                "validate_folds": validate_folds,
            }
        )

        # Save the configuration as JSON
        with open(config_file, "w") as f:
            json.dump(config_info, f, indent=4)
        print(f"Configuration saved to {config_file}")

        overall_end = datetime.now()
        print("Training ended at:", overall_end)
        print("Total training time:", overall_end - overall_start)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
