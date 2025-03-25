#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Training Script (Distributed Version)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Version     : 2.0
Created     : 2025-03-24
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
import re
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, ConcatDataset, DistributedSampler
from datetime import datetime

# Import your custom modules
from components.augmentation import augment_orientation
from components.datasets import (
    IMUDataset,
    create_balanced_subject_folds,
    load_predefined_validate_folds,
)
from components.pre_processing import hand_mirroring
from components.checkpoint import save_best_model
from components.model_mstcn import MSTCN, MSTCN_Loss

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
NUM_EPOCHS = 200
BATCH_SIZE = 64
NUM_WORKERS = 16
FLAG_AUGMENT = True
FLAG_MIRROR = False


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
        version_prefix = datetime.now().strftime("%Y%m%d%H%M")[:12]
        result_dir = os.path.join("result", version_prefix)
        os.makedirs(result_dir, exist_ok=True)
        # File paths for saving statistics and configuration
        TRAINING_STATS_FILE = os.path.join(result_dir, "train_stats.npy")
        CONFIG_FILE = os.path.join(result_dir, "config.json")

    # -------------------- Dataset Loading --------------------
    if DATASET.startswith("DX"):
        NUM_CLASSES = 2
        sub_version = DATASET.replace("DX", "").upper() or "I"
        DATA_DIR = f"./dataset/DX/DX-{sub_version}"
    elif DATASET.startswith("FD"):
        NUM_CLASSES = 3
        sub_version = DATASET.replace("FD", "").upper() or "I"
        DATA_DIR = f"./dataset/FD/FD-{sub_version}"
    else:
        raise ValueError(f"Invalid dataset: {DATASET}")

    # Define dataset file paths
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

    # Merge left-hand and right-hand data and create full dataset
    X = np.concatenate([X_L, X_R], axis=0)
    Y = np.concatenate([Y_L, Y_R], axis=0)
    full_dataset = IMUDataset(X, Y, sequence_length=WINDOW_SIZE)

    # Augment dataset with FDIII if applicable
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

    # Create balanced cross-validation folds
    validate_folds = create_balanced_subject_folds(full_dataset, num_folds=NUM_FOLDS)
    training_statistics = []

    # -------------------- Cross-Validation Loop --------------------
    for fold, validate_subjects in enumerate(validate_folds):
        # Split training and validation indices based on subject IDs
        train_indices = [
            i
            for i, subject in enumerate(full_dataset.subject_indices)
            if subject not in validate_subjects
        ]

        # Augment training data with FDIII if applicable
        if DATASET in ["FDII", "FDI"] and fdiii_dataset is not None:
            base_train_dataset = Subset(full_dataset, train_indices)
            train_dataset = ConcatDataset([base_train_dataset, fdiii_dataset])
        else:
            train_dataset = Subset(full_dataset, train_indices)

        # Create DistributedSampler for training data to partition it across GPUs
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # -------------------- Model Setup --------------------
        model = MSTCN(
            num_stages=NUM_STAGES,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            input_dim=INPUT_DIM,
            num_filters=NUM_FILTERS,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT,
        ).to(device)
        # Wrap the model with DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
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
                if FLAG_AUGMENT:
                    batch_x = augment_orientation(batch_x)
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
        np.save(TRAINING_STATS_FILE, training_statistics)
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

        # Save the configuration as JSON
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_info, f, indent=4)

        overall_end = datetime.now()
        print("Training ended at:", overall_end)
        print("Total training time:", overall_end - overall_start)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
