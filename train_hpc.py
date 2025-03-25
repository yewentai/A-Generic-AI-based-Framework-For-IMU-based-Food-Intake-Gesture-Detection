#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Training Script (Distributed Version)
-------------------------------------------------------------------------------
Author      : Joseph Yep (improved by ChatGPT)
Email       : yewentai126@gmail.com
Version     : 2.1
Created     : 2025-03-25
Description : Distributed training script for the MSTCN model on IMU data using
              cross-validation and PyTorch DistributedDataParallel (DDP). This
              script initializes a distributed process group, partitions the
              data with DistributedSampler, and ensures that only the master
              process (rank 0) handles checkpoint saving and logging.
===============================================================================
"""

import os
import json
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, ConcatDataset, DistributedSampler

# Custom modules
from components.augmentation import augment_orientation
from components.datasets import (
    IMUDataset,
    create_balanced_subject_folds,
    load_predefined_validate_folds,
)
from components.pre_processing import hand_mirroring
from components.checkpoint import save_best_model
from components.model_mstcn import MSTCN, MSTCN_Loss

# =========================== Configuration Parameters ===========================
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
WINDOW_LENGTH = 60  # seconds
WINDOW_SIZE = SAMPLING_FREQ * WINDOW_LENGTH
NUM_FOLDS = 7
NUM_EPOCHS = 200
BATCH_SIZE = 64
NUM_WORKERS = 16
FLAG_AUGMENT = True
FLAG_MIRROR = True


def main(local_rank=None, world_size=None):
    """
    Main function for distributed training of the MSTCN model.
    Initializes distributed training, loads and preprocesses the dataset,
    sets up the data loaders, and executes the training loop with checkpointing
    and logging (only on the master process, rank 0).
    """
    # Retrieve distributed training parameters from environment if not provided
    if local_rank is None or world_size is None:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    # Initialize the distributed process group using NCCL backend
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Master process: set up logging and result directories
    if local_rank == 0:
        print(f"[Rank {local_rank}] Using device: {device}")
        overall_start = datetime.now()
        print("Training started at:", overall_start)
        version_prefix = datetime.now().strftime("%Y%m%d%H%M")[:12]
        result_dir = os.path.join("result", version_prefix)
        os.makedirs(result_dir, exist_ok=True)
        training_stats_file = os.path.join(result_dir, "train_stats.npy")
        config_file = os.path.join(result_dir, "config.json")
    else:
        result_dir = None

    # ===================== Dataset Loading =====================
    # Determine dataset directory and number of classes based on the DATASET parameter
    if DATASET.startswith("DX"):
        NUM_CLASSES = 2
        dataset_version = DATASET.replace("DX", "").upper() or "I"
        data_dir = os.path.join("dataset", "DX", f"DX-{dataset_version}")
    elif DATASET.startswith("FD"):
        NUM_CLASSES = 3
        dataset_version = DATASET.replace("FD", "").upper() or "I"
        data_dir = os.path.join("dataset", "FD", f"FD-{dataset_version}")
    else:
        raise ValueError(f"Invalid dataset: {DATASET}")

    # Define file paths for the left-hand and right-hand data
    x_l_path = os.path.join(data_dir, "X_L.pkl")
    y_l_path = os.path.join(data_dir, "Y_L.pkl")
    x_r_path = os.path.join(data_dir, "X_R.pkl")
    y_r_path = os.path.join(data_dir, "Y_R.pkl")

    # Load the data from pickle files
    with open(x_l_path, "rb") as f:
        x_left = np.array(pickle.load(f), dtype=object)
    with open(y_l_path, "rb") as f:
        y_left = np.array(pickle.load(f), dtype=object)
    with open(x_r_path, "rb") as f:
        x_right = np.array(pickle.load(f), dtype=object)
    with open(y_r_path, "rb") as f:
        y_right = np.array(pickle.load(f), dtype=object)

    # Optionally apply hand mirroring to left-hand data
    if FLAG_MIRROR:
        x_left = np.array([hand_mirroring(sample) for sample in x_left], dtype=object)

    # Merge left and right data to create the full dataset
    x_data = np.concatenate([x_left, x_right], axis=0)
    y_data = np.concatenate([y_left, y_right], axis=0)
    full_dataset = IMUDataset(x_data, y_data, sequence_length=WINDOW_SIZE)

    # ===================== Optional FDIII Augmentation =====================
    fdiii_dataset = None
    if DATASET in ["FDII", "FDI"]:
        fdiii_dir = os.path.join("dataset", "FD", "FD-III")
        with open(os.path.join(fdiii_dir, "X_L.pkl"), "rb") as f:
            x_left_fdiii = np.array(pickle.load(f), dtype=object)
        with open(os.path.join(fdiii_dir, "Y_L.pkl"), "rb") as f:
            y_left_fdiii = np.array(pickle.load(f), dtype=object)
        with open(os.path.join(fdiii_dir, "X_R.pkl"), "rb") as f:
            x_right_fdiii = np.array(pickle.load(f), dtype=object)
        with open(os.path.join(fdiii_dir, "Y_R.pkl"), "rb") as f:
            y_right_fdiii = np.array(pickle.load(f), dtype=object)
        x_fdiii = np.concatenate([x_left_fdiii, x_right_fdiii], axis=0)
        y_fdiii = np.concatenate([y_left_fdiii, y_right_fdiii], axis=0)
        fdiii_dataset = IMUDataset(x_fdiii, y_fdiii, sequence_length=WINDOW_SIZE)

    # Create balanced cross-validation folds based on subject IDs
    validate_folds = create_balanced_subject_folds(full_dataset, num_folds=NUM_FOLDS)
    training_statistics = []

    # ===================== Cross-Validation Training Loop =====================
    for fold, validate_subjects in enumerate(validate_folds):
        # Identify training indices (subjects not in the current validation fold)
        train_indices = [
            idx
            for idx, subject in enumerate(full_dataset.subject_indices)
            if subject not in validate_subjects
        ]

        # Build the training dataset; optionally augment with FDIII data
        if DATASET in ["FDII", "FDI"] and fdiii_dataset is not None:
            base_train_dataset = Subset(full_dataset, train_indices)
            train_dataset = ConcatDataset([base_train_dataset, fdiii_dataset])
        else:
            train_dataset = Subset(full_dataset, train_indices)

        # Partition the training data among GPUs using DistributedSampler
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

        # ===================== Model Setup =====================
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

        # ===================== Epoch Training Loop =====================
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_sampler.set_epoch(epoch)  # Shuffle data for this epoch
            epoch_loss = 0.0
            epoch_loss_ce = 0.0
            epoch_loss_mse = 0.0

            for batch_x, batch_y in train_loader:
                # Optionally apply data augmentation
                if FLAG_AUGMENT:
                    batch_x = augment_orientation(batch_x)
                # Permute dimensions and move tensors to the GPU device
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

            # ===================== Logging and Checkpointing (Master Only) =====================
            if local_rank == 0:
                avg_loss = epoch_loss / len(train_loader)
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
                    "train_loss_ce": epoch_loss_ce / len(train_loader),
                    "train_loss_mse": epoch_loss_mse / len(train_loader),
                }
                training_statistics.append(stats)

    # ===================== Save Results and Configuration (Master Only) =====================
    if local_rank == 0:
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

        overall_end = datetime.now()
        print("Training ended at:", overall_end)
        print("Total training time:", overall_end - overall_start)

    # Clean up the distributed process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
