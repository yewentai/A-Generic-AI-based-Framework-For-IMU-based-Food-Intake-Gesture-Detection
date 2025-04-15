#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================================
MSTCN Left Hand IMU Training Script
------------------------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-14
Description : This script trains an MSTCN model on LEFT hand IMU data using cross-validation.
================================================================================================
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, ConcatDataset, DistributedSampler
from datetime import datetime
import argparse
from tqdm import tqdm
import logging

# Import custom modules from the components package
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

# ==============================================================================================
#                             Configuration Parameters
# ==============================================================================================

# Setup logger
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Dataset Settings
DATASET = "FDI"
SAMPLING_FREQ_ORIGINAL = 64
DOWNSAMPLE_FACTOR = 4
SAMPLING_FREQ = SAMPLING_FREQ_ORIGINAL // DOWNSAMPLE_FACTOR
if DATASET.startswith("DX"):
    NUM_CLASSES = 2
    sub_version = DATASET.replace("DX", "").upper() or "I"
    DATA_DIR = f"./dataset/DX/DX-{sub_version}"
    TASK = "binary"
elif DATASET.startswith("FD"):
    NUM_CLASSES = 3
    sub_version = DATASET.replace("FD", "").upper() or "I"
    DATA_DIR = f"./dataset/FD/FD-{sub_version}"
    TASK = "multiclass"
else:
    raise ValueError(f"Invalid dataset: {DATASET}")

# Dataloader Settings
WINDOW_LENGTH = 60
WINDOW_SIZE = SAMPLING_FREQ * WINDOW_LENGTH
BATCH_SIZE = 64
NUM_WORKERS = 16

# Model Settings
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

# Training Settings
LEARNING_RATE = 5e-4
NUM_FOLDS = 7
NUM_EPOCHS = 100
FLAG_AUGMENT_HAND_MIRRORING = False
FLAG_AUGMENT_AXIS_PERMUTATION = False
FLAG_AUGMENT_PLANAR_ROTATION = False
FLAG_AUGMENT_SPATIAL_ORIENTATION = False
FLAG_DATASET_MIRROR = False

# ==============================================================================================
#                            Main Training Code (Left Hand)
# ==============================================================================================

parser = argparse.ArgumentParser(description="Left Hand IMU Training Script")
parser.add_argument("--distributed", action="store_true", help="Run in distributed mode")
args = parser.parse_args()

if args.distributed:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = 0

if local_rank == 0:
    logger.info(f"[Rank {local_rank}] Using device: {device}")
    overall_start = datetime.now()
    logger.info(f"Left Hand Training started at: {overall_start}")
    version_prefix = datetime.now().strftime("%Y%m%d%H%M")[:12] + "_LEFT"
    result_dir = os.path.join("result", version_prefix)
    os.makedirs(result_dir, exist_ok=True)
    checkpoint_dir = os.path.join(result_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    training_stats_file = os.path.join(result_dir, "train_stats.npy")
else:
    checkpoint_dir = None

# Load only LEFT hand data
with open(os.path.join(DATA_DIR, "X_L.pkl"), "rb") as f:
    X = np.array(pickle.load(f), dtype=object)
with open(os.path.join(DATA_DIR, "Y_L.pkl"), "rb") as f:
    Y = np.array(pickle.load(f), dtype=object)

if FLAG_DATASET_MIRROR:
    X = np.array([hand_mirroring(sample) for sample in X], dtype=object)

full_dataset = IMUDataset(X, Y, sequence_length=WINDOW_SIZE, downsample_factor=DOWNSAMPLE_FACTOR)

# Augment Dataset with FDIII (if using FDII/FDI)
fdiii_dataset = None
if DATASET in ["FDII", "FDI"]:
    fdiii_dir = "./dataset/FD/FD-III"
    with open(os.path.join(fdiii_dir, "X_L.pkl"), "rb") as f:
        X_fdiii = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(fdiii_dir, "Y_L.pkl"), "rb") as f:
        Y_fdiii = np.array(pickle.load(f), dtype=object)
    if FLAG_DATASET_MIRROR:
        X_fdiii = np.array([hand_mirroring(sample) for sample in X_fdiii], dtype=object)
    fdiii_dataset = IMUDataset(
        X_fdiii,
        Y_fdiii,
        sequence_length=WINDOW_SIZE,
        downsample_factor=DOWNSAMPLE_FACTOR,
    )

# Create validation folds
if DATASET == "FDI":
    validate_folds = load_predefined_validate_folds()
else:
    validate_folds = create_balanced_subject_folds(full_dataset, num_folds=NUM_FOLDS)
training_statistics = []

loss_fn = {"TCN": TCN_Loss, "MSTCN": MSTCN_Loss, "CNN_LSTM": CNNLSTM_Loss}[MODEL]

for fold, validate_subjects in enumerate(validate_folds):
    train_indices = [i for i, s in enumerate(full_dataset.subject_indices) if s not in validate_subjects]
    if DATASET in ["FDII", "FDI"] and fdiii_dataset is not None:
        base_train_dataset = Subset(full_dataset, train_indices)
        train_dataset = ConcatDataset([base_train_dataset, fdiii_dataset])
    else:
        train_dataset = Subset(full_dataset, train_indices)

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        if args.distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=not args.distributed,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = {"TCN": TCN, "MSTCN": MSTCN, "CNN_LSTM": CNNLSTM}[MODEL](
        **(
            {
                "num_stages": NUM_STAGES,
                "num_layers": NUM_LAYERS,
                "num_classes": NUM_CLASSES,
                "input_dim": INPUT_DIM,
                "num_filters": NUM_FILTERS,
                "kernel_size": KERNEL_SIZE,
                "dropout": DROPOUT,
            }
            if MODEL == "MSTCN"
            else (
                {
                    "num_layers": NUM_LAYERS,
                    "num_classes": NUM_CLASSES,
                    "input_dim": INPUT_DIM,
                    "num_filters": NUM_FILTERS,
                    "kernel_size": KERNEL_SIZE,
                    "dropout": DROPOUT,
                }
                if MODEL == "TCN"
                else {
                    "input_channels": INPUT_DIM,
                    "conv_filters": CONV_FILTERS,
                    "lstm_hidden": LSTM_HIDDEN,
                    "num_classes": NUM_CLASSES,
                }
            )
        )
    ).to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float("inf")

    for epoch in tqdm(range(NUM_EPOCHS), desc=f"Fold {fold+1}", leave=False):
        model.train()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        training_loss = 0
        training_loss_ce = 0
        training_loss_mse = 0

        for batch_x, batch_y in train_loader:
            if FLAG_AUGMENT_HAND_MIRRORING:
                batch_x, batch_y = augment_hand_mirroring(batch_x, batch_y, 1, True)
            if FLAG_AUGMENT_AXIS_PERMUTATION:
                batch_x, batch_y = augment_axis_permutation(batch_x, batch_y, 0.5, True)
            if FLAG_AUGMENT_PLANAR_ROTATION:
                batch_x, batch_y = augment_planar_rotation(batch_x, batch_y, 0.5, True)
            if FLAG_AUGMENT_SPATIAL_ORIENTATION:
                batch_x, batch_y = augment_spatial_orientation(batch_x, batch_y, 0.5, True)

            batch_x = batch_x.permute(0, 2, 1).to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            ce_loss, mse_loss = loss_fn(outputs, batch_y)
            loss = ce_loss + LAMBDA_COEF * mse_loss
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_loss_ce += ce_loss.item()
            training_loss_mse += mse_loss.item()

        if local_rank == 0:
            avg_loss = training_loss / len(train_loader)
            best_loss = save_best_model(model, fold + 1, avg_loss, best_loss, checkpoint_dir, "min")
            stats = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_loss_ce": training_loss_ce / len(train_loader),
                "train_loss_mse": training_loss_mse / len(train_loader),
                "hand": "left",
            }
            training_statistics.append(stats)

if local_rank == 0:
    np.save(training_stats_file, training_statistics)
    logger.info(f"Left hand training statistics saved to {training_stats_file}")

    config_info = {
        "dataset": DATASET,
        "num_classes": NUM_CLASSES,
        "sampling_freq": SAMPLING_FREQ,
        "window_size": WINDOW_SIZE,
        "model": MODEL,
        "input_dim": INPUT_DIM,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_folds": NUM_FOLDS,
        "num_epochs": NUM_EPOCHS,
        "augmentation_hand_mirroring": FLAG_AUGMENT_HAND_MIRRORING,
        "augmentation_axis_permutation": FLAG_AUGMENT_AXIS_PERMUTATION,
        "augmentation_planar_rotation": FLAG_AUGMENT_PLANAR_ROTATION,
        "augmentation_spatial_orientation": FLAG_AUGMENT_SPATIAL_ORIENTATION,
        "dataset_mirroring": FLAG_DATASET_MIRROR,
        "validate_folds": validate_folds,
        "hand": "left",
    }

    if MODEL == "CNN_LSTM":
        config_info.update({"conv_filters": CONV_FILTERS, "lstm_hidden": LSTM_HIDDEN})
    elif MODEL in ["TCN", "MSTCN"]:
        config_info.update(
            {"num_layers": NUM_LAYERS, "num_filters": NUM_FILTERS, "kernel_size": KERNEL_SIZE, "dropout": DROPOUT}
        )
        if MODEL == "MSTCN":
            config_info["num_stages"] = NUM_STAGES

    config_file = os.path.join(result_dir, "config.json")
    with open(config_file, "w") as f:
        json.dump(config_info, f, indent=4)
    logger.info(f"Left hand configuration saved to {config_file}")

if args.distributed:
    dist.destroy_process_group()
