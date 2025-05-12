#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Training Script (Single and Distributed Combined)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-12
Description : This script trains MSTCN models on IMU data with:
              1. Support for both single-GPU and distributed multi-GPU training
              2. Cross-validation across subject folds
              3. Configurable model architectures (CNN-LSTM, TCN, MSTCN)
              4. Flexible dataset handling (DX/FD/Oreba) with hand-specific inputs
              5. Optional augmentation strategies and dataset merging (FDIII/Oreba)
              6. Detailed logging, fold-wise statistics, and checkpoint saving

              Tips: Use --distributed flag for distributed training mode.
===============================================================================
"""


# Standard library imports
import os
import json
import pickle
import logging
import argparse
from datetime import datetime

# Third-party imports
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, ConcatDataset, DistributedSampler
from tqdm import tqdm

# Local imports
from components.pre_processing import hand_mirroring
from components.checkpoint import save_best_model
from components.models.cnnlstm import CNNLSTM
from components.models.tcn import TCN, MSTCN
from components.utils import loss_fn
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

# ----------------------------------------------------------------------------------------------
# Dataset Configuration
# ----------------------------------------------------------------------------------------------
DATASET = "DXI"
if DATASET.startswith("DX"):
    NUM_CLASSES = 2
    SAMPLING_FREQ_ORIGINAL = 64
    DOWNSAMPLE_FACTOR = 4
    sub_version = DATASET.replace("DX", "").upper() or "I"
    DATA_DIR = f"./dataset/DX/DX-{sub_version}"
    TASK = "binary"
    HAND_SEPERATION = True
elif DATASET.startswith("FD"):
    SAMPLING_FREQ_ORIGINAL = 64
    DOWNSAMPLE_FACTOR = 4
    NUM_CLASSES = 3
    sub_version = DATASET.replace("FD", "").upper() or "I"
    DATA_DIR = f"./dataset/FD/FD-{sub_version}"
    TASK = "multiclass"
    HAND_SEPERATION = True
elif DATASET.startswith("OREBA"):
    SAMPLING_FREQ_ORIGINAL = 64
    DOWNSAMPLE_FACTOR = 4
    NUM_CLASSES = 3
    DATA_DIR = "./dataset/Oreba"
    TASK = "multiclass"
    HAND_SEPERATION = False
else:
    raise ValueError(f"Invalid dataset: {DATASET}")
SAMPLING_FREQ = SAMPLING_FREQ_ORIGINAL // DOWNSAMPLE_FACTOR

# ----------------------------------------------------------------------------------------------
# Dataloader Configuration
# ----------------------------------------------------------------------------------------------
WINDOW_LENGTH = 60
WINDOW_SIZE = SAMPLING_FREQ * WINDOW_LENGTH
BATCH_SIZE = 64
NUM_WORKERS = 16

# ----------------------------------------------------------------------------------------------
# Model Configuration
# ----------------------------------------------------------------------------------------------
MODEL = "MSTCN"  # Options: CNN_LSTM, TCN, MSTCN
INPUT_DIM = 6
LAMBDA_COEF = 0.15
if MODEL in ["TCN", "MSTCN"]:
    KERNEL_SIZE = 3
    NUM_LAYERS = 9
    NUM_FILTERS = 128
    DROPOUT = 0.3
    CAUSAL = False
    if MODEL == "MSTCN":
        NUM_STAGES = 2
elif MODEL == "CNN_LSTM":
    CONV_FILTERS = (32, 64, 128)
    LSTM_HIDDEN = 128
else:
    raise ValueError(f"Invalid model: {MODEL}")

# ----------------------------------------------------------------------------------------------
# Training Configuration
# ----------------------------------------------------------------------------------------------
LEARNING_RATE = 5e-4
if DATASET == "FDI":
    NUM_FOLDS = 7
else:
    NUM_FOLDS = 5
NUM_EPOCHS = 100

# ----------------------------------------------------------------------------------------------
# Augmentation Configuration
# ----------------------------------------------------------------------------------------------
FLAG_AUGMENT_HAND_MIRRORING = False
FLAG_AUGMENT_AXIS_PERMUTATION = False
FLAG_AUGMENT_PLANAR_ROTATION = False
FLAG_AUGMENT_SPATIAL_ORIENTATION = False
FLAG_DATASET_AUGMENTATION = False
FLAG_DATASET_MIRRORING = False  # If True, mirror the left hand data

# ==============================================================================================
#                                   Main Training Code
# ==============================================================================================

parser = argparse.ArgumentParser(description="IMU HAR Training Script")
parser.add_argument("--distributed", action="store_true", help="Run in distributed mode")
parser.add_argument("--smoothing", type=str, default="L1", help="Smoothing loss type")
args = parser.parse_args()
SMOOTHING = args.smoothing

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
    # Logger setup
    logger.info(f"[Rank {local_rank}] Using device: {device}")
    overall_start = datetime.now()
    logger.info(f"Training started at: {overall_start}")

    # Create result directory
    version_prefix = f"{DATASET}_{MODEL}_{SMOOTHING}"
    if FLAG_AUGMENT_HAND_MIRRORING:
        version_prefix += "_HM"
    if FLAG_DATASET_AUGMENTATION:
        version_prefix += "_DA"
    if FLAG_DATASET_MIRRORING:
        version_prefix += "_DM"

    result_dir = os.path.join("result", version_prefix)
    postfix = 1
    original_result_dir = result_dir
    while os.path.exists(result_dir):
        result_dir = f"{original_result_dir}_{postfix}"
        postfix += 1
    os.makedirs(result_dir, exist_ok=True)
    checkpoint_dir = os.path.join(result_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    training_stats_file = os.path.join(result_dir, "training_stats.npy")

# ----------------------------------------------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------------------------------------------

if HAND_SEPERATION:
    with open(os.path.join(DATA_DIR, "X_L.pkl"), "rb") as f:
        X_L = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(DATA_DIR, "Y_L.pkl"), "rb") as f:
        Y_L = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(DATA_DIR, "X_R.pkl"), "rb") as f:
        X_R = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(DATA_DIR, "Y_R.pkl"), "rb") as f:
        Y_R = np.array(pickle.load(f), dtype=object)

    if FLAG_DATASET_MIRRORING:
        X_L = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)

    # Combine left and right data into a unified dataset
    X = np.array([np.concatenate([x_l, x_r], axis=0) for x_l, x_r in zip(X_L, X_R)], dtype=object)
    Y = np.array([np.concatenate([y_l, y_r], axis=0) for y_l, y_r in zip(Y_L, Y_R)], dtype=object)
    full_dataset = IMUDataset(X, Y, sequence_length=WINDOW_SIZE, downsample_factor=DOWNSAMPLE_FACTOR)
else:
    with open(os.path.join(DATA_DIR, "X.pkl"), "rb") as f:
        X = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(DATA_DIR, "Y.pkl"), "rb") as f:
        Y = np.array(pickle.load(f), dtype=object)
    full_dataset = IMUDataset(X, Y, sequence_length=WINDOW_SIZE, downsample_factor=DOWNSAMPLE_FACTOR)

# ----------------------------------------------------------------------------------------------
# Augment Dataset
# ----------------------------------------------------------------------------------------------

fdiii_dataset = None
oreba_dataset = None
if DATASET == "FDI" and FLAG_DATASET_AUGMENTATION:
    fdiii_dir = "./dataset/FD/FD-III"
    with open(os.path.join(fdiii_dir, "X_L.pkl"), "rb") as f:
        X_L_fdiii = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(fdiii_dir, "Y_L.pkl"), "rb") as f:
        Y_L_fdiii = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(fdiii_dir, "X_R.pkl"), "rb") as f:
        X_R_fdiii = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(fdiii_dir, "Y_R.pkl"), "rb") as f:
        Y_R_fdiii = np.array(pickle.load(f), dtype=object)
    if FLAG_DATASET_MIRRORING:
        X_L_fdiii = np.array([hand_mirroring(sample) for sample in X_L_fdiii], dtype=object)
    X_fdiii = np.concatenate([X_L_fdiii, X_R_fdiii], axis=0)
    Y_fdiii = np.concatenate([Y_L_fdiii, Y_R_fdiii], axis=0)
    fdiii_dataset = IMUDataset(
        X_fdiii,
        Y_fdiii,
        sequence_length=WINDOW_SIZE,
        downsample_factor=DOWNSAMPLE_FACTOR,
    )

    oreba_dir = "./dataset/Oreba"
    with open(os.path.join(oreba_dir, "X.pkl"), "rb") as f:
        X_oreba = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(oreba_dir, "Y.pkl"), "rb") as f:
        Y_oreba = np.array(pickle.load(f), dtype=object)
    oreba_dataset = IMUDataset(
        X_oreba,
        Y_oreba,
        sequence_length=WINDOW_SIZE,
        downsample_factor=DOWNSAMPLE_FACTOR,
    )

# ----------------------------------------------------------------------------------------------
# Training loop over folds
# ----------------------------------------------------------------------------------------------
if DATASET == "FDI":
    validate_folds = load_predefined_validate_folds()
else:
    validate_folds = create_balanced_subject_folds(full_dataset, num_folds=NUM_FOLDS)
training_statistics = []
for fold, validate_subjects in enumerate(validate_folds):
    # Prepare training data
    train_indices = [i for i, s in enumerate(full_dataset.subject_indices) if s not in validate_subjects]
    if DATASET == "FDI" and FLAG_DATASET_AUGMENTATION:
        base_train_dataset = Subset(full_dataset, train_indices)
        train_dataset = ConcatDataset([base_train_dataset, fdiii_dataset, oreba_dataset])
    else:
        train_dataset = Subset(full_dataset, train_indices)

    # Setup dataloader
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

    # Training loop over epochs
    model.train()
    for epoch in tqdm(range(NUM_EPOCHS), desc=f"Fold {fold+1}", leave=False):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        training_loss = 0
        training_loss_ce = 0
        training_loss_smooth = 0

        for batch_x, batch_y in train_loader:
            # Shape of batch_x: [batch_size, seq_len, channels]
            # Shape of batch_y: [batch_size, seq_len]
            # Optionally apply data augmentation
            if FLAG_AUGMENT_HAND_MIRRORING:
                batch_x, batch_y = augment_hand_mirroring(batch_x, batch_y, 1, True)
            if FLAG_AUGMENT_AXIS_PERMUTATION:
                batch_x, batch_y = augment_axis_permutation(batch_x, batch_y, 0.5, True)
            if FLAG_AUGMENT_PLANAR_ROTATION:
                batch_x, batch_y = augment_planar_rotation(batch_x, batch_y, 0.5, True)
            if FLAG_AUGMENT_SPATIAL_ORIENTATION:
                batch_x, batch_y = augment_spatial_orientation(batch_x, batch_y, 0.5, True)
            # Rearrange dimensions because CNN in PyTorch expect the channel dimension to be the second dimension (index 1)
            # Shape of batch_x: [batch_size, channels, seq_len]
            batch_x = batch_x.permute(0, 2, 1).to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            if MODEL == "MSTCN":
                ce_loss = 0.0
                smooth_loss = 0.0
                for output in outputs:
                    ce, smooth = loss_fn(output, batch_y, smoothing=SMOOTHING)
                    ce_loss += ce
                    smooth_loss += smooth
                # ce_loss, smooth_loss = MSTCN_Loss(outputs, batch_y)
            else:
                ce_loss, smooth_loss = loss_fn(outputs, batch_y, smoothing=SMOOTHING)
            loss = ce_loss + LAMBDA_COEF * smooth_loss
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_loss_ce += ce_loss.item()
            training_loss_smooth += smooth_loss.item()

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
                "train_loss_smooth": training_loss_smooth / len(train_loader),
            }
            training_statistics.append(stats)
            if epoch % 5 == 0:
                logger.info(
                    f"[Rank {local_rank}] Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}, CE: {training_loss_ce / len(train_loader):.4f}, smooth: {training_loss_smooth / len(train_loader):.4f}"
                )

# ==============================================================================================
#                                   Save final results
# ==============================================================================================
if local_rank == 0:
    np.save(training_stats_file, training_statistics)
    logger.info(f"Training statistics saved to {training_stats_file}")
    logger.info(f"Training completed in {datetime.now() - overall_start}")

    # Save configuration
    config_info = {
        # Dataset Settings
        "dataset": DATASET,
        "task": TASK,
        "hand_separation": HAND_SEPERATION,
        "num_classes": NUM_CLASSES,
        "sampling_freq_original": SAMPLING_FREQ_ORIGINAL,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "sampling_freq": SAMPLING_FREQ,
        "data_dir": DATA_DIR,
        # Dataloader Settings
        "window_length": WINDOW_LENGTH,
        "window_size": WINDOW_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        # Model Settings
        "model": MODEL,
        "input_dim": INPUT_DIM,
        "smoothing": SMOOTHING,
        "lambda_coef": LAMBDA_COEF,
        # Training Settings
        "learning_rate": LEARNING_RATE,
        "num_folds": NUM_FOLDS,
        "num_epochs": NUM_EPOCHS,
        # Augmentation flags
        "augmentation_hand_mirroring": FLAG_AUGMENT_HAND_MIRRORING,
        "augmentation_axis_permutation": FLAG_AUGMENT_AXIS_PERMUTATION,
        "augmentation_planar_rotation": FLAG_AUGMENT_PLANAR_ROTATION,
        "augmentation_spatial_orientation": FLAG_AUGMENT_SPATIAL_ORIENTATION,
        "dataset_augmentation": FLAG_DATASET_AUGMENTATION,
        "dataset_mirroring": FLAG_DATASET_MIRRORING,
        # Model-specific parameters
        **({"conv_filters": CONV_FILTERS, "lstm_hidden": LSTM_HIDDEN} if MODEL == "CNN_LSTM" else {}),
        **(
            {
                "num_layers": NUM_LAYERS,
                "num_filters": NUM_FILTERS,
                "kernel_size": KERNEL_SIZE,
                "dropout": DROPOUT,
                "causal": CAUSAL,
            }
            if MODEL in ["TCN", "MSTCN"]
            else {}
        ),
        **({"num_stages": NUM_STAGES} if MODEL == "MSTCN" else {}),
        # Validation configuration
        "validate_folds": validate_folds,
    }

    if MODEL == "CNN_LSTM":
        config_info.update({"conv_filters": CONV_FILTERS, "lstm_hidden": LSTM_HIDDEN})
    elif MODEL in ["TCN", "MSTCN"]:
        config_info.update(
            {"num_layers": NUM_LAYERS, "num_filters": NUM_FILTERS, "kernel_size": KERNEL_SIZE, "dropout": DROPOUT}
        )
        if MODEL == "MSTCN":
            config_info["num_stages"] = NUM_STAGES

    config_file = os.path.join(result_dir, "training_config.json")
    with open(config_file, "w") as f:
        json.dump(config_info, f, indent=4)
    logger.info(f"Configuration saved to {config_file}")

if args.distributed:
    dist.destroy_process_group()
