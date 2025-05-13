#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Training Script (Single and Distributed Combined)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-13
Description : This script trains MSTCN models on IMU data with:
              1. Support for both single-GPU and distributed multi-GPU training
              2. Cross-validation across subject folds
              3. Configurable model architectures (CNN-LSTM, TCN, MSTCN)
              4. Flexible dataset handling (DX/FD) with hand-specific inputs
              5. Detailed logging, fold-wise statistics, and checkpoint saving

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
from fractions import Fraction

# Third-party imports
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, DistributedSampler
from tqdm import tqdm

# Local imports
from components.pre_processing import hand_mirroring
from components.checkpoint import save_best_model
from components.models.cnnlstm import CNNLSTM
from components.models.tcn import TCN, MSTCN
from components.models.accnet import AccNet
from components.models.resnet_bilstm import ResNetEncoder, BiLSTMHead, ResNetBiLSTM
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
elif DATASET.startswith("FD"):
    SAMPLING_FREQ_ORIGINAL = 64
    DOWNSAMPLE_FACTOR = 4
    NUM_CLASSES = 3
    sub_version = DATASET.replace("FD", "").upper() or "I"
    DATA_DIR = f"./dataset/FD/FD-{sub_version}"
    TASK = "multiclass"
else:
    raise ValueError(f"Invalid dataset: {DATASET}")
SAMPLING_FREQ = SAMPLING_FREQ_ORIGINAL // DOWNSAMPLE_FACTOR

# ----------------------------------------------------------------------------------------------
# Dataloader Configuration
# ----------------------------------------------------------------------------------------------
WINDOW_SECONDS = 60
WINDOW_SAMPLES = SAMPLING_FREQ * WINDOW_SECONDS
BATCH_SIZE = 64
NUM_WORKERS = 16

# ----------------------------------------------------------------------------------------------
# Model Configuration
# ----------------------------------------------------------------------------------------------
MODEL = "TCN"  # Options: CNN_LSTM, TCN, MSTCN, AccNet, ResNetBiLSTM
if MODEL in ["TCN", "MSTCN", "AccNet"]:
    KERNEL_SIZE = 3
if MODEL in ["TCN", "MSTCN"]:
    NUM_LAYERS = 9
    NUM_FILTERS = 128
    DROPOUT = 0.3
    CAUSAL = False
    if MODEL == "MSTCN":
        NUM_STAGES = 2
elif MODEL in ["CNN_LSTM", "AccNet"]:
    CONV_FILTERS = (32, 64, 128)
elif MODEL in ["ResNetBiLSTM", "CNN_LSTM"]:
    LSTM_HIDDEN = 128
else:
    raise ValueError(f"Invalid model: {MODEL}")

if MODEL in ["CNN_LSTM", "TCN", "MSTCN"]:
    INPUT_DIM = 6
elif MODEL in ["AccNet", "ResNetBiLSTM"]:
    INPUT_DIM = 3
else:
    raise ValueError(f"Invalid model: {MODEL}")

if MODEL in ["AccNet", "ResNetBiLSTM"]:
    SELECTED_CHANNELS = [0, 1, 2]  # Only use accelerometer data for these models
else:
    SELECTED_CHANNELS = [0, 1, 2, 3, 4, 5]


# ----------------------------------------------------------------------------------------------
# Training Configuration
# ----------------------------------------------------------------------------------------------
LEARNING_RATE = 5e-4
LAMBDA_COEF = 0.2
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
    version_prefix = f"{DATASET}_{MODEL}"
    if FLAG_DATASET_MIRRORING:
        version_prefix += "_DM"
    if FLAG_AUGMENT_HAND_MIRRORING:
        version_prefix += "_AM"
    if FLAG_AUGMENT_AXIS_PERMUTATION:
        version_prefix += "_AP"
    if FLAG_AUGMENT_PLANAR_ROTATION:
        version_prefix += "_AR"
    if FLAG_AUGMENT_SPATIAL_ORIENTATION:
        version_prefix += "_AS"

    result_dir = os.path.join(f"results", version_prefix)
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
dataset = IMUDataset(
    X, Y, sequence_length=WINDOW_SAMPLES, downsample_factor=Fraction(32, 15), selected_channels=SELECTED_CHANNELS
)


# ----------------------------------------------------------------------------------------------
# Training loop over folds
# ----------------------------------------------------------------------------------------------
if DATASET == "FDI":
    validate_folds = load_predefined_validate_folds()
else:
    validate_folds = create_balanced_subject_folds(dataset, num_folds=NUM_FOLDS)
training_statistics = []
for fold, validate_subjects in enumerate(validate_folds):
    # Prepare training data
    train_indices = [i for i, s in enumerate(dataset.subject_indices) if s not in validate_subjects]
    train_dataset = Subset(dataset, train_indices)

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
    if MODEL in ["TCN", "MSTCN", "CNN_LSTM", "AccNet"]:
        model = {"TCN": TCN, "MSTCN": MSTCN, "CNN_LSTM": CNNLSTM, "AccNet": AccNet}[MODEL](
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
                    else (
                        {
                            "input_channels": INPUT_DIM,
                            "conv_filters": CONV_FILTERS,
                            "lstm_hidden": LSTM_HIDDEN,
                            "num_classes": NUM_CLASSES,
                        }
                        if MODEL == "CNN_LSTM"
                        else {
                            "num_classes": NUM_CLASSES,
                            "input_channels": INPUT_DIM,
                            "conv_filters": CONV_FILTERS,
                            "kernel_size": KERNEL_SIZE,
                        }
                    )
                )
            )
        ).to(device)
    elif MODEL == "ResNetBiLSTM":
        encoder = ResNetEncoder(
            in_channels=INPUT_DIM,
        ).to(device)

        # figure out the encoder's actual output size
        with torch.no_grad():
            dummy = torch.zeros(1, INPUT_DIM, WINDOW_SAMPLES, device=device)
            flat_feats = encoder(dummy)
            feature_dim = flat_feats.shape[1]  # e.g. 4096

        # now build the head with the correct feature_dim
        seq_head = BiLSTMHead(
            feature_dim=feature_dim,
            seq_length=WINDOW_SAMPLES,
            num_classes=NUM_CLASSES,
            hidden_dim=LSTM_HIDDEN,
        ).to(device)

        model = ResNetBiLSTM(encoder, seq_head).to(device)
    else:
        raise ValueError(f"Invalid model: {MODEL}")

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
        "num_classes": NUM_CLASSES,
        "sampling_freq_original": SAMPLING_FREQ_ORIGINAL,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "selected_channels": SELECTED_CHANNELS,
        "sampling_freq": SAMPLING_FREQ,
        "data_dir": DATA_DIR,
        # Dataloader Settings
        "window_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SAMPLES,
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
        "dataset_mirroring": FLAG_DATASET_MIRRORING,
        # Model-specific parameters
        "conv_filters": CONV_FILTERS if MODEL in ["CNN_LSTM", "AccNet"] else None,
        "lstm_hidden": LSTM_HIDDEN if MODEL in ["CNN_LSTM", "ResNetBiLSTM"] else None,
        "num_layers": NUM_LAYERS if MODEL in ["TCN", "MSTCN"] else None,
        "num_filters": NUM_FILTERS if MODEL in ["TCN", "MSTCN"] else None,
        "kernel_size": KERNEL_SIZE if MODEL in ["TCN", "MSTCN", "AccNet"] else None,
        "dropout": DROPOUT if MODEL in ["TCN", "MSTCN"] else None,
        "causal": CAUSAL if MODEL in ["TCN", "MSTCN"] else None,
        "num_stages": NUM_STAGES if MODEL == "MSTCN" else None,
        "hidden_dim": LSTM_HIDDEN if MODEL == "ResNetBiLSTM" else None,
        "seq_length": WINDOW_SAMPLES if MODEL == "ResNetBiLSTM" else None,
        # Validation configuration
        "validate_folds": validate_folds,
    }

    config_file = os.path.join(result_dir, "training_config.json")
    with open(config_file, "w") as f:
        json.dump(config_info, f, indent=4)
    logger.info(f"Configuration saved to {config_file}")

if args.distributed:
    dist.destroy_process_group()
