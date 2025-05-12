#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Fine-Tuning Script Using Pre-Trained ResNet Encoder with Sequence Labeling
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-12
Description : This script loads a pre-trained ResNet encoder (via the harnet10
              framework and load_weights function) and attaches a sequence
              labeling head for fine-tuning on a downstream sequence labeling
              task. It performs the following:
              1. Loads and combines left/right IMU data
              2. Wraps the ResNet encoder for feature extraction
              3. Trains a sequence labeling head on top of the encoder
              4. Fine-tunes the full network and saves model + config
===============================================================================
"""

import argparse
import json
import logging
import os
import pickle
from datetime import datetime
from fractions import Fraction

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from components.datasets import IMUDataset, create_balanced_subject_folds, load_predefined_validate_folds
from components.models.resnet import ResNetEncoder
from components.models.head import BiLSTMHead, ResNetBiLSTM
from components.pre_processing import hand_mirroring
from components.checkpoint import save_best_model
from components.utils import loss_fn

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

# Dataset Configuration
DATASET = "DXI"
if DATASET.startswith("DX"):
    SAMPLING_FREQ_ORIGINAL = 64
    DOWNSAMPLE_FACTOR = Fraction(32, 15)
    NUM_CLASSES = 2
    sub_version = DATASET.replace("DX", "").upper() or "I"
    DATA_DIR = f"./dataset/DX/DX-{sub_version}"
    TASK = "binary"
    HAND_SEPERATION = True
elif DATASET.startswith("FD"):
    SAMPLING_FREQ_ORIGINAL = 64
    DOWNSAMPLE_FACTOR = Fraction(32, 15)
    NUM_CLASSES = 3
    sub_version = DATASET.replace("FD", "").upper() or "I"
    DATA_DIR = f"./dataset/FD/FD-{sub_version}"
    TASK = "multiclass"
    HAND_SEPERATION = True
elif DATASET.startswith("OREBA"):
    SAMPLING_FREQ_ORIGINAL = 64
    DOWNSAMPLE_FACTOR = Fraction(32, 15)
    NUM_CLASSES = 3
    DATA_DIR = "./dataset/Oreba"
    TASK = "multiclass"
    HAND_SEPERATION = False
else:
    raise ValueError(f"Invalid dataset: {DATASET}")
SAMPLING_FREQ = SAMPLING_FREQ_ORIGINAL // DOWNSAMPLE_FACTOR

# Dataloader Configuration
WINDOW_SECONDS = 10  # seconds
WINDOW_SAMPLES = SAMPLING_FREQ * WINDOW_SECONDS
BATCH_SIZE = 64
NUM_WORKERS = 16

# Training Configuration
FREEZE_ENCODER = False  # If True, freeze the encoder weights, only train the head
MODEL = "ResNetBiLSTM_FTHead" if FREEZE_ENCODER else "ResNetBiLSTM_FTFull"
INPUT_DIM = 3  # Only accelerometer data
LEARNING_RATE = 5e-4
LAMBDA_COEF = 1
if DATASET == "FDI":
    NUM_FOLDS = 7
else:
    NUM_FOLDS = 5
NUM_EPOCHS = 100

# Augmentation Configuration
FLAG_AUGMENT_HAND_MIRRORING = False
FLAG_DATASET_MIRRORING = False  # If True, mirror the left hand data
FLAG_DATASET_MIRRORING_ADD = False  # If True, add mirrored data to the dataset

# ==============================================================================================
#                                   Main Training Code
# ==============================================================================================

parser = argparse.ArgumentParser(description="IMU Sequence Labeling Training Script")
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
    # Logger setup
    logger.info(f"[Rank {local_rank}] Using device: {device}")
    overall_start = datetime.now()
    logger.info(f"Training started at: {overall_start}")

    # Create result directory
    version_prefix = f"{DATASET}_{MODEL}"
    if FLAG_AUGMENT_HAND_MIRRORING:
        version_prefix += "_HM"
    if FLAG_DATASET_MIRRORING:
        version_prefix += "_DM"
    if FLAG_DATASET_MIRRORING_ADD:
        version_prefix += "_DMA"

    result_dir = os.path.join("results", version_prefix)
    postfix = 1
    original_result_dir = result_dir
    while os.path.exists(result_dir):
        result_dir = f"{original_result_dir}_{postfix}"
        postfix += 1
    os.makedirs(result_dir, exist_ok=True)
    checkpoint_dir = os.path.join(result_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    training_stats_file = os.path.join(result_dir, "training_stats.npy")

# Load Dataset
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

    if FLAG_DATASET_MIRRORING_ADD:
        X_L_mirrored = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)
        X_R_mirrored = np.array([hand_mirroring(sample) for sample in X_R], dtype=object)
        X_L = np.array([np.concatenate([x_l, x_r], axis=0) for x_l, x_r in zip(X_L, X_L_mirrored)], dtype=object)
        X_R = np.array([np.concatenate([x_l, x_r], axis=0) for x_l, x_r in zip(X_R, X_R_mirrored)], dtype=object)
        Y_L = np.array([np.concatenate([y_l, y_r], axis=0) for y_l, y_r in zip(Y_L, Y_L)], dtype=object)
        Y_R = np.array([np.concatenate([y_l, y_r], axis=0) for y_l, y_r in zip(Y_R, Y_R)], dtype=object)

    X = np.array([np.concatenate([x_l, x_r], axis=0) for x_l, x_r in zip(X_L, X_R)], dtype=object)
    Y = np.array([np.concatenate([y_l, y_r], axis=0) for y_l, y_r in zip(Y_L, Y_R)], dtype=object)
else:
    with open(os.path.join(DATA_DIR, "X.pkl"), "rb") as f:
        X = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(DATA_DIR, "Y.pkl"), "rb") as f:
        Y = np.array(pickle.load(f), dtype=object)

dataset = IMUDataset(
    X, Y, sequence_length=WINDOW_SAMPLES, downsample_factor=DOWNSAMPLE_FACTOR, selected_channels=[0, 1, 2]
)

# Training loop over folds
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

    # ---------------------- Load Pre-Trained ResNet Encoder ----------------------
    # Path to the pre-trained ResNet weights
    pretrained_ckpt = "mtl_best.mdl"

    # Create the encoder with desired parameters
    encoder = ResNetEncoder(
        weight_path=pretrained_ckpt,
        n_channels=INPUT_DIM,
        class_num=NUM_CLASSES,
        my_device=device,
        freeze_encoder=FREEZE_ENCODER,
    ).to(device)
    feature_dim = encoder.out_features

    # ---------------------- Build Sequence Labeling Model ----------------------
    # Create the sequence labeling head
    seq_labeler = BiLSTMHead(
        feature_dim=feature_dim, seq_length=WINDOW_SAMPLES, num_classes=NUM_CLASSES, hidden_dim=128
    ).to(device)

    # Create the full model
    model = ResNetBiLSTM(encoder, seq_labeler).to(device)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # criterion = nn.CrossEntropyLoss()

    # ---------------------- Fine-Tuning Loop ----------------------
    model.train()
    best_loss = float("inf")

    for epoch in tqdm(range(NUM_EPOCHS), desc=f"Fold {fold+1}", leave=False):
        training_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.permute(0, 2, 1).to(device)  # → [B, C, L]
            batch_y = batch_y.long().to(device)  # → [B, L]

            optimizer.zero_grad()
            outputs = model(batch_x)  # → [B, L, num_classes]

            # prepare for loss_fn
            logits_t = outputs.permute(0, 2, 1)  # → [B, num_classes, L]

            # get CE + smoothing loss
            ce_loss, smooth_loss = loss_fn(logits_t, batch_y, smoothing="MSE", max_diff=16.0)

            loss = ce_loss + LAMBDA_COEF * smooth_loss
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        if local_rank == 0:
            avg_loss = training_loss / len(train_loader)
            best_loss = save_best_model(model, fold + 1, avg_loss, best_loss, checkpoint_dir, "min")
            stats = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": avg_loss,
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
        "dataset": DATASET,
        "hand_separation": HAND_SEPERATION,
        "num_classes": NUM_CLASSES,
        "sampling_freq_original": SAMPLING_FREQ_ORIGINAL,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "sampling_freq": SAMPLING_FREQ,
        "window_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SAMPLES,
        "batch_size": BATCH_SIZE,
        "model": MODEL,
        "input_dim": INPUT_DIM,
        "learning_rate": LEARNING_RATE,
        "num_folds": NUM_FOLDS,
        "num_epochs": NUM_EPOCHS,
        "augmentation_hand_mirroring": FLAG_AUGMENT_HAND_MIRRORING,
        "dataset_mirroring": FLAG_DATASET_MIRRORING,
        "dataset_mirroring_add": FLAG_DATASET_MIRRORING_ADD,
        "validate_folds": validate_folds,
    }

    config_file = os.path.join(result_dir, "training_config.json")
    with open(config_file, "w") as f:
        json.dump(config_info, f, indent=4)

if args.distributed:
    dist.destroy_process_group()
