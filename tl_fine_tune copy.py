#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Fine-Tuning Script Using Pre-Trained ResNet Encoder with MLP
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-11
Description : This script loads a pre-trained ResNet encoder (via the harnet10
              framework and load_weights function) and attaches a classifier
              head (MLP) for fine-tuning on a downstream classification task.
              It performs the following:
              1. Loads and combines left/right IMU data
              2. Wraps the ResNet encoder for feature extraction
              3. Trains an MLP classifier on top of the encoder
              4. Fine-tunes the full network and saves model + config
===============================================================================
"""


import argparse
import json
import logging
import os
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, DistributedSampler
from tqdm import tqdm

from components.pre_processing import hand_mirroring
from components.models.resnet import ResNetEncoder
from components.models.head import MLPClassifier, ResNetMLP
from components.datasets import IMUDatasetN21, create_balanced_subject_folds, load_predefined_validate_folds


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
    DOWNSAMPLE_FACTOR = 2
    sub_version = DATASET.replace("DX", "").upper() or "I"
    DATA_DIR = f"./dataset/DX/DX-{sub_version}"
    TASK = "binary"
    HAND_SEPERATION = True
elif DATASET.startswith("FD"):
    SAMPLING_FREQ_ORIGINAL = 64
    DOWNSAMPLE_FACTOR = 2
    NUM_CLASSES = 3
    sub_version = DATASET.replace("FD", "").upper() or "I"
    DATA_DIR = f"./dataset/FD/FD-{sub_version}"
    TASK = "multiclass"
    HAND_SEPERATION = True
elif DATASET.startswith("OREBA"):
    SAMPLING_FREQ_ORIGINAL = 64
    DOWNSAMPLE_FACTOR = 2
    NUM_CLASSES = 3
    DATA_DIR = "./dataset/Oreba"
    TASK = "multiclass"
    HAND_SEPERATION = False
else:
    raise ValueError(f"Invalid dataset: {DATASET}")
SAMPLING_FREQ = SAMPLING_FREQ_ORIGINAL // DOWNSAMPLE_FACTOR
DATASET_HAND = "BOTH"  # "LEFT" or "RIGHT" or "BOTH"

# ----------------------------------------------------------------------------------------------
# Dataloader Configuration
# ----------------------------------------------------------------------------------------------
WINDOW_LENGTH = 10  # seconds
WINDOW_SIZE = SAMPLING_FREQ * WINDOW_LENGTH
BATCH_SIZE = 64
NUM_WORKERS = 16

# ----------------------------------------------------------------------------------------------
# Model Configuration
# ----------------------------------------------------------------------------------------------
MODEL = "ResNetMLP"  # Options:
INPUT_DIM = 6
LAMBDA_COEF = 0.15

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
FLAG_DATASET_MIRRORING_ADD = False  # If True, add mirrored data to the dataset

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
    version_prefix = f"{DATASET}_{DATASET_HAND}_{MODEL}_{SMOOTHING}"
    if FLAG_AUGMENT_HAND_MIRRORING:
        version_prefix += "_HM"
    if FLAG_DATASET_AUGMENTATION:
        version_prefix += "_DA"
    if FLAG_DATASET_MIRRORING:
        version_prefix += "_DM"
    if FLAG_DATASET_MIRRORING_ADD:
        version_prefix += "_DMA"

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

    if FLAG_DATASET_MIRRORING_ADD:
        X_L_mirrored = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)
        X_R_mirrored = np.array([hand_mirroring(sample) for sample in X_R], dtype=object)
        X_L = np.array([np.concatenate([x_l, x_r], axis=0) for x_l, x_r in zip(X_L, X_L_mirrored)], dtype=object)
        X_R = np.array([np.concatenate([x_l, x_r], axis=0) for x_l, x_r in zip(X_R, X_R_mirrored)], dtype=object)
        Y_L = np.array([np.concatenate([y_l, y_r], axis=0) for y_l, y_r in zip(Y_L, Y_L)], dtype=object)
        Y_R = np.array([np.concatenate([y_l, y_r], axis=0) for y_l, y_r in zip(Y_R, Y_R)], dtype=object)

    if DATASET_HAND == "LEFT":
        dataset = IMUDatasetN21(X_L, Y_L, sequence_length=WINDOW_SIZE, downsample_factor=DOWNSAMPLE_FACTOR)
    elif DATASET_HAND == "RIGHT":
        dataset = IMUDatasetN21(X_R, Y_R, sequence_length=WINDOW_SIZE, downsample_factor=DOWNSAMPLE_FACTOR)
    elif DATASET_HAND == "BOTH":
        # Combine left and right data into a unified dataset
        X = np.array([np.concatenate([x_l, x_r], axis=0) for x_l, x_r in zip(X_L, X_R)], dtype=object)
        Y = np.array([np.concatenate([y_l, y_r], axis=0) for y_l, y_r in zip(Y_L, Y_R)], dtype=object)
        dataset = IMUDatasetN21(X, Y, sequence_length=WINDOW_SIZE, downsample_factor=DOWNSAMPLE_FACTOR)
    else:
        raise ValueError(f"Invalid DATASET_HAND value: {DATASET_HAND}")
else:
    with open(os.path.join(DATA_DIR, "X.pkl"), "rb") as f:
        X = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(DATA_DIR, "Y.pkl"), "rb") as f:
        Y = np.array(pickle.load(f), dtype=object)

    dataset = IMUDatasetN21(X, Y, sequence_length=WINDOW_SIZE, downsample_factor=DOWNSAMPLE_FACTOR)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# ---------------------- Load Pre-Trained ResNet Encoder ----------------------
# Path to the pre-trained ResNet weights (adjust as necessary)
pretrained_ckpt = "mtl_best.mdl"

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

    # Create the encoder with desired parameters.
    # For example, using 3 input channels and a pre-training classification head with 2 classes.
    encoder = ResNetEncoder(
        weight_path=pretrained_ckpt, n_channels=3, class_num=2, my_device=device, freeze_encoder=False
    ).to(device)
    feature_dim = encoder.out_features

    num_classes = 3  # Modify to match your downstream task

    # ---------------------- Build Fine-Tuning Model (ResNet + MLP) ----------------------
    classifier = MLPClassifier(feature_dim=feature_dim, num_classes=num_classes)
    model = ResNetMLP(encoder, classifier).to(device)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    # ---------------------- Fine-Tuning Loop ----------------------
    num_epochs_ft = 50
    model.train()
    for epoch in range(num_epochs_ft):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in tqdm(dataloader, desc=f"Fine-tune Epoch {epoch+1}/{num_epochs_ft}"):
            # batch_x shape: [B, sequence_length, channels] --> need to permute to [B, channels, seq_len]
            batch_x = batch_x.to(device).permute(0, 2, 1)
            # Use the first label of each sequence as the sample label
            labels = batch_y[:, 0].long().to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / len(dataloader)
        acc = correct / total
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")

    # ---------------------- Save Fine-Tuned Model and Configuration ----------------------
    version_prefix = datetime.now().strftime("%Y%m%d%H%M")
    ft_save_dir = os.path.join("result", version_prefix)
    os.makedirs(ft_save_dir, exist_ok=True)
    ckpt_path = os.path.join(ft_save_dir, "fine_tuned_resnet_mlp.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Fine-tuned model saved to {ckpt_path}")

ft_config = {
    "pretrained_ckpt": pretrained_ckpt,
    "model": "ResNetMLP",
    "feature_dim": feature_dim,
    "num_classes": num_classes,
    "sequence_length": WINDOW_SIZE,
    "batch_size": batch_size,
    "num_epochs_ft": num_epochs_ft,
    "learning_rate": 5e-4,
    "freeze_encoder": False,
}
config_path = os.path.join(ft_save_dir, "ft_config.json")
with open(config_path, "w") as f:
    json.dump(ft_config, f, indent=4)
print(f"Fine-tuning configuration saved to {config_path}")


if args.distributed:
    dist.destroy_process_group()
