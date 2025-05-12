#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================================
Event-based Segmentation GAN Training Script
------------------------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-12
Description : This script trains a GAN model for event-based IMU segmentation. It loads and
              preprocesses IMU segment data, defines and trains a 1D-CNN generator and a
              2D-CNN discriminator using WGAN-GP with label smoothing and L1 regularization.
              The script supports single and distributed GPU training, saves configuration
              and training stats, and checkpoints the best-performing generator.
================================================================================================
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import logging
import argparse
from scipy import signal
from torch.utils.data import Dataset

from components.checkpoint import save_generator, save_discriminator
from components.models.gan import Generator, Discriminator


class IMUDatasetSegment(Dataset):
    def __init__(self, X, Y, downsample_factor=1, apply_antialias=True, min_length=5):
        """
        Dataset that splits IMU data into variable-length segments where the class label is constant.

        Parameters:
            X (list of np.ndarray): IMU data for each subject, each of shape (N, 6)
            Y (list of np.ndarray): Label arrays for each subject, each of shape (N,)
            downsample_factor (int): Downsampling factor (default 1 = no downsampling)
            apply_antialias (bool): Whether to apply anti-aliasing filter before downsampling.
            min_length (int): Minimum segment length to keep.
        """
        self.data = []
        self.labels = []
        self.subject_indices = []

        for subject_idx, (imu_data, label_seq) in enumerate(zip(X, Y)):
            if downsample_factor > 1:
                imu_data = self.downsample(imu_data, downsample_factor, apply_antialias)
                label_seq = label_seq[::downsample_factor]

            imu_data = self.normalize(imu_data)
            segments = self.segment_by_class(imu_data, label_seq, min_length)

            for imu_segment, label in segments:
                self.data.append(imu_segment)
                self.labels.append(label)
                self.subject_indices.append(subject_idx)

    def segment_by_class(self, imu_data, label_seq, min_length):
        """
        Cut sequences into segments where label is constant.

        Returns:
            List of tuples: [(imu_segment, label), ...]
        """
        segments = []
        current_label = label_seq[0]
        start = 0

        for i in range(1, len(label_seq)):
            if label_seq[i] != current_label:
                if i - start >= min_length:
                    segments.append((imu_data[start:i], current_label))
                start = i
                current_label = label_seq[i]

        # Add last segment
        if len(label_seq) - start >= min_length:
            segments.append((imu_data[start:], current_label))

        return segments

    def downsample(self, data, factor, apply_antialias=True):
        if apply_antialias:
            nyquist = 0.5 * data.shape[0]
            cutoff = (0.5 / factor) * nyquist
            b, a = signal.butter(4, cutoff / nyquist, btype="low")
            data = signal.filtfilt(b, a, data, axis=0)
        return data[::factor, :]

    def normalize(self, data):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        return (data - mean) / (std + 1e-5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imu = torch.tensor(self.data[idx], dtype=torch.float32)  # Shape: (T, 6)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Single class label
        return imu, label


# ==============================================================================================
#                             Configuration Parameters
# ==============================================================================================

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
WINDOW_SECONDS = 60
WINDOW_SAMPLES = SAMPLING_FREQ * WINDOW_SECONDS
BATCH_SIZE = 64
NUM_WORKERS = 16
INPUT_DIM = 6
LATENT_DIM = 100
LAMBDA_COEF = 0.15

# Training Settings
LEARNING_RATE = 5e-4
NUM_EPOCHS = 100

# ==============================================================================================
#                            Custom Collate Function
# ==============================================================================================


def fixed_length_collate(batch, target_len=240):
    """
    Collate function that ensures all sequences have the same predefined length
    by either truncating or padding.

    Args:
        batch: List of (data, label) tuples
        target_len: Target sequence length

    Returns:
        Batched tensors with fixed length
    """
    data = [item[0] for item in batch]  # Get all IMU sequences
    labels = [item[1] for item in batch]  # Get all labels

    # Truncate or pad each sequence to target_len
    processed_data = []
    for d in data:
        curr_len = d.size(0)
        if curr_len > target_len:
            # Truncate to target_len
            processed_d = d[:target_len]
        elif curr_len < target_len:
            # Pad to target_len
            padding = torch.zeros((target_len - curr_len, d.size(1)), dtype=d.dtype)
            processed_d = torch.cat([d, padding], dim=0)
        else:
            processed_d = d
        processed_data.append(processed_d)

    # Stack into batch tensors
    data_batch = torch.stack(processed_data)
    labels_batch = torch.stack(labels)

    return data_batch, labels_batch


# ==============================================================================================
#                            Main Training Code (Inline)
# ==============================================================================================

parser = argparse.ArgumentParser(description="Event-based GAN IMU Training Script (Combined Mode)")
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
    logger.info(f"Training started at: {overall_start}")
    version_prefix = datetime.now().strftime("%Y%m%d%H%M")[:12]
    result_dir = os.path.join("results", version_prefix)
    os.makedirs(result_dir, exist_ok=True)
    checkpoint_dir = os.path.join(result_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    training_stats_file = os.path.join(result_dir, "train_stats.npy")
    config_path = os.path.join(result_dir, "config.json")
else:
    checkpoint_dir = None
    training_stats_file = None
    config_path = None

# Save config to JSON
if local_rank == 0:
    config_dict = {
        "DATASET": DATASET,
        "SAMPLING_FREQ": SAMPLING_FREQ,
        "WINDOW_SECONDS": WINDOW_SECONDS,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "LATENT_DIM": LATENT_DIM,
        "INPUT_DIM": INPUT_DIM,
        "LAMBDA_COEF": LAMBDA_COEF,
        "DOWNSAMPLE_FACTOR": DOWNSAMPLE_FACTOR,
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)

# Load data
with open(os.path.join(DATA_DIR, "X_L.pkl"), "rb") as f:
    X_L = np.array(pickle.load(f), dtype=object)
with open(os.path.join(DATA_DIR, "Y_L.pkl"), "rb") as f:
    Y_L = np.array(pickle.load(f), dtype=object)
with open(os.path.join(DATA_DIR, "X_R.pkl"), "rb") as f:
    X_R = np.array(pickle.load(f), dtype=object)
with open(os.path.join(DATA_DIR, "Y_R.pkl"), "rb") as f:
    Y_R = np.array(pickle.load(f), dtype=object)

X = np.concatenate([X_L, X_R], axis=0)
Y = np.concatenate([Y_L, Y_R], axis=0)

dataset = IMUDatasetSegment(X, Y, downsample_factor=DOWNSAMPLE_FACTOR)

# Use the custom collate function to handle variable-length segments
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=fixed_length_collate,  # Add this line to use the custom collate function
)

# Define models
generator = Generator(latent_dim=LATENT_DIM, output_channels=INPUT_DIM).to(device)
discriminator = Discriminator(input_channels=INPUT_DIM).to(device)  # Make sure to match INPUT_DIM

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Label smoothing parameters
real_label = 0.9
fake_label = 0.1  # Changed from 0.0 to 0.1 for better label smoothing
lambda_gp = 10.0  # Gradient penalty coefficient


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN-GP"""
    batch_size = real_samples.size(0)
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(batch_size, 1, 1, device=real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    # Create a tensor of ones with the same shape as d_interpolates
    fake = torch.ones_like(d_interpolates, device=real_samples.device, requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


training_stats = []

for epoch in range(NUM_EPOCHS):
    generator.train()
    discriminator.train()
    g_loss_total = 0.0
    d_loss_total = 0.0

    for i, (real_data, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
        batch_size = real_data.size(0)

        # Reshape data for 1D convolution: (batch, time, channels) -> (batch, channels, time)
        real_data = real_data.permute(0, 2, 1).to(device)  # Shape: [B, 6, T]
        real_data += 0.05 * torch.randn_like(real_data)

        # Train Discriminator
        discriminator.zero_grad()

        # Real data loss
        label_real = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        output_real = discriminator(real_data)
        # Ensure output is 1D
        if len(output_real.shape) > 1:
            output_real = output_real.mean(dim=1)  # Shape: [B]
        loss_real = criterion(output_real, label_real)

        # Fake data loss
        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_data = generator(noise)
        label_fake = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
        output_fake = discriminator(fake_data.detach())
        # Ensure output is 1D
        if len(output_fake.shape) > 1:
            output_fake = output_fake.mean(dim=1)  # Shape: [B]
        loss_fake = criterion(output_fake, label_fake)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_data.data)

        # Total discriminator loss
        loss_D = loss_real + loss_fake + lambda_gp * gradient_penalty
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        generator.zero_grad()
        label_gen = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        output_gen = discriminator(fake_data)
        # Ensure output is 1D
        if len(output_gen.shape) > 1:
            output_gen = output_gen.mean(dim=1)  # Shape: [B]

        # Generator loss with additional regularization
        loss_G = criterion(output_gen, label_gen)

        # Add L1 regularization to encourage smoothness in generated sequences
        l1_reg = torch.mean(torch.abs(fake_data[:, :, 1:] - fake_data[:, :, :-1]))
        loss_G += 0.1 * l1_reg  # Small weight for L1 regularization

        loss_G.backward()
        optimizer_G.step()

        g_loss_total += loss_G.item()
        d_loss_total += loss_D.item()

    avg_d_loss = d_loss_total / len(dataloader)
    avg_g_loss = g_loss_total / len(dataloader)
    training_stats.append({"epoch": epoch + 1, "d_loss": avg_d_loss, "g_loss": avg_g_loss})

    if local_rank == 0:
        logger.info(f"[Epoch {epoch+1}/{NUM_EPOCHS}] " f"D Loss: {avg_d_loss:.4f} | " f"G Loss: {avg_g_loss:.4f}")
        # Save the best generator based on G Loss
        if epoch == 0 or avg_g_loss < min(training_stats, key=lambda x: x["g_loss"])["g_loss"]:
            save_generator(generator, optimizer_G, epoch, checkpoint_dir)
            logger.info(f"Best generator saved at epoch {epoch+1}")


# Save training statistics
if local_rank == 0:
    np.save(training_stats_file, training_stats)

    config_info = {
        "dataset": DATASET,
        "num_classes": NUM_CLASSES,
        "sampling_freq": SAMPLING_FREQ,
        "window_samples": WINDOW_SAMPLES,
        "input_dim": INPUT_DIM,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "latent_dim": LATENT_DIM,
        "lambda_coef": LAMBDA_COEF,
    }

    config_file = os.path.join(result_dir, "config.json")
    with open(config_file, "w") as f:
        json.dump(config_info, f, indent=4)
    logger.info(f"Configuration saved to {config_file}")

if args.distributed:
    dist.destroy_process_group()
