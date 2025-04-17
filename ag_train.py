#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================================
Event-based Segmentation GAN Training Script
------------------------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-17
Description : This script trains a GAN model for augmenting event-based IMU segmentation data.
              It is structured similarly to the MSTCN training script, with inline code for
              dataset loading, GAN model definition, training, and checkpointing.
================================================================================================
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import argparse
import logging
from tqdm import tqdm

# ==============================================================================================
#                             Configuration Parameters
# ==============================================================================================

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

DATASET = "FDI"
DATA_DIR = f"./dataset/{DATASET}"
BATCH_SIZE = 32
SEQ_LEN = 60
INPUT_DIM = 6
NUM_CLASSES = 3
NOISE_DIM = 100
EPOCHS = 100
LEARNING_RATE = 1e-4
LAMBDA_GP = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================================
#                             Dataset Loader
# ==============================================================================================


class IMUGANDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.long)


with open(os.path.join(DATA_DIR, "X_L.pkl"), "rb") as f:
    X_L = np.array(pickle.load(f), dtype=object)
with open(os.path.join(DATA_DIR, "Y_L.pkl"), "rb") as f:
    Y_L = np.array(pickle.load(f), dtype=object)

X = np.concatenate(X_L, axis=0)
Y = np.concatenate(Y_L, axis=0)
dataset = IMUGANDataset(X, Y)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

# ==============================================================================================
#                             GAN Model Definitions
# ==============================================================================================


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NOISE_DIM, 256), nn.ReLU(), nn.Linear(256, SEQ_LEN * INPUT_DIM), nn.Tanh())

    def forward(self, z):
        return self.net(z).view(-1, SEQ_LEN, INPUT_DIM)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(SEQ_LEN * INPUT_DIM, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================================
#                             Gradient Penalty (WGAN-GP)
# ==============================================================================================


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1).to(DEVICE)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size()).to(DEVICE)
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


# ==============================================================================================
#                             Training Loop
# ==============================================================================================


def train():
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))

    result_dir = os.path.join("result", datetime.now().strftime("%Y%m%d%H%M")[:12])
    os.makedirs(result_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        for i, (real_data, _) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}")):
            real_data = real_data.to(DEVICE)
            batch_size = real_data.size(0)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            z = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
            fake_data = G(z).detach()
            d_real = D(real_data).mean()
            d_fake = D(fake_data).mean()
            gp = compute_gradient_penalty(D, real_data, fake_data)
            d_loss = -d_real + d_fake + LAMBDA_GP * gp

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            if i % 5 == 0:
                z = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
                gen_data = G(z)
                g_loss = -D(gen_data).mean()
                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        # Save model checkpoints
        torch.save(G.state_dict(), os.path.join(result_dir, f"generator_epoch_{epoch+1}.pt"))
        torch.save(D.state_dict(), os.path.join(result_dir, f"discriminator_epoch_{epoch+1}.pt"))

    logger.info("Training completed.")


if __name__ == "__main__":
    train()
