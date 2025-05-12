#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU SimCLR Pre-Training Script (Using TCN as Encoder)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-12
Description : This script pre-trains a SimCLR model on IMU (Inertial Measurement
              Unit) data in an unsupervised manner. It:
              1. Uses a Temporal Convolutional Network (TCN) as the encoder
              2. Wraps the encoder with global average pooling to get fixed-size features
              3. Uses an MLP projection head to map features into a contrastive space
              4. Computes NT-Xent loss between two augmented views of the same input
              5. Saves the trained model and configuration for downstream use
===============================================================================
"""


import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

from components.datasets import IMUDataset
from components.models.tcn import TCN
from components.augmentation import (
    augment_hand_mirroring,
    augment_axis_permutation,
    augment_planar_rotation,
    augment_spatial_orientation,
)


# Build TCN encoder. We wrap it with a global average pooling to get [B, D] features.
class TCN_EncoderWrapper(nn.Module):
    def __init__(self, tcn):
        super(TCN_EncoderWrapper, self).__init__()
        self.tcn = tcn
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        out = self.tcn(x)  # Expected shape: [B, num_filters, seq_len]
        out = self.pool(out)  # [B, num_filters, 1]
        out = out.squeeze(-1)  # [B, num_filters]
        return out


# Define the Projection Head for SimCLR
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim=64, hidden_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, proj_dim))

    def forward(self, x):
        return self.net(x)


# Define the SimCLR model with an encoder and a projection head
class SimCLR(nn.Module):
    def __init__(self, encoder, projector):
        super(SimCLR, self).__init__()
        self.encoder = encoder  # Should output features of shape [B, D]
        self.projector = projector  # Maps features to projection space [B, proj_dim]

    def forward(self, x):
        features = self.encoder(x)  # [B, D]
        projections = self.projector(features)  # [B, proj_dim]
        return features, projections


# Define NT-Xent Loss for SimCLR
def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = nn.functional.normalize(z, dim=1)
    sim_matrix = torch.mm(z, z.t())
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix = sim_matrix[~mask].view(2 * batch_size, -1)
    pos_sim = torch.sum(z1 * z2, dim=1)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0) / temperature
    sim_matrix = sim_matrix / temperature
    loss = -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(sim_matrix), dim=1))
    return loss.mean()


# A simple augmentation function: here adding Gaussian noise
def simclr_augment(x):
    noise = torch.randn_like(x) * 0.05
    return x + noise


def main():
    # ---------------------- Configuration ----------------------
    data_dir = "./dataset/your_dataset"  # Modify to your dataset directory
    # Assume your data is stored in X_L.pkl and X_R.pkl (Y is not needed for unsupervised training)
    X_L_PATH = os.path.join(data_dir, "X_L.pkl")
    X_R_PATH = os.path.join(data_dir, "X_R.pkl")

    with open(X_L_PATH, "rb") as f:
        X_L = np.array(pickle.load(f), dtype=object)
    with open(X_R_PATH, "rb") as f:
        X_R = np.array(pickle.load(f), dtype=object)

    X = np.concatenate([X_L, X_R], axis=0)
    sequence_length = 300  # Adjust as needed
    dataset = IMUDataset(X, Y=None, sequence_length=sequence_length)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------- Model Construction ----------------------
    tcn_encoder = TCN(
        num_layers=6,
        input_dim=6,
        num_classes=128,  # Intermediate feature dimension
        num_filters=128,
        kernel_size=3,
        dropout=0.3,
    )
    encoder = TCN_EncoderWrapper(tcn_encoder).to(device)
    # Define the projection head; assume encoder outputs 128-dim features
    projector = ProjectionHead(input_dim=128, proj_dim=64, hidden_dim=128).to(device)
    simclr_model = SimCLR(encoder, projector).to(device)

    optimizer = optim.Adam(simclr_model.parameters(), lr=1e-3)
    num_epochs = 100

    # ---------------------- Pre-Training Loop ----------------------
    simclr_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = batch.to(device).permute(0, 2, 1)  # [B, channels, seq_len]
            # Create two augmented views
            x1 = simclr_augment(x)
            x2 = simclr_augment(x)
            optimizer.zero_grad()
            _, z1 = simclr_model(x1)
            _, z2 = simclr_model(x2)
            loss = nt_xent_loss(z1, z2, temperature=0.5)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    # ---------------------- Save Pre-Trained Model ----------------------
    version_prefix = datetime.now().strftime("%Y%m%d%H%M")
    save_dir = os.path.join("result", version_prefix)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "pretrained_simclr_tcn.pth")
    torch.save(simclr_model.state_dict(), checkpoint_path)
    print(f"Pre-trained SimCLR model saved to {checkpoint_path}")

    config = {
        "model": "SimCLR_TCN",
        "num_layers": 6,
        "input_dim": 6,
        "tcn_num_filters": 128,
        "kernel_size": 3,
        "dropout": 0.3,
        "proj_hidden_dim": 128,
        "proj_output_dim": 64,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": 1e-3,
    }
    config_path = os.path.join(save_dir, "config_simclr.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")


if __name__ == "__main__":
    main()
