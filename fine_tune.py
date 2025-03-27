#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Fine-Tuning Script Using Pretrained VAE Encoder
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Version     : 1.0
Created     : 2025-03-26
Description : This script loads the pretrained VAE model weights, extracts its encoder,
              and builds a downstream classifier for the IMU data fine-tuning task.
              You can choose to freeze the encoder parameters (only training the classifier)
              or fine-tune the entire model jointly.
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

# Import dataset and preprocessing functions (assumed to be available in your project)
from components.datasets import IMUDataset
from components.pre_processing import hand_mirroring
from components.model_vae import VAE


#############################################
#         Classifier Model Definition     #
#############################################
class Classifier(nn.Module):
    def __init__(self, encoder, latent_dim, num_classes, freeze_encoder=True):
        """
        Parameters:
            encoder: Pretrained VAE model (must have an encode() method)
            latent_dim (int): Dimension of the latent space.
            num_classes (int): Number of classes.
            freeze_encoder (bool): Whether to freeze the encoder parameters.
        """
        super(Classifier, self).__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, channels, sequence_length)
        # Use the encode() method to get mu and logvar
        mu, _ = self.encoder.encode(x)
        logits = self.fc(mu)
        return logits


#############################################
#         Fine-Tuning Training Function   #
#############################################
def train_classifier(model, dataloader, device, num_epochs=50, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_model_state = None
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in tqdm(
            dataloader, desc=f"Fine-Tuning Epoch {epoch+1}/{num_epochs}"
        ):
            # Reshape input to (batch, channels, sequence_length)
            batch_x = batch_x.to(device).permute(0, 2, 1)
            # Use the first label of each sequence as the sample label.
            batch_y = batch_y[:, 0].long().to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total
        print(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
    return best_model_state


def main():
    # -------------------- Configuration --------------------
    DATASET = "FDI"  # Options: DXI/DXII or FDI/FDII/FDIII
    SAMPLING_FREQ_ORIGINAL = 64
    DOWNSAMPLE_FACTOR = 4
    SAMPLING_FREQ = SAMPLING_FREQ_ORIGINAL // DOWNSAMPLE_FACTOR
    if DATASET.startswith("FD"):
        sub_version = DATASET.replace("FD", "").upper() or "I"
        DATA_DIR = f"./dataset/FD/FD-{sub_version}"
        num_classes = 3
    elif DATASET.startswith("DX"):
        sub_version = DATASET.replace("DX", "").upper() or "I"
        DATA_DIR = f"./dataset/DX/DX-{sub_version}"
        num_classes = 2
    else:
        raise ValueError(f"Invalid dataset: {DATASET}")

    WINDOW_LENGTH = 60
    WINDOW_SIZE = SAMPLING_FREQ * WINDOW_LENGTH
    BATCH_SIZE = 64
    NUM_WORKERS = 16
    FLAG_MIRROR = False  # Set to True if mirroring is needed

    # Fine-tuning parameters
    num_epochs_ft = 50
    learning_rate_ft = 5e-4
    freeze_encoder = True  # Set to False if joint fine-tuning is desired

    # -------------------- Load Pretrained Model and Configuration --------------------
    # Modify the directory below to your pretrained model directory (timestamped folder)
    pretrained_dir = "result/202503271341"  # Change as needed
    pretrained_checkpoint = os.path.join(pretrained_dir, "pretrained_vae.pth")
    config_path = os.path.join(pretrained_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    INPUT_DIM = config["input_dim"]
    latent_dim = config["latent_dim"]
    hidden_dim = config["hidden_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build the VAE model and load pretrained weights.
    vae_model = VAE(
        input_channels=INPUT_DIM,
        sequence_length=WINDOW_SIZE,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)
    vae_model.load_state_dict(
        torch.load(pretrained_checkpoint, map_location=device, weights_only=True)
    )
    vae_model.eval()

    # -------------------- Data Loading --------------------
    X_L_PATH = os.path.join(DATA_DIR, "X_L.pkl")
    Y_L_PATH = os.path.join(DATA_DIR, "Y_L.pkl")
    X_R_PATH = os.path.join(DATA_DIR, "X_R.pkl")
    Y_R_PATH = os.path.join(DATA_DIR, "Y_R.pkl")
    with open(X_L_PATH, "rb") as f:
        X_L = np.array(pickle.load(f), dtype=object)
    with open(Y_L_PATH, "rb") as f:
        Y_L = np.array(pickle.load(f), dtype=object)
    with open(X_R_PATH, "rb") as f:
        X_R = np.array(pickle.load(f), dtype=object)
    with open(Y_R_PATH, "rb") as f:
        Y_R = np.array(pickle.load(f), dtype=object)

    if FLAG_MIRROR:
        X_L = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)

    # Merge left and right data
    X = np.concatenate([X_L, X_R], axis=0)
    Y = np.concatenate([Y_L, Y_R], axis=0)

    full_dataset = IMUDataset(
        X, Y, sequence_length=WINDOW_SIZE, downsample_factor=DOWNSAMPLE_FACTOR
    )
    dataloader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------- Build Classifier Model --------------------
    # Note: pass the entire VAE model (not just its encoder) so that we can call encode() in the classifier.
    classifier = Classifier(
        encoder=vae_model,
        latent_dim=latent_dim,
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
    ).to(device)

    # -------------------- Fine-Tuning Training --------------------
    print("Starting fine-tuning ...")
    best_state = train_classifier(
        classifier, dataloader, device, num_epochs=num_epochs_ft, lr=learning_rate_ft
    )
    classifier.load_state_dict(best_state)

    # -------------------- Save Fine-Tuned Model and Configuration --------------------
    ft_dir = os.path.join("result", datetime.now().strftime("%Y%m%d%H%M"))
    os.makedirs(ft_dir, exist_ok=True)
    classifier_ckpt = os.path.join(ft_dir, "fine_tuned_classifier.pth")
    torch.save(classifier.state_dict(), classifier_ckpt)
    print(f"Fine-tuned classifier saved to {classifier_ckpt}")

    ft_config = {
        "dataset": DATASET,
        "input_dim": INPUT_DIM,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "num_classes": num_classes,
        "sampling_freq": SAMPLING_FREQ,
        "window_size": WINDOW_SIZE,
        "batch_size": BATCH_SIZE,
        "num_epochs_ft": num_epochs_ft,
        "learning_rate_ft": learning_rate_ft,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "mirroring": FLAG_MIRROR,
        "freeze_encoder": freeze_encoder,
        "pretrained_dir": pretrained_dir,
    }
    ft_config_path = os.path.join(ft_dir, "ft_config.json")
    with open(ft_config_path, "w") as f:
        json.dump(ft_config, f, indent=4)
    print(f"Configuration saved to {ft_config_path}")


if __name__ == "__main__":
    main()
