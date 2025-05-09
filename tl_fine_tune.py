#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Fine-Tuning Script Using Pre-Trained ResNet Encoder with MLP
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-09
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

from components.models.resnet import ResNetEncoder
from components.datasets import IMUDatasetN21


# ------------------------------------------------------------------------------
# Define the MLP classifier head
# ------------------------------------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        """
        A simple MLP head with one hidden layer.

        Parameters:
            feature_dim (int): Dimensionality of the input feature vector.
            num_classes (int): Number of target classes.
        """
        super(MLPClassifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):
        logits = self.fc(x)
        return logits


# ------------------------------------------------------------------------------
# Combine the ResNet encoder and the MLP classifier into one fine-tuning model
# ------------------------------------------------------------------------------
class ResNetMLP(nn.Module):
    def __init__(self, encoder, classifier):
        """
        Parameters:
            encoder (nn.Module): Pre-trained ResNet encoder.
            classifier (nn.Module): MLP classifier head.
        """
        super(ResNetMLP, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


# ------------------------------------------------------------------------------
# Main fine-tuning pipeline
# ------------------------------------------------------------------------------
def main():
    # ---------------------- Configuration ----------------------
    # Dataset paths (modify these to suit your environment)
    data_dir = "./dataset/DX/DX-I"
    X_L_PATH = os.path.join(data_dir, "X_L.pkl")
    Y_L_PATH = os.path.join(data_dir, "Y_L.pkl")
    X_R_PATH = os.path.join(data_dir, "X_R.pkl")
    Y_R_PATH = os.path.join(data_dir, "Y_R.pkl")

    with open(X_L_PATH, "rb") as f:
        X_L = np.array(pickle.load(f), dtype=object)
    with open(X_R_PATH, "rb") as f:
        X_R = np.array(pickle.load(f), dtype=object)
    with open(Y_L_PATH, "rb") as f:
        Y_L = np.array(pickle.load(f), dtype=object)
    with open(Y_R_PATH, "rb") as f:
        Y_R = np.array(pickle.load(f), dtype=object)

    X = np.concatenate([X_L, X_R], axis=0)
    Y = np.concatenate([Y_L, Y_R], axis=0)

    sequence_length = 300  # Adjust as needed
    dataset = IMUDatasetN21(X, Y, sequence_length=sequence_length)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------- Load Pre-Trained ResNet Encoder ----------------------
    # Path to the pre-trained ResNet weights (adjust as necessary)
    pretrained_ckpt = "mtl_best.mdl"
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
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "num_epochs_ft": num_epochs_ft,
        "learning_rate": 5e-4,
        "freeze_encoder": False,
    }
    config_path = os.path.join(ft_save_dir, "ft_config.json")
    with open(config_path, "w") as f:
        json.dump(ft_config, f, indent=4)
    print(f"Fine-tuning configuration saved to {config_path}")


if __name__ == "__main__":
    main()
