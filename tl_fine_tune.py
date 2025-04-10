#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Fine-Tuning Script Using Pre-Trained Encoder
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-10
Description : This script loads a pre-trained encoder (obtained via SimCLR or VAE pre-training)
              and attaches a classifier head to fine-tune on a downstream classification task.
              You can choose whether to freeze the encoder weights. The fine-tuned model and its
              configuration are saved in a timestamped directory.
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
from components.model_tcn import TCN


# Define a wrapper to get encoder output from TCN via global average pooling
class TCN_EncoderWrapper(nn.Module):
    def __init__(self, tcn):
        super(TCN_EncoderWrapper, self).__init__()
        self.tcn = tcn
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        out = self.tcn(x)  # [B, num_filters, seq_len]
        out = self.pool(out)  # [B, num_filters, 1]
        out = out.squeeze(-1)  # [B, num_filters]
        return out


# Define the classifier that attaches to the pre-trained encoder
class Classifier(nn.Module):
    def __init__(self, encoder, feature_dim, num_classes, freeze_encoder=True):
        """
        Parameters:
            encoder: Pre-trained encoder model.
            feature_dim: Dimensionality of encoder output.
            num_classes: Number of target classes.
            freeze_encoder: If True, encoder weights are frozen.
        """
        super(Classifier, self).__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):
        features = self.encoder(x)  # [B, feature_dim]
        logits = self.fc(features)
        return logits


def main():
    # ---------------------- Configuration ----------------------
    data_dir = "./dataset/your_dataset"  # Modify to your dataset directory
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
    dataset = IMUDataset(X, Y, sequence_length=sequence_length)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------- Load Pre-Trained Encoder ----------------------
    # In this example, we use the SimCLR pre-trained model based on TCN.
    pretrained_dir = "result/202503271341"  # Modify to your pre-trained model directory
    simclr_ckpt = os.path.join(pretrained_dir, "pretrained_simclr_tcn.pth")

    # Construct the TCN encoder as in pre-training
    tcn_encoder = TCN(
        num_layers=6,
        input_dim=6,
        num_classes=128,  # Must match the pre-training configuration
        num_filters=128,
        kernel_size=3,
        dropout=0.3,
    )
    encoder = TCN_EncoderWrapper(tcn_encoder).to(device)
    state_dict = torch.load(simclr_ckpt, map_location=device)
    # Assume the saved state dict has keys prefixed with "encoder."; adjust if necessary
    encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")})
    encoder.eval()
    feature_dim = 128  # The output dimension of the encoder

    num_classes = 3  # Modify to match your downstream task

    # ---------------------- Build Classifier Model ----------------------
    model = Classifier(encoder, feature_dim, num_classes, freeze_encoder=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    num_epochs_ft = 50
    model.train()
    for epoch in range(num_epochs_ft):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in tqdm(dataloader, desc=f"Fine-tune Epoch {epoch+1}/{num_epochs_ft}"):
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

    # ---------------------- Save Fine-Tuned Model ----------------------
    version_prefix = datetime.now().strftime("%Y%m%d%H%M")
    ft_save_dir = os.path.join("result", version_prefix, "finetune")
    os.makedirs(ft_save_dir, exist_ok=True)
    ckpt_path = os.path.join(ft_save_dir, "fine_tuned_classifier.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Fine-tuned classifier saved to {ckpt_path}")

    ft_config = {
        "pretrained_dir": pretrained_dir,
        "simclr_checkpoint": simclr_ckpt,
        "model": "Classifier_Finetune_SimCLR_TCN",
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
