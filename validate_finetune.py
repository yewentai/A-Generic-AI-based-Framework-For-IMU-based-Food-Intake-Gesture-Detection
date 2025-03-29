#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Fine-Tuned Classifier Validation Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Version     : 1.1
Created     : 2025-03-26
Description : This script loads the fine-tuned classifier model and evaluates it on test data.
              It computes overall accuracy, and if scikit-learn is installed, prints out a
              detailed classification report and confusion matrix.
===============================================================================
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from components.datasets import IMUDataset
from components.model_vae import VAE

# Optionally import scikit-learn for detailed metrics.
try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    classification_report, confusion_matrix = None, None


#############################################
#         Classifier Model Definition     #
#############################################
class Classifier(nn.Module):
    def __init__(self, encoder, latent_dim, num_classes):
        """
        Parameters:
            encoder: Pretrained VAE model (must have an encode() method)
            latent_dim (int): Dimension of the latent space.
            num_classes (int): Number of classes.
        """
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        mu, _ = self.encoder.encode(x)
        logits = self.fc(mu)
        return logits


def main():
    # ------------------- Configuration -------------------
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
        raise ValueError("Invalid dataset")
    WINDOW_LENGTH = 60
    WINDOW_SIZE = SAMPLING_FREQ * WINDOW_LENGTH
    BATCH_SIZE = 64
    NUM_WORKERS = 16
    FLAG_DATASET_MIRROR = False  # Set to True if mirroring is needed

    # ------------------- Load Fine-Tuned Model and Configuration -------------------
    # Modify this directory to your fine-tuned model folder (should contain fine_tuned_classifier.pth and ft_config.json)
    ft_dir = "result/202503271347"
    classifier_ckpt = os.path.join(ft_dir, "fine_tuned_classifier.pth")
    ft_config_path = os.path.join(ft_dir, "ft_config.json")
    with open(ft_config_path, "r") as f:
        config = json.load(f)
    INPUT_DIM = config["input_dim"]
    latent_dim = config["latent_dim"]
    hidden_dim = config["hidden_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------- Load Pretrained VAE Encoder -------------------
    pretrained_dir = config["pretrained_dir"]
    pretrained_checkpoint = os.path.join(pretrained_dir, "pretrained_vae.pth")
    vae_model = VAE(
        input_channels=INPUT_DIM,
        sequence_length=WINDOW_SIZE,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)
    vae_model.load_state_dict(
        torch.load(pretrained_checkpoint, map_location=device, weights_only=True),
        strict=False,
    )
    vae_model.eval()

    # ------------------- Build Classifier and Load Fine-Tuned Weights -------------------
    classifier = Classifier(
        encoder=vae_model, latent_dim=latent_dim, num_classes=num_classes
    ).to(device)
    # Adjust the state dict keys to remove the extra "encoder." prefix if needed.
    state_dict = torch.load(classifier_ckpt, map_location=device)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder.encoder."):
            new_key = key.replace("encoder.encoder.", "encoder.")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    classifier.load_state_dict(new_state_dict)
    classifier.eval()

    # ------------------- Load Test Data -------------------
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

    if FLAG_DATASET_MIRROR:
        from components.pre_processing import hand_mirroring

        X_L = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)

    X = np.concatenate([X_L, X_R], axis=0)
    Y = np.concatenate([Y_L, Y_R], axis=0)

    test_dataset = IMUDataset(
        X, Y, sequence_length=WINDOW_SIZE, downsample_factor=DOWNSAMPLE_FACTOR
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ------------------- Evaluation -------------------
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Validating"):
            batch_x = batch_x.to(device).permute(0, 2, 1)
            # Use the first label of each sequence as the sample label.
            labels = batch_y[:, 0].long().to(device)
            logits = classifier(batch_x)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test Accuracy: {accuracy:.4f}")

    if classification_report is not None:
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, digits=4))
    else:
        print("scikit-learn not installed, skipping classification report.")

    if confusion_matrix is not None:
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
    else:
        print("scikit-learn not installed, skipping confusion matrix.")


if __name__ == "__main__":
    main()
