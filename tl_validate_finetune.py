#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Fine-Tuned Classifier Validation Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-10
Description : This script loads the fine-tuned classifier model and evaluates it on
              the test dataset. It computes overall accuracy and, if scikit-learn is
              installed, prints a detailed classification report and confusion matrix.
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
from sklearn.metrics import classification_report, confusion_matrix

from components.datasets import IMUDatasetN21


# Define the same classifier structure as in fine-tuning
class Classifier(nn.Module):
    def __init__(self, encoder, feature_dim, num_classes):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):
        features = self.encoder(x)
        logits = self.fc(features)
        return logits


def main():
    # ---------------------- Configuration ----------------------
    data_dir = "./dataset/FD/FD-III"
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

    sequence_length = 300
    # Instantiate the dataset selecting only three channels (e.g., channels 0,1,2)
    dataset = IMUDatasetN21(X, Y, sequence_length=sequence_length, selected_channels=[0, 1, 2])
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------- Load Fine-Tuned Model ----------------------
    ft_dir = "result/202504102302"  # Modify to your fine-tuned model directory
    classifier_ckpt = os.path.join(ft_dir, "fine_tuned_resnet_mlp.pth")
    ft_config_path = os.path.join(ft_dir, "ft_config.json")
    with open(ft_config_path, "r") as f:
        config = json.load(f)
    feature_dim = config["feature_dim"]
    num_classes = config["num_classes"]

    # Reconstruct the encoder (using the same TCN wrapper as in fine-tuning)
    # Updated input_dim to 3 to match the selected channels
    from components.model_tcn import TCN

    class TCN_EncoderWrapper(nn.Module):
        def __init__(self, tcn):
            super(TCN_EncoderWrapper, self).__init__()
            self.tcn = tcn
            self.pool = nn.AdaptiveAvgPool1d(1)

        def forward(self, x):
            # x shape is [B, channels, seq_len]
            out = self.tcn(x)
            out = self.pool(out)  # Shape: [B, num_filters, 1]
            out = out.squeeze(-1)  # [B, num_filters]
            return out

    # Create the TCN encoder with input_dim=3 (instead of 6)
    tcn_encoder = TCN(
        num_layers=6,
        input_dim=3,
        num_classes=128,
        num_filters=128,
        kernel_size=3,
        dropout=0.3,
    )
    encoder = TCN_EncoderWrapper(tcn_encoder).to(device)

    # Load encoder weights from the fine-tuned model checkpoint
    state_dict = torch.load(classifier_ckpt, map_location=device)
    encoder_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()

    # Build the complete classifier and load the full fine-tuned weights
    model = Classifier(encoder, feature_dim, num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # ---------------------- Evaluation ----------------------
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Validating"):
            # Permute input: from [B, seq_len, channels] to [B, channels, seq_len]
            batch_x = batch_x.to(device).permute(0, 2, 1)
            # Use the first label in each sequence as the sample label
            labels = batch_y[:, 0].long().to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    if classification_report is not None:
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, digits=4))
    if confusion_matrix is not None:
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
