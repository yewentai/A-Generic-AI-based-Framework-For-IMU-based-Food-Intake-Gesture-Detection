#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Fine-Tuned Classifier Validation Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-22
Description : Recreates the exact model architecture used during fine-tuning
              to ensure proper loading of the saved state dictionary and
              performs validation on the test dataset.
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
from components.model_resnet import Resnet


# ------------------------------------------------------------------------------
# Define the ResNetEncoder class identical to the one used in fine-tuning
# ------------------------------------------------------------------------------
class ResNetEncoder(nn.Module):
    def __init__(self, n_channels=3, class_num=2, epoch_len=10):
        """
        Creates the same ResNet encoder structure used during fine-tuning.
        This must match EXACTLY what was used in pre-training.
        """
        super(ResNetEncoder, self).__init__()
        # Create the ResNet model - IMPORTANT: use class_num=2 to match pre-training
        # regardless of the downstream task's number of classes
        self.resnet = Resnet(
            output_size=2,  # This must be 2 to match pre-training model
            n_channels=n_channels,
            is_eva=True,
            resnet_version=1,
            epoch_len=epoch_len,
        )

        # Save the feature extractor
        self.feature_extractor = self.resnet.feature_extractor
        self.out_features = 1024  # For epoch_len=10

    def forward(self, x):
        feats = self.feature_extractor(x)
        feats = feats.view(x.shape[0], -1)  # flatten
        return feats


# ------------------------------------------------------------------------------
# Define the MLP classifier head - same as in fine-tuning
# ------------------------------------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.fc(x)


# ------------------------------------------------------------------------------
# Combine the ResNet encoder and the MLP classifier - same as in fine-tuning
# ------------------------------------------------------------------------------
class ResNetMLP(nn.Module):
    def __init__(self, encoder, classifier):
        super(ResNetMLP, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


def main():
    # ---------------------- Configuration ----------------------
    data_dir = "./dataset/FD/FD-III"
    test_batch_size = 64

    # Load test data
    with open(os.path.join(data_dir, "X_L.pkl"), "rb") as f:
        X_L = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(data_dir, "Y_L.pkl"), "rb") as f:
        Y_L = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(data_dir, "X_R.pkl"), "rb") as f:
        X_R = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(data_dir, "Y_R.pkl"), "rb") as f:
        Y_R = np.array(pickle.load(f), dtype=object)

    X = np.concatenate([X_L, X_R], axis=0)
    Y = np.concatenate([Y_L, Y_R], axis=0)

    # ---------------------- Load Fine-Tuned Model Config ----------------------
    ft_dir = "result/202504102302"
    ckpt_path = os.path.join(ft_dir, "fine_tuned_resnet_mlp.pth")
    ft_config_path = os.path.join(ft_dir, "ft_config.json")

    with open(ft_config_path, "r") as f:
        config = json.load(f)

    # Get sequence length from config or use default
    sequence_length = config.get("sequence_length", 300)

    # Create dataset with the correct sequence length
    dataset = IMUDatasetN21(X, Y, sequence_length=sequence_length, stride=1, selected_channels=[0, 1, 2])
    dataloader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------- Recreate the EXACT model architecture used during fine-tuning ----------------------
    # Create the encoder with the same params as during fine-tuning
    # IMPORTANT: The ResNet must be initialized with 2 classes to match pre-training
    # even though the downstream task has config["num_classes"] classes
    encoder = ResNetEncoder(
        n_channels=3,
        class_num=2,  # Must be 2 to match the pre-trained model
        epoch_len=10,  # Must match the epoch_len used in fine-tuning
    ).to(device)

    # Create classifier
    feature_dim = encoder.out_features
    classifier = MLPClassifier(feature_dim=feature_dim, num_classes=config["num_classes"]).to(device)

    # Create the combined model with the EXACT same structure as during fine-tuning
    model = ResNetMLP(encoder, classifier).to(device)

    # Load the fine-tuned weights
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Handle missing/incompatible keys
    model_dict = model.state_dict()

    # Filter out classifier weights which may have different dimensions
    pretrained_dict = {
        k: v
        for k, v in checkpoint.items()
        if (k in model_dict and (not k.startswith("encoder.resnet.classifier") or model_dict[k].shape == v.shape))
    }

    # Update the model with compatible weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    model.eval()

    print(f"Model loaded successfully from {ckpt_path}")
    print(f"Evaluating on {len(dataset)} samples with sequence length {sequence_length}...")

    # ---------------------- Evaluation ----------------------
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Validating"):
            batch_x = batch_x.permute(0, 2, 1).to(device)  # [B, C, T]
            labels = batch_y[:, 0].long().to(device)
            logits = model(batch_x)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # Save results
    results = {"accuracy": float(accuracy), "predictions": all_preds.tolist(), "true_labels": all_labels.tolist()}

    results_path = os.path.join(ft_dir, "validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Validation results saved to {results_path}")


if __name__ == "__main__":
    main()
