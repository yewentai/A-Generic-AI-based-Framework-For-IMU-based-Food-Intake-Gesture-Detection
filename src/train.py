import os
from datetime import date
from enum import Enum
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, LeaveOneGroupOut
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from augmentation import augment_orientation
from datasets import IMUDataset
from evaluation import segment_evaluation
from post_processing import post_process_predictions
from model_cnnlstm import CNN_LSTM

# Add these imports at the top
import pickle
from tqdm import tqdm


# Configuration
class Config:
    # Dataset
    dataset_path = "dataset/DX/DX-I"
    sequence_length = 128
    batch_size = 64
    num_workers = 4

    # Training
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Post-processing
    postprocess_min_length = 16  # In downsampled units
    postprocess_merge_distance = 8  # In downsampled units

    # Experiment tracking
    save_dir = "checkpoints/dx1_cnnlstm_loso"


# Load DX-I dataset
def load_dx1_data(hand="R"):
    """Load DX-I data for specified hand"""
    base_path = f"{Config.dataset_path}/X_{hand}.pkl"
    with open(base_path, "rb") as f:
        X = pickle.load(f)
    with open(f"{Config.dataset_path}/Y_{hand}.pkl", "rb") as f:
        Y = pickle.load(f)
    return X, Y


# Training function
def train_fold(train_loader, test_loader, fold_idx, num_subjects):
    # Initialize model
    model = CNN_LSTM().to(Config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0
    for epoch in range(Config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        with tqdm(
            train_loader,
            unit="batch",
            desc=f"Fold {fold_idx+1}/{num_subjects} Epoch {epoch+1}",
        ) as tepoch:
            for x, y in tepoch:
                x = x.to(Config.device)
                y = y.to(Config.device)

                # Data augmentation
                x = augment_orientation(x)

                # Forward pass
                outputs = model(x)
                loss = criterion(outputs.squeeze(), y.squeeze())

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x.size(0)
                tepoch.set_postfix(loss=loss.item())

        # Evaluation phase
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(Config.device)
                outputs = model(x)
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(preds.squeeze())
                all_labels.append(y.numpy().squeeze())

        # Post-process predictions
        full_pred = np.concatenate(all_preds)
        full_label = np.concatenate(all_labels)
        processed_pred = post_process_predictions(
            full_pred,
            min_length=Config.postprocess_min_length,
            merge_distance=Config.postprocess_merge_distance,
        )

        # Calculate metrics
        fn, fp, tp = segment_evaluation(processed_pred, full_label)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                model.state_dict(), f"{Config.save_dir}/fold_{fold_idx}_best.pth"
            )

        print(
            f"Epoch {epoch+1}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}"
        )


# Main training loop
def main():
    # Load dataset
    X, Y = load_dx1_data(hand="R")  # Use right hand data

    # Create save directory
    os.makedirs(Config.save_dir, exist_ok=True)

    # LOSO cross-validation
    num_subjects = len(X)
    fold_metrics = []

    for fold_idx in range(num_subjects):
        print(f"\n{'='*40}")
        print(f"Training fold {fold_idx+1}/{num_subjects}")
        print(f"{'='*40}")

        # Split data
        train_X = [x for i, x in enumerate(X) if i != fold_idx]
        train_Y = [y for i, y in enumerate(Y) if i != fold_idx]
        test_X = [X[fold_idx]]
        test_Y = [Y[fold_idx]]

        # Create datasets
        train_dataset = IMUDataset(
            train_X,
            train_Y,
            sequence_length=Config.sequence_length,
        )
        test_dataset = IMUDataset(
            test_X,
            test_Y,
            sequence_length=Config.sequence_length,
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
        )

        # Train and evaluate fold
        train_fold(train_loader, test_loader, fold_idx, num_subjects)


if __name__ == "__main__":
    main()
