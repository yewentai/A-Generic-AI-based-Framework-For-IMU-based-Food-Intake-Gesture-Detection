# TODO: Cut the sqeunce before feeding to the model

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import pickle
import os
from datetime import datetime
from tqdm import tqdm
from model_tcn_mha import TCNMHA, TCNMHA_Loss
from augmentation import augment_orientation
from utils import IMUDataset, segment_f1_multiclass, post_process_predictions

# Hyperparameters
num_layers = 9
num_classes = 3
num_heads = 8
input_dim = 6
num_filters = 128
kernel_size = 3
dropout = 0.3
lambda_coef = 0.15
tau = 4
learning_rate = 0.0005
debug_plot = False

# Load data
X_L_path = "./dataset/FD/FD-I/X_L.pkl"
Y_L_path = "./dataset/FD/FD-I/Y_L.pkl"
X_R_path = "./dataset/FD/FD-I/X_R.pkl"
Y_R_path = "./dataset/FD/FD-I/Y_R.pkl"

with open(X_L_path, "rb") as f:
    X_L = np.array(pickle.load(f), dtype=object)  # Convert to NumPy array
with open(Y_L_path, "rb") as f:
    Y_L = np.array(pickle.load(f), dtype=object)
with open(X_R_path, "rb") as f:
    X_R = np.array(pickle.load(f), dtype=object)
with open(Y_R_path, "rb") as f:
    Y_R = np.array(pickle.load(f), dtype=object)

# Concatenate the left and right data
X = np.concatenate([X_L, X_R], axis=0)
Y = np.concatenate([Y_L, Y_R], axis=0)

# Create the full dataset
full_dataset = IMUDataset(X, Y)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for saving result
if not os.path.exists("result"):
    os.makedirs("result")

# File names for training and testing result
training_stats_file = "result/training_stats_tcnmha.npy"
testing_stats_file = "result/testing_stats_tcnmha.npy"

# Initialize empty lists to store result
training_statistics = []
testing_statistics = []

# LOSO cross-validation
unique_subjects = np.unique(full_dataset.subject_indices)
loso_f1_scores = []

for fold, test_subject in enumerate(
    tqdm(unique_subjects, desc="LOSO Folds", leave=True)
):
    # Create train and test indices
    train_indices = [
        i
        for i, subject in enumerate(full_dataset.subject_indices)
        if subject != test_subject
    ]
    test_indices = [
        i
        for i, subject in enumerate(full_dataset.subject_indices)
        if subject == test_subject
    ]

    # Create train and test dataset
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    # Initialize the model
    model = TCNMHA(
        num_layers=num_layers,
        num_classes=num_classes,
        num_heads=num_heads,
        input_dim=input_dim,
        num_filters=num_filters,
        kernel_size=kernel_size,
        dropout=dropout,
        lambda_coef=lambda_coef,
        tau=tau,
    ).to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    num_epochs = 10
    for epoch in tqdm(range(num_epochs), desc=f"Training Fold {fold + 1}", leave=False):
        model.train()
        training_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            # Data augmentation
            batch_x = augment_orientation(batch_x)

            batch_x, batch_y = batch_x.permute(0, 2, 1).to(device), batch_y.to(device)
            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = TCNMHA_Loss(outputs, batch_y, lambda_coef)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        avg_train_loss = training_loss / len(train_loader)
        # Save training result for each epoch
        training_statistics.append(
            {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
            }
        )

    # Testing
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x, batch_y = batch_x.permute(0, 2, 1).to(device), batch_y.to(device)
            outputs = model(batch_x)
            final_output = outputs[-1]
            probabilities = F.softmax(final_output, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            all_predictions.extend(predicted_classes.view(-1).cpu().numpy())
            all_labels.extend(batch_y.view(-1).cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    # Post-processing predictions
    all_predictions = post_process_predictions(all_predictions)

    # Calculate metrics
    fp, fn, tp = 0, 0, 0
    for i in range(len(all_predictions)):
        fp += np.sum((all_predictions[i] == 1) & (all_labels[i] == 0))
        fn += np.sum((all_predictions[i] == 0) & (all_labels[i] == 1))
        tp += np.sum((all_predictions[i] == 1) & (all_labels[i] == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_sample = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    f1_segment_1 = segment_f1_multiclass(all_predictions, all_labels, 0.1, debug_plot)
    f1_segment_2 = segment_f1_multiclass(all_predictions, all_labels, 0.25, debug_plot)
    f1_segment_3 = segment_f1_multiclass(all_predictions, all_labels, 0.5, debug_plot)

    # Save testing result for each fold
    testing_statistics.append(
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "fold": fold + 1,
            "f1_sample": f1_sample,
            "f1_segment_1": f1_segment_1,
            "f1_segment_2": f1_segment_2,
            "f1_segment_3": f1_segment_3,
        }
    )

    loso_f1_scores.append([f1_sample, f1_segment_1, f1_segment_2, f1_segment_3])

# Save result to .npy files
np.save(training_stats_file, training_statistics)
np.save(testing_stats_file, testing_statistics)
