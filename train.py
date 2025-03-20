# -*- coding: utf-8 -*-

from calendar import c
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import pickle
import os
from datetime import datetime
from tqdm import tqdm
from torchmetrics.functional import f1_score as F1S
from torchmetrics import CohenKappa
from torchmetrics import MatthewsCorrCoef

from components.augmentation import augment_orientation
from components.datasets import (
    IMUDataset,
    create_balanced_subject_folds,
    load_predefined_validate_folds,
)
from components.evaluation import segment_evaluation
from components.pre_processing import hand_mirroring
from components.post_processing import post_process_predictions
from components.checkpoint import save_checkpoint
from components.model_mstcn import MSTCN, MSTCN_Loss

# ********************** Configuration Parameters **********************
DATASET = "FDI"  # Options: DXI/DXII or FDI/FDII/FDIII
NUM_STAGES = 2
NUM_LAYERS = 9
NUM_HEADS = 8
INPUT_DIM = 6
NUM_FILTERS = 128
KERNEL_SIZE = 3
DROPOUT = 0.3
LAMBDA_COEF = 0.15
TAU = 4
LEARNING_RATE = 5e-4
SAMPLING_FREQ_ORIGINAL = 64
DOWNSAMPLE_FACTOR = 4
SAMPLING_FREQ = SAMPLING_FREQ_ORIGINAL // DOWNSAMPLE_FACTOR
WINDOW_LENGTH = 60
WINDOW_SIZE = SAMPLING_FREQ * WINDOW_LENGTH
DEBUG_PLOT = False
NUM_FOLDS = 7
NUM_EPOCHS = 20
BATCH_SIZE = 64
NUM_WORKERS = 16
FLAG_AUGMENT = False
FLAG_MIRROR = True

# Configure parameters based on dataset type
if DATASET.startswith("DX"):
    NUM_CLASSES = 2
    dataset_type = "DX"
    sub_version = (
        DATASET.replace("DX", "").upper() or "I"
    )  # Handle formats like DX/DXII
    DATA_DIR = f"./dataset/DX/DX-{sub_version}"
    TASK = "binary"
elif DATASET.startswith("FD"):
    NUM_CLASSES = 3
    dataset_type = "FD"
    sub_version = DATASET.replace("FD", "").upper() or "I"
    DATA_DIR = f"./dataset/FD/FD-{sub_version}"
    TASK = "multiclass"
else:
    raise ValueError(f"Invalid dataset: {DATASET}")

# Data paths
X_L_PATH = os.path.join(DATA_DIR, "X_L.pkl")
Y_L_PATH = os.path.join(DATA_DIR, "Y_L.pkl")
X_R_PATH = os.path.join(DATA_DIR, "X_R.pkl")
Y_R_PATH = os.path.join(DATA_DIR, "Y_R.pkl")

# Result file paths
version_prefix = datetime.now().strftime("%Y%m%d%H%M")[:12]
TRAINING_STATS_FILE = f"result/{version_prefix}_training_stats_{DATASET.lower()}.npy"
TESTING_STATS_FILE = f"result/{version_prefix}_validating_stats_{DATASET.lower()}.npy"
CONFIG_FILE = f"result/{version_prefix}_config_{DATASET.lower()}.txt"
CHECKPOINT_PATH = f"checkpoints/{version_prefix}_checkpoint_{DATASET.lower()}.pth"
# ****************************************************

# Load data
with open(X_L_PATH, "rb") as f:
    X_L = np.array(pickle.load(f), dtype=object)
with open(Y_L_PATH, "rb") as f:
    Y_L = np.array(pickle.load(f), dtype=object)
with open(X_R_PATH, "rb") as f:
    X_R = np.array(pickle.load(f), dtype=object)
with open(Y_R_PATH, "rb") as f:
    Y_R = np.array(pickle.load(f), dtype=object)

# Hand mirroring processing
if FLAG_MIRROR:
    X_L = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)

# Merge data
X = np.concatenate([X_L, X_R], axis=0)
Y = np.concatenate([Y_L, Y_R], axis=0)

# Create dataset
full_dataset = IMUDataset(X, Y, sequence_length=WINDOW_SIZE)

# If the selected dataset is FDII or FDI, load FDIII data to add more training samples.
fdiii_dataset = None
if DATASET in ["FDII", "FDI"]:
    fdiii_dir = "./dataset/FD/FD-III"
    with open(os.path.join(fdiii_dir, "X_L.pkl"), "rb") as f:
        X_L_fdiii = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(fdiii_dir, "Y_L.pkl"), "rb") as f:
        Y_L_fdiii = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(fdiii_dir, "X_R.pkl"), "rb") as f:
        X_R_fdiii = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(fdiii_dir, "Y_R.pkl"), "rb") as f:
        Y_R_fdiii = np.array(pickle.load(f), dtype=object)
    X_fdiii = np.concatenate([X_L_fdiii, X_R_fdiii], axis=0)
    Y_fdiii = np.concatenate([Y_L_fdiii, Y_R_fdiii], axis=0)
    fdiii_dataset = IMUDataset(X_fdiii, Y_fdiii, sequence_length=WINDOW_SIZE)

# Create balanced cross-validation folds
# validate_folds = create_balanced_subject_folds(full_dataset, num_folds=NUM_FOLDS)

# Use the predefined folds if available
validate_folds = load_predefined_validate_folds()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create result directory
os.makedirs("result", exist_ok=True)

# Initialize statistics records
training_statistics = []
validating_statistics = []

# Cross-validation main loop
for fold, validate_subjects in enumerate(
    tqdm(validate_folds, desc="K-Fold", leave=True)
):
    if fold != 0:
        continue
    # Split training and validating sets
    train_indices = [
        i
        for i, subject in enumerate(full_dataset.subject_indices)
        if subject not in validate_subjects
    ]
    validate_indices = [
        i
        for i, subject in enumerate(full_dataset.subject_indices)
        if subject in validate_subjects
    ]

    # If using FDII/FDI, augment training data with FDIII samples
    if DATASET in ["FDII", "FDI"] and fdiii_dataset is not None:
        train_dataset = ConcatDataset(
            [Subset(full_dataset, train_indices), fdiii_dataset]
        )
    else:
        train_dataset = Subset(full_dataset, train_indices)

    # Create data loaders
    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    validate_loader = DataLoader(
        Subset(full_dataset, validate_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize model
    model = MSTCN(
        num_stages=NUM_STAGES,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        input_dim=INPUT_DIM,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_f1_score = 0.0

    # Training loop
    for epoch in tqdm(range(NUM_EPOCHS), desc=f"Fold {fold+1}", leave=False):
        model.train()
        training_loss = 0.0
        training_loss_ce = 0.0
        training_loss_mse = 0.0

        for batch_x, batch_y in train_loader:
            if FLAG_AUGMENT:
                batch_x = augment_orientation(batch_x)
            batch_x = batch_x.permute(0, 2, 1).to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            ce_loss, mse_loss = MSTCN_Loss(outputs, batch_y)
            loss = ce_loss + LAMBDA_COEF * mse_loss
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_loss_ce += ce_loss.item()
            training_loss_mse += mse_loss.item()

        # After finishing training for the epoch, switch to evaluation mode on training data
        model.eval()
        train_predictions = []
        train_labels = []
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.permute(0, 2, 1).to(device)
                outputs = model(batch_x)
                # For MS-TCN, take the last stage's output
                last_stage_output = outputs[:, -1, :, :]
                probabilities = F.softmax(last_stage_output, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                # Flatten predictions and labels and accumulate
                train_predictions.extend(predictions.view(-1).cpu().numpy())
                train_labels.extend(batch_y.view(-1).cpu().numpy())

        # Convert to tensors and compute Matthews CorrCoef for the epoch
        train_preds_tensor = torch.tensor(train_predictions)
        train_labels_tensor = torch.tensor(train_labels)
        train_mcc = MatthewsCorrCoef(num_classes=NUM_CLASSES, task=TASK)(
            train_preds_tensor, train_labels_tensor
        ).item()

        # Record training statistics
        training_statistics.append(
            {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": training_loss / len(train_loader),
                "train_loss_ce": training_loss_ce / len(train_loader),
                "train_loss_mse": training_loss_mse / len(train_loader),
                "matthews_corrcoef": train_mcc,
            }
        )

    # Evaluation phase
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in validate_loader:
            batch_x = batch_x.permute(0, 2, 1).to(device)
            outputs = model(batch_x)

            # For MS-TCN, we take the last stage's predictions
            last_stage_output = outputs[
                :, -1, :, :
            ]  # Shape: [batch_size, num_classes, seq_len]
            probabilities = F.softmax(last_stage_output, dim=1)

            # Get predictions along the class dimension
            predictions = torch.argmax(
                probabilities, dim=1
            )  # Shape: [batch_size, seq_len]

            # Flatten both predictions and labels
            all_predictions.extend(predictions.view(-1).cpu().numpy())
            all_labels.extend(batch_y.view(-1).cpu().numpy())

    # Post-processing
    all_predictions = post_process_predictions(np.array(all_predictions))
    all_labels = np.array(all_labels)

    # Label distribution
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    label_distribution = dict(zip(unique_labels, counts))

    # Segment-wise evaluation (using the custom function)
    metrics_segment = {}
    for label in range(1, NUM_CLASSES):
        fn, fp, tp = segment_evaluation(
            all_predictions,
            all_labels,
            class_label=label,
            threshold=0.5,
            debug_plot=DEBUG_PLOT,
        )
        f1 = 2 * tp / (2 * tp + fp + fn) if (fp + fn) != 0 else 0.0
        metrics_segment[f"{label}"] = {
            "fn": int(fn),
            "fp": int(fp),
            "tp": int(tp),
            "f1": float(f1),
        }

    # Sample-wise evaluation
    preds_tensor = torch.tensor(all_predictions)
    labels_tensor = torch.tensor(all_labels)
    metrics_sample = {}
    for label in range(1, NUM_CLASSES):
        tp = torch.sum((preds_tensor == label) & (labels_tensor == label)).item()
        fp = torch.sum((preds_tensor == label) & (labels_tensor != label)).item()
        fn = torch.sum((preds_tensor != label) & (labels_tensor == label)).item()
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator != 0 else 0.0
        metrics_sample[f"{label}"] = {"fn": fn, "fp": fp, "tp": tp, "f1": f1}

    # Compute additional metrics: Cohen's Kappa and Matthews Correlation Coefficient
    cohen_kappa_val = CohenKappa(num_classes=NUM_CLASSES, task=TASK)(
        preds_tensor, labels_tensor
    ).item()
    matthews_corrcoef_val = MatthewsCorrCoef(num_classes=NUM_CLASSES, task=TASK)(
        preds_tensor, labels_tensor
    ).item()

    # Record validating statistics
    validating_statistics.append(
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "fold": fold + 1,
            "metrics_segment": metrics_segment,
            "metrics_sample": metrics_sample,
            "cohen_kappa": cohen_kappa_val,
            "matthews_corrcoef": matthews_corrcoef_val,
            "label_distribution": label_distribution,
        }
    )

# Save results and configuration
np.save(TRAINING_STATS_FILE, training_statistics)
np.save(TESTING_STATS_FILE, validating_statistics)

config_info = {
    "dataset": DATASET,
    "num_classes": NUM_CLASSES,
    "num_stages": NUM_STAGES,
    "num_layers": NUM_LAYERS,
    "num_heads": NUM_HEADS,
    "input_dim": INPUT_DIM,
    "num_filters": NUM_FILTERS,
    "kernel_size": KERNEL_SIZE,
    "dropout": DROPOUT,
    "lambda_coef": LAMBDA_COEF,
    "tau": TAU,
    "learning_rate": LEARNING_RATE,
    "sampling_freq": SAMPLING_FREQ,
    "window_size": WINDOW_SIZE,
    "num_folds": NUM_FOLDS,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "augmentation": FLAG_AUGMENT,
    "mirroring": FLAG_MIRROR,
}

with open(CONFIG_FILE, "w") as f:
    for key, value in config_info.items():
        f.write(f"{key}: {value}\n")
