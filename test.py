#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Testing Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Version     : 1.0
Created     : 2025-03-22
Description : This script tests a trained MSTCN model on the full IMU dataset.
              It loads the dataset and a saved checkpoint, runs inference,
              post-processes the predictions, and computes evaluation metrics.
Usage       : Execute the script in your terminal:
              $ python test.py
===============================================================================
"""

import os
import glob
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import CohenKappa, MatthewsCorrCoef
from torch.utils.data import DataLoader

# Import model and utility modules from components package
from components.model_mstcn import MSTCN
from components.post_processing import post_process_predictions
from components.evaluation import segment_evaluation
from components.datasets import IMUDataset
from components.pre_processing import hand_mirroring

# -----------------------------------------------------------------------------
#                   Configuration and Parameters
# -----------------------------------------------------------------------------

# Directory where checkpoints are saved
CHECKPOINT_DIR = "checkpoints"

# Dataset and model configuration (should match train.py)
DATASET = "FDI"  # Options: DXI/DXII or FDI/FDII/FDIII
if DATASET.startswith("DX"):
    NUM_CLASSES = 2
    sub_version = DATASET.replace("DX", "").upper() or "I"
    DATA_DIR = f"./dataset/DX/DX-{sub_version}"
    TASK = "binary"
elif DATASET.startswith("FD"):
    NUM_CLASSES = 3
    sub_version = DATASET.replace("FD", "").upper() or "I"
    DATA_DIR = f"./dataset/FD/FD-{sub_version}"
    TASK = "multiclass"
else:
    raise ValueError(f"Invalid dataset: {DATASET}")

# Model hyperparameters (as used in train.py)
NUM_STAGES = 2
NUM_LAYERS = 9
NUM_HEADS = 8
INPUT_DIM = 6
NUM_FILTERS = 128
KERNEL_SIZE = 3
DROPOUT = 0.3

# Sampling and window parameters
SAMPLING_FREQ_ORIGINAL = 64
DOWNSAMPLE_FACTOR = 4
SAMPLING_FREQ = SAMPLING_FREQ_ORIGINAL // DOWNSAMPLE_FACTOR
WINDOW_LENGTH = 60
WINDOW_SIZE = SAMPLING_FREQ * WINDOW_LENGTH

# Data augmentation / pre-processing flags
FLAG_MIRROR = False  # Apply hand mirroring

# Testing DataLoader parameters
BATCH_SIZE = 64
NUM_WORKERS = 4  # Fewer workers are typically sufficient for testing

# -----------------------------------------------------------------------------
#                        Data Loading and Pre-processing
# -----------------------------------------------------------------------------

# Define file paths for the dataset (left and right data)
X_L_PATH = os.path.join(DATA_DIR, "X_L.pkl")
Y_L_PATH = os.path.join(DATA_DIR, "Y_L.pkl")
X_R_PATH = os.path.join(DATA_DIR, "X_R.pkl")
Y_R_PATH = os.path.join(DATA_DIR, "Y_R.pkl")

# Load data from pickle files
with open(X_L_PATH, "rb") as f:
    X_L = np.array(pickle.load(f), dtype=object)
with open(Y_L_PATH, "rb") as f:
    Y_L = np.array(pickle.load(f), dtype=object)
with open(X_R_PATH, "rb") as f:
    X_R = np.array(pickle.load(f), dtype=object)
with open(Y_R_PATH, "rb") as f:
    Y_R = np.array(pickle.load(f), dtype=object)

# Optionally apply hand mirroring if the flag is set
if FLAG_MIRROR:
    X_L = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)

# Merge left and right data
X = np.concatenate([X_L, X_R], axis=0)
Y = np.concatenate([Y_L, Y_R], axis=0)

# Create the full dataset with the defined window size
full_dataset = IMUDataset(X, Y, sequence_length=WINDOW_SIZE)

# Create a DataLoader for the full dataset
test_loader = DataLoader(
    full_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# -----------------------------------------------------------------------------
#                      Model Initialization and Checkpoint Loading
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the MSTCN model with the specified parameters
model = MSTCN(
    num_stages=NUM_STAGES,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    input_dim=INPUT_DIM,
    num_filters=NUM_FILTERS,
    kernel_size=KERNEL_SIZE,
    dropout=DROPOUT,
).to(device)

# Automatically select the latest version based on timestamp
all_versions = glob.glob(os.path.join(CHECKPOINT_DIR, "*"))
RESULT_VERSION = max(all_versions, key=os.path.getmtime).split(os.sep)[-1]
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, RESULT_VERSION, f"best_model_fold1.pth")

if os.path.exists(CHECKPOINT_PATH):
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("Checkpoint loaded successfully.")
else:
    print(f"Checkpoint not found at {CHECKPOINT_PATH}. Exiting.")
    exit(1)

model.eval()

# -----------------------------------------------------------------------------
#                          Model Testing / Inference
# -----------------------------------------------------------------------------

all_predictions = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
        # Rearrange dimensions and send to device
        batch_x = batch_x.permute(0, 2, 1).to(device)
        outputs = model(batch_x)
        # Use the output from the last stage for predictions
        last_stage_output = outputs[:, -1, :, :]
        probabilities = F.softmax(last_stage_output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        all_predictions.extend(predictions.view(-1).cpu().numpy())
        all_labels.extend(batch_y.view(-1).cpu().numpy())

# -----------------------------------------------------------------------------
#                        Evaluation Metrics Calculation
# -----------------------------------------------------------------------------

# Compute label distribution
unique_labels, counts = np.unique(all_labels, return_counts=True)
label_distribution = {
    float(label): int(count) for label, count in zip(unique_labels, counts)
}
print("Label distribution:", label_distribution)

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

cohen_kappa_val = CohenKappa(num_classes=NUM_CLASSES, task=TASK)(
    preds_tensor, labels_tensor
).item()
matthews_corrcoef_val = MatthewsCorrCoef(num_classes=NUM_CLASSES, task=TASK)(
    preds_tensor, labels_tensor
).item()

print("Sample-wise Evaluation Metrics:")
for class_label, metrics in metrics_sample.items():
    print(f"Class {class_label}: {metrics}")
print(f"Cohen's Kappa: {cohen_kappa_val:.4f}")
print(f"Matthews CorrCoef: {matthews_corrcoef_val:.4f}")

# all_predictions = post_process_predictions(np.array(all_predictions), SAMPLING_FREQ)
# all_labels = np.array(all_labels)

# # Segment-wise evaluation
# metrics_segment = {}
# for label in range(1, NUM_CLASSES):
#     fn, fp, tp = segment_evaluation(
#         all_predictions, all_labels, class_label=label, threshold=0.5, debug_plot=True
#     )
#     f1 = 2 * tp / (2 * tp + fp + fn) if (fp + fn) != 0 else 0.0
#     metrics_segment[f"Class {label}"] = {
#         "fn": int(fn),
#         "fp": int(fp),
#         "tp": int(tp),
#         "f1": f1,
#     }

# print("Segment-wise Evaluation Metrics:")
# for class_label, metrics in metrics_segment.items():
#     print(f"{class_label}: {metrics}")

# -----------------------------------------------------------------------------
#           Two-Phase Line Search for Optimal Post-processing Parameters
# -----------------------------------------------------------------------------

# Phase 1: Optimize min_length_sec while holding merge_distance_sec fixed
default_merge_distance = 0.1
min_length_values = np.linspace(
    0.0, 1.0, 11
)  # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 seconds

best_avg_f1_phase1 = 0.0
best_min_length = None

print(
    "\nPhase 1: Optimizing min_length_sec with merge_distance_sec fixed at "
    f"{default_merge_distance:.2f} seconds..."
)

for min_length in min_length_values:
    processed_preds = post_process_predictions(
        all_predictions,
        SAMPLING_FREQ,
        min_length_sec=min_length,
        merge_distance_sec=default_merge_distance,
    )
    # Evaluate average F1 score over classes 1..NUM_CLASSES-1
    f1_list = []
    for label in range(1, NUM_CLASSES):
        fn, fp, tp = segment_evaluation(
            processed_preds,
            all_labels,
            class_label=label,
            threshold=0.1,
            debug_plot=False,
        )
        f1 = 2 * tp / (2 * tp + fp + fn) if (fp + fn) != 0 else 0.0
        f1_list.append(f1)
    avg_f1 = np.mean(f1_list)
    print(f"min_length_sec: {min_length:.2f}, avg F1: {avg_f1:.4f}")

    if avg_f1 > best_avg_f1_phase1:
        best_avg_f1_phase1 = avg_f1
        best_min_length = min_length

print(
    f"\nBest min_length_sec found: {best_min_length:.2f} seconds with avg F1: {best_avg_f1_phase1:.4f}"
)


# Phase 2: Optimize merge_distance_sec while holding min_length_sec fixed at best_min_length
merge_distance_values = np.linspace(
    0.0, 0.3, 7
)  # 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3 seconds

best_avg_f1_phase2 = 0.0
best_merge_distance = None

print(
    "\nPhase 2: Optimizing merge_distance_sec with min_length_sec fixed at "
    f"{best_min_length:.2f} seconds..."
)

for merge_distance in merge_distance_values:
    processed_preds = post_process_predictions(
        all_predictions,
        SAMPLING_FREQ,
        min_length_sec=best_min_length,
        merge_distance_sec=merge_distance,
    )
    # Evaluate average F1 score over classes 1..NUM_CLASSES-1
    f1_list = []
    for label in range(1, NUM_CLASSES):
        fn, fp, tp = segment_evaluation(
            processed_preds,
            all_labels,
            class_label=label,
            threshold=0.1,
            debug_plot=False,
        )
        f1 = 2 * tp / (2 * tp + fp + fn) if (fp + fn) != 0 else 0.0
        f1_list.append(f1)
    avg_f1 = np.mean(f1_list)
    print(f"merge_distance_sec: {merge_distance:.2f}, avg F1: {avg_f1:.4f}")

    if avg_f1 > best_avg_f1_phase2:
        best_avg_f1_phase2 = avg_f1
        best_merge_distance = merge_distance

print(
    f"\nBest merge_distance_sec found: {best_merge_distance:.2f} seconds with avg F1: {best_avg_f1_phase2:.4f}"
)

# Final Best Parameters
print("\nOverall Optimal Post-processing Parameters:")
print(f"min_length_sec = {best_min_length:.2f} seconds")
print(f"merge_distance_sec = {best_merge_distance:.2f} seconds")
print(f"Optimized segment-wise average F1 score: {best_avg_f1_phase2:.4f}")
