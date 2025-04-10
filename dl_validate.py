#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Validating Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-03-29
Description : This script validates a trained MSTCN model on the full IMU dataset.
              It loads the dataset and saved checkpoints for each fold,
              runs inference, and computes comprehensive evaluation metrics.
Usage       : Execute the script in your terminal:
              $ python validate.py
===============================================================================
"""

import os
import sys
import json
import glob
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# Import model and utility modules from components package
from components.model_mstcn import MSTCN
from components.model_tcn import TCN
from components.model_cnnlstm import CNNLSTM
from components.post_processing import post_process_predictions
from components.evaluation import segment_evaluation
from components.datasets import IMUDataset
from components.pre_processing import hand_mirroring

# -----------------------------------------------------------------------------
#                   Configuration and Parameters
# -----------------------------------------------------------------------------

if len(sys.argv) >= 2:
    result_version = sys.argv[1]

if len(sys.argv) >= 3 and sys.argv[2].lower() == "mirror":
    FLAG_DATASET_MIRROR = True
else:
    FLAG_DATASET_MIRROR = False

# result_version = max(glob.glob(os.path.join("result", "*")), key=os.path.getmtime).split(os.sep)[-1]
# result_version = "202503281533"  # <- Manually set version

result_dir = os.path.join("result", result_version)
config_file = os.path.join(result_dir, "config.json")

# Load the configuration parameters from the JSON file
with open(config_file, "r") as f:
    config_info = json.load(f)

# Set configuration parameters from the loaded JSON
DATASET = config_info["dataset"]
NUM_CLASSES = config_info["num_classes"]
MODEL = config_info["model"]
INPUT_DIM = config_info["input_dim"]
SAMPLING_FREQ = config_info["sampling_freq"]
WINDOW_SIZE = config_info["window_size"]
BATCH_SIZE = config_info["batch_size"]
NUM_WORKERS = 4
THRESHOLD_LIST = [0.1, 0.25, 0.5, 0.75]
DEBUG_PLOT = False
# FLAG_DATASET_MIRROR = False

validating_stats_file = os.path.join(
    result_dir,
    "validate_stats_mirrored.npy" if FLAG_DATASET_MIRROR else "validate_stats.npy",
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------------
    #                        Data Loading and Pre-processing
    # -----------------------------------------------------------------------------

    if DATASET.startswith("DX"):
        sub_version = DATASET.replace("DX", "").upper() or "I"
        DATA_DIR = f"./dataset/DX/DX-{sub_version}"
        TASK = "binary"
    elif DATASET.startswith("FD"):
        sub_version = DATASET.replace("FD", "").upper() or "I"
        DATA_DIR = f"./dataset/FD/FD-{sub_version}"
        TASK = "multiclass"
    else:
        raise ValueError(f"Invalid dataset: {DATASET}")

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
    if FLAG_DATASET_MIRROR:
        X_L = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)

    # Merge left and right data
    X = np.concatenate([X_L, X_R], axis=0)
    Y = np.concatenate([Y_L, Y_R], axis=0)

    # Create the full dataset with the defined window size
    full_dataset = IMUDataset(X, Y, sequence_length=WINDOW_SIZE)

    # Load cross-validation folds from the JSON configuration
    validate_folds = config_info.get("validate_folds")
    if validate_folds is None:
        raise ValueError("No 'validate_folds' found in the configuration file.")

    # -----------------------------------------------------------------------------
    #                              Validating Loop
    # -----------------------------------------------------------------------------

    # Create a results storage for cross-validation evaluations
    validating_statistics = []

    for fold, validate_subjects in enumerate(tqdm(validate_folds, desc="K-Fold", leave=True)):
        # Construct the checkpoint path for the current fold
        checkpoint_path = os.path.join(result_dir, "checkpoint", f"best_model_fold{fold+1}.pth")

        # Check if the checkpoint for this fold exists
        if not os.path.exists(checkpoint_path):
            continue

        # Instantiate the model based on the saved configuration
        if MODEL == "TCN":
            NUM_LAYERS = config_info["num_layers"]
            NUM_FILTERS = config_info["num_filters"]
            KERNEL_SIZE = config_info["kernel_size"]
            DROPOUT = config_info["dropout"]
            model = TCN(
                num_layers=NUM_LAYERS,
                num_classes=NUM_CLASSES,
                input_dim=INPUT_DIM,
                num_filters=NUM_FILTERS,
                kernel_size=KERNEL_SIZE,
                dropout=DROPOUT,
            ).to(device)
        elif MODEL == "MSTCN":
            NUM_STAGES = config_info["num_stages"]
            NUM_LAYERS = config_info["num_layers"]
            NUM_FILTERS = config_info["num_filters"]
            KERNEL_SIZE = config_info["kernel_size"]
            DROPOUT = config_info["dropout"]
            model = MSTCN(
                num_stages=NUM_STAGES,
                num_layers=NUM_LAYERS,
                num_classes=NUM_CLASSES,
                input_dim=INPUT_DIM,
                num_filters=NUM_FILTERS,
                kernel_size=KERNEL_SIZE,
                dropout=DROPOUT,
            ).to(device)
        elif MODEL == "CNN_LSTM":
            conv_filters = config_info["conv_filters"]
            lstm_hidden = config_info["lstm_hidden"]
            model = CNNLSTM(
                input_channels=INPUT_DIM,
                conv_filters=conv_filters,
                lstm_hidden=lstm_hidden,
                num_classes=NUM_CLASSES,
            ).to(device)
        else:
            raise ValueError(f"Invalid model: {MODEL}")

        # Load the checkpoint for the current fold
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Remove 'module.' prefix if it exists.
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

        # Split validation indices based on subject IDs
        validate_indices = [i for i, subject in enumerate(full_dataset.subject_indices) if subject in validate_subjects]

        # Create DataLoader for the validation subset of the current fold
        validate_loader = DataLoader(
            Subset(full_dataset, validate_indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Explicitly set the model to evaluation mode
        model.eval()

        # Inference and Prediction Collection
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in tqdm(validate_loader, desc=f"Validating Fold {fold+1}", leave=False):
                batch_x = batch_x.permute(0, 2, 1).to(device)
                outputs = model(batch_x)
                # If the model produces outputs with multiple stages (4D tensor), select the last stage.
                if outputs.ndim == 4:
                    logits = outputs[:, -1, :, :]
                else:
                    logits = outputs
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                all_predictions.extend(predictions.view(-1).cpu().numpy())
                all_labels.extend(batch_y.view(-1).cpu().numpy())

        # -----------------------------------------------------------------------------
        #                        Evaluation Metrics Calculation
        # -----------------------------------------------------------------------------

        # Calculate label distribution and weights for weighted metrics
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        label_distribution = {int(label): int(count) for label, count in zip(unique_labels, counts)}
        # Calculate weights only for non-background classes
        positive_counts = {label: count for label, count in label_distribution.items() if label > 0}
        total_positive_count = sum(positive_counts.values())
        label_weight = {label: count / total_positive_count for label, count in positive_counts.items()}

        # Convert predictions and labels to numpy arrays for efficient processing
        preds_array = np.array(all_predictions)
        labels_array = np.array(all_labels)

        # Calculate sample-wise metrics
        metrics_sample = {}
        for label in range(1, NUM_CLASSES):
            # Vectorized computation of confusion matrix elements
            tp = np.sum((preds_array == label) & (labels_array == label))
            fp = np.sum((preds_array == label) & (labels_array != label))
            fn = np.sum((preds_array != label) & (labels_array == label))

            # Safe F1 calculation
            denominator = 2 * tp + fp + fn
            f1 = (2 * tp / denominator) if denominator > 0 else 0.0

            metrics_sample[f"{label}"] = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "f1": float(f1)}

        # Calculate weighted sample F1 score
        f1_score_sample_weighted = sum(
            metrics_sample[str(label)]["f1"] * label_weight[label] for label in range(1, NUM_CLASSES)
        )
        metrics_sample["weighted_f1"] = float(f1_score_sample_weighted)

        # Post-process predictions for segment evaluation
        processed_predictions = post_process_predictions(preds_array, SAMPLING_FREQ)

        # Segment-wise evaluation across multiple thresholds
        metrics_segment_all_thresholds = {}
        for threshold in THRESHOLD_LIST:
            metrics_segment = {}

            # Calculate metrics for each class
            for label in range(1, NUM_CLASSES):
                fn, fp, tp = segment_evaluation(
                    processed_predictions, labels_array, class_label=label, threshold=threshold, debug_plot=DEBUG_PLOT
                )

                # Safe F1 calculation
                total = 2 * tp + fp + fn
                f1 = (2 * tp / total) if total > 0 else 0.0

                metrics_segment[str(label)] = {"fn": int(fn), "fp": int(fp), "tp": int(tp), "f1": float(f1)}

            # Calculate weighted segment F1 score
            f1_score_segment_weighted = sum(
                metrics_segment[str(label)]["f1"] * label_weight[label] for label in range(1, NUM_CLASSES)
            )
            metrics_segment["weighted_f1"] = float(f1_score_segment_weighted)
            metrics_segment_all_thresholds[str(threshold)] = metrics_segment

        # Record validating statistics for the current fold
        fold_statistics = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "fold": fold + 1,
            "metrics_segment": metrics_segment_all_thresholds,
            "metrics_sample": metrics_sample,
            "label_distribution": label_distribution,
        }
        validating_statistics.append(fold_statistics)

    # -----------------------------------------------------------------------------
    #                         Save Evaluation Results
    # -----------------------------------------------------------------------------

    # Save validating statistics
    np.save(validating_stats_file, validating_statistics)
    print(f"\nValidating statistics saved to {validating_stats_file}")


if __name__ == "__main__":
    main()
