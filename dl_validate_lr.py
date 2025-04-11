#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Validating Script (Separate Hands)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Edited      : 2025-04-11
Description : This script validates a trained MSTCN model on the full IMU dataset,
              separately for left and right hand data.
Usage       : Execute the script in your terminal:
              $ python validate.py <version> [mirror]
===============================================================================
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from components.model_mstcn import MSTCN
from components.model_tcn import TCN
from components.model_cnnlstm import CNNLSTM
from components.post_processing import post_process_predictions
from components.evaluation import segment_evaluation
from components.datasets import IMUDataset
from components.pre_processing import hand_mirroring

if len(sys.argv) >= 2:
    result_version = sys.argv[1]
else:
    raise ValueError("Please provide the result version as the first argument.")

FLAG_DATASET_MIRROR = len(sys.argv) >= 3 and sys.argv[2].lower() == "mirror"

result_dir = os.path.join("result", result_version)
config_file = os.path.join(result_dir, "config.json")

with open(config_file, "r") as f:
    config_info = json.load(f)

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

validating_stats_file = os.path.join(
    result_dir,
    "validate_stats_mirrored.npy" if FLAG_DATASET_MIRROR else "validate_stats_separate.npy",
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if DATASET.startswith("DX"):
        sub_version = DATASET.replace("DX", "").upper() or "I"
        DATA_DIR = f"./dataset/DX/DX-{sub_version}"
    elif DATASET.startswith("FD"):
        sub_version = DATASET.replace("FD", "").upper() or "I"
        DATA_DIR = f"./dataset/FD/FD-{sub_version}"
    else:
        raise ValueError(f"Invalid dataset: {DATASET}")

    X_L = np.array(pickle.load(open(os.path.join(DATA_DIR, "X_L.pkl"), "rb")), dtype=object)
    Y_L = np.array(pickle.load(open(os.path.join(DATA_DIR, "Y_L.pkl"), "rb")), dtype=object)
    X_R = np.array(pickle.load(open(os.path.join(DATA_DIR, "X_R.pkl"), "rb")), dtype=object)
    Y_R = np.array(pickle.load(open(os.path.join(DATA_DIR, "Y_R.pkl"), "rb")), dtype=object)

    validate_folds = config_info.get("validate_folds")
    if validate_folds is None:
        raise ValueError("No 'validate_folds' found in the configuration file.")

    validating_statistics = []

    for side_name, (X_side, Y_side) in [("left", (X_L, Y_L)), ("right", (X_R, Y_R))]:
        print(f"\n--- Validating {side_name.upper()} hand data ---")

        if side_name == "left" and FLAG_DATASET_MIRROR:
            X_side = np.array([hand_mirroring(sample) for sample in X_side], dtype=object)

        dataset = IMUDataset(X_side, Y_side, sequence_length=WINDOW_SIZE)

        for fold, validate_subjects in enumerate(tqdm(validate_folds, desc=f"K-Fold ({side_name})", leave=True)):
            checkpoint_path = os.path.join(result_dir, "checkpoint", f"best_model_fold{fold+1}.pth")
            if not os.path.exists(checkpoint_path):
                continue

            if MODEL == "TCN":
                model = TCN(
                    num_layers=config_info["num_layers"],
                    num_classes=NUM_CLASSES,
                    input_dim=INPUT_DIM,
                    num_filters=config_info["num_filters"],
                    kernel_size=config_info["kernel_size"],
                    dropout=config_info["dropout"],
                ).to(device)
            elif MODEL == "MSTCN":
                model = MSTCN(
                    num_stages=config_info["num_stages"],
                    num_layers=config_info["num_layers"],
                    num_classes=NUM_CLASSES,
                    input_dim=INPUT_DIM,
                    num_filters=config_info["num_filters"],
                    kernel_size=config_info["kernel_size"],
                    dropout=config_info["dropout"],
                ).to(device)
            elif MODEL == "CNN_LSTM":
                model = CNNLSTM(
                    input_channels=INPUT_DIM,
                    conv_filters=config_info["conv_filters"],
                    lstm_hidden=config_info["lstm_hidden"],
                    num_classes=NUM_CLASSES,
                ).to(device)
            else:
                raise ValueError(f"Invalid model: {MODEL}")

            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            model.eval()

            validate_indices = [i for i, subject in enumerate(dataset.subject_indices) if subject in validate_subjects]
            validate_loader = DataLoader(
                Subset(dataset, validate_indices),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
            )

            all_predictions, all_labels = [], []
            with torch.no_grad():
                for batch_x, batch_y in tqdm(validate_loader, desc=f"Fold {fold+1}", leave=False):
                    batch_x = batch_x.permute(0, 2, 1).to(device)
                    outputs = model(batch_x)
                    logits = outputs[:, -1, :, :] if outputs.ndim == 4 else outputs
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_predictions.extend(preds.view(-1).cpu().numpy())
                    all_labels.extend(batch_y.view(-1).cpu().numpy())

            preds_array = np.array(all_predictions)
            labels_array = np.array(all_labels)
            unique_labels, counts = np.unique(labels_array, return_counts=True)
            label_distribution = {int(l): int(c) for l, c in zip(unique_labels, counts)}
            positive_counts = {l: c for l, c in label_distribution.items() if l > 0}
            total_positive = sum(positive_counts.values())
            label_weight = {l: c / total_positive for l, c in positive_counts.items()}

            metrics_sample = {}
            for label in range(1, NUM_CLASSES):
                tp = np.sum((preds_array == label) & (labels_array == label))
                fp = np.sum((preds_array == label) & (labels_array != label))
                fn = np.sum((preds_array != label) & (labels_array == label))
                denom = 2 * tp + fp + fn
                f1 = (2 * tp / denom) if denom > 0 else 0.0
                metrics_sample[str(label)] = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "f1": float(f1)}

            weighted_f1_sample = sum(metrics_sample[str(l)]["f1"] * label_weight[l] for l in range(1, NUM_CLASSES))
            metrics_sample["weighted_f1"] = float(weighted_f1_sample)

            processed_preds = post_process_predictions(preds_array, SAMPLING_FREQ)
            metrics_segment_all = {}
            for threshold in THRESHOLD_LIST:
                metrics_segment = {}
                for label in range(1, NUM_CLASSES):
                    fn, fp, tp = segment_evaluation(
                        processed_preds, labels_array, class_label=label, threshold=threshold, debug_plot=DEBUG_PLOT
                    )
                    denom = 2 * tp + fp + fn
                    f1 = (2 * tp / denom) if denom > 0 else 0.0
                    metrics_segment[str(label)] = {"fn": int(fn), "fp": int(fp), "tp": int(tp), "f1": float(f1)}
                weighted_f1 = sum(metrics_segment[str(l)]["f1"] * label_weight[l] for l in range(1, NUM_CLASSES))
                metrics_segment["weighted_f1"] = float(weighted_f1)
                metrics_segment_all[str(threshold)] = metrics_segment

            fold_stat = {
                "side": side_name,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "fold": fold + 1,
                "metrics_segment": metrics_segment_all,
                "metrics_sample": metrics_sample,
                "label_distribution": label_distribution,
            }
            validating_statistics.append(fold_stat)

    np.save(validating_stats_file, validating_statistics)
    print(f"\nValidating statistics saved to {validating_stats_file}")


if __name__ == "__main__":
    main()
