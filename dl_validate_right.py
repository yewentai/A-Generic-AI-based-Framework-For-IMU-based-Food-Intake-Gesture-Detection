#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN Right Hand IMU Validation Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Edited      : 2025-04-14
Description : This script validates MSTCN models on RIGHT hand IMU data with:
              1. Original right hand validation
              2. Hand mirroring (if enabled in config)
              3. Planar rotation (if enabled in config)
===============================================================================
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from components.model_mstcn import MSTCN
from components.model_tcn import TCN
from components.model_cnnlstm import CNNLSTM
from components.post_processing import post_process_predictions
from components.evaluation import segment_evaluation
from components.datasets import IMUDataset
from components.pre_processing import hand_mirroring, planar_rotation

# --- Configurations ---
NUM_WORKERS = 4
THRESHOLD_LIST = [0.1, 0.25, 0.5, 0.75]
DEBUG_PLOT = False


if __name__ == "__main__":
    result_root = "result"
    # versions = ["202504102241"]  # Uncomment to manually specify versions
    versions = [d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d)) and "_RIGHT" in d]
    versions.sort()

    for version in versions:
        result_dir = os.path.join(result_root, version)
        os.makedirs(result_dir, exist_ok=True)

        # Set up logging
        log_file = os.path.join(result_dir, "validation_right.log")
        logger = logging.getLogger(f"validation_right_{version}")
        logger.setLevel(logging.INFO)

        # Remove existing handlers if any
        if logger.hasHandlers():
            logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        logger.info(f"\n=== Validating Right Hand Version: {version} ===")

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
        validate_folds = config_info.get("validate_folds")
        mirror_enabled = config_info.get("augmentation_hand_mirroring", False) or config_info.get(
            "dataset_mirroring", False
        )
        rotation_enabled = config_info.get("augmentation_planar_rotation", False)

        if validate_folds is None:
            logger.error("No 'validate_folds' found in the configuration file.")
            continue

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"\nUsing device: {device}")

        if DATASET.startswith("DX"):
            sub_version = DATASET.replace("DX", "").upper() or "I"
            DATA_DIR = f"./dataset/DX/DX-{sub_version}"
        elif DATASET.startswith("FD"):
            sub_version = DATASET.replace("FD", "").upper() or "I"
            DATA_DIR = f"./dataset/FD/FD-{sub_version}"
        else:
            logger.error(f"Invalid dataset: {DATASET}")
            continue

        # Load RIGHT hand data only
        X = np.array(pickle.load(open(os.path.join(DATA_DIR, "X_R.pkl"), "rb")), dtype=object)
        Y = np.array(pickle.load(open(os.path.join(DATA_DIR, "Y_R.pkl"), "rb")), dtype=object)

        # Prepare validation modes for RIGHT hand
        validation_modes = [
            {"name": "original_right", "X": X, "Y": Y},
        ]

        if mirror_enabled:
            X_mirrored = np.array([hand_mirroring(sample) for sample in X], dtype=object)
            validation_modes.append({"name": "mirrored_right", "X": X_mirrored, "Y": Y})

        if rotation_enabled:
            # Apply rotation to 50% of samples
            random_indices = np.random.choice(len(X), size=int(len(X) * 0.5), replace=False)
            X_rotated = np.copy(X)
            for i in random_indices:
                X_rotated[i], Y[i] = planar_rotation(X[i], Y[i])

            validation_modes.append({"name": "rotated_right", "X": X_rotated, "Y": Y})

            if mirror_enabled:
                X_rotated_mirrored = np.array([hand_mirroring(sample) for sample in X_rotated], dtype=object)
                validation_modes.append({"name": "rotated_mirrored_right", "X": X_rotated_mirrored, "Y": Y})

        all_stats = {}

        for mode in validation_modes:
            logger.info(f"\n--- Validating {mode['name']} ---")

            dataset = IMUDataset(mode["X"], mode["Y"], sequence_length=WINDOW_SIZE)
            mode_stats = []

            for fold, validate_subjects in enumerate(tqdm(validate_folds, desc=f"K-Fold ({mode['name']})", leave=True)):
                checkpoint_path = os.path.join(result_dir, "checkpoint", f"best_model_fold{fold+1}.pth")
                if not os.path.exists(checkpoint_path):
                    logger.warning(f"Checkpoint not found for fold {fold+1}, skipping...")
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
                    logger.error(f"Invalid model: {MODEL}")
                    continue

                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
                new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
                model.eval()

                validate_indices = [
                    i for i, subject in enumerate(dataset.subject_indices) if subject in validate_subjects
                ]
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
                    tn = np.sum((preds_array != label) & (labels_array != label))
                    denom = 2 * tp + fp + fn
                    f1 = (2 * tp / denom) if denom > 0 else 0.0
                    metrics_sample[str(label)] = {
                        "tp": int(tp),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tn": int(tn),
                        "f1": float(f1),
                    }

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
                    "fold": fold + 1,
                    "metrics_segment": metrics_segment_all,
                    "metrics_sample": metrics_sample,
                    "label_distribution": label_distribution,
                }
                mode_stats.append(fold_stat)

            all_stats[mode["name"]] = mode_stats

        # Save all statistics
        stats_file_npy = os.path.join(result_dir, "validation_right_stats.npy")
        stats_file_json = os.path.join(result_dir, "validation_right_stats.json")
        np.save(stats_file_npy, all_stats)
        with open(stats_file_json, "w") as f_json:
            json.dump(all_stats, f_json, indent=4)
        logger.info(f"\nRight hand validation statistics saved to {stats_file_npy} and {stats_file_json}")
