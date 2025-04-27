#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Validation Script (DX/FD Datasets)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Edited      : 2025-04-23
Description : This script validates MSTCN models on DX/FD IMU datasets with:
              1. Original left/right hand validation
              2. Hand mirroring (if enabled in config)
              3. Planar rotation (if enabled in config)
              4. Support for both DX and FD dataset formats
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
from scipy.io import savemat

from components.model_mstcn import MSTCN
from components.model_tcn import TCN
from components.model_cnnlstm import CNNLSTM
from components.post_processing import post_process_predictions
from components.evaluation import segment_evaluation
from components.datasets import IMUDataset
from components.pre_processing import hand_mirroring, planar_rotation
from components.utils import convert_for_matlab

# --- Configurations ---
NUM_WORKERS = 4
THRESHOLD_LIST = [0.1, 0.25, 0.5, 0.75]
DEBUG_PLOT = False
SAVE_LOG = True


if __name__ == "__main__":
    result_root = "result"
    versions = ["202504271525", "202504271533"]  # Uncomment to manually specify versions
    # versions = [d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d))]
    versions.sort()

    for version in versions:
        result_dir = os.path.join(result_root, version)
        os.makedirs(result_dir, exist_ok=True)

        # Set up logging
        logger = logging.getLogger(f"validation_{version}")
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            logger.handlers.clear()

        # Always add stream handler (console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(stream_handler)

        # Optionally add file handler
        if SAVE_LOG:
            log_file = os.path.join(result_dir, "validation.log")
            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(file_handler)

        logger.info(f"\n=== Validating Version: {version} ===")

        config_file = os.path.join(result_dir, "config.json")
        if not os.path.exists(config_file):
            logger.warning(f"Configuration file not found for version {version}, skipping...")
            continue
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
        if version == "202503281533":
            mirror_enabled = True
            rotation_enabled = True

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

        # Load original data
        X_L = np.array(pickle.load(open(os.path.join(DATA_DIR, "X_L.pkl"), "rb")), dtype=object)
        Y_L = np.array(pickle.load(open(os.path.join(DATA_DIR, "Y_L.pkl"), "rb")), dtype=object)
        X_R = np.array(pickle.load(open(os.path.join(DATA_DIR, "X_R.pkl"), "rb")), dtype=object)
        Y_R = np.array(pickle.load(open(os.path.join(DATA_DIR, "Y_R.pkl"), "rb")), dtype=object)

        # Prepare validation modes based on config
        validation_modes = []

        # Determine base validation modes
        validation_modes.extend(
            [{"name": "original_left", "X": X_L, "Y": Y_L}, {"name": "original_right", "X": X_R, "Y": Y_R}]
        )

        # Add mirrored validation mode
        X_L_mirrored = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)
        validation_modes.append(
            {
                "name": "mirrored_left_original_right",
                "X": np.concatenate((X_L_mirrored, X_R), axis=0),
                "Y": np.concatenate((Y_L, Y_R), axis=0),
            }
        )

        if mirror_enabled:
            validation_modes.extend(
                [
                    {"name": "original_left", "X": X_L, "Y": Y_L},
                    {"name": "original_right", "X": X_R, "Y": Y_R},
                ]
            )

        if rotation_enabled:
            # Apply rotation to 50% of samples
            random_indices = np.random.choice(len(X_L), size=int(len(X_L) * 0.5), replace=False)
            X_L_rotated = np.copy(X_L)
            X_R_rotated = np.copy(X_R)
            for i in random_indices:
                X_L_rotated[i], Y_L[i] = planar_rotation(X_L[i], Y_L[i])
                X_R_rotated[i], Y_R[i] = planar_rotation(X_R[i], Y_R[i])

            X_L_rotated_mirrored = np.array([hand_mirroring(sample) for sample in X_L_rotated], dtype=object)
            validation_modes.append(
                {
                    "name": "rotated_mirrored_left_original_right",
                    "X": np.concatenate((X_L_rotated_mirrored, X_R_rotated), axis=0),
                    "Y": np.concatenate((Y_L, Y_R), axis=0),
                }
            )

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
        stats_file_npy = os.path.join(result_dir, "validation_stats.npy")
        stats_file_json = os.path.join(result_dir, "validation_stats.json")
        stats_file_mat = os.path.join(result_dir, "validation_stats.mat")

        np.save(stats_file_npy, all_stats)
        with open(stats_file_json, "w") as f_json:
            json.dump(all_stats, f_json, indent=4)
        matlab_stats = {}
        for mode, stats in all_stats.items():
            matlab_stats[mode] = convert_for_matlab(stats)
        savemat(stats_file_mat, {f"{version}": matlab_stats})
        logger.info(f"\nAll validation statistics saved to {stats_file_npy}, {stats_file_json}, {stats_file_mat}")
