#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Validation Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Edited      : 2025-04-14
Description : This script validates MSTCN models on IMU datasets
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
from torch.utils.data import DataLoader

from components.model_mstcn import MSTCN
from components.model_tcn import TCN
from components.model_cnnlstm import CNNLSTM
from components.post_processing import post_process_predictions
from components.evaluation import segment_evaluation
from components.datasets import IMUDataset

# --- Configurations ---
NUM_WORKERS = 4
THRESHOLD_LIST = [0.1, 0.25, 0.5, 0.75]
DEBUG_PLOT = False
SAVE_LOG = False


if __name__ == "__main__":
    result_root = "result"
    # versions = ["202504211732", "202504211918"]  # Uncomment to manually specify versions
    versions = [d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d))]
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

        DATASET = "Clemson"  # or "Oreba"
        NUM_CLASSES = config_info["num_classes"]
        MODEL = config_info["model"]
        INPUT_DIM = config_info["input_dim"]
        SAMPLING_FREQ = 15
        WINDOW_SIZE = config_info["window_size"]
        BATCH_SIZE = config_info["batch_size"]
        validate_folds = config_info.get("validate_folds")
        mirror_enabled = config_info.get("augmentation_hand_mirroring", False) or config_info.get(
            "dataset_mirroring", False
        )
        rotation_enabled = config_info.get("augmentation_planar_rotation", False)
        hand = config_info.get("hand", "BOTH")

        if validate_folds is None:
            logger.error("No 'validate_folds' found in the configuration file.")
            continue

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"\nUsing device: {device}")

        if DATASET.startswith("Oreba"):
            DATA_DIR = "./dataset/Oreba"
        elif DATASET.startswith("Clemson"):
            DATA_DIR = "./dataset/Clemson"
        else:
            logger.error(f"Invalid dataset: {DATASET}")
            continue

        X = np.array(pickle.load(open(os.path.join(DATA_DIR, "X.pkl"), "rb")), dtype=object)
        Y = np.array(pickle.load(open(os.path.join(DATA_DIR, "Y.pkl"), "rb")), dtype=object)

        dataset = IMUDataset(X, Y, sequence_length=WINDOW_SIZE)
        validate_stats = []

        for fold, validate_subjects in enumerate(tqdm(validate_folds, desc=f"K-Fold", leave=True)):
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

            validate_loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
            )

            all_predictions, all_labels = [], []
            with torch.no_grad():
                for batch_x, batch_y in tqdm(validate_loader, desc=f"Fold {fold+1}", leave=False):
                    batch_x = batch_x.permute(0, 2, 1).to(device)
                    # Swap the 0-2 channels with 3-5 channels
                    # batch_x[:, 0:3, :], batch_x[:, 3:6, :] = batch_x[:, 3:6, :].clone(), batch_x[:, 0:3, :].clone()
                    outputs = model(batch_x)
                    logits = outputs[:, -1, :, :] if outputs.ndim == 4 else outputs
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_predictions.extend(preds.view(-1).cpu().numpy())
                    all_labels.extend(batch_y.view(-1).cpu().numpy())

            preds_array = np.array(all_predictions)
            # swap label 1 and 2 in predictions
            # preds_array = np.where(preds_array == 1, -1, preds_array)  # Temporarily set label 1 to -1
            # preds_array = np.where(preds_array == 2, 1, preds_array)  # Change label 2 to 1
            # preds_array = np.where(preds_array == -1, 2, preds_array)  # Change temporary -1 to 2
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
            validate_stats.append(fold_stat)

            # Save all statistics
            stats_file_npy = os.path.join(result_dir, f"validation_stats_{DATASET}.npy")
            stats_file_json = os.path.join(result_dir, f"validation_stats_{DATASET}.json")

            np.save(stats_file_npy, validate_stats)
            with open(stats_file_json, "w") as f_json:
                json.dump(validate_stats, f_json, indent=4)
