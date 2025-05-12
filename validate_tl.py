#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Fine-Tuned Classifier Validation Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-12
Description : This script validates a fine-tuned IMU classifier by recreating
              the exact model architecture used during fine-tuning, loading
              the saved state dictionary, and evaluating the model on a test
              dataset. It calculates accuracy, generates a classification
              report, and saves the validation results.
===============================================================================
"""


# Standard library imports
import os
import json
import pickle

# Third-party library imports
import numpy as np
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# Local imports
from components.models.resnet import ResNetEncoder
from components.models.head import MLPClassifier, ResNetMLP
from components.datasets import IMUDatasetN21
from components.pre_processing import hand_mirroring, planar_rotation
from components.models.head import BiLSTMHead, ResNetBiLSTM

# --- Configurations ---
NUM_WORKERS = 4
SEGMENT_VALIDATION = False
if SEGMENT_VALIDATION:
    THRESHOLD_LIST = [0.1, 0.25, 0.5, 0.75]
DEBUG_PLOT = False


if __name__ == "__main__":
    result_root = "results"
    versions = [
        "DXI_ResNetBiLSTM_3",
    ]  # Uncomment to manually specify versions
    # versions = [d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d))]
    versions.sort()

    for version in versions:
        # Set up logging
        logger = logging.getLogger(f"validation_{version}")
        logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(stream_handler)
        logger.info(f"\n=== Validating Version: {version} ===")

        # Load configuration
        result_dir = os.path.join(result_root, version)
        if not os.path.exists(result_dir):
            logger.warning(f"Result directory does not exist: {result_dir}.")
            continue
        config_file = os.path.join(result_dir, "training_config.json")
        if not os.path.exists(config_file):
            logger.warning(f"Configuration file not found for version {version}, skipping...")
            continue
        with open(config_file, "r") as f:
            config_info = json.load(f)
        validate_dataset = config_info["dataset"]  # Options: "ORIGINAL", "FDI", "FDII", "FDIII", "DXI", "DXII", "OREBA"
        num_classes = config_info["num_classes"]
        model_name = config_info["model"]
        input_dim = config_info["input_dim"]
        downsample_factor = config_info["downsample_factor"]
        sampling_freq = config_info["sampling_freq"]
        window_size = config_info["window_size"]
        batch_size = config_info["batch_size"]
        flag_augment_hand_mirroringing = config_info.get("augmentation_hand_mirroring", False)
        flag_dataset_mirroring = config_info.get("dataset_mirroring", False)
        flag_dataset_mirroring_add = config_info.get("dataset_mirroring_add", False)
        validate_folds = config_info.get("validate_folds")
        rotation_enabled = config_info.get("augmentation_planar_rotation", False)
        hand_separation = config_info.get("hand_separation", False)
        if flag_augment_hand_mirroringing or flag_dataset_mirroring or flag_dataset_mirroring_add:
            mirroring_enabled = True
        else:
            mirroring_enabled = False

        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"\nUsing device: {device}")

        # Load pretrained model
        pretrained_ckpt = "mtl_best.mdl"

        # Set up dataset directory
        if validate_dataset.startswith("DX"):
            sub_version = validate_dataset.replace("DX", "").upper() or "I"
            DATA_DIR = f"./dataset/DX/DX-{sub_version}"
        elif validate_dataset.startswith("FD"):
            sub_version = validate_dataset.replace("FD", "").upper() or "I"
            DATA_DIR = f"./dataset/FD/FD-{sub_version}"
        elif validate_dataset.startswith("OREBA"):
            DATA_DIR = f"./dataset/Oreba"
        else:
            logger.error(f"Invalid dataset: {validate_dataset}")
            continue

        # Load dataset
        validation_modes = []
        if hand_separation:
            X_L = np.array(pickle.load(open(os.path.join(DATA_DIR, "X_L.pkl"), "rb")), dtype=object)
            Y_L = np.array(pickle.load(open(os.path.join(DATA_DIR, "Y_L.pkl"), "rb")), dtype=object)
            X_R = np.array(pickle.load(open(os.path.join(DATA_DIR, "X_R.pkl"), "rb")), dtype=object)
            Y_R = np.array(pickle.load(open(os.path.join(DATA_DIR, "Y_R.pkl"), "rb")), dtype=object)
            X = np.array([np.concatenate([X_L[i], X_R[i]], axis=0) for i in range(len(X_L))], dtype=object)
            Y = np.array([np.concatenate([Y_L[i], Y_R[i]], axis=0) for i in range(len(Y_L))], dtype=object)
            validation_modes.extend(
                [
                    {"name": "original", "X": X, "Y": Y},
                    {"name": "left", "X": X_L, "Y": Y_L},
                    {"name": "right", "X": X_R, "Y": Y_R},
                ]
            )

            if mirroring_enabled:
                X_L_mirrored = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)
                validation_modes.append(
                    {
                        "name": "mirrored_left_original_right",
                        "X": np.array(
                            [np.concatenate([X_L_mirrored[i], X_R[i]], axis=0) for i in range(len(X_L))], dtype=object
                        ),
                        "Y": np.array(
                            [np.concatenate([Y_L[i], Y_R[i]], axis=0) for i in range(len(Y_L))], dtype=object
                        ),
                    }
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
                        "X": np.array(
                            [
                                np.concatenate([X_L_rotated_mirrored[i], X_R_rotated[i]], axis=0)
                                for i in range(len(X_L))
                            ],
                            dtype=object,
                        ),
                        "Y": np.array(
                            [np.concatenate([Y_L[i], Y_R[i]], axis=0) for i in range(len(Y_L))], dtype=object
                        ),
                    }
                )
        else:
            X = np.array(pickle.load(open(os.path.join(DATA_DIR, "X.pkl"), "rb")), dtype=object)
            Y = np.array(pickle.load(open(os.path.join(DATA_DIR, "Y.pkl"), "rb")), dtype=object)
            validation_modes.append({"name": "original", "X": X, "Y": Y})

        all_stats = {}
        for mode in validation_modes:
            logger.info(f"\n--- Validating {mode['name']} ---")

            dataset = IMUDatasetN21(
                mode["X"], mode["Y"], sequence_length=window_size, downsample_factor=downsample_factor
            )
            mode_stats = []

            for fold, validate_subjects in enumerate(tqdm(validate_folds, desc=f"K-Fold ({mode['name']})", leave=True)):
                checkpoint_path = os.path.join(result_dir, "checkpoint", f"best_model_fold{fold+1}.pth")
                if not os.path.exists(checkpoint_path):
                    logger.warning(f"Checkpoint not found for fold {fold+1}, skipping...")
                    continue
                encoder = ResNetEncoder(
                    weight_path=pretrained_ckpt,
                    n_channels=input_dim,
                    class_num=num_classes,
                    my_device=device,
                    freeze_encoder=True,
                ).to(device)
                feature_dim = encoder.out_features
                if model_name == "ResNetMLP":
                    classifier = MLPClassifier(feature_dim=feature_dim, num_classes=num_classes)
                    model = ResNetMLP(encoder, classifier).to(device)
                elif model_name == "ResNetBiLSTM":
                    seq_labeler = BiLSTMHead(
                        feature_dim=feature_dim, seq_length=window_size, num_classes=num_classes, hidden_dim=128
                    ).to(device)
                    model = ResNetBiLSTM(encoder, seq_labeler).to(device)
                else:
                    logger.error(f"Invalid model: {model_name}")
                    continue

                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()

                validate_indices = [
                    i for i, subject in enumerate(dataset.subject_indices) if subject in validate_subjects
                ]
                validate_loader = DataLoader(
                    Subset(dataset, validate_indices),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                )

                all_predictions, all_labels = [], []

                model.eval()
                all_predictions = []
                all_labels = []

                with torch.no_grad():
                    for batch_x, batch_y in tqdm(validate_loader, desc=f"Fold {fold+1}", leave=False):
                        # [B, L, C_in] → [B, C_in, L]
                        batch_x = batch_x.permute(0, 2, 1).to(device)
                        # [B, L] → [B, L] on device, long for indexing
                        batch_y = batch_y.long().to(device)

                        # Forward pass: get per‐frame logits [B, seq_length, num_classes]
                        logits = model(batch_x)

                        # Softmax over the class dimension (last dim) → [B, L, C]
                        probs = F.softmax(logits, dim=2)

                        # Argmax per frame → [B, L]
                        preds_seq = torch.argmax(probs, dim=2)

                        # Flatten everything to 1D: [B*L]
                        preds_flat = preds_seq.reshape(-1).cpu().numpy()
                        true_flat = batch_y.reshape(-1).cpu().numpy()

                        all_predictions.extend(preds_flat.tolist())
                        all_labels.extend(true_flat.tolist())

                # Convert to numpy arrays
                preds_array = np.array(all_predictions)
                labels_array = np.array(all_labels)

                # Compute label distribution
                unique_labels, counts = np.unique(labels_array, return_counts=True)
                label_distribution = {int(l): int(c) for l, c in zip(unique_labels, counts)}

                # Compute weights for positive classes (1..num_classes-1)
                positive_counts = {l: c for l, c in label_distribution.items() if l > 0}
                total_positive = sum(positive_counts.values())
                if total_positive == 0:
                    # No positives: uniform weights
                    label_weight = {l: 1.0 / (num_classes - 1) for l in range(1, num_classes)}
                else:
                    label_weight = {l: positive_counts.get(l, 0) / total_positive for l in range(1, num_classes)}

                # Compute per-class confusion & F1
                metrics_sample = {}
                for label in range(1, num_classes):
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

                # now weighted F1 without KeyError
                weighted_f1_sample = sum(metrics_sample[str(l)]["f1"] * label_weight[l] for l in range(1, num_classes))
                metrics_sample["weighted_f1"] = float(weighted_f1_sample)

                fold_stat = {
                    "fold": fold + 1,
                    "metrics_sample": metrics_sample,
                    "label_distribution": label_distribution,
                }
                mode_stats.append(fold_stat)

            all_stats[mode["name"]] = mode_stats

        # Save all statistics
        stats_file_npy = os.path.join(result_dir, f"validation_stats_{validate_dataset.lower()}.npy")
        stats_file_json = os.path.join(result_dir, f"validation_stats_{validate_dataset.lower()}.json")
        np.save(stats_file_npy, all_stats)
        with open(stats_file_json, "w") as f_json:
            json.dump(all_stats, f_json, indent=4)
        logger.info(f"\nAll validation statistics saved to {stats_file_npy}, {stats_file_json}")

        # Save configuration
        config_info = {
            "flag_segment_validation": SEGMENT_VALIDATION,
            "threshold_list": THRESHOLD_LIST if SEGMENT_VALIDATION else None,
        }
        config_save_path = os.path.join(result_dir, "validation_config.json")
        with open(config_save_path, "w") as f:
            json.dump(config_info, f, indent=4)
        logger.info(f"\nValidation configuration saved to {config_save_path}")
