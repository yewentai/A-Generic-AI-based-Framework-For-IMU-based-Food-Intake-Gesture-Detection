#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
MSTCN IMU Validation Script (DX/FD Datasets)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Edited      : 2025-05-14
Description : This script validates MSTCN models on DX/FD IMU datasets with:
              1. Original, left, and right hand-based validation modes
              2. Optional data augmentation: hand mirroring and planar rotation
              3. Compatibility with CNN-LSTM, TCN, and MSTCN models
              4. Sample-wise and optional segment-wise evaluation
              5. Output of per-fold metrics and optional threshold-based F1 analysis
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
from components.models.tcn import TCN, MSTCN
from components.models.cnnlstm import CNNLSTM
from components.models.accnet import AccNet
from components.models.resnet_bilstm import ResNet, BiLSTMHead, ResNetBiLSTM
from components.models.resnet import ResNetEncoder
from components.post_processing import post_process_predictions
from components.evaluation import segment_evaluation
from components.datasets import IMUDataset
from components.pre_processing import hand_mirroring, planar_rotation, axis_permutation, spatial_orientation

# --- Configurations ---
NUM_WORKERS = 4
SEGMENT_VALIDATION = True
POST_PROCESSING_SMOOTHING = False
THRESHOLD_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEBUG_PLOT = False


if __name__ == "__main__":
    result_root = "results/DXI"
    versions = ["DXI_ResNetBiLSTM_FTFull", "DXI_ResNetBiLSTM_FTHead"]  # Uncomment to manually specify versions
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
        selected_channels = config_info["selected_channels"]
        sampling_freq = config_info["sampling_freq"]
        window_samples = config_info["window_samples"]
        batch_size = config_info["batch_size"]
        validate_folds = config_info.get("validate_folds")
        mirroring_enabled = config_info.get("dataset_mirroring", False)
        planar_rotation_enabled = config_info.get("augmentation_planar_rotation", False)
        axis_permutation_enabled = config_info.get("axis_permutation", False)
        spatial_orientation_enabled = config_info.get("spatial_orientation", False)

        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"\nUsing device: {device}")

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

        # Load pretrained model
        pretrained_ckpt = "mtl_best.mdl"

        # Load dataset
        validation_modes = []
        X_L = np.array(pickle.load(open(os.path.join(DATA_DIR, "X_L.pkl"), "rb")), dtype=object)
        Y_L = np.array(pickle.load(open(os.path.join(DATA_DIR, "Y_L.pkl"), "rb")), dtype=object)
        X_R = np.array(pickle.load(open(os.path.join(DATA_DIR, "X_R.pkl"), "rb")), dtype=object)
        Y_R = np.array(pickle.load(open(os.path.join(DATA_DIR, "Y_R.pkl"), "rb")), dtype=object)
        X = np.array([np.concatenate([X_L[i], X_R[i]], axis=0) for i in range(len(X_L))], dtype=object)
        Y = np.array([np.concatenate([Y_L[i], Y_R[i]], axis=0) for i in range(len(Y_L))], dtype=object)
        validation_modes.append(
            {
                "name": "original",
                "X": X,
                "Y": Y,
            }
        )
        if mirroring_enabled:
            X_L_mirrored = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)
            validation_modes.append(
                {
                    "name": "mirrored_left_original_right",
                    "X": np.array(
                        [np.concatenate([X_L_mirrored[i], X_R[i]], axis=0) for i in range(len(X_L))], dtype=object
                    ),
                    "Y": np.array([np.concatenate([Y_L[i], Y_R[i]], axis=0) for i in range(len(Y_L))], dtype=object),
                }
            )
        if planar_rotation_enabled:
            # Apply rotation to 50% of samples
            random_indices = np.random.choice(len(X), size=int(len(X) * 0.5), replace=False)
            X_rotated = np.copy(X)
            for i in random_indices:
                X_rotated[i], Y[i] = planar_rotation(X[i], Y[i])
            validation_modes.append(
                {
                    "name": "planar_rotated",
                    "X": X_rotated,
                    "Y": Y,
                }
            )
        if axis_permutation_enabled:
            # Apply axis permutation to 50% of samples
            random_indices = np.random.choice(len(X), size=int(len(X) * 0.5), replace=False)
            X_permuted = np.copy(X)
            for i in random_indices:
                X_permuted[i], Y[i] = axis_permutation(X[i], Y[i])
            validation_modes.append(
                {
                    "name": "axis_permuted",
                    "X": X_permuted,
                    "Y": Y,
                }
            )
        if spatial_orientation_enabled:
            # Apply spatial orientation to 50% of samples
            random_indices = np.random.choice(len(X), size=int(len(X) * 0.5), replace=False)
            X_oriented = np.copy(X)
            for i in random_indices:
                X_oriented[i], Y[i] = spatial_orientation(X[i], Y[i])
            validation_modes.append(
                {
                    "name": "spatial_oriented",
                    "X": X_oriented,
                    "Y": Y,
                }
            )

        all_stats = {}
        for mode in validation_modes:
            logger.info(f"\n--- Validating {mode['name']} ---")

            dataset = IMUDataset(
                mode["X"],
                mode["Y"],
                sequence_length=window_samples,
                downsample_factor=downsample_factor,
                selected_channels=selected_channels,
            )
            mode_stats = []

            for fold, validate_subjects in enumerate(tqdm(validate_folds, desc=f"K-Fold ({mode['name']})", leave=True)):
                checkpoint_path = os.path.join(result_dir, "checkpoint", f"best_model_fold{fold+1}.pth")
                if not os.path.exists(checkpoint_path):
                    logger.warning(f"Checkpoint not found for fold {fold+1}, skipping...")
                    continue

                if model_name == "TCN":
                    model = TCN(
                        num_layers=config_info["num_layers"],
                        num_classes=num_classes,
                        input_dim=input_dim,
                        num_filters=config_info["num_filters"],
                        kernel_size=config_info["kernel_size"],
                        dropout=config_info["dropout"],
                    ).to(device)
                elif model_name == "MSTCN":
                    model = MSTCN(
                        num_stages=config_info["num_stages"],
                        num_layers=config_info["num_layers"],
                        num_classes=num_classes,
                        input_dim=input_dim,
                        num_filters=config_info["num_filters"],
                        kernel_size=config_info["kernel_size"],
                        dropout=config_info["dropout"],
                    ).to(device)
                elif model_name == "CNN_LSTM":
                    model = CNNLSTM(
                        input_channels=input_dim,
                        conv_filters=config_info["conv_filters"],
                        lstm_hidden=config_info["lstm_hidden"],
                        num_classes=num_classes,
                    ).to(device)
                elif model_name == "AccNet":
                    model = AccNet(
                        num_classes=num_classes,
                        input_channels=input_dim,
                        conv_filters=config_info["conv_filters"],
                        kernel_size=config_info["kernel_size"],
                    ).to(device)
                elif model_name == "ResNetBiLSTM":
                    encoder = ResNet(
                        in_channels=input_dim,
                    ).to(device)

                    # figure out the encoder's actual output size
                    with torch.no_grad():
                        dummy = torch.zeros(1, input_dim, window_samples, device=device)
                        flat_feats = encoder(dummy)
                        feature_dim = flat_feats.shape[1]  # e.g. 4096

                    # now build the head with the correct feature_dim
                    seq_head = BiLSTMHead(
                        feature_dim=feature_dim,
                        seq_length=window_samples,
                        num_classes=num_classes,
                        hidden_dim=config_info["lstm_hidden"],
                    ).to(device)
                    model = ResNetBiLSTM(encoder, seq_head).to(device)
                elif model_name in ["ResNetBiLSTM_FTFull", "ResNetBiLSTM_FTHead"]:
                    encoder = ResNetEncoder(
                        weight_path=pretrained_ckpt,
                        n_channels=input_dim,
                        class_num=num_classes,
                        my_device=device,
                        freeze_encoder=True,
                    ).to(device)
                    feature_dim = encoder.out_features
                    seq_labeler = BiLSTMHead(
                        feature_dim=feature_dim, seq_length=window_samples, num_classes=num_classes, hidden_dim=128
                    ).to(device)
                    model = ResNetBiLSTM(encoder, seq_labeler).to(device)

                else:
                    logger.error(f"Invalid model: {model_name}")
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
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                )

                all_predictions, all_labels = [], []

                with torch.no_grad():
                    for batch_x, batch_y in tqdm(validate_loader, desc=f"Fold {fold+1}", leave=False):
                        # [B, L, C_in] → [B, C_in, L]
                        batch_x = batch_x.permute(0, 2, 1).to(device)
                        batch_y = batch_y.to(device)
                        outputs_list = model(batch_x)
                        # pick the last stage's logits
                        if isinstance(outputs_list, list):
                            logits = outputs_list[-1]  # -> [B, num_classes, L]
                        else:
                            logits = outputs_list  # for single-stage TCN
                        # softmax + argmax over class-dim
                        probs = F.softmax(logits, dim=1)  # [B, C, L]
                        preds = torch.argmax(probs, dim=1)  # [B, L]
                        # flatten and store
                        all_predictions.extend(preds.reshape(-1).cpu().numpy())
                        all_labels.extend(batch_y.reshape(-1).cpu().numpy())

                preds_array = np.array(all_predictions)
                labels_array = np.array(all_labels)

                unique_labels, counts = np.unique(labels_array, return_counts=True)
                label_distribution = {int(l): int(c) for l, c in zip(unique_labels, counts)}
                positive_counts = {l: c for l, c in label_distribution.items() if l > 0}
                total_positive = sum(positive_counts.values())
                label_weight = {l: c / total_positive for l, c in positive_counts.items()}

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

                weighted_f1_sample = sum(metrics_sample[str(l)]["f1"] * label_weight[l] for l in range(1, num_classes))
                metrics_sample["weighted_f1"] = float(weighted_f1_sample)

                metrics_segment_all = None
                if SEGMENT_VALIDATION:
                    if POST_PROCESSING_SMOOTHING:
                        preds_array = post_process_predictions(preds_array, sampling_freq)
                    metrics_segment_all = {}
                    for threshold in THRESHOLD_LIST:
                        metrics_segment = {}
                        for label in range(1, num_classes):
                            fn, fp, tp = segment_evaluation(
                                preds_array,
                                labels_array,
                                class_label=label,
                                threshold=threshold,
                                debug_plot=DEBUG_PLOT,
                            )
                            denom = 2 * tp + fp + fn
                            f1 = (2 * tp / denom) if denom > 0 else 0.0
                            metrics_segment[str(label)] = {"fn": int(fn), "fp": int(fp), "tp": int(tp), "f1": float(f1)}
                        weighted_f1 = sum(
                            metrics_segment[str(l)]["f1"] * label_weight[l] for l in range(1, num_classes)
                        )
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
