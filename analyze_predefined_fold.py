#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================================
Event-based Segmentation GAN Training Script
------------------------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-02
Description : This script analyzes the fold-wise and global label distributions in the FD-I
              dataset used for event-based IMU segmentation. It loads preprocessed label data
              for both left and right hands, applies predefined validation folds, and computes
              detailed statistics on sample counts and class distributions across folds.
================================================================================================
"""


import os
import pickle
import numpy as np
from components.datasets import load_predefined_validate_folds

# Dataset Configuration
DATASET = "FDI"
NUM_CLASSES = 3
DATA_DIR = "./dataset/FD/FD-I"
TASK = "multiclass"


def main():
    # Load dataset
    with open(os.path.join(DATA_DIR, "Y_L.pkl"), "rb") as f:
        Y_L = np.array(pickle.load(f), dtype=object)
    with open(os.path.join(DATA_DIR, "Y_R.pkl"), "rb") as f:
        Y_R = np.array(pickle.load(f), dtype=object)

    # Create subject indices for left and right separately
    subject_indices_L = np.array([i for i in range(len(Y_L)) for _ in range(len(Y_L[i]))])
    subject_indices_R = np.array([i for i in range(len(Y_R)) for _ in range(len(Y_R[i]))])
    all_labels_L = np.concatenate(Y_L)
    all_labels_R = np.concatenate(Y_R)

    # Load predefined folds
    validate_folds = load_predefined_validate_folds()

    # Calculate global statistics
    total_subjects = len(Y_L)  # Assuming same number of subjects in Y_L and Y_R
    total_samples_L = len(all_labels_L)
    total_samples_R = len(all_labels_R)
    total_samples = total_samples_L + total_samples_R

    global_counts_L = {label: np.sum(all_labels_L == label) for label in range(NUM_CLASSES)}
    global_counts_R = {label: np.sum(all_labels_R == label) for label in range(NUM_CLASSES)}
    global_counts = {label: global_counts_L[label] + global_counts_R[label] for label in range(NUM_CLASSES)}

    print("\n" + "=" * 80)
    print("FD-I Dataset Fold Statistics (Separate Left/Right)".center(80))
    print("=" * 80)
    print(f"\nTotal subjects in dataset: {total_subjects}")
    print(f"Total samples (windows) in dataset: {total_samples:,} (L: {total_samples_L:,}, R: {total_samples_R:,})")
    print("\nGlobal label counts:")
    for label in range(NUM_CLASSES):
        print(
            f"  Label {label}: {global_counts[label]:,} samples "
            f"(L: {global_counts_L[label]:,}, R: {global_counts_R[label]:,})"
        )

    # Analyze each fold
    for fold_idx, validate_subjects in enumerate(validate_folds, 1):
        print("\n" + "-" * 80)
        print(f" Fold {fold_idx} Statistics ".center(80, "-"))
        print("-" * 80)

        # Get validation set indices for left and right
        val_mask_L = np.isin(subject_indices_L, validate_subjects)
        val_mask_R = np.isin(subject_indices_R, validate_subjects)

        val_labels_L = all_labels_L[val_mask_L]
        val_labels_R = all_labels_R[val_mask_R]

        fold_samples_L = len(val_labels_L)
        fold_samples_R = len(val_labels_R)
        fold_samples = fold_samples_L + fold_samples_R

        # Calculate fold counts
        fold_counts_L = {label: np.sum(val_labels_L == label) for label in range(NUM_CLASSES)}
        fold_counts_R = {label: np.sum(val_labels_R == label) for label in range(NUM_CLASSES)}
        fold_counts = {label: fold_counts_L[label] + fold_counts_R[label] for label in range(NUM_CLASSES)}

        print(f"\nSubjects in validation: {len(validate_subjects)}")
        print(f"Subject IDs: {sorted(validate_subjects)}")
        print(f"\nSamples in fold: {fold_samples:,} (L: {fold_samples_L:,}, R: {fold_samples_R:,})")
        print(f"  Proportion of total samples: {fold_samples/total_samples:.1%}")
        print(f"  Left proportion: {fold_samples_L/total_samples_L:.1%}")
        print(f"  Right proportion: {fold_samples_R/total_samples_R:.1%}")

        print("\nLabel distribution in fold:")
        for label in range(NUM_CLASSES):
            count = fold_counts[label]
            count_L = fold_counts_L[label]
            count_R = fold_counts_R[label]
            global_count = global_counts[label]

            fold_proportion = count / fold_samples if fold_samples > 0 else 0
            global_proportion = count / global_count if global_count > 0 else 0
            left_proportion = count_L / count if count > 0 else 0
            right_proportion = count_R / count if count > 0 else 0

            print(
                f"  Label {label}: {count:,} samples "
                f"(L: {count_L:,}, R: {count_R:,})\n"
                f"    {fold_proportion:.1%} of fold, "
                f"{global_proportion:.1%} of global {label}s\n"
                f"    Left proportion: {left_proportion:.1%}\n"
                f"    Right proportion: {right_proportion:.1%}"
            )


if __name__ == "__main__":
    main()
