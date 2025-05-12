#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Dataset Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-12
Description : This script defines multiple dataset classes for loading and preprocessing
              IMU data, including support for segment-based, balanced, sliding window,
              and unlabeled variants. It also includes helper functions for generating
              subject-balanced cross-validation folds and loading predefined test splits.
===============================================================================
"""


import os
from fractions import Fraction

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample_poly


class IMUDataset(Dataset):
    def __init__(
        self,
        X,
        Y,
        sequence_length=128,
        downsample_factor=None,
        stride=None,
        selected_channels=None,
    ):
        """
        Initialize the IMUDataset.

        Parameters:
            X (list of np.ndarray): IMU data for each subject, each array has shape (N, C)
            Y (list of np.ndarray): Label arrays for each subject, each array has shape (N,)
            sequence_length (int): Length of each sequence segment.
            downsample_factor (float): Downsampling factor (can be non-integer).
            apply_antialias (bool): Whether to apply anti-aliasing filter (resample_poly does it internally).
            stride (int): Step size for the sliding window.
            selected_channels (list or None): Indices of channels to select; None → all channels.
        """
        self.data = []
        self.labels = []
        self.subject_indices = []
        self.sequence_length = sequence_length
        self.downsample_factor = downsample_factor if downsample_factor is not None else 1
        self.stride = stride if stride is not None else sequence_length

        # infer channels
        if selected_channels is None:
            if len(X) == 0 or X[0].ndim < 2:
                raise ValueError("Cannot infer number of channels from X")
            n_channels = X[0].shape[1]
            self.selected_channels = list(range(n_channels))
        else:
            self.selected_channels = selected_channels

        if downsample_factor <= 1:
            raise ValueError("downsample_factor must be > 1.")

        for subject_idx, (imu_data, labels) in enumerate(zip(X, Y)):
            # --- Downsample signal via resample_poly ---
            imu_data_ds = self.downsample(imu_data, downsample_factor)
            # align labels by nearest-neighbor on the new time grid
            old_len = labels.shape[0]
            new_len = imu_data_ds.shape[0]
            orig_pos = np.linspace(0, old_len - 1, new_len)
            idx_nn = np.round(orig_pos).astype(int)
            labels_ds = labels[idx_nn]

            # --- Normalize and channel-select ---
            imu_data_ds = self.normalize(imu_data_ds)
            imu_data_ds = imu_data_ds[:, self.selected_channels]

            num_samples = labels_ds.shape[0]

            # --- sliding windows ---
            for i in range(0, num_samples - sequence_length + 1, self.stride):
                # The tail will be dropped if not enough samples
                seg_x = imu_data_ds[i : i + sequence_length]
                seg_y = labels_ds[i : i + sequence_length]
                self.data.append(seg_x)
                self.labels.append(seg_y)
                self.subject_indices.append(subject_idx)

    def downsample(self, data, factor):
        """
        Downsample by arbitrary factor >1 using polyphase filtering.
        factor = old_rate / new_rate.
        """
        # convert to Fraction so we get integer up/down
        frac = Fraction(factor).limit_denominator()
        up = frac.denominator
        down = frac.numerator
        # resample_poly applies its own anti-alias filter
        return resample_poly(data, up, down, axis=0)

    def normalize(self, data):
        """
        Perform Z-score normalization on the input data.
        """
        # Compute the mean of each feature (column-wise)
        mean = np.mean(data, axis=0, keepdims=True)
        # Compute the standard deviation of each feature (column-wise)
        std = np.std(data, axis=0, keepdims=True)
        # Normalize the data using Z-score normalization
        return (data - mean) / (std + 1e-5)  # Add a small value to avoid division by zero

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # long for CE loss
        return x, y


def create_balanced_subject_folds(dataset, num_folds=7):
    """
    Create balanced folds where each fold has approximately the same number of samples.

    Args:
        dataset: IMUDataset object with subject_indices attribute
        num_folds: Number of folds to create

    Returns:
        List of lists, where each inner list contains subject IDs for that fold
    """
    # Count samples per subject
    subject_counts = {}
    unique_subjects = sorted(set(dataset.subject_indices))

    for subject in unique_subjects:
        subject_count = sum(1 for idx in dataset.subject_indices if idx == subject)
        subject_counts[subject] = subject_count

    # Sort subjects by sample count (descending)
    sorted_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)

    # Initialize folds with empty lists and zero counts
    folds = [[] for _ in range(num_folds)]
    fold_sample_counts = [0] * num_folds

    # Distribute subjects to folds using greedy approach
    for subject, count in sorted_subjects:
        # Find the fold with the fewest samples
        min_fold_idx = fold_sample_counts.index(min(fold_sample_counts))

        # Add the subject to this fold
        folds[min_fold_idx].append(subject)
        fold_sample_counts[min_fold_idx] += count

    return folds


def load_predefined_validate_folds(num_folds=7, base_dir="dataset/FD/FD1_7_fold_id_list"):
    """
    Load predefined test folds from .npy files.

    Each fold is stored in a subdirectory (named "0", "1", ..., etc.) under base_dir.
    The test split is stored in a file named "test.npy" in each fold directory.

    Args:
        num_folds (int): Total number of folds.
        base_dir (str): Directory containing the fold subdirectories.

    Returns:
        List[List[int]]: A list of test folds, where each inner list contains subject IDs for that fold.
    """
    test_folds = []
    for i in range(num_folds):
        fold_dir = os.path.join(base_dir, str(i))
        test_path = os.path.join(fold_dir, "test.npy")
        test_subjects = np.load(test_path)
        # Convert to a regular Python list if needed
        test_folds.append(test_subjects.tolist())
    return test_folds
