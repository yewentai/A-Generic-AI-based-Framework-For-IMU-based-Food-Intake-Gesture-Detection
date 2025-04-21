#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Dataset Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-15
Description : This script defines the IMUDataset class for loading and preprocessing
              IMU data, as well as functions for creating balanced cross-validation folds.
===============================================================================
"""

import numpy as np
import torch
import scipy.signal as signal
from torch.utils.data import Dataset
import os

from components.pre_processing import (
    hand_mirroring,
    planar_rotation,
    axis_permutation,
    spatial_orientation,
)


class IMUDataset(Dataset):
    def __init__(self, X, Y, sequence_length=128, downsample_factor=4, apply_antialias=True, augmentations=None):
        """
        Initialize the IMUDataset with optional augmentations.

        Parameters:
            X (list of np.ndarray): IMU data for each subject
            Y (list of np.ndarray): Label arrays for each subject
            sequence_length (int): Length of each sequence segment
            downsample_factor (int): Downsampling factor
            apply_antialias (bool): Whether to apply anti-aliasing before downsampling
            augmentations (dict): Dictionary of augmentation configurations:
                {
                    'hand_mirroring': {'probability': float, 'is_additive': bool},
                    'planar_rotation': {'probability': float, 'is_additive': bool},
                    'axis_permutation': {'probability': float, 'is_additive': bool},
                    'spatial_orientation': {'probability': float, 'is_additive': bool}
                }
        """
        self.data = []
        self.labels = []
        self.sequence_length = sequence_length
        self.subject_indices = []
        self.downsample_factor = downsample_factor
        self.augmentations = augmentations if augmentations else {}

        if downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1.")

        # Apply augmentations at the subject level before segmentation
        augmented_X, augmented_Y = self._apply_augmentations(X, Y)

        for subject_idx, (imu_data, labels) in enumerate(zip(augmented_X, augmented_Y)):
            if downsample_factor > 1:
                imu_data = self.downsample(imu_data, downsample_factor, apply_antialias)
                labels = labels[::downsample_factor]

            imu_data = self.normalize(imu_data)
            num_samples = len(labels)

            # Process complete sequence segments
            for i in range(0, num_samples - sequence_length + 1, sequence_length):
                imu_segment = imu_data[i : i + sequence_length]
                label_segment = labels[i : i + sequence_length]
                self.data.append(imu_segment)
                self.labels.append(label_segment)
                self.subject_indices.append(subject_idx)

            # Process remaining segments with zero-padding
            remainder = num_samples % sequence_length
            if remainder > 0:
                start = num_samples - remainder
                imu_segment = imu_data[start:]
                label_segment = labels[start:]
                pad_length = sequence_length - remainder

                imu_segment_padded = np.pad(
                    imu_segment,
                    pad_width=((0, pad_length), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                label_segment_padded = np.pad(
                    label_segment,
                    pad_width=(0, pad_length),
                    mode="constant",
                    constant_values=0,
                )

                self.data.append(imu_segment_padded)
                self.labels.append(label_segment_padded)
                self.subject_indices.append(subject_idx)

    def _apply_augmentations(self, X, Y):
        """Apply configured augmentations at the subject level."""
        if not self.augmentations:
            return X, Y

        augmented_X = []
        augmented_Y = []

        for subject_x, subject_y in zip(X, Y):
            current_x = subject_x.copy()
            current_y = subject_y.copy()

            # Apply each augmentation in sequence
            if "hand_mirroring" in self.augmentations:
                cfg = self.augmentations["hand_mirroring"]
                current_x, current_y = hand_mirroring(
                    np.array([current_x], dtype=object),
                    np.array([current_y], dtype=object),
                    probability=cfg["probability"],
                    is_additive=cfg["is_additive"],
                )
                current_x, current_y = current_x[0], current_y[0]

            if "planar_rotation" in self.augmentations:
                cfg = self.augmentations["planar_rotation"]
                current_x, current_y = planar_rotation(
                    np.array([current_x], dtype=object),
                    np.array([current_y], dtype=object),
                    probability=cfg["probability"],
                    is_additive=cfg["is_additive"],
                )
                current_x, current_y = current_x[0], current_y[0]

            if "axis_permutation" in self.augmentations:
                cfg = self.augmentations["axis_permutation"]
                current_x, current_y = axis_permutation(
                    np.array([current_x], dtype=object),
                    np.array([current_y], dtype=object),
                    probability=cfg["probability"],
                    is_additive=cfg["is_additive"],
                )
                current_x, current_y = current_x[0], current_y[0]

            if "spatial_orientation" in self.augmentations:
                cfg = self.augmentations["spatial_orientation"]
                current_x, current_y = spatial_orientation(
                    np.array([current_x], dtype=object),
                    np.array([current_y], dtype=object),
                    probability=cfg["probability"],
                    is_additive=cfg["is_additive"],
                )
                current_x, current_y = current_x[0], current_y[0]

            augmented_X.append(current_x)
            augmented_Y.append(current_y)

        return augmented_X, augmented_Y

    def downsample(self, data, factor, apply_antialias=True):
        """Downsample with optional anti-aliasing filter."""
        if apply_antialias:
            nyquist = 0.5 * data.shape[0]
            cutoff = (0.5 / factor) * nyquist
            b, a = signal.butter(4, cutoff / nyquist, btype="low", analog=False)
            data = signal.filtfilt(b, a, data, axis=0)
        return data[::factor, :]

    def normalize(self, data):
        """Z-score normalization."""
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        return (data - mean) / (std + 1e-5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class IMUDatasetN21(Dataset):
    def __init__(
        self,
        X,
        Y,
        sequence_length=128,
        stride=None,
        downsample_factor=4,
        apply_antialias=True,
        selected_channels=[0, 1, 2],
    ):
        """
        Initialize the IMUDataset.

        Parameters:
            X (list of np.ndarray): IMU data for each subject, each array has shape (N, 6)
            Y (list of np.ndarray): Label arrays for each subject, each array has shape (N,)
            sequence_length (int): Length of each sequence segment.
            stride (int): Step size for the sliding window. Defaults to sequence_length (non-overlapping).
            downsample_factor (int): Downsampling factor.
            apply_antialias (bool): Whether to apply anti-aliasing filter before downsampling.
            selected_channels (list): Indices of channels to select from the 6 available.
        """
        self.data = []
        self.labels = []
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length
        self.subject_indices = []
        self.downsample_factor = downsample_factor
        self.selected_channels = selected_channels

        if downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1.")

        for subject_idx, (imu_data, labels) in enumerate(zip(X, Y)):
            # Downsample data if needed
            if downsample_factor > 1:
                imu_data = self.downsample(imu_data, downsample_factor, apply_antialias)
                labels = labels[::downsample_factor]

            # Normalize the IMU data (z-score normalization)
            imu_data = self.normalize(imu_data)

            # Select only the desired channels (e.g., accelerometer channels)
            imu_data = imu_data[:, self.selected_channels]

            num_samples = len(labels)

            # Create sequence segments with specified stride
            for i in range(0, num_samples - sequence_length + 1, self.stride):
                imu_segment = imu_data[i : i + sequence_length]
                label_segment = labels[i : i + sequence_length]
                self.data.append(imu_segment)
                self.labels.append(label_segment)
                self.subject_indices.append(subject_idx)

            # For samples that do not fill a complete segment, pad with zeros.
            remainder = (num_samples - sequence_length) % self.stride
            if remainder > 0 and num_samples >= sequence_length:
                start = num_samples - remainder
                imu_segment = imu_data[start:]
                label_segment = labels[start:]
                pad_length = sequence_length - imu_segment.shape[0]

                imu_segment_padded = np.pad(
                    imu_segment,
                    pad_width=((0, pad_length), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                label_segment_padded = np.pad(
                    label_segment,
                    pad_width=(0, pad_length),
                    mode="constant",
                    constant_values=0,
                )

                self.data.append(imu_segment_padded)
                self.labels.append(label_segment_padded)
                self.subject_indices.append(subject_idx)

    def downsample(self, data, factor, apply_antialias=True):
        if apply_antialias:
            nyquist = 0.5 * data.shape[0]
            cutoff = (0.5 / factor) * nyquist
            b, a = signal.butter(4, cutoff / nyquist, btype="low", analog=False)
            data = signal.filtfilt(b, a, data, axis=0)
        return data[::factor, :]

    def normalize(self, data):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        return (data - mean) / (std + 1e-5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class IMUDatasetSegment(Dataset):
    def __init__(self, X, Y, downsample_factor=1, apply_antialias=True, min_length=5):
        """
        Dataset that splits IMU data into variable-length segments where the class label is constant.

        Parameters:
            X (list of np.ndarray): IMU data for each subject, each of shape (N, 6)
            Y (list of np.ndarray): Label arrays for each subject, each of shape (N,)
            downsample_factor (int): Downsampling factor (default 1 = no downsampling)
            apply_antialias (bool): Whether to apply anti-aliasing filter before downsampling.
            min_length (int): Minimum segment length to keep.
        """
        self.data = []
        self.labels = []
        self.subject_indices = []

        for subject_idx, (imu_data, label_seq) in enumerate(zip(X, Y)):
            if downsample_factor > 1:
                imu_data = self.downsample(imu_data, downsample_factor, apply_antialias)
                label_seq = label_seq[::downsample_factor]

            imu_data = self.normalize(imu_data)
            segments = self.segment_by_class(imu_data, label_seq, min_length)

            for imu_segment, label in segments:
                self.data.append(imu_segment)
                self.labels.append(label)
                self.subject_indices.append(subject_idx)

    def segment_by_class(self, imu_data, label_seq, min_length):
        """
        Cut sequences into segments where label is constant.

        Returns:
            List of tuples: [(imu_segment, label), ...]
        """
        segments = []
        current_label = label_seq[0]
        start = 0

        for i in range(1, len(label_seq)):
            if label_seq[i] != current_label:
                if i - start >= min_length:
                    segments.append((imu_data[start:i], current_label))
                start = i
                current_label = label_seq[i]

        # Add last segment
        if len(label_seq) - start >= min_length:
            segments.append((imu_data[start:], current_label))

        return segments

    def downsample(self, data, factor, apply_antialias=True):
        if apply_antialias:
            nyquist = 0.5 * data.shape[0]
            cutoff = (0.5 / factor) * nyquist
            b, a = signal.butter(4, cutoff / nyquist, btype="low")
            data = signal.filtfilt(b, a, data, axis=0)
        return data[::factor, :]

    def normalize(self, data):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        return (data - mean) / (std + 1e-5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imu = torch.tensor(self.data[idx], dtype=torch.float32)  # Shape: (T, 6)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Single class label
        return imu, label


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
