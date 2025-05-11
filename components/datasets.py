#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Dataset Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-11
Description : This script defines multiple dataset classes for loading and preprocessing
              IMU data, including support for segment-based, balanced, sliding window,
              and unlabeled variants. It also includes helper functions for generating
              subject-balanced cross-validation folds and loading predefined test splits.
===============================================================================
"""


import numpy as np
import torch
import scipy.signal as signal
from torch.utils.data import Dataset
import os


class IMUDataset(Dataset):
    def __init__(self, X, Y, sequence_length=128, downsample_factor=1, apply_antialias=True):
        """
        Initialize the IMUDataset.

        Parameters:
            X (list of np.ndarray): IMU data for each subject, each array has shape (N, 6)
            Y (list of np.ndarray): Label arrays for each subject, each array has shape (N,)
            sequence_length (int): Length of each sequence segment.
            downsample_factor (int): Downsampling factor.
            apply_antialias (bool): Whether to apply anti-aliasing filter before downsampling.
        """
        self.data = []
        self.labels = []
        self.sequence_length = sequence_length
        self.subject_indices = []
        self.downsample_factor = downsample_factor

        if downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1.")

        for subject_idx, (imu_data, labels) in enumerate(zip(X, Y)):
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

            # Process remaining segments that are less than sequence_length, zero-padding
            remainder = num_samples % sequence_length
            if remainder > 0:
                start = num_samples - remainder
                imu_segment = imu_data[start:]
                label_segment = labels[start:]
                pad_length = sequence_length - remainder

                # Zero-pad imu_segment in 2D (pad rows, keep 6 features unchanged)
                imu_segment_padded = np.pad(
                    imu_segment,
                    pad_width=((0, pad_length), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                # Zero-pad label_segment in 1D
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
        """
        Apply anti-aliasing filter and downsample the IMU data.

        Parameters:
            data (np.ndarray): IMU data.
            factor (int): Downsampling factor.
            apply_antialias (bool): Whether to apply a low-pass filter before downsampling.

        Returns:
            np.ndarray: Downsampled IMU data.
        """
        if apply_antialias:
            nyquist = 0.5 * data.shape[0]  # Original Nyquist frequency
            cutoff = (0.5 / factor) * nyquist  # Limit to 80% of the new Nyquist frequency
            b, a = signal.butter(4, cutoff / nyquist, btype="low", analog=False)
            data = signal.filtfilt(b, a, data, axis=0)

        return data[::factor, :]

    def normalize(self, data):
        """
        Z-score normalization.

        Parameters:
            data (np.ndarray): IMU data.

        Returns:
            np.ndarray: Normalized IMU data.
        """
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        return (data - mean) / (std + 1e-9)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class IMUDatasetBalanced(Dataset):
    def __init__(
        self,
        X,
        Y,
        sequence_length=128,
        downsample_factor=1,
        apply_antialias=True,
    ):
        """
        Dataset that balances long zero-label segments across entire recordings before
        slicing into fixed-length windows, ensuring each window is exactly `sequence_length`.

        Steps:
          1. For each subject, optional anti-alias + downsample + normalize.
          2. Identify runs of constant label; crop any zero-only runs to half the length
             of the neighboring runs (ignoring missing neighbors).
          3. Reconstruct a new sequence by concatenating the balanced runs.
          4. Window the balanced sequence into fixed-length segments of `sequence_length`, zero-
             padding the final remainder if necessary.

        Parameters:
            X (list of np.ndarray): IMU data per subject, shape (N, 6)
            Y (list of np.ndarray): Label arrays per subject, shape (N,)
            sequence_length (int): Window length after balancing.
            downsample_factor (int): Factor for decimation.
            apply_antialias (bool): Whether to low-pass filter before decimation.
        """
        if downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1.")

        self.sequence_length = sequence_length
        self.downsample_factor = downsample_factor
        self.apply_antialias = apply_antialias

        self.data = []
        self.labels = []
        self.subject_indices = []

        for subject_idx, (imu_data, label_seq) in enumerate(zip(X, Y)):
            # Step 1: downsample + normalize
            if downsample_factor > 1:
                imu_data = self._downsample(imu_data)
                label_seq = label_seq[::downsample_factor]
            imu_data = self._normalize(imu_data)

            # Step 2: segment by class then balance zero runs
            runs = self._segment_by_class(imu_data, label_seq)
            balanced_runs = self._balance_runs(runs)

            # Step 3: reconstruct balanced full sequence
            imu_balanced = np.concatenate([seg for seg, _ in balanced_runs], axis=0)
            lbl_balanced = np.concatenate([lbl for _, lbl in balanced_runs], axis=0)

            # Step 4: slice into fixed windows
            num_samples = len(lbl_balanced)
            end_full = num_samples - (num_samples % sequence_length)
            for start in range(0, end_full, sequence_length):
                x = imu_balanced[start : start + sequence_length]
                y = lbl_balanced[start : start + sequence_length]
                self.data.append(x)
                self.labels.append(y)
                self.subject_indices.append(subject_idx)

            # Step 5: zero-pad the last segment if needed
            rem = num_samples - end_full
            if rem > 0:
                x = imu_balanced[end_full:]
                y = lbl_balanced[end_full:]
                pad_len = sequence_length - rem
                x = np.pad(x, ((0, pad_len), (0, 0)), mode="constant", constant_values=0)
                y = np.pad(y, (0, pad_len), mode="constant", constant_values=0)
                self.data.append(x)
                self.labels.append(y)
                self.subject_indices.append(subject_idx)

    def _segment_by_class(self, data, labels):
        """
        Splits continuous arrays into runs of constant label.
        Returns list of (imu_run, label_run).
        """
        runs = []
        current_label = labels[0]
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                runs.append((data[start:i], labels[start:i]))
                start = i
                current_label = labels[i]
        runs.append((data[start:], labels[start:]))
        return runs

    def _balance_runs(self, runs):
        """
        Crop zero-only runs based on neighbors.
        """
        balanced = []
        n = len(runs)
        for idx, (seg, lbl) in enumerate(runs):
            if np.all(lbl == 0):
                left_len = len(runs[idx - 1][0]) // 2 if idx > 0 else 0
                right_len = len(runs[idx + 1][0]) // 2 if idx < n - 1 else 0
                target = left_len + right_len
                L = len(lbl)
                start = max((L - target) // 2, 0)
                seg = seg[start : start + target]
                lbl = lbl[start : start + target]
            balanced.append((seg, lbl))
        return balanced

    def _downsample(self, data):
        if self.apply_antialias:
            nyq = 0.5 * data.shape[0]
            cutoff = (0.5 / self.downsample_factor) * nyq
            b, a = signal.butter(4, cutoff / nyq, btype="low", analog=False)
            data = signal.filtfilt(b, a, data, axis=0)
        return data[:: self.downsample_factor]

    def _normalize(self, data):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        return (data - mean) / (std + 1e-9)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class IMUDatasetN21(Dataset):
    def __init__(
        self,
        X,
        Y,
        sequence_length=300,
        stride=20,
        downsample_factor=2,
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
        self.stride = stride
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

            # For samples that do not fill a complete segment, discard them.
            remainder = (num_samples - sequence_length) % self.stride
            if remainder > 0 and num_samples >= sequence_length:
                num_samples -= remainder

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


class IMUDatasetX(Dataset):
    def __init__(self, X, sequence_length=128, downsample_factor=1, apply_antialias=True):
        """
        Dataset for IMU data without labels, used in unsupervised learning tasks (e.g. VAE pretraining).

        Parameters:
            X (list of np.ndarray): IMU data per subject, each array shape (N, 6)
            sequence_length (int): Length of each sequence segment
            downsample_factor (int): Downsampling factor
            apply_antialias (bool): Whether to apply anti-aliasing filter before downsampling
        """
        self.sequence_length = sequence_length
        self.downsample_factor = downsample_factor
        self.data = []

        if downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1.")

        for imu_data in X:
            if downsample_factor > 1:
                imu_data = self.downsample(imu_data, downsample_factor, apply_antialias)
            imu_data = self.normalize(imu_data)
            num_samples = imu_data.shape[0]

            # Only process complete sequence segments
            for i in range(0, num_samples - sequence_length + 1, sequence_length):
                segment = imu_data[i : i + sequence_length]
                self.data.append(segment)

        # Print the total number of subjects and segments
        print(f"Total subjects: {len(X)}, Total segments: {len(self.data)}")
        print(f"Data shape: {self.data[0].shape}")

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
        return (data - mean) / (std + 1e-9)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        return x


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
