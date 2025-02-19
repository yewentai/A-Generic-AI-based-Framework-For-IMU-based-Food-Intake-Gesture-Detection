import numpy as np
import torch
import scipy.signal as signal
from torch.utils.data import Dataset


class IMUDataset(Dataset):
    def __init__(
        self, X, Y, sequence_length=128, downsample_factor=4, apply_antialias=True
    ):
        """
        Initialize the IMUDataset.

        Parameters:
            X (list of np.ndarray): List of IMU data arrays for each subject.
            Y (list of np.ndarray): List of label arrays for each subject.
            sequence_length (int): Length of each sequence segment.
            downsample_factor (int): Factor by which to downsample the data.
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

            for i in range(0, num_samples - sequence_length + 1, sequence_length):
                imu_segment = imu_data[i : i + sequence_length]
                label_segment = labels[i : i + sequence_length]

                if len(imu_segment) == sequence_length:
                    self.data.append(imu_segment)
                    self.labels.append(label_segment)
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
            cutoff = (
                0.5 / factor
            ) * nyquist  # Limit to 80% of the new Nyquist frequency
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
        return (data - mean) / (std + 1e-5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
