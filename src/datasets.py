import numpy as np
import torch
from torch.utils.data import Dataset


# Dataset class
class IMUDataset(Dataset):
    def __init__(self, X, Y, sequence_length=128):
        """
        Initialize the IMUDataset.

        Parameters:
            X (list of np.ndarray): List of IMU data arrays for each subject.
            Y (list of np.ndarray): List of label arrays for each subject.
            sequence_length (int): Length of each sequence segment.
        """
        self.data = []
        self.labels = []
        self.sequence_length = sequence_length
        self.subject_indices = []  # Record which subject each sample belongs to

        # Processing data for each session
        for subject_idx, (imu_data, labels) in enumerate(zip(X, Y)):
            imu_data = self.normalize(imu_data)
            num_samples = len(labels)

            # Segment the data into sequences of length `sequence_length`
            for i in range(0, num_samples, sequence_length):
                imu_segment = imu_data[i : i + sequence_length]
                label_segment = labels[i : i + sequence_length]

                # Only add segments that are exactly `sequence_length` long
                if len(imu_segment) == sequence_length:
                    self.data.append(imu_segment)
                    self.labels.append(label_segment)
                    self.subject_indices.append(subject_idx)

    def normalize(self, data):
        """
        Z-score normalization of the IMU data.

        Parameters:
            data (np.ndarray): IMU data array.

        Returns:
            np.ndarray: Normalized IMU data.
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-5)

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the IMU data and the corresponding labels.
        """
        x = self.data[idx]
        y = self.labels[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y
