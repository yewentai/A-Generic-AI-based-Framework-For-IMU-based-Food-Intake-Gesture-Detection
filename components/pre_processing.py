#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Pre-Processing Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-12
Description : This script provides IMU data augmentation functions, including
              hand mirroring and planar rotation. These transformations are
              useful for improving model robustness in activity recognition tasks.
===============================================================================
"""


import numpy as np
import torch


def hand_mirroring(data):
    """
    Apply hand mirroring transformation to the input data.

    Args:
        data (np.ndarray or torch.Tensor): Input data of shape (M, 6), where M is the sequence length
                                           and 6 corresponds to the 3 accelerometer and 3 gyroscope axes.

    Returns:
        np.ndarray or torch.Tensor: Transformed data with the same shape as input.
    """
    # Define the mirroring transformation matrix

    mirroring_matrix = np.array(
        [
            [-1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, -1],
        ]
    )

    if isinstance(data, torch.Tensor):
        # Convert the mirroring matrix to a tensor and move it to the same device as the data
        mirroring_matrix = torch.tensor(mirroring_matrix, dtype=torch.float32, device=data.device)
        # Apply the transformation
        data = torch.matmul(data, mirroring_matrix.T)
    else:
        # Apply the transformation for numpy arrays
        data = np.dot(data, mirroring_matrix.T)

    return data


def planar_rotation(data, labels, degree=None):
    """
    Apply planar rotation (90°, 180°, or 270°) on a single IMU sequence using NumPy.

    Args:
        data (np.ndarray): IMU data of shape [seq_len, 6], with channels
                           [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        labels (np.ndarray): Corresponding labels of shape [seq_len]
        degree (int, optional): Rotation degree to apply. One of [90, 180, 270].
                                If None, a random choice is made.

    Returns:
        rotated_data (np.ndarray): Rotated IMU data of shape [seq_len, 6]
        labels (np.ndarray): Unchanged labels (for convenience)
    """
    assert data.shape[1] == 6, "Expected 6-channel IMU input [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]"
    assert degree in [None, 90, 180, 270], "Degree must be one of [None, 90, 180, 270]"

    if degree is None:
        degree = np.random.choice([90, 180, 270])

    if degree == 90:
        rotation_matrix = np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
    elif degree == 180:
        rotation_matrix = np.array(
            [
                [-1, 0, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, -1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
    elif degree == 270:
        rotation_matrix = np.array(
            [
                [0, -1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, -1, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
    else:
        raise ValueError(f"Unsupported rotation angle: {degree}")

    rotated_data = np.matmul(data, rotation_matrix.T)
    return rotated_data, labels


def axis_permutation(data, labels=None, option=None):
    """
    Apply a random (or specified) X↔Y axis swap (with optional sign flips) to IMU data.

    Args:
        data (np.ndarray or torch.Tensor): shape (M, 6), channels [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        labels (np.ndarray or torch.Tensor, optional): shape (M,) or None. Returned unchanged.
        option (int, optional): index of the permutation to apply (0–3). If None, one is chosen at random:
            0) swap X/Y
            1) swap X/Y + flip new X
            2) swap X/Y + flip new Y
            3) swap X/Y + flip both
    Returns:
        permuted_data, labels
    """
    # define the four 6×6 permutation matrices
    perms = [
        np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        ),
        np.array(
            [
                [0, -1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, -1, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        ),
        np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        ),
        np.array(
            [
                [0, -1, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, -1, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        ),
    ]
    idx = option if option in range(4) else np.random.randint(4)
    P = perms[idx]

    if isinstance(data, torch.Tensor):
        P_t = torch.tensor(P, dtype=torch.float32, device=data.device)
        permuted = torch.matmul(data, P_t.T)
    else:
        permuted = data.dot(P.T)

    return (permuted, labels) if labels is not None else permuted


def spatial_orientation(data, labels=None, std_dev_deg=10.0):
    """
    Apply small random rotations (±std_dev_deg) about the X and Z axes to simulate placement noise.

    Args:
        data (np.ndarray or torch.Tensor): shape (M, 6)
        labels (np.ndarray or torch.Tensor, optional): shape (M,) or None. Returned unchanged.
        std_dev_deg (float): standard deviation of rotation (in degrees)

    Returns:
        noisy_data, labels
    """
    # sample angles in radians
    sigma = std_dev_deg * np.pi / 180.0
    theta_x = np.random.normal(0.0, sigma)
    theta_z = np.random.normal(0.0, sigma)

    # 3×3 rot mats
    Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])

    # choose order randomly
    choice = np.random.randint(4)
    if choice == 0:
        R = Rx
    elif choice == 1:
        R = Rz
    elif choice == 2:
        R = Rx.dot(Rz)
    else:
        R = Rz.dot(Rx)

    # build 6×6 block-diagonal for acc & gyro
    T = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])

    if isinstance(data, torch.Tensor):
        T_t = torch.tensor(T, dtype=torch.float32, device=data.device)
        noisy = torch.matmul(data, T_t.T)
    else:
        noisy = data.dot(T.T)

    return (noisy, labels) if labels is not None else noisy
