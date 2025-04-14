#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Pre-Processing Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-14
Description : This script contains functions for pre-processing IMU data,
              including transformations such as hand mirroring.
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
