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


def left_hand_mirroring(x_l):
    """
    Apply hand mirroring transformation to the input data.

    Args:
        data (np.ndarray): Input data of shape (M, 6), where M is the sequence length
                           and 6 corresponds to the 3 accelerometer and 3 gyroscope axes.

    Returns:
        np.ndarray: Transformed data with the same shape as input.
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

    # Apply the transformation for numpy arrays
    x_l = np.dot(x_l, mirroring_matrix.T)

    return x_l


def hand_mirroring(x, y, probability=0.5, is_additive=False):
    """
    Augment data by mirroring IMU data to simulate wearing the device on the opposite hand.

    In many real-world datasets, the hand on which the IMU is worn (left or right) is not
    explicitly labeled. This augmentation applies a mirroring operation to simulate data
    from the opposite hand, enhancing model robustness to hand placement variations.

    Args:
        x: Input tensor of shape [size, seq_len, channels] where channels
                typically represent [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        y: Input labels of shape [size, seq_len]
        probability: Probability of applying the mirroring to each sample (default: 0.5)
        is_additive: If True, concatenate original data with mirrored data.
                          If False, return only mirrored data based on probability (default: True)

    Returns:
        augmented_x: Augmented input data
        augmented_y: Corresponding labels
    """
    size = x.shape[0]
    augmented_x = x.copy()

    mask = np.random.rand(size) < probability

    if np.any(mask):
        # Define the mirroring transformation matrix
        mirroring_matrix = np.array(
            [
                [-1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, -1, 0],
                [0, 0, 0, 0, 0, -1],
            ],
            dtype=np.float32,
        )

        for i in range(size):
            if mask[i]:
                augmented_x[i] = np.dot(augmented_x[i], mirroring_matrix.T)

    if is_additive and np.any(mask):
        augmented_x = np.concatenate([x, augmented_x[mask]], axis=0)
        augmented_y = np.concatenate([y, y[mask]], axis=0)
    else:
        augmented_y = y.copy()

    return augmented_x, augmented_y


def planar_rotation(x, y, probability=0.5, is_additive=False):
    """
    Augments IMU data by simulating different planar rotations of the sensor device on the wrist.

    In wearable IMU applications, sensor placement can vary significantly between users. When
    participants attach sensors themselves (as in the FD dataset with Shimmer3 IMU modules),
    the device orientation on the wrist can rotate in the xy-plane by 90°, 180°, or 270°.
    This inconsistency causes substantial variability in IMU signals for identical actions.

    This augmentation simulates these rotational variations by applying planar rotations to
    the accelerometer and gyroscope data. By exposing the model to these orientation variations
    during training, it learns to recognize activities regardless of how the device is positioned
    on the wrist, improving robustness for real-world deployments.

    Args:
        x: Input tensor of shape [size, seq_len, channels] where channels
                typically represent [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        y: Input labels of shape [size, seq_len]
        probability: Probability of applying rotation to each sample (default: 0.5)
        is_additive: If True, concatenate original data with rotated data to expand the dataset.
                     If False, return only rotated data based on probability (default: False)

    Returns:
        augmented_x: Tensor containing augmented sensor data
        augmented_y: Corresponding labels for the augmented data
    """
    size = x.shape[0]
    augmented_x = x.copy()

    mask = np.random.rand(size) < probability

    if np.any(mask):
        rotation_options = [
            # 90° rotation in xy-plane
            np.array(
                [
                    [0, 1, 0, 0, 0, 0],
                    [-1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=np.float32,
            ),
            # 180° rotation in xy-plane
            np.array(
                [
                    [-1, 0, 0, 0, 0, 0],
                    [0, -1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, -1, 0, 0],
                    [0, 0, 0, 0, -1, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=np.float32,
            ),
            # 270° rotation in xy-plane
            np.array(
                [
                    [0, -1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, -1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=np.float32,
            ),
        ]

        for i in range(size):
            if mask[i]:
                rot_idx = np.random.randint(0, len(rotation_options))
                augmented_x[i] = np.dot(augmented_x[i], rotation_options[rot_idx].T)

    if is_additive and np.any(mask):
        augmented_x = np.concatenate([x, augmented_x[mask]], axis=0)
        augmented_y = np.concatenate([y, y[mask]], axis=0)
    else:
        augmented_y = y.copy()

    return augmented_x, augmented_y


def axis_permutation(x, y, probability=0.5, is_additive=False):
    """
    Augments IMU data by permuting sensor axes to improve model robustness against device variability.

    Different IMU devices often use inconsistent axis conventions and orientations, causing the same
    physical movement to produce different sensor readings across devices. This augmentation simulates
    these variations by applying transformations between the X and Y axes (swapping and/or sign flipping)
    while preserving the Z axis, which typically aligns with gravity or vertical movement.

    This technique helps the model learn device-agnostic features and improves generalization to data
    collected from various IMU sensor configurations without requiring explicit calibration or
    standardization during deployment.

    Args:
        x: Input tensor of shape [size, seq_len, channels] where channels
                typically represent [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        y: Input labels of shape [size, seq_len]
        probability: Probability of applying the permutation to each sample (default: 0.5)
        is_additive: If True, concatenate original data with permuted data to expand the dataset.
                     If False, return only permuted data based on probability (default: False)

    Returns:
        augmented_x: Tensor containing augmented sensor data
        augmented_y: Corresponding labels for the augmented data
    """
    size = x.shape[0]
    augmented_x = x.copy()

    mask = np.random.rand(size) < probability

    if np.any(mask):
        permutation_options = [
            # Swap X and Y axes
            np.array(
                [
                    [0, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=np.float32,
            ),
            # Other permutation options...
            # (Include all other permutation matrices from original code)
        ]

        for i in range(size):
            if mask[i]:
                perm_idx = np.random.randint(0, len(permutation_options))
                augmented_x[i] = np.dot(augmented_x[i], permutation_options[perm_idx].T)

    if is_additive and np.any(mask):
        augmented_x = np.concatenate([x, augmented_x[mask]], axis=0)
        augmented_y = np.concatenate([y, y[mask]], axis=0)
    else:
        augmented_y = y.copy()

    return augmented_x, augmented_y


def planar_rotation(x, y, probability=0.5, is_additive=False):
    """NumPy version of planar rotation augmentation."""
    size = x.shape[0]
    augmented_x = x.copy()

    mask = np.random.rand(size) < probability

    if np.any(mask):
        rotation_options = [
            # 90° rotation in xy-plane
            np.array(
                [
                    [0, 1, 0, 0, 0, 0],
                    [-1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=np.float32,
            ),
            # Other rotation options...
            # (Include all other rotation matrices from original code)
        ]

        for i in range(size):
            if mask[i]:
                rot_idx = np.random.randint(0, len(rotation_options))
                augmented_x[i] = np.dot(augmented_x[i], rotation_options[rot_idx].T)

    if is_additive and np.any(mask):
        augmented_x = np.concatenate([x, augmented_x[mask]], axis=0)
        augmented_y = np.concatenate([y, y[mask]], axis=0)
    else:
        augmented_y = y.copy()

    return augmented_x, augmented_y


def spatial_orientation(x, y, probability=0.5, is_additive=False):
    """
    Augments IMU data by simulating subtle variations in sensor placement and orientation.

    Even when using identical devices, small differences in sensor placement—such as positioning
    on the wrist, strap tightness, or attachment angle—can significantly affect accelerometer
    and gyroscope readings. These spatial variations create signal inconsistencies across users
    that can degrade model generalization.

    This augmentation applies small random rotations around the x and z axes (typically ±10°)
    to simulate these natural placement variations. By training on data with these subtle
    orientation differences, the model becomes more robust to the inevitable variability in
    how users wear sensors in real-world conditions.

    Args:
        x: Input tensor of shape [size, seq_len, channels] where channels
                typically represent [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        y: Input labels of shape [size, seq_len]
        probability: Probability of applying orientation change to each sample (default: 0.5)
        is_additive: If True, concatenate original data with augmented data to expand the dataset.
                     If False, return only augmented data based on probability (default: False)

    Returns:
        augmented_x: Tensor containing augmented sensor data
        augmented_y: Corresponding labels for the augmented data
    """
    size = x.shape[0]
    augmented_x = x.copy()

    mask = np.random.rand(size) < probability

    if np.any(mask):
        for i in range(size):
            if mask[i]:
                # Generate random rotation angles (~10° in radians)
                theta_x = np.random.normal(0, 10 * np.pi / 180)
                theta_z = np.random.normal(0, 10 * np.pi / 180)

                # Create rotation matrices
                rot_x = np.array(
                    [[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]]
                )

                rot_z = np.array(
                    [[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]]
                )

                # Randomly select transformation order
                choice = np.random.randint(0, 4)
                if choice == 0:
                    transformation = rot_x
                elif choice == 1:
                    transformation = rot_z
                elif choice == 2:
                    transformation = np.dot(rot_x, rot_z)
                else:
                    transformation = np.dot(rot_z, rot_x)

                # Build 6x6 block-diagonal transformation matrix
                zeros = np.zeros((3, 3))
                full_transformation = np.block([[transformation, zeros], [zeros, transformation]])

                # Apply transformation to each time step
                for t in range(augmented_x.shape[1]):
                    augmented_x[i, t] = np.dot(full_transformation, augmented_x[i, t])

    if is_additive and np.any(mask):
        augmented_x = np.concatenate([x, augmented_x[mask]], axis=0)
        augmented_y = np.concatenate([y, y[mask]], axis=0)
    else:
        augmented_y = y.copy()

    return augmented_x, augmented_y
