#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Data Augmentation Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-14
Description : This script provides functions for augmenting IMU data, including
              hand mirroring, axis permutation, planar rotation, and spatial orientation
              transformations.
===============================================================================
"""

import numpy as np
import torch


def rotation_matrix_x(angle_rad):
    """
    Creates a 3D rotation matrix for a rotation around the x-axis.

    Args:
        angle_rad (float): Rotation angle in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )


def rotation_matrix_z(angle_rad):
    """
    Creates a 3D rotation matrix for a rotation around the z-axis.

    Args:
        angle_rad (float): Rotation angle in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    return np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1],
        ]
    )


def augment_hand_mirroring(batch_x, batch_y, probability=0.5, is_additive=False):
    """
    Augment data by mirroring IMU data to simulate wearing the device on the opposite hand.

    In many real-world datasets, the hand on which the IMU is worn (left or right) is not
    explicitly labeled. This augmentation applies a mirroring operation to simulate data
    from the opposite hand, enhancing model robustness to hand placement variations.

    Args:
        batch_x: Input tensor of shape [batch_size, seq_len, channels] where channels
                typically represent [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        batch_y: Input labels of shape [batch_size, seq_len]
        probability: Probability of applying the mirroring to each sample (default: 0.5)
        is_additive: If True, concatenate original data with mirrored data.
                          If False, return only mirrored data based on probability (default: True)

    Returns:
        augmented_batch_x: Augmented input data
        augmented_batch_y: Corresponding labels
    """
    batch_size = batch_x.shape[0]
    device = batch_x.device

    # Define the mirroring transformation matrix
    mirroring_matrix = torch.tensor(
        [
            [-1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, -1],
        ],
        dtype=torch.float32,
        device=device,
    )

    # Generate random mask based on probability
    mask = torch.rand(batch_size, device=device) < probability

    # Create a copy of the input batch
    augmented_batch_x = batch_x.clone()

    # Apply mirroring transformation only to selected samples
    if mask.any():
        # Apply the transformation only to samples selected by the mask
        augmented_batch_x[mask] = torch.matmul(augmented_batch_x[mask], mirroring_matrix.T)

    # If include_original is True, concatenate original and augmented data
    if is_additive and mask.any():
        augmented_batch_x = torch.cat([batch_x, augmented_batch_x[mask]], dim=0)
        augmented_batch_y = torch.cat([batch_y, batch_y[mask]], dim=0)
    else:
        # Otherwise, just return the modified batch
        augmented_batch_y = batch_y.clone()

    return augmented_batch_x, augmented_batch_y


def augment_axis_permutation(batch_x, batch_y, probability=0.5, is_additive=False):
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
        batch_x: Input tensor of shape [batch_size, seq_len, channels] where channels
                typically represent [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        batch_y: Input labels of shape [batch_size, seq_len]
        probability: Probability of applying the permutation to each sample (default: 0.5)
        is_additive: If True, concatenate original data with permuted data to expand the dataset.
                     If False, return only permuted data based on probability (default: False)

    Returns:
        augmented_batch_x: Tensor containing augmented sensor data
        augmented_batch_y: Corresponding labels for the augmented data
    """
    batch_size = batch_x.shape[0]
    device = batch_x.device

    # Create a copy of the input batch
    augmented_batch_x = batch_x.clone()

    # Generate random mask based on probability
    mask = torch.rand(batch_size, device=device) < probability

    if mask.any():
        # Apply axis permutation to selected samples
        # Assuming channels are [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        # Swap X and Y axes and potentially flip signs
        # Create a list of possible permutation matrices
        permutation_options = [
            # Swap X and Y axes
            torch.tensor(
                [
                    [0, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.float32,
                device=device,
            ),
            # Swap X and Y axes and flip X sign
            torch.tensor(
                [
                    [0, -1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, -1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.float32,
                device=device,
            ),
            # Swap X and Y axes and flip Y sign
            torch.tensor(
                [
                    [0, 1, 0, 0, 0, 0],
                    [-1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.float32,
                device=device,
            ),
            # Swap X and Y axes and flip both signs
            torch.tensor(
                [
                    [0, -1, 0, 0, 0, 0],
                    [-1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, -1, 0],
                    [0, 0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.float32,
                device=device,
            ),
        ]

        # For each sample to be permuted, randomly choose one permutation matrix
        for i in range(batch_size):
            if mask[i]:
                # Randomly select one of the permutation options
                perm_idx = torch.randint(0, len(permutation_options), (1,), device=device).item()
                perm_matrix = permutation_options[perm_idx]

                # Apply the permutation
                augmented_batch_x[i] = torch.matmul(augmented_batch_x[i], perm_matrix.T)

    # If is_additive is True, concatenate original and permuted data
    if is_additive and mask.any():
        augmented_batch_x = torch.cat([batch_x, augmented_batch_x[mask]], dim=0)
        augmented_batch_y = torch.cat([batch_y, batch_y[mask]], dim=0)
    else:
        # Otherwise, just return the modified batch
        augmented_batch_y = batch_y.clone()

    return augmented_batch_x, augmented_batch_y


def augment_planar_rotation(batch_x, batch_y, probability=0.5, is_additive=False):
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
        batch_x: Input tensor of shape [batch_size, seq_len, channels] where channels
                typically represent [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        batch_y: Input labels of shape [batch_size, seq_len]
        probability: Probability of applying rotation to each sample (default: 0.5)
        is_additive: If True, concatenate original data with rotated data to expand the dataset.
                     If False, return only rotated data based on probability (default: False)

    Returns:
        augmented_batch_x: Tensor containing augmented sensor data
        augmented_batch_y: Corresponding labels for the augmented data
    """
    batch_size = batch_x.shape[0]
    device = batch_x.device

    # Create a copy of the input batch
    augmented_batch_x = batch_x.clone()

    # Generate random mask based on probability
    mask = torch.rand(batch_size, device=device) < probability

    if mask.any():
        # Define rotation matrices for 90°, 180°, and 270° in the xy-plane
        # These matrices will apply to both accelerometer and gyroscope data
        rotation_options = [
            # 90° rotation in xy-plane
            torch.tensor(
                [
                    [0, 1, 0, 0, 0, 0],
                    [-1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.float32,
                device=device,
            ),
            # 180° rotation in xy-plane
            torch.tensor(
                [
                    [-1, 0, 0, 0, 0, 0],
                    [0, -1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, -1, 0, 0],
                    [0, 0, 0, 0, -1, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.float32,
                device=device,
            ),
            # 270° rotation in xy-plane
            torch.tensor(
                [
                    [0, -1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, -1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.float32,
                device=device,
            ),
        ]

        # For each sample to be rotated, randomly choose one rotation angle
        for i in range(batch_size):
            if mask[i]:
                # Randomly select one of the rotation options
                rot_idx = torch.randint(0, len(rotation_options), (1,), device=device).item()
                rot_matrix = rotation_options[rot_idx]

                # Apply the rotation
                augmented_batch_x[i] = torch.matmul(augmented_batch_x[i], rot_matrix.T)

    # If is_additive is True, concatenate original and rotated data
    if is_additive and mask.any():
        augmented_batch_x = torch.cat([batch_x, augmented_batch_x[mask]], dim=0)
        augmented_batch_y = torch.cat([batch_y, batch_y[mask]], dim=0)
    else:
        # Otherwise, just return the modified batch
        augmented_batch_y = batch_y.clone()

    return augmented_batch_x, augmented_batch_y


def augment_spatial_orientation(batch_x, batch_y, probability=0.5, is_additive=False):
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
        batch_x: Input tensor of shape [batch_size, seq_len, channels] where channels
                typically represent [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        batch_y: Input labels of shape [batch_size, seq_len]
        probability: Probability of applying orientation change to each sample (default: 0.5)
        is_additive: If True, concatenate original data with augmented data to expand the dataset.
                     If False, return only augmented data based on probability (default: False)

    Returns:
        augmented_batch_x: Tensor containing augmented sensor data
        augmented_batch_y: Corresponding labels for the augmented data
    """
    batch_size = batch_x.shape[0]
    device = batch_x.device

    # Create a copy of the input batch
    augmented_batch_x = batch_x.clone()

    # Generate random mask based on probability
    mask = torch.rand(batch_size, device=device) < probability

    if mask.any():
        for i in range(batch_size):
            if mask[i]:
                # Generate random rotation angles (with standard deviation ~10° converted to radians)
                theta_x = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(10.0 * torch.pi / 180.0)).item()
                theta_z = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(10.0 * torch.pi / 180.0)).item()

                # Create rotation matrices
                rot_x = torch.tensor(
                    [
                        [1, 0, 0],
                        [
                            0,
                            torch.cos(torch.tensor(theta_x)),
                            -torch.sin(torch.tensor(theta_x)),
                        ],
                        [
                            0,
                            torch.sin(torch.tensor(theta_x)),
                            torch.cos(torch.tensor(theta_x)),
                        ],
                    ],
                    dtype=torch.float32,
                    device=device,
                )

                rot_z = torch.tensor(
                    [
                        [
                            torch.cos(torch.tensor(theta_z)),
                            -torch.sin(torch.tensor(theta_z)),
                            0,
                        ],
                        [
                            torch.sin(torch.tensor(theta_z)),
                            torch.cos(torch.tensor(theta_z)),
                            0,
                        ],
                        [0, 0, 1],
                    ],
                    dtype=torch.float32,
                    device=device,
                )

                # Randomly select one of four transformation orders
                choice = torch.randint(0, 4, (1,), device=device).item()
                if choice == 0:
                    transformation = rot_x
                elif choice == 1:
                    transformation = rot_z
                elif choice == 2:
                    transformation = torch.matmul(rot_x, rot_z)
                else:
                    transformation = torch.matmul(rot_z, rot_x)

                # Build a 6x6 block-diagonal transformation matrix for both accelerometer and gyroscope
                zeros = torch.zeros((3, 3), device=device)
                full_transformation = torch.cat(
                    [
                        torch.cat([transformation, zeros], dim=1),
                        torch.cat([zeros, transformation], dim=1),
                    ],
                    dim=0,
                )

                # Apply transformation to each time step in the sequence
                for t in range(augmented_batch_x.shape[1]):  # Loop through sequence length
                    augmented_batch_x[i, t] = torch.matmul(full_transformation, augmented_batch_x[i, t])

    # If is_additive is True, concatenate original and augmented data
    if is_additive and mask.any():
        augmented_batch_x = torch.cat([batch_x, augmented_batch_x[mask]], dim=0)
        augmented_batch_y = torch.cat([batch_y, batch_y[mask]], dim=0)
    else:
        # Otherwise, just return the modified batch
        augmented_batch_y = batch_y.clone()

    return augmented_batch_x, augmented_batch_y
