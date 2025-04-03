#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Data Augmentation Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-03
Description : This script provides functions for augmenting IMU data, including
              hand mirroring, axis permutation, planar rotation, and spatial orientation
              transformations.
===============================================================================
"""

import numpy as np
import torch
from components.pre_processing import hand_mirroring


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


def augment_hand_mirroring(batch_x, batch_y):
    """
    Augments the input batch by adding mirrored versions of each sample.
    It applies the hand_mirroring transformation (which performs the mirroring operation)
    to generate the mirrored samples. The returned data contains both the original
    and the mirrored samples, with labels remaining the same.

    Args:
        batch_x: Input tensor of shape (batch_size, sequence_length, features)
        batch_y: Input labels of shape (batch_size, ...)

    Returns:
        augmented_batch_x: Tensor of shape (2 * batch_size, sequence_length, features)
        augmented_batch_y: Tensor of shape (2 * batch_size, ...)
    """
    # Generate mirrored data using the hand_mirroring function
    mirrored_batch_x = hand_mirroring(batch_x)

    # Concatenate the original and mirrored samples along the batch dimension
    augmented_batch_x = torch.cat([batch_x, mirrored_batch_x], dim=0)

    # Duplicate the labels for the mirrored samples
    augmented_batch_y = torch.cat([batch_y, batch_y], dim=0)

    return augmented_batch_x, augmented_batch_y


def augment_axis_permutation(batch_x, batch_y):
    """
    Augments IMU data by randomly swapping axes and flipping their signs.
    This simulates differences in axis arrangement and sensor direction definitions
    across devices. The transformation is applied to every sample.

    Args:
        batch_x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, 3).
        batch_y (torch.Tensor): Input labels of shape (batch_size, ...).

    Returns:
        tuple: Augmented tensor of shape (2 * batch_size, sequence_length, 3)
               and augmented labels of shape (2 * batch_size, ...), where the augmented
               batch is the concatenation of the original and augmented samples.
    """
    batch_size, seq_len, num_axes = batch_x.shape
    augmented = batch_x.clone()

    for i in range(batch_size):
        # Generate a random permutation of axes and random sign flips.
        perm = np.random.permutation(num_axes)
        signs = np.random.choice([-1, 1], size=num_axes)
        # Build a transformation matrix where each row has one non-zero entry.
        transform = np.zeros((num_axes, num_axes))
        for j in range(num_axes):
            transform[j, perm[j]] = signs[j]
        sample_np = augmented[i].numpy()  # shape: (seq_len, 3)
        sample_np = sample_np @ transform.T
        augmented[i] = torch.tensor(sample_np, dtype=batch_x.dtype)

    augmented_batch_x = torch.cat([batch_x, augmented], dim=0)
    augmented_batch_y = torch.cat([batch_y, batch_y], dim=0)
    return augmented_batch_x, augmented_batch_y


def augment_planar_rotation(batch_x, batch_y):
    """
    Augments IMU data by applying a discrete random rotation (0°, 90°, 180°, or 270°)
    to the x-y components of each sample while keeping the z component unchanged.
    This simulates differences in sensor alignment across devices.
    The transformation is applied to every sample.

    Args:
        batch_x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, 3).
        batch_y (torch.Tensor): Input labels of shape (batch_size, ...).

    Returns:
        tuple: Augmented tensor of shape (2 * batch_size, sequence_length, 3)
               and augmented labels of shape (2 * batch_size, ...), where the augmented
               batch is the concatenation of the original and augmented samples.
    """
    batch_size, seq_len, _ = batch_x.shape
    augmented = batch_x.clone()

    for i in range(batch_size):
        # Choose a random rotation angle from {0, 90, 180, 270} degrees.
        angle_deg = np.random.choice([0, 90, 180, 270])
        angle_rad = np.deg2rad(angle_deg)
        # Construct a 2D rotation matrix for the x-y plane.
        rotation_2d = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
        sample_np = augmented[i].numpy()  # shape: (seq_len, 3)
        sample_np[:, :2] = sample_np[:, :2] @ rotation_2d.T
        augmented[i] = torch.tensor(sample_np, dtype=batch_x.dtype)

    augmented_batch_x = torch.cat([batch_x, augmented], dim=0)
    augmented_batch_y = torch.cat([batch_y, batch_y], dim=0)
    return augmented_batch_x, augmented_batch_y


def augment_spatial_orientation(batch_x, batch_y):
    """
    Augments IMU data by applying a random 3D rotation to both accelerometer and gyroscope readings.
    This simulates variations in device orientation due to different wearing positions.
    The input is assumed to have 6 features per timestep (3 for accelerometer, 3 for gyroscope).
    The transformation is applied to every sample.

    Args:
        batch_x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, 6).
        batch_y (torch.Tensor): Input labels of shape (batch_size, ...).

    Returns:
        tuple: Augmented tensor of shape (2 * batch_size, sequence_length, 6)
               and augmented labels of shape (2 * batch_size, ...), where the augmented
               batch is the concatenation of the original and augmented samples.
    """
    batch_size, seq_len, features = batch_x.shape
    augmented = batch_x.clone()

    for i in range(batch_size):
        # Generate random rotation angles (with standard deviation ~10° converted to radians).
        theta_x = np.random.normal(0, 10 * np.pi / 180)
        theta_z = np.random.normal(0, 10 * np.pi / 180)

        # Create basic rotation matrices.
        rot_x = rotation_matrix_x(theta_x)
        rot_z = rotation_matrix_z(theta_z)

        # Randomly select one of four transformation orders.
        choice = np.random.choice([0, 1, 2, 3])
        if choice == 0:
            transformation = rot_x
        elif choice == 1:
            transformation = rot_z
        elif choice == 2:
            transformation = np.dot(rot_x, rot_z)
        else:
            transformation = np.dot(rot_z, rot_x)

        # Build a 6x6 block-diagonal transformation matrix for both accelerometer and gyroscope.
        full_transformation = np.block(
            [[transformation, np.zeros((3, 3))], [np.zeros((3, 3)), transformation]]
        )

        sample_np = augmented[i].numpy()  # shape: (seq_len, 6)
        transformed_sample = np.dot(full_transformation, sample_np.T).T
        augmented[i] = torch.tensor(transformed_sample, dtype=batch_x.dtype)

    augmented_batch_x = torch.cat([batch_x, augmented], dim=0)
    augmented_batch_y = torch.cat([batch_y, batch_y], dim=0)
    return augmented_batch_x, augmented_batch_y
