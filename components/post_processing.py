#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Post-Processing Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-03
Description : This script defines functions for post-processing IMU prediction masks,
              such as removing short segments and merging close segments.
===============================================================================
"""

import numpy as np


def nonzero_intervals(x):
    """
    Extract start and end indices of nonzero intervals in a binary array.

    Parameters:
        x (np.ndarray): Input binary array (1D).

    Returns:
        np.ndarray: An array of shape (N, 2), where each row represents the
                    [start, end) indices of a contiguous nonzero interval.
    """
    results = []
    in_segment = False
    start_idx = 0

    for i in range(len(x)):
        if x[i] != 0 and not in_segment:
            in_segment = True
            start_idx = i
        elif x[i] == 0 and in_segment:
            in_segment = False
            results.append([start_idx, i])
    if in_segment:
        results.append([start_idx, len(x)])
    return np.array(results, dtype=int)


def post_process_binary_mask(binary_mask, sampling_freq, min_length_sec, merge_distance_sec):
    """
    Post-process a binary mask by removing short segments and merging close segments.

    Parameters:
        binary_mask (np.ndarray): Binary array (0/1) for a single class.
        sampling_freq (float): Sampling frequency in Hz.
        min_length_sec (float): Minimum duration (in seconds) of a segment to keep.
        merge_distance_sec (float): Maximum gap (in seconds) between segments to merge.

    Returns:
        np.ndarray: The refined binary mask after post-processing.
    """
    min_length_samples = int(sampling_freq * min_length_sec)
    merge_distance_samples = int(sampling_freq * merge_distance_sec)

    # Get intervals where the mask is nonzero.
    intervals = nonzero_intervals(binary_mask)

    # Remove segments shorter than the minimum length.
    intervals = [interval for interval in intervals if (interval[1] - interval[0]) >= min_length_samples]

    # Merge intervals that are close to each other.
    merged_intervals = []
    for interval in intervals:
        if not merged_intervals or (interval[0] - merged_intervals[-1][1]) > merge_distance_samples:
            merged_intervals.append(interval)
        else:
            merged_intervals[-1][1] = interval[1]

    # Create a new binary mask based on the merged intervals.
    new_mask = np.zeros_like(binary_mask)
    for start, end in merged_intervals:
        new_mask[start:end] = 1

    return new_mask


def post_process_predictions(predictions, sampling_freq, min_length_sec=1.0, merge_distance_sec=0.1):
    """
    Post-process predictions by removing short segments and merging close segments.
    Supports both binary (0/1) and multi-class predictions.

    For binary predictions, the function behaves as before. For multi-class predictions,
    it processes each class separately and returns a new prediction array.

    Parameters:
        predictions (np.ndarray): Predictions array (1D) containing class labels.
        sampling_freq (float): Sampling frequency in Hz (samples per second).
        min_length_sec (float): Minimum duration (in seconds) of a segment to keep.
        merge_distance_sec (float): Maximum gap (in seconds) between segments to merge.

    Returns:
        np.ndarray: Post-processed predictions array.
    """
    unique_labels = np.unique(predictions)

    # If predictions are binary (0/1), process directly.
    if len(unique_labels) <= 2 and set(unique_labels) == {0, 1}:
        return post_process_binary_mask(predictions, sampling_freq, min_length_sec, merge_distance_sec)

    # Multi-class case: process each class separately.
    new_predictions = np.zeros_like(predictions)
    for label in unique_labels:
        # Create a binary mask for the current class.
        binary_mask = (predictions == label).astype(np.int32)
        # Post-process the binary mask.
        processed_mask = post_process_binary_mask(binary_mask, sampling_freq, min_length_sec, merge_distance_sec)
        # Assign the label to the refined intervals.
        new_predictions[processed_mask == 1] = label

    return new_predictions
