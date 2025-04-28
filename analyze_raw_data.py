#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Raw Data Analysis Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-28
Description : This script performs comprehensive analysis on raw IMU data with:
              1. Per-fold dataset statistics and label distribution analysis
              2. Subject-wise sample distribution visualization
              3. Gesture segmentation and time-domain analysis
              4. Support for multiple datasets (DX/FD, Oreba/Clemson)
              5. Combined left/right hand data analysis

              Outputs include statistical reports and visualization plots.
===============================================================================
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Configuration
DATASET = "DXI"  # Options: "DXI", "DXII", "FDI", "FDII", "FDIII", "Clemson", "Oreba"

if DATASET.startswith("DX"):
    NUM_CLASSES = 2
    sub_version = DATASET.replace("DX", "").upper() or "I"
    DATA_PATH = f"./dataset/DX/DX-{sub_version}"
    SAVE_DIR = f"./analysis/DX/DX-{sub_version}"
    SAMPLING_FREQ = 64
elif DATASET.startswith("FD"):
    NUM_CLASSES = 3
    sub_version = DATASET.replace("FD", "").upper() or "I"
    DATA_PATH = f"./dataset/FD/FD-{sub_version}"
    SAVE_DIR = f"./analysis/FD/FD-{sub_version}"
    SAMPLING_FREQ = 64
elif DATASET == "Clemson":
    NUM_CLASSES = 3
    DATA_PATH = "./dataset/Clemson"
    SAVE_DIR = "./analysis/Clemson"
    SAMPLING_FREQ = 15
elif DATASET == "Oreba":
    NUM_CLASSES = 3
    DATA_PATH = "./dataset/Oreba"
    SAVE_DIR = "./analysis/Oreba"
    SAMPLING_FREQ = 16

os.makedirs(SAVE_DIR, exist_ok=True)


def load_data():
    try:
        # Attempt to load combined X and Y
        with open(os.path.join(DATA_PATH, "X.pkl"), "rb") as f:
            X = pickle.load(f)
        with open(os.path.join(DATA_PATH, "Y.pkl"), "rb") as f:
            Y = pickle.load(f)
    except FileNotFoundError:
        # If combined data is not found, load left and right separately and merge
        with open(os.path.join(DATA_PATH, "X_L.pkl"), "rb") as f:
            X_L = np.array(pickle.load(f), dtype=object)
        with open(os.path.join(DATA_PATH, "Y_L.pkl"), "rb") as f:
            Y_L = np.array(pickle.load(f), dtype=object)
        with open(os.path.join(DATA_PATH, "X_R.pkl"), "rb") as f:
            X_R = np.array(pickle.load(f), dtype=object)
        with open(os.path.join(DATA_PATH, "Y_R.pkl"), "rb") as f:
            Y_R = np.array(pickle.load(f), dtype=object)
        X = np.array([np.concatenate([x_l, x_r], axis=0) for x_l, x_r in zip(X_L, X_R)], dtype=object)
        Y = np.array([np.concatenate([y_l, y_r], axis=0) for y_l, y_r in zip(Y_L, Y_R)], dtype=object)
    return X, Y


def basic_statistics(X, Y):
    """Print detailed dataset statistics and write to a txt file"""
    stats_file = os.path.join(SAVE_DIR, "basic_statistics.txt")

    # Get unique labels across all subjects
    all_labels = np.concatenate(Y)
    unique_labels = np.unique(all_labels)
    num_labels = len(unique_labels)

    # Compute label distribution per subject
    label_counts_per_subject = []

    for subj_idx, y in enumerate(Y):
        # Initialize counts for all possible labels
        label_counts = {label: 0 for label in range(num_labels)}
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            label_counts[label] = count

        # Create tuple with subject index and counts for each label
        counts_tuple = (subj_idx + 1,) + tuple(label_counts[label] for label in range(num_labels))
        label_counts_per_subject.append(counts_tuple)

    # Compute overall label counts
    total_counts = np.sum([[count[i] for i in range(1, len(count))] for count in label_counts_per_subject], axis=0)

    # Write to file
    with open(stats_file, "w") as f:  # Open file in write mode
        f.write(f"Number of subjects: {len(X)}\n")
        f.write(f"Input features: {X[0].shape[1]}\n")  # Assuming [time, features]

        total_samples = sum(len(subj) for subj in Y)
        f.write(f"Total samples: {total_samples:,}\n")

        f.write(f"Unique labels: {unique_labels}\n")

        # Write overall label-specific counts
        for label in range(num_labels):
            f.write(f"Samples with label {label}: {total_counts[label]:,}\n")

        # Write per-subject statistics
        f.write("\nSubject-wise Sample Distribution:\n")
        header = "Subject_ID | " + " | ".join(f"Label_{i}" for i in range(num_labels))
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for counts in label_counts_per_subject:
            subj_idx = counts[0]
            label_counts = counts[1:]
            row = f"{subj_idx:<10} | " + " | ".join(f"{count:<7}" for count in label_counts)
            f.write(row + "\n")


def segment_by_label(Y):
    """
    Identify continuous segments of label 0, 1, and 2.

    Returns:
        segments_dict: {label_value: [(start_idx, end_idx), ...]}
    """
    segments_dict = {0: [], 1: [], 2: []}

    current_label = Y[0]
    start_idx = 0

    for i in range(1, len(Y)):
        if Y[i] != current_label:  # Label changed
            segments_dict[current_label].append((start_idx, i))  # Store segment
            start_idx = i
            current_label = Y[i]

    # Add last segment
    segments_dict[current_label].append((start_idx, len(Y)))

    return segments_dict


def plot_segment_length_distribution_by_label(Y):
    """
    For each non-zero label (1 and 2), plot the distribution of segment lengths.
    Each label gets its own histogram plot.
    """
    # Initialize a dictionary to hold segment lengths per label
    segment_lengths_per_label = {1: [], 2: []}

    # Collect segment lengths
    for labels in Y:
        segments_dict = segment_by_label(labels)
        for label in [1, 2]:
            for start_idx, end_idx in segments_dict[label]:
                length_in_seconds = (end_idx - start_idx) / SAMPLING_FREQ
                segment_lengths_per_label[label].append(length_in_seconds)

    # Create a plot for each label
    for label, lengths in segment_lengths_per_label.items():
        if not lengths:
            continue  # Skip if no segments for this label

        bin_width = 0.1  # seconds
        max_length = max(lengths)
        bins = np.arange(0, max_length + bin_width, bin_width)

        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=bins, color="skyblue", edgecolor="black")
        plt.title(f"Segment Length Distribution - Label {label}")
        plt.xlabel("Segment Length (seconds)")
        plt.ylabel("Count")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.savefig(os.path.join(SAVE_DIR, f"segment_length_distribution_label{label}.png"), dpi=300)
        plt.close()


def plot_shortest_segments(X, Y):
    """
    For each non-zero label (1 and 2), find the top 5 shortest segments across all subjects,
    and plot their corresponding sensor data (accelerometer and gyroscope).
    """
    save_dir = os.path.join(SAVE_DIR, "shortest_segments")
    os.makedirs(save_dir, exist_ok=True)

    segment_info_per_label = {1: [], 2: []}  # {label: [(length, subj_idx, start_idx, end_idx), ...]}

    # Collect all segments
    for subj_idx, labels in enumerate(Y):
        segments_dict = segment_by_label(labels)
        for label in [1, 2]:
            for start_idx, end_idx in segments_dict[label]:
                length = end_idx - start_idx
                segment_info_per_label[label].append((length, subj_idx, start_idx, end_idx))

    # For each label, find top 5 shortest segments
    for label, segments in segment_info_per_label.items():
        if not segments:
            continue  # Skip if no segments for this label

        # Sort by length ascending
        segments_sorted = sorted(segments, key=lambda x: x[0])
        top5_segments = segments_sorted[:5]

        for idx, (length, subj_idx, start_idx, end_idx) in enumerate(top5_segments):
            imu_data = X[subj_idx]
            labels_subj = Y[subj_idx]

            # Extract the segment
            segment_data = imu_data[start_idx:end_idx, :]
            time_axis = np.arange(segment_data.shape[0]) / SAMPLING_FREQ

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            # Plot accelerometer data (features 0-2)
            ax1.set_ylabel("Accelerometer (m/s²)", color="blue")
            accel_colors = ["blue", "royalblue", "dodgerblue"]
            for feature_idx in range(3):
                ax1.plot(
                    time_axis,
                    segment_data[:, feature_idx],
                    label=f"Accel {['X', 'Y', 'Z'][feature_idx]}",
                    color=accel_colors[feature_idx],
                    alpha=0.8,
                )
            ax1.tick_params(axis="y", labelcolor="blue")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"Label {label} - Shortest #{idx+1} - Subject {subj_idx+1}")

            # Plot gyroscope data (features 3-5)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Gyroscope (rad/s)", color="red")
            gyro_colors = ["red", "tomato", "firebrick"]
            for feature_idx in range(3, 6):
                ax2.plot(
                    time_axis,
                    segment_data[:, feature_idx],
                    label=f"Gyro {['X', 'Y', 'Z'][feature_idx-3]}",
                    color=gyro_colors[feature_idx - 3],
                    alpha=0.8,
                )
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.legend(loc="upper left")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"label{label}_shortest{idx+1}_subj{subj_idx+1}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()


def plot_longest_segments(X, Y):
    """
    For each non-zero label (1 and 2), find the top 5 longest segments across all subjects,
    and plot their corresponding sensor data (accelerometer and gyroscope).
    """
    save_dir = os.path.join(SAVE_DIR, "longest_segments")
    os.makedirs(save_dir, exist_ok=True)

    segment_info_per_label = {1: [], 2: []}  # {label: [(length, subj_idx, start_idx, end_idx), ...]}

    # Collect all segments
    for subj_idx, labels in enumerate(Y):
        segments_dict = segment_by_label(labels)
        for label in [1, 2]:
            for start_idx, end_idx in segments_dict[label]:
                length = end_idx - start_idx
                segment_info_per_label[label].append((length, subj_idx, start_idx, end_idx))

    # For each label, find top 5 longest segments
    for label, segments in segment_info_per_label.items():
        if not segments:
            continue  # Skip if no segments for this label

        # Sort by length descending
        segments_sorted = sorted(segments, key=lambda x: x[0], reverse=True)
        top5_segments = segments_sorted[:5]

        for idx, (length, subj_idx, start_idx, end_idx) in enumerate(top5_segments):
            imu_data = X[subj_idx]
            labels_subj = Y[subj_idx]

            # Extract the segment
            segment_data = imu_data[start_idx:end_idx, :]
            time_axis = np.arange(start_idx, end_idx) / SAMPLING_FREQ

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            # Plot accelerometer data (features 0-2)
            ax1.set_ylabel("Accelerometer (m/s²)", color="blue")
            accel_colors = ["blue", "royalblue", "dodgerblue"]
            for feature_idx in range(3):
                ax1.plot(
                    time_axis,
                    segment_data[:, feature_idx],
                    label=f"Accel {['X', 'Y', 'Z'][feature_idx]}",
                    color=accel_colors[feature_idx],
                    alpha=0.8,
                )
            ax1.tick_params(axis="y", labelcolor="blue")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)
            ax1.axvspan(time_axis[0], time_axis[-1], color="red", alpha=0.2)
            ax1.set_title(f"Label {label} - Longest #{idx+1} - Subject {subj_idx+1}")

            # Plot gyroscope data (features 3-5)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Gyroscope (rad/s)", color="red")
            gyro_colors = ["red", "tomato", "firebrick"]
            for feature_idx in range(3, 6):
                ax2.plot(
                    time_axis,
                    segment_data[:, feature_idx],
                    label=f"Gyro {['X', 'Y', 'Z'][feature_idx-3]}",
                    color=gyro_colors[feature_idx - 3],
                    alpha=0.8,
                )
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.legend(loc="upper left")
            ax2.grid(True, alpha=0.3)
            ax2.axvspan(time_axis[0], time_axis[-1], color="red", alpha=0.2)

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"label{label}_top{idx+1}_subj{subj_idx+1}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()


def main():
    # Load and analyze both sides together
    X, Y = load_data()
    basic_statistics(X, Y)
    # plot_sample_distribution(Y)
    # plot_segment_length_distribution_by_label(Y)
    # plot_longest_segments(X, Y)
    # plot_shortest_segments(X, Y)


if __name__ == "__main__":
    main()
    print(f"Analysis saved to: {os.path.abspath(SAVE_DIR)}")
