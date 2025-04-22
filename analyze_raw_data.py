#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Raw Data Analysis Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
P25-04-22
Description : This script performs comprehensive analysis on raw IMU data with:
              1. Dataset statistics and label distribution analysis
              2. Subject-wise sample distribution visualization
              3. Gesture segmentation and time-domain analysis
              4. Support for multiple datasets (DX/FD, Oreba/Clemson)
              5. Combined left/right hand data analysis
              
              Outputs include statistical reports and visualization plots.
===============================================================================
"""

import os
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt

# Configuration
DATASET = "FDIII"

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

    # Compute label distribution per subject
    label_counts_per_subject = []

    for subj_idx, y in enumerate(Y):
        label_counts = {0: 0, 1: 0, 2: 0}
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            label_counts[label] = count

        label_counts_per_subject.append((subj_idx + 1, label_counts[0], label_counts[1], label_counts[2]))

    # Compute overall label counts
    total_counts = np.sum([[count[1], count[2], count[3]] for count in label_counts_per_subject], axis=0)

    # Write to file
    with open(stats_file, "w") as f:  # Open file in write mode
        f.write(f"Number of subjects: {len(X)}\n")
        f.write(f"Input features: {X[0].shape[1]}\n")  # Assuming [time, features]

        total_samples = sum(len(subj) for subj in Y)
        f.write(f"Total samples: {total_samples:,}\n")

        unique_labels = np.unique(np.concatenate(Y))
        f.write(f"Unique labels: {unique_labels}\n")

        # Write overall label-specific counts
        f.write(f"Samples with label 0: {total_counts[0]:,}\n")
        f.write(f"Samples with label 1: {total_counts[1]:,}\n")
        f.write(f"Samples with label 2: {total_counts[2]:,}\n")

        # Write per-subject statistics
        f.write("\nSubject-wise Sample Distribution:\n")
        f.write("Subject_ID | Label_0 | Label_1 | Label_2\n")
        f.write("------------------------------------\n")
        for subj_idx, count_0, count_1, count_2 in label_counts_per_subject:
            f.write(f"{subj_idx:<10} | {count_0:<7} | {count_1:<7} | {count_2:<7}\n")


def plot_sample_distribution(Y):
    """Plot distribution of samples per subject with label breakdown, sorted by total number of samples"""
    label_colors = {0: "blue", 1: "green", 2: "red"}
    label_names = {0: "Label 0", 1: "Label 1", 2: "Label 2"}

    num_subjects = len(Y)
    label_counts = {0: [], 1: [], 2: []}
    total_counts = []

    # Count samples for each label per subject
    for y in Y:
        counts = {label: np.sum(np.array(y) == label) for label in [0, 1, 2]}
        for label in [0, 1, 2]:
            label_counts[label].append(counts[label])
        total_counts.append(sum(counts.values()))

    # Sort subjects by total number of samples
    sorted_indices = np.argsort(total_counts)
    for label in [0, 1, 2]:
        label_counts[label] = np.array(label_counts[label])[sorted_indices]
    total_counts = np.array(total_counts)[sorted_indices]

    # Stack plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(num_subjects)  # Track the bottom for stacking

    for label in [0, 1, 2]:
        ax.bar(
            range(num_subjects),
            label_counts[label],
            label=label_names[label],
            color=label_colors[label],
            bottom=bottom,
        )
        bottom += np.array(label_counts[label])  # Update bottom for next stack

    ax.set_title(f"Sample Distribution by Label (Sorted by Total Samples)")
    ax.set_xlabel("Subject ID (Sorted)")
    ax.set_ylabel("Number of Samples")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.savefig(os.path.join(SAVE_DIR, f"sample_distribution_sorted.png"), dpi=300)
    plt.close()


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


def analyze_segments(X, Y):
    """
    1. Segment continuous 1, 2 segments based on `Y`.
    2. For each subject (subjects 5,6,7), randomly select one segment for each label (1, 2) for visualization.
    3. For each selected segment, also plot the before and after period of the same length as the segment, and highlight the during period.
    """
    save_dir = os.path.join(SAVE_DIR, "segments")
    os.makedirs(save_dir, exist_ok=True)

    subject_indices = [4, 5, 6]

    for subj_idx in subject_indices:
        imu_data = X[subj_idx]  # (T, F)
        labels = Y[subj_idx]  # (T,)

        # Get segments for labels 1, 2 based on Y
        segments_dict = segment_by_label(labels)

        # Randomly select one segment for each label (1, 2)
        selected_segments = {}
        for label in [1, 2]:
            if len(segments_dict[label]) > 0:
                selected_segments[label] = random.choice(segments_dict[label])

        # For each selected segment, plot the before, during, and after periods.
        for label, (start, end) in selected_segments.items():
            seg_length = end - start

            # Define the before and after segments with the same length.
            before_start = max(0, start - seg_length)
            after_end = min(len(imu_data), end + seg_length)

            # Extract the combined data: before, during, and after.
            combined_data = imu_data[before_start:after_end, :]
            time_combined = np.arange(before_start, after_end) / SAMPLING_FREQ

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            # Plot accelerometer data (features 1-3) on the first subplot
            ax1.set_ylabel("Accelerometer (m/s²)", color="blue")
            accel_colors = ["blue", "royalblue", "dodgerblue"]
            for feature_idx in range(3):  # Features 1-3
                ax1.plot(
                    time_combined,
                    combined_data[:, feature_idx],
                    label=f"Accel {['X', 'Y', 'Z'][feature_idx]}",
                    color=accel_colors[feature_idx],
                    alpha=0.8,
                )
            ax1.tick_params(axis="y", labelcolor="blue")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)
            ax1.axvspan(start / SAMPLING_FREQ, end / SAMPLING_FREQ, color="red", alpha=0.2, label="During period")
            ax1.set_title(f"Subject {subj_idx+1} - Label {label} - Time Domain")

            # Plot gyroscope data (features 4-6) on the second subplot
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Gyroscope (rad/s)", color="red")
            gyro_colors = ["red", "tomato", "firebrick"]
            for feature_idx in range(3, 6):  # Features 4-6
                ax2.plot(
                    time_combined,
                    combined_data[:, feature_idx],
                    label=f"Gyro {['X', 'Y', 'Z'][feature_idx - 3]}",
                    color=gyro_colors[feature_idx - 3],
                    alpha=0.8,
                )
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.legend(loc="upper left")
            ax2.grid(True, alpha=0.3)
            ax2.axvspan(start / SAMPLING_FREQ, end / SAMPLING_FREQ, color="red", alpha=0.2, label="During period")

            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f"subj{subj_idx+1}_label{label}_time.png"),
                dpi=300,
            )
            plt.close()


def main():
    # Load and analyze both sides together
    X, Y = load_data()
    basic_statistics(X, Y)
    plot_sample_distribution(Y)
    analyze_segments(X, Y)


if __name__ == "__main__":
    main()
    print(f"Analysis saved to: {os.path.abspath(SAVE_DIR)}")
