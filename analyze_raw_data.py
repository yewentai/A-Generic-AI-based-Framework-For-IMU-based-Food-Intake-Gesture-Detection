#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Raw Data Analysis Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-07
Description : This script performs analysis on raw IMU data, computes basic statistics,
              visualizes sample distributions, and segments gestures for further inspection.
===============================================================================
"""

import os
import sys
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Configuration
DATA_PATH = "./dataset/FD/FD-I/"
SAVE_DIR = "./analysis/FD-I/"
os.makedirs(SAVE_DIR, exist_ok=True)


def load_data(side="L"):
    """Load data for specified body side"""
    with open(os.path.join(DATA_PATH, f"X_{side}.pkl"), "rb") as f:
        X = pickle.load(f)
    with open(os.path.join(DATA_PATH, f"Y_{side}.pkl"), "rb") as f:
        Y = pickle.load(f)
    return X, Y


def basic_statistics(X, Y, side):
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
    with open(stats_file, "a") as f:  # Append to existing file
        f.write(f"\nBasic Statistics ({side} side):\n")
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

    print(f"Basic Statistics ({side} side) are saved to:", stats_file)


def plot_sample_distribution(Y, side):
    """Plot distribution of samples per subject with label breakdown"""
    label_colors = {0: "blue", 1: "green", 2: "red"}
    label_names = {0: "Label 0", 1: "Label 1", 2: "Label 2"}

    num_subjects = len(Y)
    label_counts = {0: [], 1: [], 2: []}

    # Count samples for each label per subject
    for y in Y:
        counts = {label: np.sum(np.array(y) == label) for label in [0, 1, 2]}
        for label in [0, 1, 2]:
            label_counts[label].append(counts[label])

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

    ax.set_title(f"Sample Distribution by Label ({side} side)")
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Number of Samples")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.savefig(os.path.join(SAVE_DIR, f"sample_distribution_{side}.png"), dpi=150)
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


def analyze_segments(X, Y, side="L"):
    """
    1. Segment continuous 1, 2 segments based on `Y`.
    2. For each subject (subjects 5,6,7), randomly select one segment for each label (1, 2) for visualization.
    3. For each selected segment, also plot the before and after period of the same length as the segment, and highlight the during period.
    """
    save_dir = os.path.join(SAVE_DIR, "segments")
    os.makedirs(save_dir, exist_ok=True)

    fs = 16  # Sampling frequency (Hz)
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
            time_combined = np.arange(before_start, after_end) / fs

            plt.figure(figsize=(12, 6))
            num_features = combined_data.shape[1]
            for feature_idx in range(num_features):
                plt.plot(
                    time_combined,
                    combined_data[:, feature_idx],
                    label=f"Feature {feature_idx+1}",
                    alpha=0.7,
                )

            # Highlight the "during" period.
            plt.axvspan(start / fs, end / fs, color="red", alpha=0.2, label="During period")

            plt.title(f"Subject {subj_idx+1} - Label {label} - Time Domain ({side})")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(
                os.path.join(save_dir, f"subj{subj_idx+1}_label{label}_time_{side}.png"),
                dpi=150,
            )
            plt.close()


def main():
    # Load and analyze both sides
    for side in ["L", "R"]:
        print(f"\nAnalyzing {side} side data...")
        X, Y = load_data(side)

        # Basic statistics
        basic_statistics(X, Y, side)
        plot_sample_distribution(Y, side)
        # analyze_segments(X, Y, side=side)


if __name__ == "__main__":
    main()
    print(f"Analysis saved to: {os.path.abspath(SAVE_DIR)}")
