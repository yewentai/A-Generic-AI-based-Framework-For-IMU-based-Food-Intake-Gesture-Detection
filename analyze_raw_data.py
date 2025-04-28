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
from matplotlib.patches import Patch

# Configuration
DATASET = "FDI"  # Options: "DXI", "DXII", "FDI", "FDII", "FDIII", "Clemson", "Oreba"

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


def segment_by_label(Y):
    """
    Identify continuous segments by labels (auto-detect labels).

    Returns:
        segments_dict: {label_value: [(start_idx, end_idx), ...]}
    """
    unique_labels = np.unique(Y)
    segments_dict = {label: [] for label in unique_labels}

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


def basic_statistics(Y):
    """Print detailed dataset statistics and write to a txt file, including segment counts."""
    stats_file = os.path.join(SAVE_DIR, "basic_statistics.txt")

    # 1. Dynamically retrieve all unique labels
    unique_labels = np.unique(np.concatenate(Y))

    # 2. Per-subject statistics
    sample_counts_per_subject = []
    segment_counts_per_subject = []

    for subj_idx, y in enumerate(Y):
        # Count samples per label
        sample_counts = {label: 0 for label in unique_labels}
        uniq, cnts = np.unique(y, return_counts=True)
        for lab, c in zip(uniq, cnts):
            sample_counts[lab] = c
        sample_counts_per_subject.append((subj_idx + 1,) + tuple(sample_counts[lab] for lab in unique_labels))

        # Count segments per label
        segs = segment_by_label(y)
        segment_counts = {label: len(segs[label]) for label in unique_labels}
        segment_counts_per_subject.append((subj_idx + 1,) + tuple(segment_counts[lab] for lab in unique_labels))

    # 3. Aggregate totals
    sample_arr = np.array([tup[1:] for tup in sample_counts_per_subject])
    total_sample_counts = np.sum(sample_arr, axis=0)
    segment_arr = np.array([tup[1:] for tup in segment_counts_per_subject])
    total_segment_counts = np.sum(segment_arr, axis=0)

    # 4. Write statistics to file
    with open(stats_file, "w") as f:
        f.write(f"Number of subjects: {len(Y)}\n")
        f.write(f"Sequence length (first subject): {len(Y[0])}\n\n")

        f.write("Overall Sample Counts:\n")
        for lab, tot in zip(unique_labels, total_sample_counts):
            f.write(f"Samples with label {lab}: {tot:,}\n")

        f.write("\nOverall Segment Counts:\n")
        for lab, tot in zip(unique_labels, total_segment_counts):
            f.write(f"Segments with label {lab}: {tot:,}\n")

        # Per-subject sample distribution
        header = "Subject_ID | " + " | ".join(f"Label_{int(lab)}" for lab in unique_labels)
        f.write("\nSubject-wise Sample Distribution:\n")
        f.write(header + "\n" + "-" * len(header) + "\n")
        for tup in sample_counts_per_subject:
            sid, *counts = tup
            f.write(f"{sid:<10} | " + " | ".join(f"{c:<7}" for c in counts) + "\n")

        # Per-subject segment distribution
        header2 = "Subject_ID | " + " | ".join(f"Label_{int(lab)}" for lab in unique_labels)
        f.write("\nSubject-wise Segment Distribution:\n")
        f.write(header2 + "\n" + "-" * len(header2) + "\n")
        for tup in segment_counts_per_subject:
            sid, *counts = tup
            f.write(f"{sid:<10} | " + " | ".join(f"{c:<7}" for c in counts) + "\n")


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


def plot_all_subjects_labels(Y, sampling_freq=SAMPLING_FREQ):
    """
    Plot all subjects' label sequences with:
        • A gray outline showing the full duration of each subject
        • Filled bars for non-zero labels only
    """
    # Dynamically retrieve all unique labels and assign colors (using a colormap)
    unique_labels = np.unique(np.concatenate(Y))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    color_map = {lab: cmap[i] for i, lab in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(18, len(Y) * 0.7 + 4))
    ax.set_facecolor("#FFFFFF")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E0E0E0", linestyle="-", linewidth=0.8)

    # Draw the full duration outline for each subject
    for subj_idx, y in enumerate(Y):
        duration = len(y) / sampling_freq
        y_center = subj_idx + 0.5
        ax.broken_barh(
            [(0, duration)],
            (y_center - 0.4, 0.8),
            facecolors="none",
            edgecolor="#888888",
            linewidth=1.0,
            zorder=1,
        )

    # Draw filled bars for non-zero labels
    for subj_idx, y in enumerate(Y):
        segments = segment_by_label(y)
        y_center = subj_idx + 0.5
        for lab, seg_list in segments.items():
            if lab == 0:
                continue
            for s, e in seg_list:
                start_sec = s / sampling_freq
                length_sec = (e - s) / sampling_freq
                ax.broken_barh(
                    [(start_sec, length_sec)],
                    (y_center - 0.4, 0.8),
                    facecolors=color_map[lab],
                    edgecolor=None,
                    linewidth=0,
                    alpha=1.0,
                    zorder=2,
                )

    # Add a legend for non-zero labels
    legend_elements = [Patch(facecolor=color_map[lab], label=f"Label {lab}") for lab in unique_labels if lab != 0]
    ax.legend(
        handles=legend_elements,
        title="Activity Labels",
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        fontsize=12,
        title_fontsize=14,
        borderpad=1,
    )

    # Configure axes
    ax.invert_yaxis()
    ax.set_ylim(0, len(Y))
    max_time = max(len(y) for y in Y) / sampling_freq
    ax.set_xlim(0, max_time)

    ax.set_xlabel("Time (seconds)", fontsize=14, labelpad=10, fontweight="semibold")
    ax.set_ylabel("Subject ID", fontsize=14, labelpad=10, fontweight="semibold")
    ax.set_yticks(np.arange(len(Y)) + 0.5)
    ax.set_yticklabels([f"Subject {i+1}" for i in range(len(Y))], fontsize=12)
    ax.tick_params(axis="x", labelsize=12, length=0)

    # Add minute-based ticks if the timeline exceeds 60 seconds
    if max_time > 60:

        def time_formatter(x, _):
            return f"{int(x//60)}:{int(x%60):02d}"

        ax.xaxis.set_major_formatter(plt.FuncFormatter(time_formatter))
        ax_top = ax.secondary_xaxis("top")
        ax_top.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{round(x/60, 1)} min"))
        ax_top.tick_params(labelsize=10)

    # Beautify the plot borders
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#808080")

    ax.set_title("Subject Activity Timeline", fontsize=16, pad=20, fontweight="bold", color="#333333")

    plt.tight_layout()
    plt.savefig(
        os.path.join(SAVE_DIR, "all_subjects_labels.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close()


def main():
    # Load and analyze both sides together
    X, Y = load_data()
    # basic_statistics(Y)
    plot_all_subjects_labels(Y)
    # plot_sample_distribution(Y)
    # plot_segment_length_distribution_by_label(Y)
    # plot_longest_segments(X, Y)
    # plot_shortest_segments(X, Y)


if __name__ == "__main__":
    main()
    print(f"Analysis saved to: {os.path.abspath(SAVE_DIR)}")
