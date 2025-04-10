#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Experiment Comparison Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-10
Description : This script compares results across multiple experiment versions,
              analyzing both mirrored and non-mirrored data conditions. It
              generates comparative visualizations and statistical analyses.
===============================================================================
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Constants - assumed to be the same across all experiments
THRESHOLD_LIST = [0.1, 0.25, 0.5, 0.75]


def load_experiment_data(result_dir, version, mirror_flag):
    """Load validation statistics and config for a specific experiment version."""
    stats_file = os.path.join(
        result_dir, version, "validate_stats_mirrored.npy" if mirror_flag else "validate_stats.npy"
    )

    config_file = os.path.join(result_dir, version, "config.json")

    if not os.path.exists(stats_file):
        print(f"Warning: Stats file not found for {version} (mirror={mirror_flag}): {stats_file}")
        return None, None

    if not os.path.exists(config_file):
        print(f"Warning: Config file not found for {version}: {config_file}")
        return None, None

    try:
        stats = np.load(stats_file, allow_pickle=True).tolist()

        with open(config_file, "r") as f:
            config = json.load(f)

        return stats, config
    except Exception as e:
        print(f"Error loading data for {version} (mirror={mirror_flag}): {e}")
        return None, None


def extract_metrics(stats):
    """Extract key metrics from validation statistics."""
    if not stats:
        return None

    # Sample-wise metrics
    sample_f1_scores = [fold_stat["metrics_sample"]["weighted_f1"] for fold_stat in stats]

    # Segment-wise metrics at different thresholds
    segment_f1_scores = {}
    for threshold in THRESHOLD_LIST:
        segment_f1_scores[threshold] = [
            fold_stat["metrics_segment"][str(threshold)]["weighted_f1"] for fold_stat in stats
        ]

    # Extract per-class F1 scores (using threshold=0.5 for segment-wise)
    classes = set()
    for fold_stat in stats:
        for class_id in fold_stat["metrics_sample"].keys():
            if class_id.isdigit() and int(class_id) > 0:  # Skip background class and non-numeric keys
                classes.add(int(class_id))

    per_class_f1_sample = {}
    per_class_f1_segment = {}

    for class_id in sorted(classes):
        per_class_f1_sample[class_id] = [fold_stat["metrics_sample"][str(class_id)]["f1"] for fold_stat in stats]

        per_class_f1_segment[class_id] = [
            fold_stat["metrics_segment"]["0.5"][str(class_id)]["f1"] for fold_stat in stats  # Using threshold 0.5
        ]

    return {
        "sample_f1_weighted": sample_f1_scores,
        "segment_f1_weighted": segment_f1_scores,
        "per_class_f1_sample": per_class_f1_sample,
        "per_class_f1_segment": per_class_f1_segment,
    }


def compute_summary_statistics(metrics):
    """Compute summary statistics for the metrics."""
    if not metrics:
        return None

    summary = {}

    # Sample-wise weighted F1
    summary["sample_f1_weighted"] = {
        "mean": np.mean(metrics["sample_f1_weighted"]),
        "std": np.std(metrics["sample_f1_weighted"]),
        "median": np.median(metrics["sample_f1_weighted"]),
        "min": np.min(metrics["sample_f1_weighted"]),
        "max": np.max(metrics["sample_f1_weighted"]),
    }

    # Segment-wise weighted F1 at different thresholds
    summary["segment_f1_weighted"] = {}
    for threshold, scores in metrics["segment_f1_weighted"].items():
        summary["segment_f1_weighted"][threshold] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "median": np.median(scores),
            "min": np.min(scores),
            "max": np.max(scores),
        }

    # Per-class F1 scores
    summary["per_class_f1_sample"] = {}
    for class_id, scores in metrics["per_class_f1_sample"].items():
        summary["per_class_f1_sample"][class_id] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "median": np.median(scores),
            "min": np.min(scores),
            "max": np.max(scores),
        }

    summary["per_class_f1_segment"] = {}
    for class_id, scores in metrics["per_class_f1_segment"].items():
        summary["per_class_f1_segment"][class_id] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "median": np.median(scores),
            "min": np.min(scores),
            "max": np.max(scores),
        }

    return summary


def plot_comparison_boxplot(all_results, metric_name, output_dir, mirror_condition="both"):
    """Create boxplot comparison across experiment versions."""
    plt.figure(figsize=(14, 8))

    data = []
    labels = []

    for version, results in all_results.items():
        if mirror_condition == "both" or mirror_condition == "no_mirror":
            if "no_mirror" in results and results["no_mirror"]["metrics"]:
                if metric_name.startswith("segment_f1_weighted"):
                    # Extract threshold from metric name
                    _, threshold = metric_name.rsplit("_", 1)
                    scores = results["no_mirror"]["metrics"]["segment_f1_weighted"][float(threshold)]
                    data.append(scores)
                    labels.append(f"{version} (No Mirror)")
                else:
                    data.append(results["no_mirror"]["metrics"][metric_name])
                    labels.append(f"{version} (No Mirror)")

        if mirror_condition == "both" or mirror_condition == "mirror":
            if "mirror" in results and results["mirror"]["metrics"]:
                if metric_name.startswith("segment_f1_weighted"):
                    # Extract threshold from metric name
                    _, threshold = metric_name.rsplit("_", 1)
                    scores = results["mirror"]["metrics"]["segment_f1_weighted"][float(threshold)]
                    data.append(scores)
                    labels.append(f"{version} (Mirror)")
                else:
                    data.append(results["mirror"]["metrics"][metric_name])
                    labels.append(f"{version} (Mirror)")

    if not data:
        print(f"No data available for metric: {metric_name}")
        plt.close()
        return

    sns.boxplot(data=data)

    metric_title = metric_name
    if metric_name.startswith("segment_f1_weighted"):
        _, threshold = metric_name.rsplit("_", 1)
        metric_title = f"Segment-wise Weighted F1 (Threshold={threshold})"
    elif metric_name == "sample_f1_weighted":
        metric_title = "Sample-wise Weighted F1"

    plt.title(f"Comparison of {metric_title} Across Experiments")
    plt.ylabel("F1 Score")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.tight_layout()

    # Create filename
    filename = f"comparison_{metric_name.replace('.', '')}"
    if mirror_condition != "both":
        filename += f"_{mirror_condition}"
    filename += ".png"

    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def plot_per_class_comparison(all_results, output_dir, mirror_condition="both", metric_type="sample"):
    """Create bar plot comparing per-class F1 scores across experiment versions."""
    # Gather all class IDs across all experiments
    all_classes = set()
    for version, results in all_results.items():
        for mirror_type in ["mirror", "no_mirror"]:
            if mirror_type in results and results[mirror_type]["metrics"]:
                all_classes.update(results[mirror_type]["metrics"][f"per_class_f1_{metric_type}"].keys())

    all_classes = sorted(all_classes)
    if not all_classes:
        print(f"No class data available for metric type: {metric_type}")
        return

    # Create plot
    plt.figure(figsize=(14, 8))

    bar_width = 0.8 / (len(all_results) * (2 if mirror_condition == "both" else 1))
    bar_positions = np.arange(len(all_classes))

    index = 0
    for version, results in all_results.items():
        if mirror_condition == "both" or mirror_condition == "no_mirror":
            if "no_mirror" in results and results["no_mirror"]["metrics"]:
                means = [
                    np.mean(results["no_mirror"]["metrics"][f"per_class_f1_{metric_type}"].get(class_id, [0]))
                    for class_id in all_classes
                ]
                plt.bar(bar_positions + index * bar_width, means, bar_width, label=f"{version} (No Mirror)")
                index += 1

        if mirror_condition == "both" or mirror_condition == "mirror":
            if "mirror" in results and results["mirror"]["metrics"]:
                means = [
                    np.mean(results["mirror"]["metrics"][f"per_class_f1_{metric_type}"].get(class_id, [0]))
                    for class_id in all_classes
                ]
                plt.bar(bar_positions + index * bar_width, means, bar_width, label=f"{version} (Mirror)")
                index += 1

    plt.title(f"Comparison of Per-Class {metric_type.capitalize()}-wise F1 Scores")
    plt.xlabel("Class ID")
    plt.ylabel("Mean F1 Score")
    plt.xticks(bar_positions + bar_width * (index - 1) / 2, all_classes)
    plt.legend(title="Experiment Version")
    plt.tight_layout()

    filename = f"class_comparison_{metric_type}"
    if mirror_condition != "both":
        filename += f"_{mirror_condition}"
    filename += ".png"

    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def create_summary_tables(all_results, output_dir):
    """Create summary tables of metrics across experiment versions."""
    # Table for weighted F1 scores
    weighted_f1_data = []

    for version, results in all_results.items():
        for mirror_type in ["no_mirror", "mirror"]:
            if mirror_type in results and results[mirror_type]["summary"]:
                summary = results[mirror_type]["summary"]
                config = results[mirror_type]["config"]

                row = {
                    "Version": version,
                    "Mirror": "Yes" if mirror_type == "mirror" else "No",
                    "Dataset": config.get("dataset", "Unknown"),
                    "Model": config.get("model", "Unknown"),
                    "Sample F1 (Mean)": f"{summary['sample_f1_weighted']['mean']:.4f}",
                    "Sample F1 (Std)": f"{summary['sample_f1_weighted']['std']:.4f}",
                }

                # Add segment-wise metrics for different thresholds
                for threshold in THRESHOLD_LIST:
                    if threshold in summary["segment_f1_weighted"]:
                        row[f"Segment F1 T={threshold} (Mean)"] = (
                            f"{summary['segment_f1_weighted'][threshold]['mean']:.4f}"
                        )
                        row[f"Segment F1 T={threshold} (Std)"] = (
                            f"{summary['segment_f1_weighted'][threshold]['std']:.4f}"
                        )

                weighted_f1_data.append(row)

    # Convert to DataFrame and save as CSV
    if weighted_f1_data:
        df_weighted = pd.DataFrame(weighted_f1_data)
        df_weighted.to_csv(os.path.join(output_dir, "weighted_f1_summary.csv"), index=False)

    # Tables for per-class F1 scores
    for metric_type in ["sample", "segment"]:
        per_class_data = []

        for version, results in all_results.items():
            for mirror_type in ["no_mirror", "mirror"]:
                if mirror_type in results and results[mirror_type]["summary"]:
                    summary = results[mirror_type]["summary"]
                    config = results[mirror_type]["config"]

                    if f"per_class_f1_{metric_type}" in summary:
                        class_summary = summary[f"per_class_f1_{metric_type}"]

                        for class_id, stats in class_summary.items():
                            row = {
                                "Version": version,
                                "Mirror": "Yes" if mirror_type == "mirror" else "No",
                                "Dataset": config.get("dataset", "Unknown"),
                                "Model": config.get("model", "Unknown"),
                                "Class": class_id,
                                "F1 (Mean)": f"{stats['mean']:.4f}",
                                "F1 (Std)": f"{stats['std']:.4f}",
                                "F1 (Min)": f"{stats['min']:.4f}",
                                "F1 (Max)": f"{stats['max']:.4f}",
                            }
                            per_class_data.append(row)

        if per_class_data:
            df_per_class = pd.DataFrame(per_class_data)
            df_per_class.to_csv(os.path.join(output_dir, f"per_class_f1_{metric_type}_summary.csv"), index=False)


def create_model_specs_table(all_results, output_dir):
    """Create a table of model specifications for each experiment."""
    model_specs = []

    for version, results in all_results.items():
        for mirror_type in ["no_mirror", "mirror"]:
            if mirror_type in results and results[mirror_type]["config"]:
                config = results[mirror_type]["config"]

                # Extract model-specific parameters
                model_type = config.get("model", "Unknown")
                row = {
                    "Version": version,
                    "Mirror": "Yes" if mirror_type == "mirror" else "No",
                    "Dataset": config.get("dataset", "Unknown"),
                    "Model": model_type,
                    "Input Dim": config.get("input_dim", "N/A"),
                    "Num Classes": config.get("num_classes", "N/A"),
                    "Window Size": config.get("window_size", "N/A"),
                    "Sampling Freq": config.get("sampling_freq", "N/A"),
                    "Batch Size": config.get("batch_size", "N/A"),
                }

                # Add model-specific parameters
                if model_type == "MSTCN":
                    row.update(
                        {
                            "Num Stages": config.get("num_stages", "N/A"),
                            "Num Layers": config.get("num_layers", "N/A"),
                            "Num Filters": config.get("num_filters", "N/A"),
                            "Kernel Size": config.get("kernel_size", "N/A"),
                            "Dropout": config.get("dropout", "N/A"),
                        }
                    )
                elif model_type == "TCN":
                    row.update(
                        {
                            "Num Layers": config.get("num_layers", "N/A"),
                            "Num Filters": config.get("num_filters", "N/A"),
                            "Kernel Size": config.get("kernel_size", "N/A"),
                            "Dropout": config.get("dropout", "N/A"),
                        }
                    )
                elif model_type == "CNN_LSTM":
                    row.update(
                        {
                            "Conv Filters": config.get("conv_filters", "N/A"),
                            "LSTM Hidden": config.get("lstm_hidden", "N/A"),
                        }
                    )

                model_specs.append(row)

    if model_specs:
        df_specs = pd.DataFrame(model_specs)
        df_specs.to_csv(os.path.join(output_dir, "model_specifications.csv"), index=False)


def main():
    """Main function to compare results across experiment versions."""
    # Define directories
    result_root = "result"
    comparison_dir = os.path.join(result_root, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Get all result version folders
    versions = [d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d)) and d != "comparison"]
    versions.sort()

    print(f"Found {len(versions)} experiment versions: {versions}")

    # Container for all results
    all_results = {}

    # Load data for each version and mirror condition
    for version in versions:
        all_results[version] = {}

        # Load non-mirrored data
        stats_no_mirror, config_no_mirror = load_experiment_data(result_root, version, False)
        if stats_no_mirror and config_no_mirror:
            metrics_no_mirror = extract_metrics(stats_no_mirror)
            summary_no_mirror = compute_summary_statistics(metrics_no_mirror)
            all_results[version]["no_mirror"] = {
                "stats": stats_no_mirror,
                "config": config_no_mirror,
                "metrics": metrics_no_mirror,
                "summary": summary_no_mirror,
            }

        # Load mirrored data
        stats_mirror, config_mirror = load_experiment_data(result_root, version, True)
        if stats_mirror and config_mirror:
            metrics_mirror = extract_metrics(stats_mirror)
            summary_mirror = compute_summary_statistics(metrics_mirror)
            all_results[version]["mirror"] = {
                "stats": stats_mirror,
                "config": config_mirror,
                "metrics": metrics_mirror,
                "summary": summary_mirror,
            }

    # Generate comparison plots and tables
    print("Generating comparison visualizations...")

    # Sample-wise weighted F1 comparison
    plot_comparison_boxplot(all_results, "sample_f1_weighted", comparison_dir)
    plot_comparison_boxplot(all_results, "sample_f1_weighted", comparison_dir, "mirror")
    plot_comparison_boxplot(all_results, "sample_f1_weighted", comparison_dir, "no_mirror")

    # Segment-wise weighted F1 comparison for different thresholds
    for threshold in THRESHOLD_LIST:
        metric_name = f"segment_f1_weighted_{threshold}"
        plot_comparison_boxplot(all_results, metric_name, comparison_dir)
        plot_comparison_boxplot(all_results, metric_name, comparison_dir, "mirror")
        plot_comparison_boxplot(all_results, metric_name, comparison_dir, "no_mirror")

    # Per-class F1 comparisons
    plot_per_class_comparison(all_results, comparison_dir, "both", "sample")
    plot_per_class_comparison(all_results, comparison_dir, "both", "segment")
    plot_per_class_comparison(all_results, comparison_dir, "mirror", "sample")
    plot_per_class_comparison(all_results, comparison_dir, "mirror", "segment")
    plot_per_class_comparison(all_results, comparison_dir, "no_mirror", "sample")
    plot_per_class_comparison(all_results, comparison_dir, "no_mirror", "segment")

    # Generate summary tables
    print("Creating summary tables...")
    create_summary_tables(all_results, comparison_dir)
    create_model_specs_table(all_results, comparison_dir)

    print(f"Comparison completed! Results saved to {comparison_dir}")


if __name__ == "__main__":
    main()
