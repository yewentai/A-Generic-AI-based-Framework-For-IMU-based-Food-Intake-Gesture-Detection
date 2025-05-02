#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Segmentation Evaluation Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-02
Description : This script provides evaluation functions for IMU segmentation tasks.
              It matches predicted and ground truth intervals for a target class,
              computes overlap-based metrics (e.g., precision, recall, F1), and supports
              optional debugging plots to visualize true/false positives and negatives.
===============================================================================
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def get_target_intervals(x, class_label):
    """
    Extract start and end indices of intervals where x[i] equals class_label.

    Parameters:
        x (np.ndarray): Input 1D array.
        class_label (int): Target class label.

    Returns:
        np.ndarray: Array of [start, end) interval indices.
    """
    results = []
    in_segment = False
    start_idx = 0

    for i in range(len(x)):
        if x[i] == class_label and not in_segment:
            in_segment = True
            start_idx = i
        elif x[i] != class_label and in_segment:
            in_segment = False
            results.append([start_idx, i])

    if in_segment:
        results.append([start_idx, len(x)])

    return np.array(results, dtype=int)


def get_union_intervals(pred_intervals, gt_intervals):
    """
    Merge prediction and ground truth intervals into union intervals,
    tracking which original intervals from each contribute to the merged intervals.

    Parameters:
        pred_intervals (np.ndarray): Array of [start, end) intervals for predictions.
        gt_intervals (np.ndarray): Array of [start, end) intervals for ground truth.

    Returns:
        list of dict: Each dict contains the merged 'range', 'pred_ranges', and 'gt_ranges'.
    """
    if len(pred_intervals) == 0 and len(gt_intervals) == 0:
        return []

    # Combine intervals with labels indicating their source
    labeled_intervals = []
    for p in pred_intervals:
        labeled_intervals.append((p[0], p[1], "pred"))
    for g in gt_intervals:
        labeled_intervals.append((g[0], g[1], "gt"))

    # Sort intervals by their start time
    labeled_intervals.sort(key=lambda x: x[0])

    merged = []
    if not labeled_intervals:
        return merged

    current_start, current_end, current_type = labeled_intervals[0]
    pred_ranges = []
    gt_ranges = []

    # Initialize the first interval
    if current_type == "pred":
        pred_ranges.append([current_start, current_end])
    else:
        gt_ranges.append([current_start, current_end])

    for interval in labeled_intervals[1:]:
        start, end, type_ = interval
        if start <= current_end:
            # Overlapping or adjacent, merge them
            current_end = max(current_end, end)
            if type_ == "pred":
                pred_ranges.append([start, end])
            else:
                gt_ranges.append([start, end])
        else:
            # Non-overlapping, finalize the current merged interval
            merged.append(
                {
                    "pred_ranges": pred_ranges.copy(),
                    "gt_ranges": gt_ranges.copy(),
                }
            )
            # Start new interval
            current_start, current_end, current_type = start, end, type_
            pred_ranges = [[start, end]] if type_ == "pred" else []
            gt_ranges = [[start, end]] if type_ == "gt" else []

    # Add the last merged interval
    merged.append(
        {
            "pred_ranges": pred_ranges.copy(),
            "gt_ranges": gt_ranges.copy(),
        }
    )

    return merged


def _plot_span(ax, interval, text, color):
    """
    Helper function for debug visualization.
    Plots a span on the given axis with the specified text and color.
    """
    ax.axvspan(interval[0], interval[1], alpha=0.3, color=color)
    ax.text(
        (interval[0] + interval[1]) / 2,
        0.5,
        text,
        ha="center",
        va="center",
        color=color,
    )


def segment_evaluation(pred, gt, class_label, threshold=0.5, debug_plot=False):
    """
    Calculate segmentation metrics using interval matching.

    Parameters:
        pred (np.ndarray): Predicted segmentation.
        gt (np.ndarray): Ground truth.
        class_label (int): Class to evaluate.
        threshold (float): IoU threshold for TP.
        debug_plot (bool): Enable visualization.

    Returns:
        tuple: (false_negatives, false_positives, true_positives).
    """
    if debug_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        ax1.step(range(len(gt)), gt, where="post", label="GT", color="blue")
        ax2.step(range(len(pred)), pred, where="post", label="Pred", color="red")

    fn, fp, tp = 0, 0, 0

    # Extract intervals for target class
    gt_intervals = get_target_intervals(gt, class_label)
    pred_intervals = get_target_intervals(pred, class_label)
    union = get_union_intervals(pred_intervals, gt_intervals)

    for interval in union:
        gt_subset = interval["gt_ranges"]
        pred_subset = interval["pred_ranges"]

        if len(gt_subset) == 0:
            fp += 1
            if debug_plot:
                for s, e in pred_subset:
                    _plot_span(ax2, (s, e), "FP", "red")
            continue

        if len(pred_subset) == 0:
            fn += 1
            if debug_plot:
                for s, e in gt_subset:
                    _plot_span(ax1, (s, e), "FN", "blue")
            continue

        # Build IoU matrix
        cost_matrix = np.zeros((len(gt_subset), len(pred_subset)))
        for i, (gs, ge) in enumerate(gt_subset):
            for j, (ps, pe) in enumerate(pred_subset):
                inter_start = max(gs, ps)
                inter_end = min(ge, pe)
                inter = max(0, inter_end - inter_start)
                union = (ge - gs) + (pe - ps) - inter
                cost_matrix[i, j] = inter / union if union else 0

        # Optimal matching
        gt_idx, pred_idx = linear_sum_assignment(-cost_matrix)
        matched = set()
        for r, c in zip(gt_idx, pred_idx):
            if cost_matrix[r, c] >= threshold:
                tp += 1
                matched.add((r, c))
                if debug_plot:
                    gs, ge = gt_subset[r][0], gt_subset[r][1]
                    ps, pe = pred_subset[c][0], pred_subset[c][1]
                    _plot_span(ax1, (gs, ge), "TP", "blue")
                    _plot_span(ax2, (ps, pe), "TP", "red")

        # Identify leftovers
        remaining_gt = [(i, (s, e)) for i, (s, e) in enumerate(gt_subset) if i not in {r for r, _ in matched}]
        remaining_pred = [(j, (s, e)) for j, (s, e) in enumerate(pred_subset) if j not in {c for _, c in matched}]

        # Greedy overlap matching for leftovers
        used_gt = set()
        used_pred = set()

        # Match remaining GT to Pred
        for gt_id, (gs, ge) in remaining_gt:
            max_overlap = 0
            best_pred = None

            for pred_id, (ps, pe) in remaining_pred:
                if pred_id in used_pred:
                    continue

                overlap = max(0, min(ge, pe) - max(gs, ps))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_pred = pred_id

            if best_pred is not None and max_overlap > 0:
                used_gt.add(gt_id)
                used_pred.add(best_pred)
                remaining_pred_dict = {j: (s, e) for j, (s, e) in remaining_pred}
                ps, pe = remaining_pred_dict.get(best_pred, (None, None))

                # Compare lengths
                if (ge - gs) > (pe - ps):
                    fn += 1
                    if debug_plot:
                        _plot_span(ax1, (gs, ge), "FN", "blue")
                        _plot_span(ax2, (ps, pe), "FN", "red")
                else:
                    fp += 1
                    if debug_plot:
                        _plot_span(ax1, (gs, ge), "FP", "blue")
                        _plot_span(ax2, (ps, pe), "FP", "red")

        # Count completely unpaired intervals
        for gt_id, (gs, ge) in remaining_gt:
            if gt_id not in used_gt:
                fn += 1
                if debug_plot:
                    _plot_span(ax1, (gs, ge), "FN", "blue")

        for pred_id, (ps, pe) in remaining_pred:
            if pred_id not in used_pred:
                fp += 1
                if debug_plot:
                    _plot_span(ax2, (ps, pe), "FP", "red")

    if debug_plot:
        plt.tight_layout()
        plt.show()

    return fn, fp, tp
