import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from utils import nonzero_intervals_value


def _plot_span(ax, interval, text, color):
    """Helper for debug visualization"""
    ax.axvspan(interval[0], interval[1], alpha=0.3, color=color)
    ax.text(
        (interval[0] + interval[1]) / 2,
        0.5,
        text,
        ha="center",
        va="center",
        color=color,
    )


def nonzero_intervals_value(x):
    # TODO: add input class_label to filter by class
    """
    Extract start and end indices of nonzero intervals in an array and their corresponding values.

    Parameters:
        x (np.ndarray): Input 1D array.

    Returns:
        np.ndarray: An array of shape (N, 3), where each row represents
                    [start, end, value] for a contiguous interval.
    """
    # Prepare a list to collect results
    results = []

    # Track the current interval
    in_segment = False
    start_idx = 0
    current_value = 0

    for i in range(len(x)):
        if x[i] != 0 and not in_segment:
            # Start of a new interval
            in_segment = True
            start_idx = i
            current_value = x[i]
        elif (x[i] != current_value or x[i] == 0) and in_segment:
            # End of the current interval
            results.append([start_idx, i, current_value])
            in_segment = False
            # Restart a new interval if the current value is nonzero
            if x[i] != 0:
                in_segment = True
                start_idx = i
                current_value = x[i]

    # Handle the case where the sequence ends with a nonzero interval
    if in_segment:
        results.append([start_idx, len(x), current_value])

    # Convert the results list to a NumPy array
    results_array = np.array(results, dtype=int)

    # Ensure the result is always 2D
    if results_array.ndim == 1:  # If it's a 1D array (e.g., empty or a single interval)
        results_array = results_array.reshape(-1, 3)

    return results_array


def segment_evaluation(pred, gt, class_label, threshold=0.5, debug_plot=False):
    """
    Unified segmentation confusion matrix calculator using value-aware interval matching.

    Parameters:
        pred (np.ndarray): Predicted segmentation array
        gt (np.ndarray): Ground truth segmentation array
        class_label (int): The class label for which to compute metrics
        threshold (float): IoU threshold for true positive classification
        debug_plot (bool): Enable visualization of matching logic

    Returns:
        tuple: (false_negatives, false_positives, true_positives) for the specified class
    """
    fn, fp, tp = 0, 0, 0

    # Extract intervals for the specified class
    gt_intervals = nonzero_intervals_value(gt)
    pred_intervals = nonzero_intervals_value(pred)

    # Filter intervals to include only the specified class
    gt_class_intervals = [
        interval for interval in gt_intervals if interval[2] == class_label
    ]
    pred_class_intervals = [
        interval for interval in pred_intervals if interval[2] == class_label
    ]

    if debug_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        ax1.step(range(len(gt)), gt, where="post", label="GT", color="blue")
        ax2.step(range(len(pred)), pred, where="post", label="Pred", color="red")

    # TODO: Implement Union before cost matrix calculation

    # Build cost matrix for Hungarian matching
    cost_matrix = np.zeros((len(gt_class_intervals), len(pred_class_intervals)))
    for i, (gs, ge, gv) in enumerate(gt_class_intervals):
        for j, (ps, pe, pv) in enumerate(pred_class_intervals):
            # Since we've filtered by class_label, gv and pv should both be class_label
            # Proceed to calculate IoU
            inter_start = max(gs, ps)
            inter_end = min(ge, pe)
            inter = max(0, inter_end - inter_start)
            union = (ge - gs) + (pe - ps) - inter
            cost_matrix[i, j] = inter / union if union else 0

    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    matched = set()
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] >= threshold:
            tp += 1
            matched.add((r, c))
            if debug_plot:
                _plot_span(ax1, gt_class_intervals[r][:2], "TP", "green")
                _plot_span(ax2, pred_class_intervals[c][:2], "TP", "green")

    # Identify remaining intervals
    matched_gt = set(r for r, _ in matched)
    matched_pred = set(c for _, c in matched)

    remaining_gt = [
        (i, (s, e, v))
        for i, (s, e, v) in enumerate(gt_class_intervals)
        if i not in matched_gt
    ]
    remaining_pred = [
        (j, (s, e, v))
        for j, (s, e, v) in enumerate(pred_class_intervals)
        if j not in matched_pred
    ]

    # Greedy matching for remaining intervals
    gt_dict = defaultdict(list)
    for idx, (s, e, v) in remaining_gt:
        gt_dict[v].append((s, e, idx))

    pred_dict = defaultdict(list)
    for idx, (s, e, v) in remaining_pred:
        pred_dict[v].append((s, e, idx))

    used_gt = set()
    used_pred = set()

    # Process each class separately (in this case, only the specified class)
    for value in gt_dict:
        if value != class_label:
            continue  # Skip if not the specified class
        if value not in pred_dict:
            continue

        # Find best overlaps between remaining GT and Pred
        for gt_idx, (gs, ge, _) in enumerate(gt_dict[value]):
            max_overlap = 0
            best_pred = None
            for pred_idx, (ps, pe, _) in enumerate(pred_dict[value]):
                overlap_start = max(gs, ps)
                overlap_end = min(ge, pe)
                overlap = max(0, overlap_end - overlap_start)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_pred = pred_idx

            if best_pred is not None and max_overlap > 0:
                # Mark as paired
                used_gt.add(gt_dict[value][gt_idx][2])  # original index
                used_pred.add(pred_dict[value][best_pred][2])

                # Compare lengths of GT and Pred intervals
                gt_length = ge - gs
                pred_length = pe - ps

                if gt_length > pred_length:
                    # GT is longer, mark as FN
                    fn += 1
                    if debug_plot:
                        _plot_span(ax1, (gs, ge), "FN", "blue")
                        _plot_span(ax2, (ps, pe), "FN", "red")
                else:
                    # Pred is longer, mark as FP
                    fp += 1
                    if debug_plot:
                        _plot_span(ax2, (ps, pe), "FP", "red")
                        _plot_span(ax1, (gs, ge), "FP", "blue")

    # Count unpaired intervals
    for idx, (s, e, v) in remaining_gt:
        if idx not in used_gt:
            fn += 1
            if debug_plot:
                _plot_span(ax1, (s, e), "FN", "blue")

    for idx, (s, e, v) in remaining_pred:
        if idx not in used_pred:
            fp += 1
            if debug_plot:
                _plot_span(ax2, (s, e), "FP", "red")

    if debug_plot:
        plt.tight_layout()
        plt.show()

    return fn, fp, tp
