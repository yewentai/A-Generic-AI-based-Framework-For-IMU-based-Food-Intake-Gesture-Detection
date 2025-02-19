import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


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


def get_target_intervals(x, class_label):
    """
    Extract start and end indices of intervals where x[i] equals class_label.

    Parameters:
        x (np.ndarray): Input 1D array
        class_label (int): Target class label

    Returns:
        np.ndarray: Array of [start, end) interval indices
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
    Merge overlapping intervals from predictions and ground truth.

    Parameters:
        pred_intervals (np.ndarray): Prediction intervals
        gt_intervals (np.ndarray): Ground truth intervals

    Returns:
        list: Merged intervals with source indices
    """
    if len(pred_intervals) == 0 and len(gt_intervals) == 0:
        return []

    all_intervals = np.vstack((pred_intervals, gt_intervals))
    all_intervals = all_intervals[np.argsort(all_intervals[:, 0])]

    merged = []
    current_start, current_end = all_intervals[0]
    pred_ids = set()
    gt_ids = set()

    # Initialize first interval
    if 0 < len(pred_intervals):
        pred_ids.add(0)
    else:
        gt_ids.add(0 - len(pred_intervals))

    for i in range(1, len(all_intervals)):
        start, end = all_intervals[i]

        if start <= current_end:
            current_end = max(current_end, end)
            # Track source indices
            if i < len(pred_intervals):
                pred_ids.add(i)
            else:
                gt_ids.add(i - len(pred_intervals))
        else:
            merged.append(
                {
                    "range": (current_start, current_end),
                    "pred_indices": list(pred_ids),
                    "gt_indices": list(gt_ids),
                }
            )
            current_start, current_end = start, end
            pred_ids = set()
            gt_ids = set()
            if i < len(pred_intervals):
                pred_ids.add(i)
            else:
                gt_ids.add(i - len(pred_intervals))

    merged.append(
        {
            "range": (current_start, current_end),
            "pred_indices": list(pred_ids),
            "gt_indices": list(gt_ids),
        }
    )

    return merged


def segment_evaluation(pred, gt, class_label, threshold=0.5, debug_plot=False):
    """
    Calculate segmentation metrics using interval matching.

    Parameters:
        pred (np.ndarray): Predicted segmentation
        gt (np.ndarray): Ground truth
        class_label (int): Class to evaluate
        threshold (float): IoU threshold for TP
        debug_plot (bool): Enable visualization

    Returns:
        tuple: (false_negatives, false_positives, true_positives)
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
        # Get contributing intervals
        if debug_plot:
            print("In interval")
            print(interval["range"])
        gt_subset = [gt_intervals[i] for i in interval["gt_indices"]]
        pred_subset = [pred_intervals[i] for i in interval["pred_indices"]]

        if len(gt_subset) == 0 and len(pred_subset) == 1:
            fp += 1
            if debug_plot:
                for _, (s, e) in pred_subset:
                    _plot_span(ax2, (s, e), "FP", "red")
            continue

        elif len(pred_subset) == 0 and len(gt_subset) == 1:
            fn += 1
            if debug_plot:
                for _, (s, e) in gt_subset:
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

        # Identify leftovers
        remaining_gt = [
            (i, (s, e))
            for i, (s, e) in enumerate(gt_subset)
            if i not in {r for r, _ in matched}
        ]
        remaining_pred = [
            (j, (s, e))
            for j, (s, e) in enumerate(pred_subset)
            if j not in {c for _, c in matched}
        ]

        # Greedy overlap matching for leftovers
        used_gt = set()
        used_pred = set()

        # Match remaining GT to Pred
        for gt_id, (gs, ge) in remaining_gt:
            max_overlap = 0
            best_pred = None

            for pred_id, (ps, pe) in enumerate(remaining_pred):
                if pred_id in used_pred:
                    continue

                overlap = max(0, min(ge, pe) - max(gs, ps))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_pred = pred_id

            if best_pred is not None and max_overlap > 0:
                used_gt.add(gt_id)
                used_pred.add(best_pred)
                ps, pe = remaining_pred[best_pred][1]

                # Compare lengths
                if (ge - gs) > (pe - ps):
                    fn += 1
                    if debug_plot:
                        _plot_span(ax1, (gs, ge), "FN", "blue")
                else:
                    fp += 1
                    if debug_plot:
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
