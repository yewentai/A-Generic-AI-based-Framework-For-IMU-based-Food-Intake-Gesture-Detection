import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from utils import nonzero_intervals, nonzero_intervals_value


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


def segment_evaluation(pred, gt, threshold=0.5, debug_plot=False):
    """
    Unified segmentation confusion matrix calculator using value-aware interval matching.

    Parameters:
        pred (np.ndarray): Predicted segmentation array (0/1 for binary, any int for multi-class)
        gt (np.ndarray): Ground truth segmentation array (same format as pred)
        threshold (float): IoU threshold for true positive classification
        debug_plot (bool): Enable visualization of matching logic

    Returns:
        tuple: (false_negatives, false_positives, true_positives)
    """
    fn, fp, tp = 0, 0, 0
    union = np.logical_or(pred, gt).astype(int)
    union_intervals = nonzero_intervals(union)

    if debug_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        ax1.step(range(len(gt)), gt, where="post", label="Ground Truth", color="blue")
        ax2.step(range(len(pred)), pred, where="post", label="Prediction", color="red")
        ax3.step(range(len(union)), union, where="post", color="green")

    for start, end in union_intervals:
        gt_intv = nonzero_intervals_value(gt[start:end]) + [start, start, 0]
        pred_intv = nonzero_intervals_value(pred[start:end]) + [start, start, 0]

        # Handle edge cases first
        if not gt_intv.size:
            fp += len(pred_intv)
            if debug_plot:
                for p in pred_intv:
                    _plot_span(ax3, p, "FP", "red")
            continue

        if not pred_intv.size:
            fn += len(gt_intv)
            if debug_plot:
                for g in gt_intv:
                    _plot_span(ax3, g, "FN", "blue")
            continue

        # Match intervals using Hungarian algorithm with IoU scores
        cost_matrix = np.zeros((len(gt_intv), len(pred_intv)))
        for i, (gs, ge, gv) in enumerate(gt_intv):
            for j, (ps, pe, pv) in enumerate(pred_intv):
                if gv != pv:
                    cost_matrix[i, j] = -1  # Prevent cross-class matches
                    continue
                inter = max(0, min(ge, pe) - max(gs, ps))
                union = max(ge, pe) - min(gs, ps)
                cost_matrix[i, j] = inter / union if union else 0

        # Solve optimal matching using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        matched = set()
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] >= threshold:
                tp += 1
                matched.add(j)
                if debug_plot:
                    _plot_span(ax3, pred_intv[j], "TP", "green")

        # Count mismatches
        for j in range(len(pred_intv)):
            if j not in matched:
                fp += 1
                if debug_plot:
                    _plot_span(ax3, pred_intv[j], "FP", "red")

        for i in range(len(gt_intv)):
            if i not in row_ind[cost_matrix[row_ind, col_ind] >= threshold]:
                fn += 1
                if debug_plot:
                    _plot_span(ax3, gt_intv[i], "FN", "blue")

    if debug_plot:
        plt.tight_layout()
        plt.show()

    return fn, fp, tp
