import numpy as np
import matplotlib.pyplot as plt
from utils import nonzero_intervals, nonzero_intervals_value


def segment_confusion_matrix_binary(pred, gt, threshold=0.5, debug_plot=False):

    # Initialize counters
    f_n, f_p, t_p = 0, 0, 0
    # Compute the union of the two sequences
    union = np.logical_or(pred, gt).astype(int)
    union_intervals = nonzero_intervals(union)

    if debug_plot:
        # Plot the ground truth, prediction, and union sequences
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        ax1.step(range(len(gt)), gt, where="post", label="Ground Truth", color="blue")
        ax1.set_title("Ground Truth")
        ax2.step(range(len(pred)), pred, where="post", label="Prediction", color="red")
        ax2.set_title("Prediction")
        ax3.step(range(len(union)), union, where="post", label="Result", color="green")
        ax3.set_title(f"Result(Threshold={threshold})")

    for start_idx, end_idx in union_intervals:
        # Initialize a flag to avoid double counting
        flag_t_p = 0
        # Extract the nonzero intervals for the ground truth and prediction
        gt_interval = nonzero_intervals(gt[start_idx:end_idx])
        pred_interval = nonzero_intervals(pred[start_idx:end_idx])
        # Shift the interval indices to the global index
        gt_interval = gt_interval + start_idx
        pred_interval = pred_interval + start_idx

        if debug_plot:
            # Highlight the ground truth and prediction intervals
            for gs, ge in gt_interval:
                ax1.axvspan(gs, ge, alpha=0.3, color="blue")
            for ps, pe in pred_interval:
                ax2.axvspan(ps, pe, alpha=0.3, color="red")

        if len(gt_interval) == 0 and len(pred_interval) == 1:
            f_p += 1
            if debug_plot:
                ax3.axvspan(
                    pred_interval[0][0], pred_interval[0][1], alpha=0.5, color="red"
                )
                ax3.text(
                    (pred_interval[0][0] + pred_interval[0][1]) / 2,
                    0.9,
                    "FP",
                    color="red",
                    ha="center",
                )
        elif len(gt_interval) == 1 and len(pred_interval) == 0:
            f_n += 1
            if debug_plot:
                ax3.axvspan(
                    gt_interval[0][0], gt_interval[0][1], alpha=0.5, color="blue"
                )
                ax3.text(
                    (gt_interval[0][0] + gt_interval[0][1]) / 2,
                    0.9,
                    "FN",
                    color="blue",
                    ha="center",
                )
        elif len(gt_interval) == 1 and len(pred_interval) == 1:
            gt_start, gt_end = gt_interval[0]
            pred_start, pred_end = pred_interval[0]
            intersection = min(gt_end, pred_end) - max(gt_start, pred_start)
            union = max(gt_end, pred_end) - min(gt_start, pred_start)
            iou = intersection / union
            len_gt = gt_interval[0][1] - gt_interval[0][0]
            len_pred = pred_interval[0][1] - pred_interval[0][0]
            if iou >= threshold:
                t_p += 1
                if debug_plot:
                    if len_gt >= len_pred:
                        ax3.axvspan(pred_start, pred_end, alpha=0.5, color="red")
                        ax3.text(
                            (pred_start + pred_end) / 2,
                            0.9,
                            "TP",
                            color="red",
                            ha="center",
                        )
                    else:
                        ax3.axvspan(gt_start, gt_end, alpha=0.5, color="blue")
                        ax3.text(
                            (gt_start + gt_end) / 2,
                            0.9,
                            "TP",
                            color="blue",
                            ha="center",
                        )
            elif len_gt > len_pred:
                f_n += 1
                if debug_plot:
                    ax3.axvspan(gt_start, gt_end, alpha=0.5, color="blue")
                    ax3.text(
                        (gt_start + gt_end) / 2, 0.9, "FN", color="blue", ha="center"
                    )
            else:
                f_p += 1
                if debug_plot:
                    ax3.axvspan(pred_start, pred_end, alpha=0.5, color="red")
                    ax3.text(
                        (pred_start + pred_end) / 2, 0.9, "FP", color="red", ha="center"
                    )
        elif len(gt_interval) >= 2 and len(pred_interval) == 1:
            pred_start, pred_end = pred_interval[0]
            for gt_start, gt_end in gt_interval:
                intersection = min(gt_end, pred_end) - max(gt_start, pred_start)
                union = max(gt_end, pred_end) - min(gt_start, pred_start)
                iou = intersection / union
                if iou >= threshold and flag_t_p == 0:
                    flag_t_p += 1
                    t_p += 1
                    if debug_plot:
                        ax3.axvspan(gt_start, gt_end, alpha=0.5, color="green")
                        ax3.text(
                            (gt_start + gt_end) / 2,
                            0.9,
                            "TP",
                            color="green",
                            ha="center",
                        )
                else:
                    f_p += 1
                    if debug_plot:
                        ax3.axvspan(gt_start, gt_end, alpha=0.5, color="blue")
                        ax3.text(
                            (gt_start + gt_end) / 2,
                            0.9,
                            "FP",
                            color="blue",
                            ha="center",
                        )
        elif len(gt_interval) == 1 and len(pred_interval) >= 2:
            gt_start, gt_end = gt_interval[0]
            for pred_start, pred_end in pred_interval:
                intersection = min(gt_end, pred_end) - max(gt_start, pred_start)
                union = max(gt_end, pred_end) - min(gt_start, pred_start)
                iou = intersection / union
                if iou >= threshold and flag_t_p == 0:
                    flag_t_p += 1
                    t_p += 1
                    if debug_plot:
                        ax3.axvspan(pred_start, pred_end, alpha=0.5, color="green")
                        ax3.text(
                            (pred_start + pred_end) / 2,
                            0.9,
                            "TP",
                            color="green",
                            ha="center",
                        )
                else:
                    f_n += 1
                    if debug_plot:
                        ax3.axvspan(pred_start, pred_end, alpha=0.5, color="blue")
                        ax3.text(
                            (pred_start + pred_end) / 2,
                            0.9,
                            "FN",
                            color="blue",
                            ha="center",
                        )
        else:
            # Handle multiple GT and Pred intervals
            matched = set()
            for pred_start, pred_end in pred_interval:
                best_iou = 0
                best_match = None

                # Find the best matching ground truth interval
                for idx, (gt_start, gt_end) in enumerate(gt_interval):
                    if idx in matched:
                        continue  # Skip already matched ground truths

                    intersection = max(
                        0, min(gt_end, pred_end) - max(gt_start, pred_start)
                    )
                    union = max(gt_end, pred_end) - min(gt_start, pred_start)
                    iou = intersection / union

                    if iou > best_iou:
                        best_iou = iou
                        best_match = idx

                # Decide based on the IoU threshold
                if best_iou >= threshold:
                    t_p += 1
                    matched.add(best_match)
                    if debug_plot:
                        ax3.axvspan(pred_start, pred_end, alpha=0.5, color="green")
                        ax3.text(
                            (pred_start + pred_end) / 2,
                            0.9,
                            "TP",
                            color="green",
                            ha="center",
                        )
                else:
                    f_p += 1
                    if debug_plot:
                        ax3.axvspan(pred_start, pred_end, alpha=0.5, color="red")
                        ax3.text(
                            (pred_start + pred_end) / 2,
                            0.9,
                            "FP",
                            color="red",
                            ha="center",
                        )

    if debug_plot:
        # Finalize the plot
        for ax in (ax1, ax2, ax3):
            ax.set_ylim(-0.1, 1.1)
            ax.legend()
        ax3.set_xlabel("Time")
        plt.tight_layout()
        plt.show()

    return f_n, f_p, t_p


def segment_confusion_matrix(pred, gt, threshold=0.5, debug_plot=False):
    """
    Compute the confusion matrix for segmentation tasks.

    Parameters:
        pred (np.ndarray): Array of predicted segmentation.
        gt (np.ndarray): Array of ground truth segmentation.
        classes (list): List of classes to evaluate.
        threshold (float): IoU threshold to determine true positives.
        debug_plot (bool): Whether to plot the segmentation results.

    Returns:
        tuple: A tuple containing the number of false negatives, false positives,
               and true positives.
    """
    # Initialize counters
    f_n, f_p, t_p = 0, 0, 0
    # Compute the union of the two sequences
    union = np.logical_or(pred, gt).astype(int)
    union_intervals = nonzero_intervals(union)

    if debug_plot:
        # Plot the ground truth, prediction, and result sequences
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        ax1.step(range(len(gt)), gt, where="post", label="Ground Truth", color="blue")
        ax1.set_title("Ground Truth")
        ax2.step(range(len(pred)), pred, where="post", label="Prediction", color="red")
        ax2.set_title("Prediction")
        ax3.step(range(len(union)), union, where="post", label="Result", color="green")
        ax3.set_title(f"Result(Threshold={threshold})")

    for start_idx, end_idx in union_intervals:
        # Initialize a flag to avoid double counting
        flag_t_p = 0
        # Extract the nonzero intervals for the ground truth and prediction
        gt_interval = nonzero_intervals_value(gt[start_idx:end_idx])
        pred_interval = nonzero_intervals_value(pred[start_idx:end_idx])
        # Shift the interval indices to the global index
        gt_interval[:, :2] += start_idx
        pred_interval[:, :2] += start_idx

        if len(gt_interval) == 0 and len(pred_interval) != 0:  # False Positive case
            f_p += len(pred_interval)  # Count all false positive intervals
            if debug_plot:
                for pred in pred_interval:
                    ax3.axvspan(pred[0], pred[1], alpha=0.5, color="red")
                    ax3.text(
                        (pred[0] + pred[1]) / 2,
                        0.9,
                        "FP",
                        color="red",
                        ha="center",
                    )
        elif len(gt_interval) != 0 and len(pred_interval) == 0:  # False Negative case
            f_n += len(gt_interval)  # Count all false negative intervals
            if debug_plot:
                for gt in gt_interval:
                    ax3.axvspan(gt[0], gt[1], alpha=0.5, color="blue")
                    ax3.text(
                        (gt[0] + gt[1]) / 2,
                        0.9,
                        "FN",
                        color="blue",
                        ha="center",
                    )

        elif len(gt_interval) == 1 and len(pred_interval) == 1:
            gt_start, gt_end, gt_value = gt_interval[0]
            pred_start, pred_end, pred_value = pred_interval[0]
            len_gt = gt_interval[0][1] - gt_interval[0][0]
            len_pred = pred_interval[0][1] - pred_interval[0][0]
            if gt_value != pred_value:
                f_p += 1
                if debug_plot:
                    ax3.axvspan(gt_start, gt_end, alpha=0.5, color="blue")
                    ax3.text(
                        (gt_start + gt_end) / 2, 0.9, "FN", color="blue", ha="center"
                    )
                    ax3.axvspan(pred_start, pred_end, alpha=0.5, color="red")
                    ax3.text(
                        (pred_start + pred_end) / 2, 0.9, "FP", color="red", ha="center"
                    )
            else:
                intersection = min(gt_end, pred_end) - max(gt_start, pred_start)
                union = max(gt_end, pred_end) - min(gt_start, pred_start)
                iou = intersection / union
                if iou >= threshold:
                    t_p += 1
                    if debug_plot:
                        if len_gt >= len_pred:
                            ax3.axvspan(pred_start, pred_end, alpha=0.5, color="red")
                            ax3.text(
                                (pred_start + pred_end) / 2,
                                0.9,
                                "TP",
                                color="red",
                                ha="center",
                            )
                        else:
                            ax3.axvspan(gt_start, gt_end, alpha=0.5, color="blue")
                            ax3.text(
                                (gt_start + gt_end) / 2,
                                0.9,
                                "TP",
                                color="blue",
                                ha="center",
                            )
                elif len_gt > len_pred:
                    f_n += 1
                    if debug_plot:
                        ax3.axvspan(gt_start, gt_end, alpha=0.5, color="blue")
                        ax3.text(
                            (gt_start + gt_end) / 2,
                            0.9,
                            "FN",
                            color="blue",
                            ha="center",
                        )
                else:
                    f_p += 1
                    if debug_plot:
                        ax3.axvspan(pred_start, pred_end, alpha=0.5, color="red")
                        ax3.text(
                            (pred_start + pred_end) / 2,
                            0.9,
                            "FP",
                            color="red",
                            ha="center",
                        )

        elif len(gt_interval) >= 2 and len(pred_interval) == 1:
            pred_start, pred_end, pred_value = pred_interval[0]
            for gt_start, gt_end, gt_value in gt_interval:
                if gt_value != pred_value:
                    f_p += 1
                    if debug_plot:
                        ax3.axvspan(gt_start, gt_end, alpha=0.5, color="blue")
                        ax3.text(
                            (gt_start + gt_end) / 2,
                            0.9,
                            "FP",
                            color="blue",
                            ha="center",
                        )
                    continue
                intersection = min(gt_end, pred_end) - max(gt_start, pred_start)
                union = max(gt_end, pred_end) - min(gt_start, pred_start)
                iou = intersection / union
                if iou >= threshold and flag_t_p == 0:
                    flag_t_p += 1
                    t_p += 1
                    if debug_plot:
                        ax3.axvspan(gt_start, gt_end, alpha=0.5, color="green")
                        ax3.text(
                            (gt_start + gt_end) / 2,
                            0.9,
                            "TP",
                            color="green",
                            ha="center",
                        )
                else:
                    f_p += 1
                    if debug_plot:
                        ax3.axvspan(gt_start, gt_end, alpha=0.5, color="blue")
                        ax3.text(
                            (gt_start + gt_end) / 2,
                            0.9,
                            "FP",
                            color="blue",
                            ha="center",
                        )

        elif len(gt_interval) == 1 and len(pred_interval) >= 2:
            gt_start, gt_end, gt_value = gt_interval[0]
            for pred_start, pred_end, pred_value in pred_interval:
                if gt_value != pred_value:
                    f_p += 1
                    if debug_plot:
                        ax3.axvspan(pred_start, pred_end, alpha=0.5, color="red")
                        ax3.text(
                            (pred_start + pred_end) / 2,
                            0.9,
                            "FP",
                            color="red",
                            ha="center",
                        )
                    continue
                intersection = min(gt_end, pred_end) - max(gt_start, pred_start)
                union = max(gt_end, pred_end) - min(gt_start, pred_start)
                iou = intersection / union
                if iou >= threshold and flag_t_p == 0:
                    flag_t_p += 1
                    t_p += 1
                    if debug_plot:
                        ax3.axvspan(pred_start, pred_end, alpha=0.5, color="green")
                        ax3.text(
                            (pred_start + pred_end) / 2,
                            0.9,
                            "TP",
                            color="green",
                            ha="center",
                        )
                else:
                    f_n += 1
                    if debug_plot:
                        ax3.axvspan(pred_start, pred_end, alpha=0.5, color="blue")
                        ax3.text(
                            (pred_start + pred_end) / 2,
                            0.9,
                            "FN",
                            color="blue",
                            ha="center",
                        )
        else:
            # Handle multiple GT and Pred intervals
            matched = set()
            for (
                pred_start,
                pred_end,
                pred_value,
            ) in pred_interval:  # Fixed: Unpack three values
                best_iou = 0
                best_match = None

                # Find the best matching ground truth interval
                for idx, (gt_start, gt_end, gt_value) in enumerate(gt_interval):
                    if idx in matched:
                        continue

                    if gt_value != pred_value:  # Check class match
                        continue

                    intersection = max(
                        0, min(gt_end, pred_end) - max(gt_start, pred_start)
                    )
                    union = max(gt_end, pred_end) - min(gt_start, pred_start)
                    iou = intersection / union

                    if iou > best_iou:
                        best_iou = iou
                        best_match = idx

                # Decide based on the IoU threshold
                if best_iou >= threshold:
                    t_p += 1
                    matched.add(best_match)
                    if debug_plot:
                        ax3.axvspan(pred_start, pred_end, alpha=0.5, color="green")
                        ax3.text(
                            (pred_start + pred_end) / 2,
                            0.9,
                            "TP",
                            color="green",
                            ha="center",
                        )
                else:
                    f_p += 1
                    if debug_plot:
                        ax3.axvspan(pred_start, pred_end, alpha=0.5, color="red")
                        ax3.text(
                            (pred_start + pred_end) / 2,
                            0.9,
                            "FP",
                            color="red",
                            ha="center",
                        )

    if debug_plot:
        # Finalize the plot
        for ax in (ax1, ax2, ax3):
            ax.legend()
        ax3.set_xlabel("Time")
        plt.tight_layout()
        plt.show()

    return f_n, f_p, t_p
