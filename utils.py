import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


# Dataset class
class IMUDataset(Dataset):
    def __init__(self, X, Y, sequence_length=128):
        self.data = []
        self.labels = []
        self.sequence_length = sequence_length
        self.subject_indices = []  # Record which subject each sample belongs to

        # Processing data for each session
        for subject_idx, (imu_data, labels) in enumerate(zip(X, Y)):
            # imu_data = self.normalize(imu_data)
            num_samples = len(labels)

            for i in range(0, num_samples, sequence_length):
                imu_segment = imu_data[i : i + sequence_length]
                label_segment = labels[i : i + sequence_length]

                if len(imu_segment) == sequence_length:
                    self.data.append(imu_segment)
                    self.labels.append(label_segment)
                    self.subject_indices.append(subject_idx)

    def normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


# Evaluation functions
def nonzero_intervals(x):
    """
    Extract start and end indices of nonzero intervals in a binary array.

    Parameters:
        x (np.ndarray): Input binary array (1D).

    Returns:
        np.ndarray: An array of shape (N, 2), where each row represents the
                    [start, end) indices of a contiguous nonzero interval.
    """
    # Prepare a list to collect results
    results = []

    # Track when we are inside a nonzero segment
    in_segment = False
    start_idx = 0

    for i in range(len(x)):
        if x[i] != 0 and not in_segment:
            # Encounter the start of a nonzero segment
            in_segment = True
            start_idx = i
        elif x[i] == 0 and in_segment:
            # Encounter the end of a nonzero segment
            in_segment = False
            results.append([start_idx, i])

    # Handle the case where the sequence ends with a nonzero segment
    if in_segment:
        results.append([start_idx, len(x)])

    # Convert the results list to a NumPy array
    return np.array(results, dtype=int)


def segment_f1_binary(pred, gt, threshold=0.5, debug_plot=False):
    """
    Compute the F1 score for binary segmentation tasks.

    Parameters:
        pred (np.ndarray): Binary array of predicted segmentation.
        gt (np.ndarray): Binary array of ground truth segmentation.
        threshold (float): IoU threshold to determine true positives.
        debug_plot (bool): Whether to plot the segmentation results.

    Returns:
        float: F1 score of the segmentation task.
    """
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
        ax3.step(range(len(union)), union, where="post", label="Union", color="green")
        ax3.set_title("Union")

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

    precision = t_p / (t_p + f_p) if (t_p + f_p) > 0 else 0
    recall = t_p / (t_p + f_n) if (t_p + f_n) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    if debug_plot:
        # Finalize the plot
        for ax in (ax1, ax2, ax3):
            ax.set_ylim(-0.1, 1.1)
            ax.legend()
        ax3.set_xlabel("Time")
        plt.tight_layout()
        plt.show()

    return f1


def segment_f1_multiclass(pred, gt, classes=[1, 2], threshold=0.5, debug_plot=False):
    """
    Compute the F1 score for multi-class segmentation tasks.

    Parameters:
        pred (np.ndarray): Array of predicted segmentation.
        gt (np.ndarray): Array of ground truth segmentation.
        classes (list): List of classes to evaluate.
        threshold (float): IoU threshold to determine true positives.
        debug_plot (bool): Whether to plot the segmentation results.

    Returns:
        dict: A dictionary with F1 scores for each class in `classes`.
    """
    f1_scores = {}

    for cls in classes:
        # Prepare binary arrays for current class
        pred_class = (pred == cls).astype(int)
        gt_class = (gt == cls).astype(int)

        # Initialize counters
        f_n, f_p, t_p = 0, 0, 0

        # Compute the union of the two sequences
        union = np.logical_or(pred_class, gt_class).astype(int)
        union_intervals = nonzero_intervals(union)

        if debug_plot:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
            ax1.step(
                range(len(gt_class)),
                gt_class,
                where="post",
                label="Ground Truth",
                color="blue",
            )
            ax1.set_title(f"Ground Truth - Class {cls}")
            ax2.step(
                range(len(pred_class)),
                pred_class,
                where="post",
                label="Prediction",
                color="red",
            )
            ax2.set_title(f"Prediction - Class {cls}")
            ax3.step(
                range(len(union)), union, where="post", label="Union", color="green"
            )
            ax3.set_title(f"Union - Class {cls}")

        for start_idx, end_idx in union_intervals:
            flag_t_p = 0
            gt_interval = nonzero_intervals(gt_class[start_idx:end_idx])
            pred_interval = nonzero_intervals(pred_class[start_idx:end_idx])
            gt_interval += start_idx
            pred_interval += start_idx

            if debug_plot:
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
                matched = set()
                for pred_start, pred_end in pred_interval:
                    best_iou = 0
                    best_match = None
                    for idx, (gt_start, gt_end) in enumerate(gt_interval):
                        if idx in matched:
                            continue
                        intersection = max(
                            0, min(gt_end, pred_end) - max(gt_start, pred_start)
                        )
                        union = max(gt_end, pred_end) - min(gt_start, pred_start)
                        iou = intersection / union
                        if iou > best_iou:
                            best_iou = iou
                            best_match = idx
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

        precision = t_p / (t_p + f_p) if (t_p + f_p) > 0 else 0
        recall = t_p / (t_p + f_n) if (t_p + f_n) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        if debug_plot:
            for ax in (ax1, ax2, ax3):
                ax.set_ylim(-0.1, 1.1)
                ax.legend()
            ax3.set_xlabel("Time")
            plt.tight_layout()
            plt.show()

        f1_scores[cls] = f1

    return f1_scores


def post_process_predictions(predictions, min_length=64, merge_distance=32):
    """
    Post-process predictions by removing short segments and merging close segments.

    Parameters:
        predictions (np.ndarray): Binary predictions array.
        min_length (int): Minimum length of a segment to keep.
        merge_distance (int): Maximum distance between segments to merge.

    Returns:
        np.ndarray: Post-processed predictions array.
    """
    # Get the intervals of nonzero predictions
    intervals = nonzero_intervals(predictions)

    # Remove short segments
    intervals = [
        interval for interval in intervals if interval[1] - interval[0] >= min_length
    ]

    # Merge close segments
    merged_intervals = []
    for interval in intervals:
        if (
            not merged_intervals
            or interval[0] - merged_intervals[-1][1] > merge_distance
        ):
            merged_intervals.append(interval)
        else:
            merged_intervals[-1][1] = interval[1]

    # Create a new predictions array based on the merged intervals
    new_predictions = np.zeros_like(predictions)
    for start, end in merged_intervals:
        new_predictions[start:end] = 1

    return new_predictions
