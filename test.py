import numpy as np


def get_union_intervals(pred_intervals, gt_intervals):
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
                    "range": [current_start, current_end],
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
            "range": [current_start, current_end],
            "pred_ranges": pred_ranges.copy(),
            "gt_ranges": gt_ranges.copy(),
        }
    )

    return merged


def get_target_intervals(x, class_label):
    """
    Extract start and end ranges of intervals where x[i] equals class_label.

    Parameters:
        x (np.ndarray): Input 1D array
        class_label (int): Target class label

    Returns:
        np.ndarray: Array of [start, end) interval ranges
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


# Generate test sequences
def generate_test_sequence(length, num_segments, classes=[0, 1]):
    arr = np.zeros(length, dtype=int)
    for _ in range(num_segments):
        cls = np.random.choice(classes[1:])  # Exclude background class 0
        start = np.random.randint(0, length - 5)
        end = start + np.random.randint(1, 5)
        arr[start:end] = cls
    return arr


# Generate binary test case
np.random.seed(42)
binary_gt = generate_test_sequence(50, 8, [0, 1])
binary_pred = generate_test_sequence(50, 10, [0, 1])
gt_intervals = get_target_intervals(binary_gt, class_label=1)
pred_intervals = get_target_intervals(binary_pred, class_label=1)
union_intervals = get_union_intervals(pred_intervals, gt_intervals)
for interval in union_intervals:
    start, end = interval["range"]
    pred_ranges = interval["pred_ranges"]
    gt_ranges = interval["gt_ranges"]

    print(f"Interval: {start}-{end}")
    print(f"Pred Indices: {pred_ranges}")
    print(f"GT Indices: {gt_ranges}")
