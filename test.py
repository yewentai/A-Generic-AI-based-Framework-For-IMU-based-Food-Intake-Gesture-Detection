import numpy as np


def get_union_intervals(pred_intervals, gt_intervals):
    """
    Find overlapping intervals from two sets: pred and gt, and return the union with indices.

    Parameters:
        pred_intervals (np.ndarray): An array of shape (N, 2), each row is [start, end).
        gt_intervals (np.ndarray): An array of shape (M, 2), each row is [start, end).

    Returns:
        list: A list of dictionaries, each with 'range', 'pred_indices', and 'gt_indices'.
    """
    # Create a list of all intervals with their type (pred/gt) and original index
    all_intervals = []
    # Add pred intervals with their indices
    for idx, interval in enumerate(pred_intervals):
        start, end = interval
        all_intervals.append((start, end, "pred", idx))
    # Add gt intervals with their indices
    for idx, interval in enumerate(gt_intervals):
        start, end = interval
        all_intervals.append((start, end, "gt", idx))

    # Sort the intervals based on their start time
    all_intervals.sort(key=lambda x: x[0])

    merged = []
    if not all_intervals:
        return merged

    # Initialize the first interval
    current_start, current_end = all_intervals[0][0], all_intervals[0][1]
    pred_indices = set()
    gt_indices = set()
    source, idx = all_intervals[0][2], all_intervals[0][3]
    if source == "pred":
        pred_indices.add(idx)
    else:
        gt_indices.add(idx)

    # Iterate through the sorted intervals
    for interval in all_intervals[1:]:
        start, end, src, idx = interval
        if start <= current_end:
            # Overlapping or contiguous, merge them
            current_end = max(current_end, end)
            if src == "pred":
                pred_indices.add(idx)
            else:
                gt_indices.add(idx)
        else:
            # Add the merged interval so far
            merged.append(
                {
                    "range": (current_start, current_end),
                    "pred_indices": sorted(pred_indices),
                    "gt_indices": sorted(gt_indices),
                }
            )
            # Reset for the new interval
            current_start, current_end = start, end
            pred_indices = set()
            gt_indices = set()
            if src == "pred":
                pred_indices.add(idx)
            else:
                gt_indices.add(idx)

    # Add the last merged interval
    merged.append(
        {
            "range": (current_start, current_end),
            "pred_indices": sorted(pred_indices),
            "gt_indices": sorted(gt_indices),
        }
    )

    return merged


# Example usage:
pred_intervals = np.array([[1, 5], [10, 15], [20, 25]])
gt_intervals = np.array([[3, 7], [16, 21]])
union_intervals = get_union_intervals(pred_intervals, gt_intervals)
for interval in union_intervals:
    start, end = interval["range"]
    pred_indices = interval["pred_indices"]
    gt_indices = interval["gt_indices"]

    print(f"Interval: {start}-{end}")
    print(f"Pred Indices: {pred_indices}")
    print(f"GT Indices: {gt_indices}")
