import numpy as np
from utils import nonzero_intervals


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
