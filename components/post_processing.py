import numpy as np


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


def post_process_predictions(
    predictions, sampling_freq, min_length_sec=0.25, merge_distance_sec=0.1
):
    """
    Post-process predictions by removing short segments and merging close segments.

    Parameters:
        predictions (np.ndarray): Binary predictions array (0/1).
        sampling_freq (float): Sampling frequency in Hz (samples per second).
        min_length_sec (float): Minimum duration (in seconds) of a segment to keep.
        merge_distance_sec (float): Maximum gap (in seconds) between segments to merge.

    Returns:
        np.ndarray: Post-processed predictions array.
    """
    # Convert duration in seconds to number of samples
    min_length_samples = int(sampling_freq * min_length_sec)
    merge_distance_samples = int(sampling_freq * merge_distance_sec)

    # Get the intervals where predictions are nonzero
    intervals = nonzero_intervals(predictions)

    # Remove segments shorter than the minimum length (in samples)
    intervals = [
        interval
        for interval in intervals
        if interval[1] - interval[0] >= min_length_samples
    ]

    # Merge intervals that are close to each other (within merge_distance_samples)
    merged_intervals = []
    for interval in intervals:
        if (
            not merged_intervals
            or (interval[0] - merged_intervals[-1][1]) > merge_distance_samples
        ):
            merged_intervals.append(interval)
        else:
            # Extend the previous interval to include this one
            merged_intervals[-1][1] = interval[1]

    # Create a new predictions array based on the merged intervals
    new_predictions = np.zeros_like(predictions)
    for start, end in merged_intervals:
        new_predictions[start:end] = 1

    return new_predictions
