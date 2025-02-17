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


def nonzero_intervals_value(x):
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
