import numpy as np
import torch


def hand_mirroring(data):
    """
    Apply hand mirroring transformation to the input data.

    Args:
        data (np.ndarray or torch.Tensor): Input data of shape (M, 6), where M is the sequence length
                                           and 6 corresponds to the 3 accelerometer and 3 gyroscope axes.

    Returns:
        np.ndarray or torch.Tensor: Transformed data with the same shape as input.
    """
    # Define the mirroring transformation matrix
    mirroring_matrix = np.array(
        [
            [-1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, -1],
        ]
    )

    if isinstance(data, torch.Tensor):
        # Convert the mirroring matrix to a tensor and move it to the same device as the data
        mirroring_matrix = torch.tensor(
            mirroring_matrix, dtype=torch.float32, device=data.device
        )
        # Apply the transformation
        data = torch.matmul(data, mirroring_matrix.T)
    else:
        # Apply the transformation for numpy arrays
        data = np.dot(data, mirroring_matrix.T)

    return data
