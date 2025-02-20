import numpy as np
import torch


def rotation_matrix_x(theta):
    """
    Creates a 3D rotation matrix for rotation around the x-axis.
    Args:
        theta: Rotation angle in radians
    Returns:
        3x3 rotation matrix
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def rotation_matrix_z(theta):
    """
    Creates a 3D rotation matrix for rotation around the z-axis.
    Args:
        theta: Rotation angle in radians
    Returns:
        3x3 rotation matrix
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def augment_orientation(batch_x, probability=0.5):
    """
    Performs data augmentation by applying random rotations to IMU sensor data.

    Args:
        batch_x: Input tensor of shape (batch_size, sequence_length, features)
        probability: Probability of applying augmentation to each sample

    Returns:
        Augmented tensor of the same shape as input
    """
    batch_size, seq_len, features = batch_x.shape
    augmented_batch = batch_x.clone()

    for i in range(batch_size):
        # Apply augmentation with given probability
        if np.random.random() < probability:
            # Generate random rotation angles with standard deviation of 10 degrees
            theta_x = np.random.normal(
                0, 10 * np.pi / 180
            )  # Convert 10 degrees to radians
            theta_z = np.random.normal(0, 10 * np.pi / 180)

            # Create basic rotation matrices
            Q_x = rotation_matrix_x(theta_x)
            Q_z = rotation_matrix_z(theta_z)

            # Randomly choose one of four possible transformations:
            # 0: rotation around x-axis only
            # 1: rotation around z-axis only
            # 2: rotation around x-axis followed by z-axis
            # 3: rotation around z-axis followed by x-axis
            transformation_choice = np.random.choice([0, 1, 2, 3])
            if transformation_choice == 0:
                transformation = Q_x
            elif transformation_choice == 1:
                transformation = Q_z
            elif transformation_choice == 2:
                transformation = np.dot(Q_x, Q_z)
            else:
                transformation = np.dot(Q_z, Q_x)

            # Create 6x6 transformation matrix that applies the same rotation
            # to both accelerometer and gyroscope data
            full_transformation = np.block(
                [[transformation, np.zeros((3, 3))], [np.zeros((3, 3)), transformation]]
            )

            # Apply transformation to the sample and convert back to torch tensor
            augmented_batch[i] = torch.tensor(
                np.dot(full_transformation, augmented_batch[i].numpy().T).T,
                dtype=torch.float32,
            )

    return augmented_batch
