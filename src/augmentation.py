import numpy as np
import torch


def rotation_matrix_x(theta):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def rotation_matrix_z(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def augment_orientation(batch_x, probability=0.5):
    batch_size, seq_len, features = batch_x.shape
    augmented_batch = batch_x.clone()

    for i in range(batch_size):
        if np.random.random() < probability:
            # Generate random rotation angles
            theta_x = np.random.normal(
                0, 10 * np.pi / 180
            )  # Convert 10 degrees to radians
            theta_z = np.random.normal(0, 10 * np.pi / 180)

            # Create rotation matrices
            Q_x = rotation_matrix_x(theta_x)
            Q_z = rotation_matrix_z(theta_z)

            # Randomly choose transformation
            transformation_choice = np.random.choice([0, 1, 2, 3])
            if transformation_choice == 0:
                transformation = Q_x
            elif transformation_choice == 1:
                transformation = Q_z
            elif transformation_choice == 2:
                transformation = np.dot(Q_x, Q_z)
            else:
                transformation = np.dot(Q_z, Q_x)

            # Create the full transformation matrix
            full_transformation = np.block(
                [[transformation, np.zeros((3, 3))], [np.zeros((3, 3)), transformation]]
            )

            # Apply transformation
            augmented_batch[i] = torch.tensor(
                np.dot(full_transformation, augmented_batch[i].numpy().T).T,
                dtype=torch.float32,
            )

    return augmented_batch


def augment_mirror(batch_x, probability=0.5):
    batch_size, seq_len, features = batch_x.shape
    augmented_batch = batch_x.clone()

    for i in range(batch_size):
        if np.random.random() < probability:
            # Generate random mirror
            mirror_choice = np.random.choice([0, 1])
            if mirror_choice == 0:
                augmented_batch[i] = torch.flip(augmented_batch[i], [1])

    return augmented_batch
