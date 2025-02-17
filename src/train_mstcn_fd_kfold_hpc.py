import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
import numpy as np
import pickle
from datetime import datetime
from model_tcn_mha import TCNMHA, TCNMHA_Loss
from augmentation import augment_orientation
from utils import IMUDataset, segment_confusion_matrix, post_process_predictions

# Hyperparameters
num_layers = 9
num_classes = 3
num_heads = 8
input_dim = 6
num_filters = 64
kernel_size = 3
dropout = 0.3
lambda_coef = 0.15
tau = 4
learning_rate = 0.0005
sampling_frequency = 16
window_length = 60
window_size = sampling_frequency * window_length  # 16 Hz * 60 s = 960
debug_plot = False
batch_size = 64  # Per-GPU batch size
num_epochs = 10

# Path to the dataset
X_L_path = "./dataset/FD/FD-I/X_L.pkl"
Y_L_path = "./dataset/FD/FD-I/Y_L.pkl"
X_R_path = "./dataset/FD/FD-I/X_R.pkl"
Y_R_path = "./dataset/FD/FD-I/Y_R.pkl"


# Initialize distributed training
def setup(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


# Function to segment data into 60-second windows
def segment_data(data, labels, window_size):
    segmented_data = []
    segmented_labels = []
    for d, l in zip(data, labels):
        for i in range(0, len(d) - window_size + 1, window_size):
            segmented_data.append(d[i : i + window_size])
            segmented_labels.append(l[i : i + window_size])
    return np.array(segmented_data), np.array(segmented_labels)


# Main training function
def main(rank, world_size):
    setup(rank, world_size)

    # Load data and convert to NumPy array
    if rank == 0:
        with open(X_L_path, "rb") as f:
            X_L = np.array(pickle.load(f), dtype=object)
        with open(Y_L_path, "rb") as f:
            Y_L = np.array(pickle.load(f), dtype=object)
        with open(X_R_path, "rb") as f:
            X_R = np.array(pickle.load(f), dtype=object)
        with open(Y_R_path, "rb") as f:
            Y_R = np.array(pickle.load(f), dtype=object)

        # Concatenate the left and right data
        X = np.concatenate([X_L, X_R], axis=0)
        Y = np.concatenate([Y_L, Y_R], axis=0)

        # Segment data
        X, Y = segment_data(X, Y, window_size)
        full_dataset = IMUDataset(X, Y)
    else:
        full_dataset = None

    # Broadcast dataset to all ranks
    full_dataset = dist.broadcast_object(full_dataset, src=0)

    # 7-Fold Test Splits
    test_folds = [
        list(range(0, 10)),  # Fold 1: Subjects 0-9
        list(range(10, 20)),  # Fold 2: Subjects 10-19
        list(range(20, 30)),  # Fold 3: Subjects 20-29
        list(range(30, 40)),  # Fold 4: Subjects 30-39
        list(range(40, 50)),  # Fold 5: Subjects 40-49
        list(range(50, 60)),  # Fold 6: Subjects 50-59
        list(range(60, 68)),  # Fold 7: Subjects 60-67 (remaining subjects)
    ]

    # Create directories for saving results
    if rank == 0:
        if not os.path.exists("result"):
            os.makedirs("result")

    # File names for training and testing results
    training_stats_file = "result/training_stats_tcnmha.npy"
    testing_stats_file = "result/testing_stats_tcnmha.npy"

    # Initialize empty lists to store results
    training_statistics = []
    testing_statistics = []

    # 7-Fold Cross-Validation
    for fold, test_subjects in enumerate(test_folds):
        # Create train and test indices
        train_indices = [
            i
            for i, subject in enumerate(full_dataset.subject_indices)
            if subject not in test_subjects
        ]
        test_indices = [
            i
            for i, subject in enumerate(full_dataset.subject_indices)
            if subject in test_subjects
        ]

        # Create train and test datasets
        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)

        # Distributed samplers
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

        # Initialize the model
        model = TCNMHA(
            input_dim=input_dim,
            hidden_dim=num_filters,
            num_classes=num_classes,
            num_heads=num_heads,
            d_model=128,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(rank)
        model = DDP(model, device_ids=[rank])

        # Loss and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate * world_size)
        scaler = torch.amp.GradScaler()

        # Training
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            training_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = augment_orientation(batch_x)
                batch_x, batch_y = batch_x.permute(0, 2, 1).to(rank), batch_y.to(rank)

                optimizer.zero_grad()
                with torch.amp.autocast():
                    outputs = model(batch_x)
                    loss = TCNMHA_Loss(outputs, batch_y, lambda_coef)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                training_loss += loss.item()

            avg_train_loss = training_loss / len(train_loader)
            if rank == 0:
                training_statistics.append(
                    {
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "fold": fold + 1,
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                    }
                )

        # Testing
        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.permute(0, 2, 1).to(rank), batch_y.to(rank)
                outputs = model(batch_x)
                final_output = outputs[-1]
                probabilities = F.softmax(final_output, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                all_predictions.extend(predicted_classes.view(-1).cpu().numpy())
                all_labels.extend(batch_y.view(-1).cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_predictions = post_process_predictions(all_predictions)

        # Calculate confusion matrix and F1 scores
        fn_10, fp_10, tp_10 = segment_confusion_matrix(
            all_predictions, all_labels, 0.1, debug_plot
        )
        fn_25, fp_25, tp_25 = segment_confusion_matrix(
            all_predictions, all_labels, 0.25, debug_plot
        )
        fn_50, fp_50, tp_50 = segment_confusion_matrix(
            all_predictions, all_labels, 0.5, debug_plot
        )

        f1_segment_10 = 2 * tp_10 / (2 * tp_10 + fp_10 + fn_10) if tp_10 > 0 else 0
        f1_segment_25 = 2 * tp_25 / (2 * tp_25 + fp_25 + fn_25) if tp_25 > 0 else 0
        f1_segment_50 = 2 * tp_50 / (2 * tp_50 + fp_50 + fn_50) if tp_50 > 0 else 0

        if rank == 0:
            testing_statistics.append(
                {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "fold": fold + 1,
                    "f1_segment_10": f1_segment_10,
                    "f1_segment_25": f1_segment_25,
                    "f1_segment_50": f1_segment_50,
                }
            )

    # Save results to .npy files
    if rank == 0:
        np.save(training_stats_file, training_statistics)
        np.save(testing_stats_file, testing_statistics)

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
