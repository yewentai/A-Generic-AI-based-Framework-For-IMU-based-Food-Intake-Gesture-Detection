import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import pickle
import csv
from datetime import datetime
from tqdm import tqdm
from model_mstcn import MSTCN, MSTCN_Loss
from augmentation import augment_orientation
from utils import IMUDataset, segment_f1_multiclass, post_process_predictions

# Hyperparameters
num_stages = 2
num_layers = 9
num_classes = 3
input_dim = 6
num_filters = 128
kernel_size = 3
dropout = 0.3
lambda_coef = 0.15
tau = 4
learning_rate = 0.0005
debug_plot = False

# Load data
X_L_path = "./dataset/FD/FD-I/X_L.pkl"
Y_L_path = "./dataset/FD/FD-I/Y_L.pkl"
X_R_path = "./dataset/FD/FD-I/X_R.pkl"
Y_R_path = "./dataset/FD/FD-I/Y_R.pkl"

with open(X_L_path, "rb") as f:
    X_L = np.array(pickle.load(f), dtype=object)  # Convert to NumPy array
with open(Y_L_path, "rb") as f:
    Y_L = np.array(pickle.load(f), dtype=object)
with open(X_R_path, "rb") as f:
    X_R = np.array(pickle.load(f), dtype=object)
with open(Y_R_path, "rb") as f:
    Y_R = np.array(pickle.load(f), dtype=object)

# Concatenate the left and right data
X = np.concatenate([X_L, X_R], axis=0)
Y = np.concatenate([Y_L, Y_R], axis=0)

# Create the full dataset
full_dataset = IMUDataset(X, Y)

# LOSO cross-validation
unique_subjects = np.unique(full_dataset.subject_indices)
loso_f1_scores = []

# Open CSV files for writing
with open(
    "result/training_log_fdi_mirrored_mstcn_attention.csv", mode="w", newline=""
) as train_csvfile, open(
    "result/testing_log_fdi_mirrored_mstcn_attention.csv", mode="w", newline=""
) as test_csvfile:

    train_csv_writer = csv.writer(train_csvfile)
    test_csv_writer = csv.writer(test_csvfile)

    # Write headers
    train_csv_writer.writerow(["Date", "Time", "Fold", "Epoch", "Training Loss"])
    test_csv_writer.writerow(
        [
            "Date",
            "Time",
            "Fold",
            "F1 Sample",
            "F1 Segment 0.1",
            "F1 Segment 0.25",
            "F1 Segment 0.5",
        ]
    )

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for fold, test_subject in enumerate(
        tqdm(unique_subjects, desc="LOSO Folds", leave=True)
    ):
        # Create train and test indices
        train_indices = [
            i
            for i, subject in enumerate(full_dataset.subject_indices)
            if subject != test_subject
        ]
        test_indices = [
            i
            for i, subject in enumerate(full_dataset.subject_indices)
            if subject == test_subject
        ]

        # Create Subset datasets
        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)

        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, pin_memory=True
        )

        # Model initialization
        model = MSTCN(
            num_stages=num_stages,
            num_layers=num_layers,
            num_classes=num_classes,
            input_dim=input_dim,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        num_epochs = 10
        for epoch in tqdm(
            range(num_epochs), desc=f"Training Fold {fold + 1}", leave=False
        ):
            model.train()
            training_loss = 0.0
            for i, (batch_x, batch_y) in enumerate(train_loader):
                # Data augmentation
                batch_x = augment_orientation(batch_x)

                batch_x, batch_y = batch_x.permute(0, 2, 1).to(device), batch_y.to(
                    device
                )
                optimizer.zero_grad()

                outputs = model(batch_x)
                loss = MSTCN_Loss(outputs, batch_y, lambda_coef)

                loss.backward()
                optimizer.step()

                training_loss += loss.item()

            avg_train_loss = training_loss / len(train_loader)

            # Save training data into CSV
            now = datetime.now()
            train_csv_writer.writerow(
                [
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    fold + 1,
                    epoch + 1,
                    avg_train_loss,
                ]
            )

        # Testing
        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x, batch_y = batch_x.permute(0, 2, 1).to(device), batch_y.to(
                    device
                )
                outputs = model(batch_x)
                final_output = outputs[-1]
                probabilities = F.softmax(final_output, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                all_predictions.extend(predicted_classes.view(-1).cpu().numpy())
                all_labels.extend(batch_y.view(-1).cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        # Post-processing predictions
        all_predictions = post_process_predictions(all_predictions)

        # Calculate metrics
        fp, fn, tp = 0, 0, 0
        for i in range(len(all_predictions)):
            fp += np.sum((all_predictions[i] == 1) & (all_labels[i] == 0))
            fn += np.sum((all_predictions[i] == 0) & (all_labels[i] == 1))
            tp += np.sum((all_predictions[i] == 1) & (all_labels[i] == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_sample = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        f1_segment_1 = segment_f1_multiclass(
            all_predictions, all_labels, 0.1, debug_plot
        )
        f1_segment_2 = segment_f1_multiclass(
            all_predictions, all_labels, 0.25, debug_plot
        )
        f1_segment_3 = segment_f1_multiclass(
            all_predictions, all_labels, 0.5, debug_plot
        )

        loso_f1_scores.append([f1_sample, f1_segment_1, f1_segment_2, f1_segment_3])

        # Save the F1 scores after testing
        now = datetime.now()
        test_csv_writer.writerow(
            [
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                fold + 1,
                f1_sample,
                f1_segment_1,
                f1_segment_2,
                f1_segment_3,
            ]
        )

        # Save the best model
        # torch.save(model.state_dict(), f"models/best_model_fold_{fold}.pth")
