import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import pickle
import os
from datetime import datetime
from tqdm import tqdm

from augmentation import augment_orientation
from datasets import IMUDataset
from evaluation import segment_evaluation
from post_processing import post_process_predictions
from model_mstcn import MSTCN, MSTCN_Loss

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


# Function to segment data into 60-second windows
def segment_data(data, labels, window_size):
    segmented_data = []
    segmented_labels = []
    for d, l in zip(data, labels):
        for i in range(0, len(d) - window_size + 1, window_size):
            segmented_data.append(d[i : i + window_size])
            segmented_labels.append(l[i : i + window_size])
    return np.array(segmented_data), np.array(segmented_labels)


# Segment data
X, Y = segment_data(X, Y, window_size)

# Create the full dataset
full_dataset = IMUDataset(X, Y)

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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for saving result
if not os.path.exists("result"):
    os.makedirs("result")

# File names for training and testing result
training_stats_file = "result/training_stats_tcnmha.npy"
testing_stats_file = "result/testing_stats_tcnmha.npy"

# Initialize empty lists to store result
training_statistics = []
testing_statistics = []

# 7-Fold Cross-Validation
loso_f1_scores = []

for fold, test_subjects in enumerate(tqdm(test_folds, desc="7-Fold", leave=True)):
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

    # Create train and test dataset
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    # Initialize the model
    model = MSTCN(
        num_stages=2,
        num_layers=num_layers,
        num_classes=num_classes,
        input_dim=input_dim,
        num_filters=128,
        kernel_size=3,
        dropout=0.3,
    ).to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    num_epochs = 10
    for epoch in tqdm(range(num_epochs), desc=f"Training Fold {fold + 1}", leave=False):
        model.train()
        training_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            # Data augmentation
            batch_x = augment_orientation(batch_x)

            batch_x, batch_y = batch_x.permute(0, 2, 1).to(device), batch_y.to(device)
            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = MSTCN_Loss(outputs, batch_y, lambda_coef)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        avg_train_loss = training_loss / len(train_loader)
        # Save training result for each epoch
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
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x, batch_y = batch_x.permute(0, 2, 1).to(device), batch_y.to(device)
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

    # Calculate confusion matrix for threshold 0.5
    fn, fp, tp = segment_evaluation(all_predictions, all_labels, 0.5, debug_plot)

    # Calculate F1-score for the segment and avoid zero division
    f1_segment = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0

    # Save testing result for each fold
    testing_statistics.append(
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "fold": fold + 1,
            "f1_segment": f1_segment,
        }
    )

    loso_f1_scores.append(f1_segment)

# Save result to .npy files
np.save(training_stats_file, training_statistics)
np.save(testing_stats_file, testing_statistics)
