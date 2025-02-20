import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import pickle
import os
from datetime import datetime
from tqdm import tqdm

from components.augmentation import augment_orientation
from components.datasets import IMUDataset, create_balanced_subject_folds
from components.evaluation import segment_evaluation
from components.post_processing import post_process_predictions
from components.checkpoint import save_checkpoint
from components.model_cnnlstm import CNN_LSTM

# Hyperparameters
HYPERPARAMS = {
    "learning_rate": 0.0005,
    "batch_size": 16,
    "num_epochs": 10,
    "sampling_freq_original": 64,
    "downsample_factor": 4,
    "window_length": 60,
    "num_folds": 5,
    "num_workers": 16,
}

SAMPLING_FREQ = (
    HYPERPARAMS["sampling_freq_original"] // HYPERPARAMS["downsample_factor"]
)
WINDOW_SIZE = SAMPLING_FREQ * HYPERPARAMS["window_length"]
DEBUG_PLOT = False

# Load data
X_L_PATH = "./dataset/FD/FD-I/X_L.pkl"
Y_L_PATH = "./dataset/FD/FD-I/Y_L.pkl"
X_R_PATH = "./dataset/FD/FD-I/X_R.pkl"
Y_R_PATH = "./dataset/FD/FD-I/Y_R.pkl"

with open(X_L_PATH, "rb") as f:
    X_L = np.array(pickle.load(f), dtype=object)
with open(Y_L_PATH, "rb") as f:
    Y_L = np.array(pickle.load(f), dtype=object)
with open(X_R_PATH, "rb") as f:
    X_R = np.array(pickle.load(f), dtype=object)
with open(Y_R_PATH, "rb") as f:
    Y_R = np.array(pickle.load(f), dtype=object)

# Concatenate the left and right data
X = np.concatenate([X_L, X_R], axis=0)
Y = np.concatenate([Y_L, Y_R], axis=0)

# Create the full dataset
full_dataset = IMUDataset(X, Y, sequence_length=WINDOW_SIZE)

# Create balanced subject folds
test_folds = create_balanced_subject_folds(
    full_dataset, num_folds=HYPERPARAMS["num_folds"]
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for saving results
os.makedirs("result", exist_ok=True)

# Initialize lists to store statistics
training_statistics = []
testing_statistics = []

# K-Fold Cross-Validation
loso_f1_scores = []

for fold, test_subjects in enumerate(tqdm(test_folds, desc="K-Fold", leave=True)):
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

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=HYPERPARAMS["batch_size"],
        shuffle=True,
        num_workers=HYPERPARAMS["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=HYPERPARAMS["batch_size"],
        shuffle=False,
        num_workers=HYPERPARAMS["num_workers"],
        pin_memory=True,
    )

    # Initialize model
    model = CNN_LSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMS["learning_rate"])
    loss_fn = torch.nn.BCELoss()

    best_f1_score = 0.0

    for epoch in tqdm(
        range(HYPERPARAMS["num_epochs"]), desc=f"Fold {fold+1}", leave=False
    ):
        model.train()
        training_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = augment_orientation(batch_x)
            batch_x = batch_x.permute(0, 2, 1).to(device)
            batch_y = batch_y.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        avg_train_loss = training_loss / len(train_loader)
        training_statistics.append(
            {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
            }
        )

        # Evaluation
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.permute(0, 2, 1).to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x).squeeze()
                predicted_classes = (outputs > 0.5).float()

                all_predictions.extend(predicted_classes.view(-1).cpu().numpy())
                all_labels.extend(batch_y.view(-1).cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_predictions = post_process_predictions(all_predictions)

        f1_segment = segment_evaluation(
            all_predictions,
            all_labels,
            class_label=1,
            threshold=0.5,
            debug_plot=DEBUG_PLOT,
        )

        is_best = f1_segment > best_f1_score
        if is_best:
            best_f1_score = f1_segment

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            fold=fold,
            f1_score=f1_segment,
            is_best=is_best,
        )

    loso_f1_scores.append(best_f1_score)
    testing_statistics.append(
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "fold": fold + 1,
            "f1_segment": best_f1_score,
        }
    )

# Save results
np.save("result/training_stats_cnn_lstm.npy", training_statistics)
np.save("result/testing_stats_cnn_lstm.npy", testing_statistics)
