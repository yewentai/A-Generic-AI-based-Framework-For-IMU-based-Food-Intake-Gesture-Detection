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
from components.pre_processing import hand_mirroring
from components.post_processing import post_process_predictions
from components.checkpoint import save_checkpoint
from components.model_mstcn import MSTCN, MSTCN_Loss

# ********************** Configuration Parameters **********************
DATASET = "DXI"  # Options: DX/DXI/DXII or FD/FDI/FDII
NUM_STAGES = 2
NUM_LAYERS = 9
NUM_HEADS = 8
INPUT_DIM = 6
NUM_FILTERS = 128
KERNEL_SIZE = 3
DROPOUT = 0.3
LAMBDA_COEF = 0.15
TAU = 4
LEARNING_RATE = 1e-3
SAMPLING_FREQ_ORIGINAL = 64
DOWNSAMPLE_FACTOR = 4
SAMPLING_FREQ = SAMPLING_FREQ_ORIGINAL // DOWNSAMPLE_FACTOR
WINDOW_LENGTH = 60
WINDOW_SIZE = SAMPLING_FREQ * WINDOW_LENGTH
DEBUG_PLOT = False
NUM_FOLDS = 13
NUM_EPOCHS = 20
BATCH_SIZE = 32
NUM_WORKERS = 16
FLAG_AUGMENT = False
FLAG_MIRROR = True

# Configure parameters based on dataset type
if DATASET.startswith("DX"):
    NUM_CLASSES = 2
    dataset_type = "DX"
    sub_version = (
        DATASET.replace("DX", "").upper() or "I"
    )  # Handle formats like DX/DXII
    DATA_DIR = f"./dataset/DX/DX-{sub_version}"
elif DATASET.startswith("FD"):
    NUM_CLASSES = 3
    dataset_type = "FD"
    sub_version = DATASET.replace("FD", "").upper() or "I"
    DATA_DIR = f"./dataset/FD/FD-{sub_version}"
else:
    raise ValueError(f"Invalid dataset: {DATASET}")

# Data paths
X_L_PATH = os.path.join(DATA_DIR, "X_L.pkl")
Y_L_PATH = os.path.join(DATA_DIR, "Y_L.pkl")
X_R_PATH = os.path.join(DATA_DIR, "X_R.pkl")
Y_R_PATH = os.path.join(DATA_DIR, "Y_R.pkl")

# Result file paths
version_suffix = datetime.now().strftime("%Y%m%d%H%M")[-4:]
TRAINING_STATS_FILE = f"result/training_stats_{DATASET.lower()}_{version_suffix}.npy"
TESTING_STATS_FILE = f"result/testing_stats_{DATASET.lower()}_{version_suffix}.npy"
CONFIG_FILE = f"result/config_{DATASET.lower()}_{version_suffix}.txt"
# ****************************************************

# Load data
with open(X_L_PATH, "rb") as f:
    X_L = np.array(pickle.load(f), dtype=object)
with open(Y_L_PATH, "rb") as f:
    Y_L = np.array(pickle.load(f), dtype=object)
with open(X_R_PATH, "rb") as f:
    X_R = np.array(pickle.load(f), dtype=object)
with open(Y_R_PATH, "rb") as f:
    Y_R = np.array(pickle.load(f), dtype=object)

# Hand mirroring processing
if FLAG_MIRROR:
    X_L = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)

# Merge data
X = np.concatenate([X_L, X_R], axis=0)
Y = np.concatenate([Y_L, Y_R], axis=0)

# Create dataset
full_dataset = IMUDataset(X, Y, sequence_length=WINDOW_SIZE)

# Create balanced cross-validation folds
test_folds = create_balanced_subject_folds(full_dataset, num_folds=NUM_FOLDS)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create result directory
os.makedirs("result", exist_ok=True)

# Initialize statistics records
training_statistics = []
testing_statistics = []
loso_f1_scores = []

# Cross-validation main loop
for fold, test_subjects in enumerate(tqdm(test_folds, desc="K-Fold", leave=True)):
    # Split training and testing sets
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

    # Create data loaders
    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize model
    model = MSTCN(
        num_stages=NUM_STAGES,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        input_dim=INPUT_DIM,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_f1_score = 0.0

    # Training loop
    for epoch in tqdm(range(NUM_EPOCHS), desc=f"Fold {fold+1}", leave=False):
        model.train()
        training_loss = 0.0
        training_loss_ce = 0.0
        training_loss_mse = 0.0

        for batch_x, batch_y in train_loader:
            if FLAG_AUGMENT:
                batch_x = augment_orientation(batch_x)
            batch_x = batch_x.permute(0, 2, 1).to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            ce_loss, mse_loss = MSTCN_Loss(outputs, batch_y)
            loss = ce_loss + LAMBDA_COEF * mse_loss
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_loss_ce += ce_loss.item()
            training_loss_mse += mse_loss.item()

        # Record training statistics
        training_statistics.append(
            {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": training_loss / len(train_loader),
                "train_loss_ce": training_loss_ce / len(train_loader),
                "train_loss_mse": training_loss_mse / len(train_loader),
            }
        )

    # Evaluation phase
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.permute(0, 2, 1).to(device)
            outputs = model(batch_x)
            probabilities = F.softmax(outputs[-1], dim=1)
            all_predictions.extend(
                torch.argmax(probabilities, dim=1).view(-1).cpu().numpy()
            )
            all_labels.extend(batch_y.view(-1).cpu().numpy())

    # Post-processing
    all_predictions = post_process_predictions(np.array(all_predictions))
    all_labels = np.array(all_labels)

    # Multi-class evaluation
    class_metrics = {}
    class_f1_scores = []
    for class_label in range(1, NUM_CLASSES):
        fn, fp, tp = segment_evaluation(
            all_predictions,
            all_labels,
            class_label=class_label,
            threshold=0.5,
            debug_plot=DEBUG_PLOT,
        )
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator != 0 else 0.0

        class_metrics[f"class_{class_label}"] = {
            "fn": int(fn),
            "fp": int(fp),
            "tp": int(tp),
            "f1": float(f1),
        }
        class_f1_scores.append(f1)

    avg_f1 = np.mean(class_f1_scores)

    # Record testing statistics
    testing_statistics.append(
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "fold": fold + 1,
            "average_f1": float(avg_f1),
            "class_metrics": class_metrics,
        }
    )
    loso_f1_scores.append(avg_f1)

    # Save best model
    if avg_f1 > best_f1_score:
        best_f1_score = avg_f1
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            fold=fold,
            f1_score=avg_f1,
            is_best=True,
        )

# Save results and configuration
np.save(TRAINING_STATS_FILE, training_statistics)
np.save(TESTING_STATS_FILE, testing_statistics)

config_info = {
    "dataset": DATASET,
    "num_classes": NUM_CLASSES,
    "num_stages": NUM_STAGES,
    "num_layers": NUM_LAYERS,
    "num_heads": NUM_HEADS,
    "input_dim": INPUT_DIM,
    "num_filters": NUM_FILTERS,
    "kernel_size": KERNEL_SIZE,
    "dropout": DROPOUT,
    "lambda_coef": LAMBDA_COEF,
    "tau": TAU,
    "learning_rate": LEARNING_RATE,
    "sampling_freq": SAMPLING_FREQ,
    "window_size": WINDOW_SIZE,
    "num_folds": NUM_FOLDS,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "augmentation": FLAG_AUGMENT,
    "mirroring": FLAG_MIRROR,
    "best_avg_f1": float(np.mean(loso_f1_scores)),
    "f1_std": float(np.std(loso_f1_scores)),
}

with open(CONFIG_FILE, "w") as f:
    for key, value in config_info.items():
        f.write(f"{key}: {value}\n")
