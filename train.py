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

# Hyperparameters
NUM_STAGES = 2
NUM_LAYERS = 9
NUM_CLASSES = 3
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
WINDOW_SIZE = SAMPLING_FREQ * WINDOW_LENGTH  # 16 Hz * 60 s = 960
DEBUG_PLOT = False
NUM_FOLDS = 13
NUM_EPOCHS = 20
BATCH_SIZE = 32
NUM_WORKERS = 16
FLAG_AUGMENT = False
FLAG_MIRROR = True

# Load data
X_L_PATH = "./dataset/DX/DX-I/X_L.pkl"
Y_L_PATH = "./dataset/DX/DX-I/Y_L.pkl"
X_R_PATH = "./dataset/DX/DX-I/X_R.pkl"
Y_R_PATH = "./dataset/DX/DX-I/Y_R.pkl"

# File paths for results
TRAINING_STATS_FILE = "result/training_stats_tcnmha_dxi_v3.npy"
TESTING_STATS_FILE = "result/testing_stats_tcnmha_dxi_v3.npy"
CONFIG_FILE = "result/config_tcnmha_dxi_v3.txt"

with open(X_L_PATH, "rb") as f:
    X_L = np.array(pickle.load(f), dtype=object)
with open(Y_L_PATH, "rb") as f:
    Y_L = np.array(pickle.load(f), dtype=object)
with open(X_R_PATH, "rb") as f:
    X_R = np.array(pickle.load(f), dtype=object)
with open(Y_R_PATH, "rb") as f:
    Y_R = np.array(pickle.load(f), dtype=object)

# Hand mirroring
if FLAG_MIRROR:
    X_L = np.array([hand_mirroring(sample) for sample in X_L], dtype=object)

# Concatenate the left and right data
X = np.concatenate([X_L, X_R], axis=0)
Y = np.concatenate([Y_L, Y_R], axis=0)

# Create the full dataset
full_dataset = IMUDataset(X, Y, sequence_length=WINDOW_SIZE)

# Create balanced subject folds
test_folds = create_balanced_subject_folds(full_dataset, num_folds=NUM_FOLDS)

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

    # Create datasets
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
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

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Track best F1 score for this fold
    best_f1_score = 0.0

    # Training loop
    for epoch in tqdm(range(NUM_EPOCHS), desc=f"Fold {fold+1}", leave=False):
        model.train()
        training_loss = 0.0
        training_loss_ce = 0.0
        training_loss_mse = 0.0

        for batch_x, batch_y in train_loader:
            # Data augmentation
            if FLAG_AUGMENT:
                batch_x = augment_orientation(batch_x)
            batch_x = batch_x.permute(0, 2, 1).to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            ce_loss, mse_loss = MSTCN_Loss(outputs, batch_y)
            loss = ce_loss + LAMBDA_COEF * mse_loss
            # loss = ce_loss
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_loss_ce += ce_loss.item()
            training_loss_mse += mse_loss.item()

        # Record training statistics
        avg_train_loss = training_loss / len(train_loader)
        avg_train_loss_ce = training_loss_ce / len(train_loader)
        avg_train_loss_mse = training_loss_mse / len(train_loader)
        training_statistics.append(
            {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_loss_ce": avg_train_loss_ce,
                "train_loss_mse": avg_train_loss_mse,
            }
        )

        # Evaluation after each epoch
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.permute(0, 2, 1).to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                final_output = outputs[-1]
                probabilities = F.softmax(final_output, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)

                all_predictions.extend(predicted_classes.view(-1).cpu().numpy())
                all_labels.extend(batch_y.view(-1).cpu().numpy())

        # Post-processing and evaluation
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_predictions = post_process_predictions(all_predictions)

        fn_1, fp_1, tp_1 = segment_evaluation(
            all_predictions,
            all_labels,
            class_label=1,
            threshold=0.5,
            debug_plot=DEBUG_PLOT,
        )

        fn_2, fp_2, tp_2 = segment_evaluation(
            all_predictions,
            all_labels,
            class_label=2,
            threshold=0.5,
            debug_plot=DEBUG_PLOT,
        )

        f1_segment_1 = 2 * tp_1 / (2 * tp_1 + fp_1 + fn_1) if tp_1 > 0 else 0
        f1_segment_2 = 2 * tp_2 / (2 * tp_2 + fp_2 + fn_2) if tp_2 > 0 else 0
        f1_segment = (f1_segment_1 + f1_segment_2) / 2

        # Save checkpoint and check if this is the best model
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

    # Record best F1 score for this fold
    loso_f1_scores.append(best_f1_score)

    # Record testing statistics
    testing_statistics.append(
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "fold": fold + 1,
            "f1_segment": best_f1_score,
        }
    )

# Save results
np.save(TRAINING_STATS_FILE, training_statistics)
np.save(TESTING_STATS_FILE, testing_statistics)
np.save(
    CONFIG_FILE,
    {
        "num_stages": NUM_STAGES,
        "num_layers": NUM_LAYERS,
        "num_classes": NUM_CLASSES,
        "num_heads": NUM_HEADS,
        "input_dim": INPUT_DIM,
        "num_filters": NUM_FILTERS,
        "kernel_size": KERNEL_SIZE,
        "dropout": DROPOUT,
        "lambda_coef": LAMBDA_COEF,
        "tau": TAU,
        "learning_rate": LEARNING_RATE,
        "sampling_freq_original": SAMPLING_FREQ_ORIGINAL,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "sampling_freq": SAMPLING_FREQ,
        "window_length": WINDOW_LENGTH,
        "window_size": WINDOW_SIZE,
        "debug_plot": DEBUG_PLOT,
        "num_folds": NUM_FOLDS,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "flag_augment": FLAG_AUGMENT,
        "flag_mirror": FLAG_MIRROR,
    },
)
