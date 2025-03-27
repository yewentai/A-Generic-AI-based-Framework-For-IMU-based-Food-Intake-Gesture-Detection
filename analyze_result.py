import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# -------------------------
# Configuration
RESULT_DIR = "result"  # Root folder for all results

# Automatically select the latest version based on timestamp
all_versions = glob.glob(os.path.join(RESULT_DIR, "*"))
RESULT_VERSION = max(all_versions, key=os.path.getmtime).split(os.sep)[-1]

# Or manually set the version
# RESULT_VERSION = "202503251515"

RESULT_PATH = os.path.join(RESULT_DIR, RESULT_VERSION)  # Path to this version

# -------------------------
# Load Training and Validation Stats
train_stats_file = os.path.join(RESULT_PATH, f"train_stats.npy")
train_stats = np.load(train_stats_file, allow_pickle=True).tolist()

# -------------------------
# Training Curve: Plot Loss Per Fold

# Identify all unique folds
folds = sorted(set(entry["fold"] for entry in train_stats))

for fold in folds:
    # Filter entries belonging to the current fold
    stats_fold = [entry for entry in train_stats if entry["fold"] == fold]

    # Extract all unique epochs in this fold
    epochs = sorted(set(entry["epoch"] for entry in stats_fold))

    # Initialize per-epoch loss containers
    loss_per_epoch = {epoch: [] for epoch in epochs}
    loss_ce_per_epoch = {epoch: [] for epoch in epochs}
    loss_mse_per_epoch = {epoch: [] for epoch in epochs}

    # Group loss values by epoch
    for entry in stats_fold:
        loss_per_epoch[entry["epoch"]].append(entry["train_loss"])
        loss_ce_per_epoch[entry["epoch"]].append(entry["train_loss_ce"])
        loss_mse_per_epoch[entry["epoch"]].append(entry["train_loss_mse"])

    # Compute mean loss per epoch
    mean_loss = [np.mean(loss_per_epoch[epoch]) for epoch in epochs]
    mean_ce = [np.mean(loss_ce_per_epoch[epoch]) for epoch in epochs]
    mean_mse = [np.mean(loss_mse_per_epoch[epoch]) for epoch in epochs]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_loss, linestyle="-", color="blue", label="Total Loss")
    plt.plot(epochs, mean_ce, linestyle="--", color="red", label="CE Loss")
    plt.plot(epochs, mean_mse, linestyle=":", color="green", label="MSE Loss")
    plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.title(f"Training Losses Over Epochs (Fold {fold})")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, f"train_loss_fold{fold}.png"), dpi=300)
    plt.close()
