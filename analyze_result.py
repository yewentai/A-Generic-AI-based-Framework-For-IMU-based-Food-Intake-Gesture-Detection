import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# Configuration

DATASET = "FDI"  # Dataset name, must match training
RESULT_VERSION = "202503212330"  # Timestamp/version identifier
RESULT_DIR = "result"  # Root folder for all results
RESULT_PATH = os.path.join(RESULT_DIR, RESULT_VERSION)  # Path to this version

# -------------------------
# Load Training and Validation Stats

training_stats_file = os.path.join(RESULT_PATH, f"training_stats_{DATASET.lower()}.npy")
validating_stats_file = os.path.join(
    RESULT_PATH, f"validating_stats_{DATASET.lower()}.npy"
)

training_stats = list(np.load(training_stats_file, allow_pickle=True))
validating_stats = list(np.load(validating_stats_file, allow_pickle=True))

# -------------------------
# Training Curve: Plot Loss Per Fold

# Identify all unique folds
folds = sorted(set(entry["fold"] for entry in training_stats))

for fold in folds:
    # Filter entries belonging to the current fold
    stats_fold = [entry for entry in training_stats if entry["fold"] == fold]

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
    plt.savefig(os.path.join(RESULT_PATH, f"train_metrics_fold{fold}.png"), dpi=300)
    plt.close()

# -------------------------
# Validation Metrics: Fold-wise Aggregated Scores

# Initialize containers
label_distribution = []
f1_scores_sample = []
f1_scores_segment = []
cohen_kappa_scores = []
matthews_corrcoef_scores = []

# Process each fold's validation results
for entry in validating_stats:
    label_dist = entry["label_distribution"]  # e.g., {0: count, 1: count}
    label_distribution.append(label_dist)

    # Weighted F1 (sample-wise)
    total_weight_sample = sum(
        label_dist.get(int(k), 0) for k in entry["metrics_sample"]
    )
    weighted_f1_sample = sum(
        entry["metrics_sample"][k]["f1"] * label_dist.get(int(k), 0)
        for k in entry["metrics_sample"]
    )
    f1_scores_sample.append(
        weighted_f1_sample / total_weight_sample if total_weight_sample > 0 else 0.0
    )

    # Weighted F1 (segment-wise)
    total_weight_segment = sum(
        label_dist.get(int(k), 0) for k in entry["metrics_segment"]
    )
    weighted_f1_segment = sum(
        entry["metrics_segment"][k]["f1"] * label_dist.get(int(k), 0)
        for k in entry["metrics_segment"]
    )
    f1_scores_segment.append(
        weighted_f1_segment / total_weight_segment if total_weight_segment > 0 else 0.0
    )

    # Other metrics
    cohen_kappa_scores.append(entry["cohen_kappa"])
    matthews_corrcoef_scores.append(entry["matthews_corrcoef"])

# -------------------------
# Plot Validation Metrics (Bar Chart)

plt.figure(figsize=(12, 6))
width = 0.2
fold_indices = np.arange(1, len(cohen_kappa_scores) + 1)

plt.bar(
    fold_indices - width * 1.5,
    cohen_kappa_scores,
    width=width,
    color="orange",
    label="Cohen Kappa",
)
plt.bar(
    fold_indices - width / 2,
    matthews_corrcoef_scores,
    width=width,
    color="purple",
    label="MCC",
)
plt.bar(
    fold_indices + width / 2,
    f1_scores_sample,
    width=width,
    color="blue",
    label="Sample-wise F1",
)
plt.bar(
    fold_indices + width * 1.5,
    f1_scores_segment,
    width=width,
    color="green",
    label="Segment-wise F1",
)

plt.xticks(fold_indices)
plt.xlabel("Fold")
plt.ylabel("Score")
plt.title("Fold-wise Performance Metrics")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, "validate_metrics_plot.png"), dpi=300)
plt.close()
