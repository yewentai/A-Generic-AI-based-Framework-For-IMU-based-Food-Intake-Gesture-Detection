import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
DATASET = "FDI"  # Ensure this matches the dataset used in training
RESULT_VERSION = "202503201553"  # Version identifier for the saved results
RESULT_DIR = "result"  # Base directory where result files are stored

# Define the specific directory for this version
RESULT_PATH = os.path.join(RESULT_DIR, RESULT_VERSION)

# Load training and validation statistics from saved .npy files
training_stats_file = os.path.join(RESULT_PATH, f"training_stats_{DATASET.lower()}.npy")
validating_stats_file = os.path.join(
    RESULT_PATH, f"validating_stats_{DATASET.lower()}.npy"
)
training_stats = np.load(training_stats_file, allow_pickle=True)
validating_stats = np.load(validating_stats_file, allow_pickle=True)

# Convert loaded data to lists for easier processing
training_stats = list(training_stats)
validating_stats = list(validating_stats)

# Extract epoch-wise training statistics
# Initialize dictionaries to store losses and metrics for each epoch
epochs = sorted(set(entry["epoch"] for entry in training_stats))
loss_per_epoch = {epoch: [] for epoch in epochs}
loss_ce_per_epoch = {epoch: [] for epoch in epochs}  # Cross-entropy loss
loss_mse_per_epoch = {epoch: [] for epoch in epochs}  # Mean squared error loss
matthews_corrcoef_per_epoch = {epoch: [] for epoch in epochs}  # MCC metric

# Populate dictionaries with data from training statistics
for entry in training_stats:
    loss_per_epoch[entry["epoch"]].append(entry["train_loss"])
    loss_ce_per_epoch[entry["epoch"]].append(entry["train_loss_ce"])
    loss_mse_per_epoch[entry["epoch"]].append(entry["train_loss_mse"])
    matthews_corrcoef_per_epoch[entry["epoch"]].append(entry["matthews_corrcoef"])

# Compute mean values for each epoch
mean_loss_per_epoch = [np.mean(loss_per_epoch[epoch]) for epoch in epochs]
mean_loss_ce_per_epoch = [np.mean(loss_ce_per_epoch[epoch]) for epoch in epochs]
mean_loss_mse_per_epoch = [np.mean(loss_mse_per_epoch[epoch]) for epoch in epochs]
mean_matthews_corrcoef_per_epoch = [
    np.mean(matthews_corrcoef_per_epoch[epoch]) for epoch in epochs
]

# Extract fold-wise validation metrics
# Initialize lists to store metrics for each fold
label_distribution = []  # Label distribution
f1_scores_sample = []  # Sample-wise F1 scores
f1_scores_segment = []  # Segment-wise F1 scores
cohen_kappa_scores = []  # Cohen's kappa scores
matthews_corrcoef_scores = []  # MCC scores

# Populate lists with data from validation statistics
for entry in validating_stats:
    label_dist = entry["label_distribution"]  # dictionary: {label: count}

    # Compute weighted average F1 for sample-wise metrics
    total_weight_sample = 0.0
    weighted_f1_sample = 0.0
    for label_str, stats in entry["metrics_sample"].items():
        label_int = int(label_str)  # Convert key to int if necessary
        weight = label_dist.get(label_int, 0)
        weighted_f1_sample += stats["f1"] * weight
        total_weight_sample += weight
    f1_sample_weighted = (
        weighted_f1_sample / total_weight_sample if total_weight_sample > 0 else 0.0
    )

    # Compute weighted average F1 for segment-wise metrics
    total_weight_segment = 0.0
    weighted_f1_segment = 0.0
    for label_str, stats in entry["metrics_segment"].items():
        label_int = int(label_str)
        weight = label_dist.get(label_int, 0)
        weighted_f1_segment += stats["f1"] * weight
        total_weight_segment += weight
    f1_segment_weighted = (
        weighted_f1_segment / total_weight_segment if total_weight_segment > 0 else 0.0
    )

    # Append results
    label_distribution.append(label_dist)
    f1_scores_sample.append(f1_sample_weighted)
    f1_scores_segment.append(f1_segment_weighted)
    cohen_kappa_scores.append(entry["cohen_kappa"])
    matthews_corrcoef_scores.append(entry["matthews_corrcoef"])

# Plot1.1: Training Losses Over Epochs
plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.plot(
    epochs,
    mean_loss_per_epoch,
    marker="o",
    linestyle="-",
    color="blue",
    label="Total Loss",
)
plt.plot(
    epochs,
    mean_loss_ce_per_epoch,
    marker="s",
    linestyle="--",
    color="red",
    label="CE Loss",
)
plt.plot(
    epochs,
    mean_loss_mse_per_epoch,
    marker="^",
    linestyle=":",
    color="green",
    label="MSE Loss",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Losses Over Epochs")
plt.grid()
plt.legend()

# Plot1.2: Matthews Correlation Coefficient Over Epochs
plt.subplot(122)
plt.plot(
    epochs,
    mean_matthews_corrcoef_per_epoch,
    marker="o",
    linestyle="-",
    color="purple",
)
plt.xlabel("Epoch")
plt.ylabel("Matthews Correlation Coefficient")
plt.title("Matthews Correlation Coefficient Over Epochs")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, f"{RESULT_VERSION}_matthews_corrcoef_plot.png"))

# Plot2: Fold-wise Performance Metrics
# Bar plot for Cohen Kappa, MCC, Sample-wise F1, and Segment-wise F1 scores
plt.figure(figsize=(12, 6))
width = 0.2  # Width of each bar
folds = np.arange(1, len(cohen_kappa_scores) + 1)  # Fold indices

plt.bar(
    folds - width * 1.5,
    cohen_kappa_scores,
    width=width,
    label="Cohen Kappa",
    color="orange",
)
plt.bar(
    folds - width / 2,
    matthews_corrcoef_scores,
    width=width,
    label="MCC",
    color="purple",
)
plt.bar(
    folds + width / 2,
    f1_scores_sample,
    width=width,
    label="Sample-wise F1",
    color="blue",
)
plt.bar(
    folds + width * 1.5,
    f1_scores_segment,
    width=width,
    label="Segment-wise F1",
    color="green",
)

plt.xticks(folds)
plt.xlabel("Fold")
plt.ylabel("Score")
plt.title("Fold-wise Performance Metrics")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, f"{RESULT_VERSION}_test_metrics_plot.png"))
