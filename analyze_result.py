import numpy as np
import matplotlib.pyplot as plt

# Load results
TESTING_STATS_FILE = "result/testing_stats_tcnmha_dxii_v1.npy"
TRAINING_STATS_FILE = "result/training_stats_tcnmha_dxii_v1.npy"

testing_stats = np.load(TESTING_STATS_FILE, allow_pickle=True)
training_stats = np.load(TRAINING_STATS_FILE, allow_pickle=True)

# Extract F1 scores and folds
folds = [entry["fold"] for entry in testing_stats]
f1_scores = [entry["f1_segment"] for entry in testing_stats]

# Compute average F1 score
avg_f1 = np.mean(f1_scores)
print(f"Average F1 Score: {avg_f1:.4f}")

# Plot F1 scores per fold
plt.figure(figsize=(8, 5))
plt.bar(folds, f1_scores, color="skyblue", edgecolor="black")
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.title("F1 Score per Fold")
plt.xticks(folds)
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Extract training loss per epoch
fold_epochs = {}  # Dictionary to hold epochs and loss per fold
for entry in training_stats:
    fold = entry["fold"]
    epoch = entry["epoch"]
    loss = entry["train_loss"]
    loss_ce = entry["train_loss_ce"]
    loss_mse = entry["train_loss_mse"]

    if fold not in fold_epochs:
        fold_epochs[fold] = {"epochs": [], "losses_ce": [], "losses_mse": []}

    fold_epochs[fold]["epochs"].append(epoch)
    fold_epochs[fold]["losses_ce"].append(loss_ce)
    fold_epochs[fold]["losses_mse"].append(loss_mse)

# Plot training loss per epoch (two subplots)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot CE Loss
for fold, data in fold_epochs.items():
    ax1.plot(data["epochs"], data["losses_ce"], label=f"Fold {fold}")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("CE Loss")
ax1.set_title("Cross Entropy Loss per Epoch for Each Fold")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.7)

# Plot MSE Loss
for fold, data in fold_epochs.items():
    ax2.plot(data["epochs"], data["losses_mse"], label=f"Fold {fold}")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE Loss")
ax2.set_title("MSE Loss per Epoch for Each Fold")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
