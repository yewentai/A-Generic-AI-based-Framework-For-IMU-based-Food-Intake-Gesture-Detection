import numpy as np
import matplotlib.pyplot as plt

# Load results
TESTING_STATS_FILE = "result/testing_stats_tcnmha.npy"
TRAINING_STATS_FILE = "result/training_stats_tcnmha.npy"

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

    if fold not in fold_epochs:
        fold_epochs[fold] = {"epochs": [], "losses": []}

    fold_epochs[fold]["epochs"].append(epoch)
    fold_epochs[fold]["losses"].append(loss)

# Plot training loss per epoch
plt.figure(figsize=(10, 6))
for fold, data in fold_epochs.items():
    plt.plot(data["epochs"], data["losses"], marker="o", label=f"Fold {fold}")

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss per Epoch for Each Fold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
