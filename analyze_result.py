import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
DATASET = "DXI"  # Adjust if needed; ensure it matches the saved results
RESULT_VERSION = "20250313"  # Use the version prefix from your result files
result_dir = "result"

# Load data using the corrected file naming scheme
testing_stats = np.load(
    os.path.join(result_dir, f"{RESULT_VERSION}_testing_stats_{DATASET.lower()}.npy"),
    allow_pickle=True,
)
training_stats = np.load(
    os.path.join(result_dir, f"{RESULT_VERSION}_training_stats_{DATASET.lower()}.npy"),
    allow_pickle=True,
)

# Convert testing_stats to a list for easier processing (if not already)
testing_stats = list(testing_stats)

# Statistical analysis
# Compute average sample-wise F1 per fold using the "f1_scores_sample" dictionary.
# Each entry in "f1_scores_sample" is assumed to be a dict like {"1": {"f1": value}, "2": {"f1": value}, ...}
f1_scores = []
for e in testing_stats:
    fold_scores = [v["f1"] for v in e["f1_scores_sample"].values()]
    f1_scores.append(np.mean(fold_scores))

# Extract Cohen Kappa and Matthews CorrCoef scores from each fold
cohen_kappa_scores = [e["cohen_kappa"] for e in testing_stats]
matthews_corrcoef_scores = [e["matthews_corrcoef"] for e in testing_stats]

# For segment metrics, use class "1" as an example.
tp_total = sum(e["metrics_segment"].get("1", {}).get("tp", 0) for e in testing_stats)
fp_total = sum(e["metrics_segment"].get("1", {}).get("fp", 0) for e in testing_stats)
fn_total = sum(e["metrics_segment"].get("1", {}).get("fn", 0) for e in testing_stats)

print(f"\n{' Results Summary ':=^40}")
print(f"Average F1 (sample-wise): {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Total TP (Segment Class 1): {tp_total}  FP: {fp_total}  FN: {fn_total}")
print(
    f"Average Cohen Kappa: {np.mean(cohen_kappa_scores):.4f} ± {np.std(cohen_kappa_scores):.4f}"
)
print(
    f"Average Matthews CorrCoef: {np.mean(matthews_corrcoef_scores):.4f} ± {np.std(matthews_corrcoef_scores):.4f}"
)

# Visualization
plt.figure(figsize=(12, 6))

# F1 distribution boxplot
plt.subplot(131)
plt.boxplot(
    f1_scores,
    vert=False,
    widths=0.6,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue"),
)
plt.title("F1 Score Distribution")
plt.xlim(0, 1)

# Fold-wise performance bar plot for F1 scores
plt.subplot(132)
plt.bar(range(1, len(f1_scores) + 1), f1_scores, color="skyblue")
plt.xticks(range(1, len(f1_scores) + 1))
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.title("F1 Score per Fold")
plt.ylim(0, 1)

# Additional visualization: Cohen Kappa and Matthews CorrCoef per fold
plt.subplot(133)
width = 0.35
folds = np.arange(1, len(cohen_kappa_scores) + 1)
plt.bar(folds - width / 2, cohen_kappa_scores, width=width, label="Cohen Kappa")
plt.bar(
    folds + width / 2, matthews_corrcoef_scores, width=width, label="Matthews CorrCoef"
)
plt.xticks(folds)
plt.xlabel("Fold")
plt.ylabel("Score")
plt.title("Other Metrics per Fold")
plt.legend()

plt.tight_layout()
plt.show()
