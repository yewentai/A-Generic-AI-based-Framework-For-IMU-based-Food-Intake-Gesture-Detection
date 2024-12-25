import numpy as np
import matplotlib.pyplot as plt
from utils import segment_f1_binary, post_process_predictions

# Ground Truth sequence
gt = np.array(
    [
        0,
        0,
        1,
        1,
        1,
        0,
        0,  # Normal gesture
        0,
        0,
        0,
        0,
        0,
        0,
        0,  # No gesture (for insertion test)
        1,
        1,
        1,
        1,
        0,
        0,
        0,  # Gesture for underfill and overfill tests
        0,
        1,
        1,
        0,
        1,
        1,
        0,  # Two gestures for merge test
        1,
        1,
        1,
        1,
        1,
        1,
        1,  # Long gesture for fragmentation test
        0,
        0,
        1,
        1,
        1,
        0,
        0,  # Gesture for deletion test
        1,
        1,
        0,
        1,
        1,
        1,
        0,
    ]
)

# Prediction sequence
pred = np.array(
    [
        0,
        1,
        1,
        1,
        1,
        1,
        0,  # Overfill
        0,
        1,
        1,
        0,
        0,
        0,
        0,  # Insertion
        1,
        1,
        0,
        0,
        0,
        0,
        0,  # Underfill
        0,
        1,
        1,
        1,
        1,
        1,
        0,  # Merge
        1,
        0,
        1,
        0,
        1,
        0,
        1,  # Fragmentation
        0,
        0,
        0,
        0,
        0,
        0,
        0,  # Deletion
        0,
        1,
        1,
        1,
        0,
        1,
        1,  # Mismatch
    ]
)

# Plot the sequences
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

# Plot ground truth
ax1.step(range(len(gt)), gt, where="post", label="Ground Truth", color="blue")
ax1.set_ylabel("State")
ax1.set_ylim(-0.1, 1.1)
ax1.legend()
ax1.set_title("Ground Truth")

# Plot prediction
ax2.step(range(len(pred)), pred, where="post", label="Prediction", color="red")
ax2.set_xlabel("Time")
ax2.set_ylabel("State")
ax2.set_ylim(-0.1, 1.1)
ax2.legend()
ax2.set_title("Prediction")

# Add labels for different situations
situations = [
    "Overfill",
    "Insertion",
    "Underfill",
    "Merge",
    "Fragmentation",
    "Deletion",
    "Mismatch",
]
for i, situation in enumerate(situations):
    ax1.text(i * 7 + 3, 1.15, situation, ha="center", va="center", rotation=45)
    ax2.text(i * 7 + 3, -0.15, situation, ha="center", va="center", rotation=45)

plt.suptitle(f"Ground Truth vs Prediction")
plt.tight_layout()
plt.show()

# Calculate F1 score sample-wise
t_p = np.sum(np.logical_and(pred == 1, gt == 1))
f_p = np.sum(np.logical_and(pred == 1, gt == 0))
f_n = np.sum(np.logical_and(pred == 0, gt == 1))
precision = t_p / (t_p + f_p)
recall = t_p / (t_p + f_n)
f1_score = 2 * precision * recall / (precision + recall)
print(f"F1 Score (sample): {f1_score:.4f}")

# Post-processing predictions
# pred = post_process_predictions(pred, 2, 1)

# Calculate F1 score segment-wise
f1_score_seg = segment_f1_binary(pred, gt, 0.3, debug_plot=True)
print(f"F1 Score (seg): {f1_score_seg:.4f}")
