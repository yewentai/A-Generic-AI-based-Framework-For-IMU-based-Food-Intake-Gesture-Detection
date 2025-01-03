import numpy as np
import matplotlib.pyplot as plt
from utils import segment_f1_multiclass

# Ground Truth sequence (3 classes: 0=background, 1=class1, 2=class2, 3=class3)
# fmt: off
gt = np.array([
    0, 0, 1, 1, 1, 0, 0,  # Normal class 1 gesture
    0, 0, 2, 2, 2, 0, 0,  # Normal class 2 gesture
    3, 3, 3, 3, 0, 0, 0,  # Class 3 gesture for testing
    0, 1, 1, 0, 2, 2, 0,  # Mixed classes
    3, 3, 3, 3, 3, 3, 3,  # Long class 3 gesture
    0, 0, 1, 2, 3, 0, 0,  # Sequential classes
    1, 2, 0, 3, 2, 1, 0   # Mixed pattern
])

# Prediction sequence
pred = np.array([
    0, 1, 1, 1, 1, 1, 0,  # Overfill class 1
    0, 2, 2, 1, 2, 0, 0,  # Class mismatch
    3, 3, 2, 2, 0, 0, 0,  # Wrong class prediction
    0, 1, 1, 1, 2, 2, 0,  # Correct mixed classes
    3, 0, 3, 0, 3, 0, 3,  # Fragmented class 3
    0, 0, 0, 0, 0, 0, 0,  # Missing detections
    1, 1, 1, 3, 3, 3, 0   # Class confusion
])
# fmt: on

# Plot the sequences
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

# Plot ground truth
ax1.step(range(len(gt)), gt, where="post", label="Ground Truth")
ax1.set_ylabel("Class")
ax1.set_ylim(-0.5, 3.5)
ax1.legend()
ax1.set_title("Ground Truth")
ax1.grid(True)

# Plot prediction
ax2.step(range(len(pred)), pred, where="post", label="Prediction")
ax2.set_xlabel("Time")
ax2.set_ylabel("Class")
ax2.set_ylim(-0.5, 3.5)
ax2.legend()
ax2.set_title("Prediction")
ax2.grid(True)

# Add labels for different situations
situations = [
    "Overfill",
    "Mismatch",
    "Wrong Class",
    "Correct Mix",
    "Fragmentation",
    "Missing",
    "Confusion",
]
for i, situation in enumerate(situations):
    ax1.text(i * 7 + 3, 3.7, situation, ha="center", va="center", rotation=45)
    ax2.text(i * 7 + 3, -0.7, situation, ha="center", va="center", rotation=45)

plt.suptitle("Multi-class Segmentation Test")
plt.tight_layout()
plt.show()

# Calculate F1 score for each class
f1_score = segment_f1_multiclass(pred, gt, threshold=0.5, debug_plot=True)
print(f"F1 Score: {f1_score:.4f}")
