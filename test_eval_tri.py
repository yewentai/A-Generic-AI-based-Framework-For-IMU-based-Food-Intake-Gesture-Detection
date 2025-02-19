import numpy as np
import matplotlib.pyplot as plt
from datasets import segment_confusion_matrix

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

# Calculate confusion matrix for each threshold
fn, fp, tp = segment_confusion_matrix(pred, gt, threshold=0.5, debug_plot=True)
f1_score = 2 * tp / (2 * tp + fp + fn)
print(f"F1 Score: {f1_score:.4f}")
