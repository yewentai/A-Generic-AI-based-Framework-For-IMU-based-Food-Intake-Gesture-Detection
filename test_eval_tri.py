import numpy as np
import matplotlib.pyplot as plt
from components.evaluation import segment_evaluation

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


# Calculate F1 score sample-wise
tp = np.sum(np.logical_and(pred == 1, gt == 1))
fp = np.sum(np.logical_and(pred == 1, gt != 1))
fn = np.sum(np.logical_and(pred != 1, gt == 1))
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)
print(f"F1 Score (sample): {f1_score:.4f}")

# Calculate F1 score segment-wise
fn_seg, fp_seg, tp_seg = segment_evaluation(pred, gt, 1, 0.5, debug_plot=True)
precision = tp_seg / (tp_seg + fp_seg)
recall = tp_seg / (tp_seg + fn_seg)
f1_score_seg = 2 * precision * recall / (precision + recall)
print(f"F1 Score (seg): {f1_score_seg:.4f}")

# Calculate F1 score segment-wise
fn_seg, fp_seg, tp_seg = segment_evaluation(pred, gt, 2, 0.5, debug_plot=True)
precision = tp_seg / (tp_seg + fp_seg)
recall = tp_seg / (tp_seg + fn_seg)
f1_score_seg = 2 * precision * recall / (precision + recall)
print(f"F1 Score (seg): {f1_score_seg:.4f}")

# Calculate F1 score segment-wise
fn_seg, fp_seg, tp_seg = segment_evaluation(pred, gt, 3, 0.5, debug_plot=True)
precision = tp_seg / (tp_seg + fp_seg)
recall = tp_seg / (tp_seg + fn_seg)
f1_score_seg = 2 * precision * recall / (precision + recall)
print(f"F1 Score (seg): {f1_score_seg:.4f}")
