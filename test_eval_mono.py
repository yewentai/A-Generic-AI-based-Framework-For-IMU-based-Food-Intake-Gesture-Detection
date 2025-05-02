#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Test Evaluation Script (Binary Case)
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-02
Description : This script evaluates binary segmentation predictions for IMU
              gesture recognition. It:
              1. Compares ground truth vs predicted labels
              2. Visualizes both sequences with annotated error types
              3. Computes sample-wise F1 score
              4. Computes segment-wise F1 score using overlap-based evaluation
===============================================================================
"""


import numpy as np
import matplotlib.pyplot as plt
from components.evaluation import segment_evaluation

# Ground Truth sequence
# fmt: off
gt = np.array([
    0, 0, 1, 1, 1, 0, 0,  # Normal gesture
    0, 0, 0, 0, 0, 0, 0,  # No gesture (for insertion test)
    1, 1, 1, 1, 0, 0, 0,  # Gesture for underfill and overfill tests
    0, 1, 1, 0, 1, 1, 0,  # Two gestures for merge test
    1, 1, 1, 1, 1, 1, 1,  # Long gesture for fragmentation test
    0, 0, 1, 1, 1, 0, 0,  # Gesture for deletion test
    1, 1, 0, 1, 1, 1, 0
])

# Prediction sequence
pred = np.array([
    0, 1, 1, 1, 1, 1, 0,  # Overfill
    0, 1, 1, 0, 0, 0, 0,  # Insertion
    1, 1, 0, 0, 0, 0, 0,  # Underfill
    0, 1, 1, 1, 1, 1, 0,  # Merge
    1, 0, 1, 0, 1, 0, 1,  # Fragmentation
    0, 0, 0, 0, 0, 0, 0,  # Deletion
    0, 1, 1, 1, 0, 1, 1   # Mismatch
])
# fmt: on

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
tp = np.sum(np.logical_and(pred == 1, gt == 1))
fp = np.sum(np.logical_and(pred == 1, gt == 0))
fn = np.sum(np.logical_and(pred == 0, gt == 1))
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)
print(f"Sample-wise - TP: {tp}, FP: {fp}, FN: {fn}")
print(f"F1 Score (sample): {f1_score:.4f}")

# Calculate F1 score segment-wise
fn_seg, fp_seg, tp_seg = segment_evaluation(pred, gt, 1, 0.5, debug_plot=True)
precision = tp_seg / (tp_seg + fp_seg)
recall = tp_seg / (tp_seg + fn_seg)
f1_score_seg = 2 * precision * recall / (precision + recall)
print(f"Segment-wise - TP: {tp_seg}, FP: {fp_seg}, FN: {fn_seg}")
print(f"F1 Score (seg): {f1_score_seg:.4f}")
