import numpy as np
import matplotlib.pyplot as plt
from components.evaluation import segment_evaluation

# Load gt_intervals.npy and pred_intervals.npy
gt_intervals = np.load("gt_intervals.npy")
pred_intervals = np.load("pred_intervals.npy")

# Determine sequence length
max_len = max(gt_intervals.max(), pred_intervals.max()) + 1

# Initialize sequences
gt = np.zeros(max_len, dtype=int)
pred = np.zeros(max_len, dtype=int)

# Mark ground truth intervals as 1
for start, end in gt_intervals:
    gt[start:end] = 1

# Mark predicted intervals as 1
for start, end in pred_intervals:
    pred[start:end] = 1

# Perform segment evaluation
fn_seg, fp_seg, tp_seg = segment_evaluation(pred, gt, 1, 0.5, debug_plot=True)
